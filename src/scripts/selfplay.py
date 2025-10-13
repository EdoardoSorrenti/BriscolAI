import logging
from pathlib import Path
from time import perf_counter

import torch

from trainconfig import *


torch.set_default_device(device)
torch.set_default_dtype(dtype)

if tensor_f32:  
    torch.set_float32_matmul_precision('high')

from model import PolicyNetwork
from tensorgames import Games


class ColorFormatter(logging.Formatter):
    """Adds ANSI colors to log levels for terminal readability."""

    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }

    RESET = "\033[0m"

    def format(self, record):
        levelname = record.levelname
        color = self.COLORS.get(record.levelno)

        if color:
            record.levelname = f"{color}{levelname}{self.RESET}"

        try:
            return super().format(record)
        finally:
            record.levelname = levelname


def setup_logging():
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    formatter = logging.Formatter(log_format, datefmt=log_datefmt)

    handlers = []

    stream_handler = logging.StreamHandler()
    try:
        stream_stream = stream_handler.stream
        stream_supports_color = getattr(stream_stream, "isatty", lambda: False)()
    except Exception:
        stream_supports_color = False

    if stream_supports_color:
        stream_handler.setFormatter(ColorFormatter(log_format, datefmt=log_datefmt))
    else:
        stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    if log_to_file:
        log_path = Path(log_file_path).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(level)
    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger.debug("Logging initialized.")


setup_logging()
logger = logging.getLogger(__name__)


# Instantiate training and evaluation models
train_model = PolicyNetwork()
eval_model = PolicyNetwork()

try:
    state_dict = torch.load(save_path)
    train_model.load_state_dict(state_dict)
    eval_model.load_state_dict(state_dict)
    logger.info("Loaded model weights from %s", save_path)
except FileNotFoundError:
    logger.info("No saved model found at %s, starting fresh.", save_path)

try:
    train_model = torch.compile(train_model, fullgraph=False)
except Exception as e:
    logger.warning("torch.compile failed, continuing without compile optimization", exc_info=e)

def _get_module(m):
    return getattr(m, "_orig_mod", m)

train_model.to(device, dtype=dtype)
eval_model.to(device, dtype=dtype)
eval_model.eval()

optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate)

self_play_train_interval = globals().get("self_play_train_interval", log_freq)
self_play_eval_batches = globals().get("self_play_eval_batches", 50)
self_play_win_threshold = globals().get("self_play_win_threshold", 0.55)


def get_states(session, player_id=0):
    """Returns the current state of the game as a tensor."""
    opponent_id = 1 - player_id

    my_hand = session.hands[player_id]
    my_taken = session.taken[player_id]
    opponent_taken = session.taken[opponent_id]
    briscole = session.briscole_cards
    on_table = session.on_table

    return torch.cat([my_hand, my_taken, opponent_taken, briscole, on_table], dim=1)


def get_action_masks(session, player_id=0):
    """Returns a mask of valid actions for the current player."""
    return session.hands[player_id].bool()


def get_moves(session, model, player_id=0, mask=None):
    """Returns a move and associated statistics for the current player."""
    states = get_states(session, player_id)
    masks = get_action_masks(session, player_id)

    if mask is not None:
        states = states[mask]
        masks = masks[mask]

    probs = model(states, masks)
    probs_fp32 = probs.to(torch.float32)
    actions = torch.multinomial(probs_fp32, 1).squeeze(1)
    selected_probs = probs_fp32.gather(1, actions.unsqueeze(1)).squeeze(1)
    log_probs = selected_probs.clamp_min(1e-12).log()
    entropy = -(probs_fp32 * probs_fp32.clamp_min(1e-12).log()).sum(dim=1)
    return actions, log_probs.to(probs.dtype), entropy.to(probs.dtype)


def tensorloop_model_vs_model(games, model_train, model_eval, collect_log_probs=True):
    """Plays a batch of games with the training model against the evaluation model."""
    n_games = games.N
    device = games.hands.device
    log_prob_sums = (
        torch.zeros(n_games, device=device, dtype=torch.float32)
        if collect_log_probs
        else None
    )
    entropy_sums = (
        torch.zeros(n_games, device=device, dtype=torch.float32)
        if collect_log_probs
        else None
    )

    card1 = torch.full((n_games,), -1, dtype=torch.int64)
    card2 = torch.full((n_games,), -1, dtype=torch.int64)

    for i in range(20):
        (turn0_mask, turn0_idx), (turn1_mask, turn1_idx) = games.split_turns()
        games.refresh_random_buffer()

        # Model player (P0) leads
        if turn0_idx.numel():
            actions0, log_probs0, entropy0 = get_moves(games, model_train, player_id=0, mask=turn0_mask)
            card1.index_copy_(0, turn0_idx, actions0)
            games.place_on_table(turn0_idx, actions0)
            if collect_log_probs:
                log_prob_sums.index_add_(0, turn0_idx, log_probs0.to(torch.float32))
                entropy_sums.index_add_(0, turn0_idx, entropy0.to(torch.float32))

            with torch.no_grad():
                actions_eval, _, _ = get_moves(games, model_eval, player_id=1, mask=turn0_mask)
            card2.index_copy_(0, turn0_idx, actions_eval)

        # Evaluation player (P1) leads
        if turn1_idx.numel():
            with torch.no_grad():
                actions_eval, _, _ = get_moves(games, model_eval, player_id=1, mask=turn1_mask)
            card2.index_copy_(0, turn1_idx, actions_eval)
            games.place_on_table(turn1_idx, actions_eval)

            actions1, log_probs1, entropy1 = get_moves(games, model_train, player_id=0, mask=turn1_mask)
            card1.index_copy_(0, turn1_idx, actions1)
            if collect_log_probs:
                log_prob_sums.index_add_(0, turn1_idx, log_probs1.to(torch.float32))
                entropy_sums.index_add_(0, turn1_idx, entropy1.to(torch.float32))

        games.remove_played_cards(card1, card2)
        winners = games.compare_cards(card1, card2)
        games.record_taken_cards(winners, card1, card2)
        games.clear_table()
        games.draw()

    if optimize_points:
        p1s, p2s, winners = games.check_winners()
        rewards = (p1s - p2s) / 60
    else:
        _, _, winners = games.check_winners()
        rewards = torch.zeros_like(winners).masked_fill_(winners == 0, 1.0).masked_fill_(winners == 1, -1.0)

    if collect_log_probs:
        policy_losses = -log_prob_sums * rewards.to(torch.float32)
        entropy_mean = entropy_sums.mean()
        return policy_losses.mean(), entropy_mean, winners

    zero = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    return zero, zero, winners


def tensorloop_model_vs_random(games, model_train, collect_log_probs=False):
    """Plays a batch of games with the training model against a random opponent."""
    n_games = games.N
    device = games.hands.device
    log_prob_sums = (
        torch.zeros(n_games, device=device, dtype=torch.float32)
        if collect_log_probs
        else None
    )
    entropy_sums = (
        torch.zeros(n_games, device=device, dtype=torch.float32)
        if collect_log_probs
        else None
    )

    card1 = torch.full((n_games,), -1, dtype=torch.int64)
    card2 = torch.full((n_games,), -1, dtype=torch.int64)

    for i in range(20):
        (turn0_mask, turn0_idx), (turn1_mask, turn1_idx) = games.split_turns()
        games.refresh_random_buffer()

        if turn0_idx.numel():
            actions0, log_probs0, entropy0 = get_moves(games, model_train, player_id=0, mask=turn0_mask)
            card1.index_copy_(0, turn0_idx, actions0)
            games.place_on_table(turn0_idx, actions0)
            if collect_log_probs:
                log_prob_sums.index_add_(0, turn0_idx, log_probs0.to(torch.float32))
                entropy_sums.index_add_(0, turn0_idx, entropy0.to(torch.float32))

            card2_turn0 = games.pick_random_cards(player_id=1, indices=turn0_idx)
            card2.index_copy_(0, turn0_idx, card2_turn0)

        if turn1_idx.numel():
            card2_turn1 = games.pick_random_cards(player_id=1, indices=turn1_idx)
            card2.index_copy_(0, turn1_idx, card2_turn1)
            games.place_on_table(turn1_idx, card2_turn1)

            actions1, log_probs1, entropy1 = get_moves(games, model_train, player_id=0, mask=turn1_mask)
            card1.index_copy_(0, turn1_idx, actions1)
            if collect_log_probs:
                log_prob_sums.index_add_(0, turn1_idx, log_probs1.to(torch.float32))
                entropy_sums.index_add_(0, turn1_idx, entropy1.to(torch.float32))

        games.remove_played_cards(card1, card2)
        winners = games.compare_cards(card1, card2)
        games.record_taken_cards(winners, card1, card2)
        games.clear_table()
        games.draw()

    if optimize_points:
        p1s, p2s, winners = games.check_winners()
        rewards = (p1s - p2s) / 60
    else:
        _, _, winners = games.check_winners()
        rewards = torch.zeros_like(winners).masked_fill_(winners == 0, 1.0).masked_fill_(winners == 1, -1.0)

    if collect_log_probs:
        policy_losses = -log_prob_sums * rewards.to(torch.float32)
        entropy_mean = entropy_sums.mean()
        return policy_losses.mean(), entropy_mean, winners

    zero = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    return zero, zero, winners


def evaluate_models(eval_games, candidate_model, reference_model, eval_batches):
    """Evaluates candidate_model against reference_model and returns candidate win rate."""
    baseline_wins = 0
    baseline_total = 0
    random_wins = 0
    random_total = 0

    candidate_was_training = candidate_model.training
    candidate_model.eval()
    reference_model.eval()

    with torch.no_grad():
        for _ in range(eval_batches):
            eval_games.reset()
            _, _, winners = tensorloop_model_vs_model(eval_games, candidate_model, reference_model, collect_log_probs=False)
            baseline_wins += winners.eq(0).sum().item()
            baseline_total += winners.numel() - winners.eq(2).sum().item()

        for _ in range(eval_batches):
            eval_games.reset()
            _, _, winners = tensorloop_model_vs_random(eval_games, candidate_model, collect_log_probs=False)
            random_wins += winners.eq(0).sum().item()
            random_total += winners.numel() - winners.eq(2).sum().item()

    candidate_model.train(candidate_was_training)
    baseline_rate = baseline_wins / baseline_total if baseline_total else 0.0
    random_rate = random_wins / random_total if random_total else 0.0
    return baseline_rate, random_rate


def promote_candidate(candidate_model, reference_model):
    """Copies the candidate model weights into the reference model."""
    source = _get_module(candidate_model)
    target = _get_module(reference_model)
    target.load_state_dict(source.state_dict())
    reference_model.to(device, dtype=dtype)
    reference_model.eval()
    logger.info("Promoted candidate model to new evaluation baseline.")


def train_self_play(total_batches, batch_size):
    start_time_total = perf_counter()
    games = Games(batch_size, alternate_turns=True)
    eval_games = Games(batch_size, alternate_turns=True)

    logger.info(
        "Starting self-play training | target batches: %s | batch size: %d | log freq: %d",
        "unbounded" if total_batches is None else total_batches,
        batch_size,
        log_freq,
    )

    batch = 0
    wins = 0
    losses = 0
    tot_games = 0
    batches_since_eval = 0

    while True:
        batch += 1
        batches_since_eval += 1

        try:
            optimizer.zero_grad()
            games.reset()

            policy_loss, entropy_mean, winners_array = tensorloop_model_vs_model(
                games,
                train_model,
                eval_model,
                collect_log_probs=True,
            )

            total_loss = None
            if torch.is_tensor(policy_loss):
                total_loss = policy_loss - entropy_coef * entropy_mean
                total_loss.backward()
                optimizer.step()

            if batch % log_freq == 0 and batch > 0:
                p1s, p2s, winners_array = games.check_winners()
                wins = winners_array.eq(0).sum()
                draws = winners_array.eq(2).sum()
                tot_games = winners_array.size(0)

                win_rate = (wins.sum().item() / (tot_games-draws.sum().item())) * 100

                end_time = perf_counter()
                tot_time = end_time - start_time_total
                games_processed = batch_size * log_freq
                games_per_second = games_processed / tot_time if tot_time > 0 else 0

                avg_p1 = p1s.mean().item()
                avg_p2 = p2s.mean().item()

                logger.info(
                    "Batch %d\n"
                    "  Win Rate:  %.3f%%\n"
                    "  Avg Points: %.3f vs %.3f\n"
                    "  Speed:     %.0f games/sec\n"
                    "  Loss:      %.4f\n"
                    "  Entropy:   %.4f\n"
                    "  Total:     %.4f",
                    batch,
                    win_rate,
                    avg_p1,
                    avg_p2,
                    games_per_second,
                    policy_loss.item() if torch.is_tensor(policy_loss) else float('nan'),
                    entropy_mean.item() if torch.is_tensor(entropy_mean) else float('nan'),
                    total_loss.item() if torch.is_tensor(total_loss) else float('nan'),
                )

                if torch.is_tensor(policy_loss):
                    logger.debug(
                        "Policy loss (x1e3): %.4f | Entropy mean: %.4f | Total loss: %.4f",
                        policy_loss.item() * 1000,
                        entropy_mean.item(),
                        total_loss.item() if torch.is_tensor(total_loss) else float('nan'),
                    )

                start_time_total = perf_counter()
                wins = 0
                tot_games = 0

            if batch % save_freq == 0 and batch > 0:
                if save_results:
                    module_to_save = _get_module(train_model)
                    torch.save(module_to_save.state_dict(), save_path)
                    logger.info("Model checkpoint saved to %s", save_path)

            if batches_since_eval >= self_play_train_interval:
                eval_win_rate, random_win_rate = evaluate_models(
                    eval_games,
                    train_model,
                    eval_model,
                    self_play_eval_batches,
                )
                logger.info(
                    "Evaluation over %d batches\n"
                    "  vs Baseline: %.2f%% wins\n"
                    "  vs Random:   %.2f%% wins\n\n",
                    self_play_eval_batches,
                    eval_win_rate * 100,
                    random_win_rate * 100,
                )

                if eval_win_rate > self_play_win_threshold:
                    promote_candidate(train_model, eval_model)

                batches_since_eval = 0

        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Saving final model...")
            break

    if save_results:
        module_to_save = _get_module(train_model)
        torch.save(module_to_save.state_dict(), save_path)
        logger.info("Final model saved to %s", save_path)


if __name__ == '__main__':
    train_self_play(batches, batch_size)
