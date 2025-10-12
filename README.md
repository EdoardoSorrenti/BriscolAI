# BriscolAI

An AI that plays Briscola (classic Italian card game), with a simple PyTorch policy network, self-play training scripts, and a Pygame GUI to play against the model.

## Features
- PyTorch policy network and training loop (self-play)
- Fast tensorized environment for many parallel games
- Pygame GUI to challenge the agent
- Utilities for model dtype conversion and lookup tables

## Repository layout
- `src/scripts/` — training, evaluation, and utilities
  - `selfplay.py`, `tensorgames.py`, `trainconfig.py`
  - `utils/` helper scripts (lookup tables, miscellaneous)
- `src/briscolai/gui/` — Pygame UI (play against the agent)
  - `gui.py`, `game.py`, `model.py`, `utils.py`, `assets/`
- `models/` — model definition, example/pretrained weights

## Requirements
- Python 3.10–3.13
- PyTorch (CUDA optional). Install from pytorch.org for your platform/GPU.
- Pygame for the GUI

Install Python deps:

```bash
pip install -r requirements.txt
# If torch fails or you want a specific CUDA build, follow https://pytorch.org/get-started/locally/
```

## Quickstart

Train via self-play (writes logs and checkpoints under `src/scripts/`):

```bash
# From repo root
cd src/scripts
python selfplay.py
```

Launch the GUI and play against the agent:

```bash
# Note: imports in the GUI are relative to this folder
cd src/briscolai/gui
python gui.py
```

If you have your own weights file, place it under `src/briscolai/gui/weights/model.pth` or update the path in the GUI code.
```

## License
This project is licensed under the MIT License — see `LICENSE` for details.

## Acknowledgments
- Briscola rules and assets are included for demonstration. Ensure you have the rights to any assets you add.
