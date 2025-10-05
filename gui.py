import pygame, sys
import game, random

"""Test features"""
SMALL_DECK = False
WAITS = True
CARTE_SCOPERTE = False

"""Pygame parameters"""
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

CARD_WIDTH = 80
CARD_HEIGHT = 150

SMALL_FONT = 32

"""Hidden Card filename"""
HIDDEN_CARD = ("Z",0)

def randmove(hand):
    """Produce valid randomized moves."""
    if len(hand) <= 1:
        return 0
    return random.randint(0, len(hand)-1)

pygame.init()
clock = pygame.time.Clock()

class CardSprite(pygame.sprite.Sprite):
    def __init__(self, card, x , y, orizzontale = False):
        super().__init__()
        self.image = pygame.transform.smoothscale(pygame.image.load(f"imgs/{card[1]}{card[0]}.jpg").convert_alpha(), (CARD_WIDTH, CARD_HEIGHT))
        if orizzontale:
            self.image = pygame.transform.rotate(self.image, 90)
        self.rect = self.image.get_rect(topleft=(x, y))
        self.round_corners()

    def round_corners(self):
        w, h = self.image.get_size()
        mask = pygame.Surface((w, h), pygame.SRCALPHA)

        # Draw a filled rounded rectangle as a mask
        pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h), border_radius=int((w*0.1)//1))

        # Copy original image with mask applied
        result = pygame.Surface((w, h), pygame.SRCALPHA)
        result.blit(self.image, (0, 0))
        result.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        self.image = result

def drawCards(hand, enemy_hand):
    hand_sprites = []
    enemy_hand_sprites = []
    for n, card in enumerate(hand):
        hand_sprites.append(CardSprite(card, 250+100*n, SCREEN_HEIGHT-0.8*CARD_HEIGHT))
    for n, card in enumerate(enemy_hand):
        if CARTE_SCOPERTE:
            enemy_hand_sprites.append(CardSprite(card, 250+100*n, -0.2*CARD_HEIGHT))
        else:
            enemy_hand_sprites.append(CardSprite(HIDDEN_CARD, 250+100*n, -0.2*CARD_HEIGHT))

    return hand_sprites, enemy_hand_sprites

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Briscola")

font = pygame.font.Font(None, SMALL_FONT)

screen.fill((53,101,77))
    
session = game.Game()
session.reset()

if SMALL_DECK:
    session.deck = session.deck[32:]

running = True
hand = []
enemy_hand = []
turned_card = CardSprite(HIDDEN_CARD, SCREEN_WIDTH*5/6, SCREEN_HEIGHT/2-CARD_HEIGHT/2)
briscola = CardSprite(session.briscola, SCREEN_WIDTH*5/6-CARD_HEIGHT/4*3, SCREEN_HEIGHT/2-CARD_WIDTH/2, orizzontale=True)
card1 = None
card2 = None
hand, enemy_hand = drawCards(session.hands[0], session.hands[1])
wait = 0



while running:
    screen.fill((53,101,77))        

    score1, score2 = session.count_points()

    screen.blit(font.render(f"P2: {score2} pts", True, (255,255,255)), (50,20))
    screen.blit(font.render(f"P1: {score1} pts", True, (255,255,255)), (50,SCREEN_HEIGHT-50))
    
    if (session.turno == 1 or card1 != None) and card2 == None:
        move = randmove(session.hands[1])
        card2 = session.hands[1].pop(move)
        enemy_hand.pop(move)
        card2sprite = CardSprite(card2, SCREEN_WIDTH/2-CARD_WIDTH*0.2, SCREEN_HEIGHT/2-CARD_HEIGHT/2)        


    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and card1 == None:
            if not card1:
                if event.key == pygame.K_1:
                    card1 = session.hands[0].pop(0)
                    hand.pop(0)
                if event.key == pygame.K_2:
                    if len(hand) > 1:
                        card1 = session.hands[0].pop(1)
                        hand.pop(1)
                if event.key == pygame.K_3:
                    if len(hand) > 2:
                        card1 = session.hands[0].pop(2)
                        hand.pop(2)

        if event.type == pygame.QUIT:  # Close window
            running = False


    for n, card_sprite in enumerate(hand):
        screen.blit(card_sprite.image, card_sprite.rect.topleft)
    for n in range(3):
        if len(session.hands[0]) >= n+1 or len(session.deck)>0:
            screen.blit(font.render(f"{n+1}", True, (255,255,255)), (250+100*n+CARD_WIDTH/2-SMALL_FONT/2, SCREEN_HEIGHT-CARD_HEIGHT))
        
    for card_sprite in enemy_hand:
        screen.blit(card_sprite.image, card_sprite.rect.topleft)

    if len(session.deck) >= 1:
        screen.blit(briscola.image, briscola.rect.topleft)
    
    if len(session.deck) >= 2:
        screen.blit(turned_card.image, turned_card.rect.topleft)
    
    if card1:
        card1sprite = CardSprite(card1, SCREEN_WIDTH/2-CARD_WIDTH*1.5, SCREEN_HEIGHT/2-CARD_HEIGHT/2)
        screen.blit(card1sprite.image, card1sprite.rect.topleft)

    if card2:
        card2sprite = CardSprite(card2, SCREEN_WIDTH/2-CARD_WIDTH*0.2, SCREEN_HEIGHT/2-CARD_HEIGHT/2)
        screen.blit(card2sprite.image, card2sprite.rect.topleft)


    if card1 and card2:
        if WAITS:
            wait = 600
        winner = session.compare_hands(card1, card2)
        session.taken[winner] += (card1, card2)
        session.turno = winner
        card1= None
        card2 = None


        session.draw()
        hand, enemy_hand = drawCards(session.hands[0], session.hands[1])

    if session.check_finished():
        winner = session.check_winner()
        screen.fill((53,101,77))
        score1, score2 = session.count_points()
        if winner != -1:
            screen.blit(font.render(f"Player {winner+1} wins, {score1}-{score2}", True, (255,255,255)), (SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
        else:
            screen.blit(font.render(f"DRAW!", True, (255,255,255)), (SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
        pygame.display.flip()
        wait = 1000
        running = False

        

    # Update display
    pygame.display.flip()
    clock.tick(60)
    pygame.time.wait(wait)
    wait = 0

# Quit pygame
pygame.quit()
sys.exit()