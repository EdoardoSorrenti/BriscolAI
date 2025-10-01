import pygame, sys
import game


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

CARD_WIDTH = 80
CARD_HEIGHT = 150

HIDDEN_CARD = ("Z",0)

pygame.init()
clock = pygame.time.Clock()

class CardSprite(pygame.sprite.Sprite):
    def __init__(self, card, x , y, orizzontale = False):
        super().__init__()
        self.image = pygame.transform.smoothscale(pygame.image.load(f"imgs/{card[1]}{card[0]}.jpg").convert_alpha(), (CARD_WIDTH, CARD_HEIGHT))
        if orizzontale:
            self.image = pygame.transform.rotate(self.image, 90)
        self.rect = self.image.get_rect(topleft=(x, y))


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Briscola")

font = pygame.font.Font(None, 48)

screen.fill((53,101,77))
    
session = game.Game.simulationInit()
running = True
hand = []
turned_card = CardSprite(HIDDEN_CARD, SCREEN_WIDTH*4/5, SCREEN_HEIGHT/2-CARD_HEIGHT/2)
enemy_hand = []
briscola = CardSprite(session.briscola, SCREEN_WIDTH*4/5-CARD_HEIGHT/4*3, SCREEN_HEIGHT/2-CARD_WIDTH/2, orizzontale=True)
while running:
    hand = []
    enemy_hand = []
    move = None
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                move = 0
            if event.key == pygame.K_2:
                if len(hand) > 1:
                    move = 1
            if event.key == pygame.K_3:
                if len(hand) > 2:
                    move = 2

        if event.type == pygame.QUIT:  # Close window
            running = False

    # Fill screen with black
    
    for n, card in enumerate(session.players[0].hand):
        hand.append(CardSprite(card, 250+100*n, SCREEN_HEIGHT-0.8*CARD_HEIGHT))
    
    for n, card in enumerate(session.players[1].hand):
        enemy_hand.append(CardSprite(HIDDEN_CARD, 250+100*n, -0.2*CARD_HEIGHT))

    for n, card_sprite in enumerate(hand):
        screen.blit(card_sprite.image, card_sprite.rect.topleft)
        screen.blit(font.render(f"{n+1}", True, (255,0,0)), card_sprite.rect.topleft)

    for card_sprite in enemy_hand:
        screen.blit(card_sprite.image, card_sprite.rect.topleft)

    if len(session.deck) >= 1:
        screen.blit(briscola.image, briscola.rect.topleft)
    
    if len(session.deck) >= 2:
        screen.blit(turned_card.image, turned_card.rect.topleft)
    




    # Update display
    pygame.display.flip()
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()