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
turned_card = CardSprite(HIDDEN_CARD, SCREEN_WIDTH*5/6, SCREEN_HEIGHT/2-CARD_HEIGHT/2)
enemy_hand = []
briscola = CardSprite(session.briscola, SCREEN_WIDTH*5/6-CARD_HEIGHT/4*3, SCREEN_HEIGHT/2-CARD_WIDTH/2, orizzontale=True)
card1 = None
card2 = None
hand = []
enemy_hand = []
player1 = session.players[0]
player2 = session.players[1]
for n, card in enumerate(session.players[0].hand):
    hand.append(CardSprite(card, 250+100*n, SCREEN_HEIGHT-0.8*CARD_HEIGHT))

for n, card in enumerate(session.players[1].hand):
    enemy_hand.append(CardSprite(HIDDEN_CARD, 250+100*n, -0.2*CARD_HEIGHT))
while running:
    
    if (session.turno == 1 or card1 != None) and card2 == None:
        pygame.time.wait(300)
        card2 = session.players[1].hand[session.players[1].randmove()] 


    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and card1 == None:
            if event.key == pygame.K_1:
                card1 = player1.hand.pop(0)
            if event.key == pygame.K_2:
                if len(hand) > 1:
                    card1 = player1.hand.pop(1)
            if event.key == pygame.K_3:
                if len(hand) > 2:
                    card1 = player1.pop(2)

        if event.type == pygame.QUIT:  # Close window
            running = False

    # Fill screen with black
    


    for n, card_sprite in enumerate(hand):
        screen.blit(card_sprite.image, card_sprite.rect.topleft)
        screen.blit(font.render(f"{n+1}", True, (255,0,0)), card_sprite.rect.topleft)

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


    # Update display
    pygame.display.flip()
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()