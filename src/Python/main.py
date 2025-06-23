import pygame
import sys

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 360, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Slime Wallpaper Proof of Concept")

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))  # Fill screen with black
    pygame.display.flip()

pygame.quit()
sys.exit()