import pygame
import numpy as np
import sys
import math
from Agent import Agent

# Initialize pygame
pygame.init()

# Window dimensions
WIDTH = 360
HEIGHT = 720

# Create window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cloud Chamber Wallpaper")

# Create a 3D array to store RGB values for each pixel (width x height x 3)
# Initialize with black (0, 0, 0)
pixel_grid = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)



agent_list = [Agent(180, 60, 5.0, math.pi/6, 2, velocity=2500.0, color=(255, 0, 0))]



def draw_pixel_grid():
    # Create a surface from the numpy array
    surface = pygame.surfarray.make_surface(pixel_grid)
    screen.blit(surface, (0, 0))


 


# Game loop
running = True
clock = pygame.time.Clock()
# Main game loop
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    
    # Draw everything
    screen.fill((0, 0, 0))  # Clear screen
    draw_pixel_grid()       # Draw pixel grid
    
    for agent in agent_list:
        # Update each agent
        if not agent.update(1/60, pixel_grid):
            agent_list.remove(agent)
    
    
    # Update display
    pygame.display.flip()
    
    # Cap at 60 FPS
    clock.tick(60)

# Clean up
pygame.quit()
sys.exit()