import pygame
import numpy as np
import sys
import math
import random
from Agent import Agent

# Initialize pygame
pygame.init()

# Window dimensions
WIDTH = 360
HEIGHT = 800

# Create window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cloud Chamber Wallpaper")

# Create a 3D array to store RGB values for each pixel (width x height x 3)
# Initialize with black (0, 0, 0)
pixel_grid = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)




def create_agent(mode="alpha"):
    if mode == "alpha":
        x = np.random.randint(0, WIDTH)
        y = np.random.randint(0, HEIGHT)
        decay = np.random.uniform(8, 18)
        angle = np.random.uniform(0, 2 * math.pi)
        size = np.random.randint(7, 12)
        velocity = np.random.uniform(1500, 3000)
        color = (255, 255, 255)

    elif mode == "beta":
        x = np.random.randint(0, WIDTH)
        y = np.random.randint(0, HEIGHT)
        decay = np.random.uniform(7, 12)
        angle = np.random.uniform(0, 2 * math.pi)
        size = np.random.randint(2, 6)
        velocity = np.random.uniform(2500, 5000)
        color = (255, 255, 255)

    return Agent(x, y, decay, angle, size, velocity, color)

def draw_pixel_grid():
    # Create a surface from the numpy array
    surface = pygame.surfarray.make_surface(pixel_grid)
    screen.blit(surface, (0, 0))


def diffuse_grid():
    global pixel_grid
    # Apply a simple diffusion effect using NumPy operations
    kernel = np.array([[0.02, 0.05, 0.02],
                       [0.05, 0.72, 0.05],
                       [0.02, 0.05, 0.02]])
    
    # Create padded version of the grid
    padded = np.pad(pixel_grid, ((1, 1), (1, 1), (0, 0)), mode='constant')
    
    # Apply the kernel using numpy operations
    result = np.zeros_like(pixel_grid, dtype=np.float32)
    
    # For each position in the kernel, multiply and add
    for i in range(3):
        for j in range(3):
            result += padded[i:i+WIDTH, j:j+HEIGHT, :] * kernel[i, j]
    
    # Clip values and convert back to uint8
    pixel_grid = np.clip(result, 0, 255).astype(np.uint8)

def evaporate_grid():
    global pixel_grid
    # Evaporate the grid by reducing the intensity of each pixel
    evaporation_rate = 0.995  # Adjust this value to control evaporation speed
    pixel_grid = (pixel_grid * evaporation_rate).astype(np.uint8)

 
 
#Agent(180, 60, 10, math.pi*.5, 2, velocity=2500.0, color=(255, 0, 0))
agent_list = [create_agent() for _ in range(1)]
 


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
    diffuse_grid()          # Apply diffusion effect
    evaporate_grid()        # Apply evaporation effect
    
    for agent in agent_list:
        # Update each agent
        if not agent.update(1/60, pixel_grid):
            agent_list.remove(agent)
           
    if random.random() < 0.5:  # Randomly create new agents
        if random.random() < 0.1:  # 10% chance for alpha, 90% for beta
            agent_list.append(create_agent("alpha"))
        else:
            agent_list.append(create_agent("beta"))



    # Update display
    pygame.display.flip()
    
    # Cap at 60 FPS
    clock.tick(60)

# Clean up
pygame.quit()
sys.exit()