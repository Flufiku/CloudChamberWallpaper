import pygame
import sys
import random
import numpy as np
import threading
import math
from Agent import Agent
from Hash import hash

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 360, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Slime Wallpaper Proof of Concept")

# Create a list of agents
def create_agents(n):
    agents = []
    for _ in range(n):
        # Random position and angle
        x = random.randint(0, WIDTH - 1)
        y = random.randint(0, HEIGHT - 1)
        angle = random.uniform(0, 2 * math.pi)  # Random angle in radians
        speed = 1
        agents.append(Agent(x, y, angle, speed))
    return agents

# Function to update agent position in a separate thread
def update_agent(agent, agent_index, width, height):
    # Generate hash value
    random_val = hash(int(agent.y * width + agent.x + agent_index))
    
    # Calculate new position
    new_x = agent.x + math.cos(agent.angle) * agent.speed
    new_y = agent.y + math.sin(agent.angle) * agent.speed
    
    # Check if out of bounds
    out_of_bounds = False
    if new_x < 0:
        new_x = 0
        out_of_bounds = True
    elif new_x >= width:
        new_x = width - 1
        out_of_bounds = True
    
    if new_y < 0:
        new_y = 0
        out_of_bounds = True
    elif new_y >= height:
        new_y = height - 1
        out_of_bounds = True
    
    # Update position
    agent.x = new_x
    agent.y = new_y
    
    # Generate new angle if out of bounds
    if out_of_bounds:
        # Generate new angle based on hash
        max_int = 0x7FFFFFFF  # Maximum value for a signed 32-bit integer
        agent.angle = (random_val / max_int) * 2 * math.pi
        
    grid[int(agent.x), int(agent.y)] = [255, 255, 255]  # Set pixel to white


# Update all agents using threading
def update_agents_threaded(agents, width, height):
    threads = []
    for i, agent in enumerate(agents):
        thread = threading.Thread(target=update_agent, args=(agent, i, width, height))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

# Create a 2D list (grid) of RGB values
def create_grid(width, height):
    # Initialize with zeros (black)
    return np.zeros((width, height, 3), dtype=np.uint8)



# Draw the grid to the screen
def draw_grid(screen, grid):
    # Convert numpy array to pygame surface
    surface = pygame.surfarray.make_surface(grid)
    screen.blit(surface, (0, 0))





# Number of agents
n_agents = 300
agents = create_agents(n_agents)



grid = create_grid(WIDTH, HEIGHT)
# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update agent positions using threading
    update_agents_threaded(agents, WIDTH, HEIGHT)
    
    # Draw the grid to the screen
    draw_grid(screen, grid)
    
    pygame.display.flip()

pygame.quit()
sys.exit()