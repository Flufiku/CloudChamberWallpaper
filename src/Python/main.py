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
WIDTH, HEIGHT = 1080, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Slime Wallpaper Proof of Concept")

# Create a list of agents
def create_agents(n):
    agents = []
    for i in range(n):
        # Random position and angle
        box_size = 200

        x = random.randint(WIDTH//2-box_size, WIDTH//2+box_size)
        y = random.randint(HEIGHT//2-box_size, HEIGHT//2+box_size)
        angle = random.uniform(0, 2 * math.pi)  # Random angle in radians
        speed = 1
        color = (255, 255, 255)  # Default color (white)
        turn_rate = 0.1  # Rate at which the agent can turn
        agents.append(Agent(x, y, angle, speed, color, turn_rate))
    return agents

# Function to update agent position in a separate thread
def update_agent(agent, agent_index, width, height):
    # Generate hash value
    random_val = hash(int(agent.y * width + agent.x + agent_index))
    
    # Define the box size for similarity calculation
    box_size = 10
    half_box = box_size // 2
    
    # Calculate the box boundaries, ensuring they stay within grid boundaries
    box_left = max(0, int(agent.x) - half_box)
    box_right = min(width - 1, int(agent.x) + half_box)
    box_top = max(0, int(agent.y) - half_box)
    box_bottom = min(height - 1, int(agent.y) + half_box)
    
    # Variables to track the center of similarity
    total_similarity = 0.0
    weighted_x_sum = 0.0
    weighted_y_sum = 0.0
    
    # Iterate through each pixel in the box
    for x in range(box_left, box_right + 1):
        for y in range(box_top, box_bottom + 1):
            # Skip the agent's own position
            if x == int(agent.x) and y == int(agent.y):
                continue
                
            # Calculate similarity for each channel
            similarity_sum = 0
            for c in range(3):  # 3 channels: R, G, B
                # Get the channel values
                agent_channel = agent.color[c]
                pixel_channel = grid[x, y, c]
                
                # Calculate channel similarity: 255 - abs(channel_pixel - channel_agent)
                channel_similarity = 255 - abs(int(pixel_channel) - agent_channel)
                similarity_sum += channel_similarity
            
            # Normalize similarity (divide by 3*255)
            pixel_similarity = similarity_sum / (3 * 255)
            
            # Add to weighted sums for center calculation
            weighted_x_sum += x * pixel_similarity
            weighted_y_sum += y * pixel_similarity
            total_similarity += pixel_similarity
    
    # Only update direction if we found some similarity in the box
    if total_similarity > 0:
        # Calculate the center of similarity
        center_x = weighted_x_sum / total_similarity
        center_y = weighted_y_sum / total_similarity
        
        # Calculate the angle to the center of similarity
        dx = center_x - agent.x
        dy = center_y - agent.y
        
        # Only update the angle if we have a significant vector
        if abs(dx) > 0.001 or abs(dy) > 0.001:
            new_angle = math.atan2(dy, dx)
            # Smoothly turn towards the new angle, limited by turn_rate
            angle_diff = (new_angle - agent.angle + math.pi) % (2 * math.pi) - math.pi
            angle_step = max(-agent.turn_rate, min(agent.turn_rate, angle_diff))
            agent.angle += angle_step
            
    
    # Calculate new position based on current angle
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
        
    grid[int(agent.x), int(agent.y)] = agent.color  # Set pixel to the agent's color


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





# Simulate a compute shader to evaporate pixels
def evaporate_pixels(grid, evaporation_rate=0.02):
    # Vectorized operation to reduce all RGB values by the evaporation rate
    # This is much faster than looping through each pixel
    grid = grid.astype(np.float32)  # Convert to float for the calculation
    grid = grid * (1.0 - evaporation_rate)  # Reduce intensity by evaporation rate
    return grid.astype(np.uint8)  # Convert back to uint8 for pygame




# Apply a box blur diffusion effect to the grid
def diffuse_grid(grid, kernel_size=3):
    # Create a temporary copy to avoid affecting the original during calculation
    temp_grid = grid.copy().astype(np.float32)
    
    # Get the dimensions of the grid
    width, height, channels = grid.shape
    
    # Calculate the half size of the kernel (for neighborhood radius)
    half = kernel_size // 2
    
    # Define weights for the box blur
    # Use a simple box kernel where all weights are equal
    weight = 1.0 / (kernel_size * kernel_size)
    
    # Create a new grid for the result
    result = np.zeros_like(temp_grid)
    
    # Apply the box blur to each pixel
    for x in range(width):
        for y in range(height):
            # For each pixel, calculate the average of its neighborhood
            for c in range(channels):
                # Initialize sum for this channel
                sum_val = 0.0
                count = 0
                
                # Process the neighborhood
                for dx in range(-half, half + 1):
                    nx = x + dx
                    # Skip if outside grid boundaries
                    if nx < 0 or nx >= width:
                        continue
                        
                    for dy in range(-half, half + 1):
                        ny = y + dy
                        # Skip if outside grid boundaries
                        if ny < 0 or ny >= height:
                            continue
                            
                        # Add this neighbor's value
                        sum_val += temp_grid[nx, ny, c]
                        count += 1
                
                # Calculate average (normalize by actual count of neighbors used)
                if count > 0:
                    result[x, y, c] = sum_val / count
    
    return result.astype(np.uint8)

# More efficient diffusion using NumPy's built-in functions
def fast_diffuse_grid(grid, kernel_size=3):
    # Use scipy's convolve function for better performance
    from scipy import ndimage
    
    # Create a uniform kernel (box blur)
    kernel = np.ones((kernel_size, kernel_size, 1)) / (kernel_size * kernel_size)
    
    # Apply the convolution separately to each color channel
    # Mode 'reflect' handles edges by reflecting the image at boundaries
    return ndimage.convolve(grid.astype(np.float32), kernel, mode='reflect').astype(np.uint8)

# Linear interpolation (lerp) between two states
def lerp_grids(grid1, grid2, t=0.5):
    """
    Linear interpolation between two grids.
    grid1: The first grid (unblurred)
    grid2: The second grid (blurred)
    t: Interpolation factor (0.0 = grid1, 1.0 = grid2, 0.5 = 50% blend)
    """
    # Convert to float for the calculation
    grid1 = grid1.astype(np.float32)
    grid2 = grid2.astype(np.float32)
    
    # Perform the linear interpolation
    result = grid1 * (1.0 - t) + grid2 * t
    
    # Convert back to uint8 for pygame
    return result.astype(np.uint8)




# Number of agents
n_agents = 1000
agents = create_agents(n_agents)

grid = create_grid(WIDTH, HEIGHT)
# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False    # Update agent positions using threading    # Update agent positions using threading
    update_agents_threaded(agents, WIDTH, HEIGHT)
    
    original_grid = grid.copy()  # Keep a copy of the original grid for blending
    # Diffuse the grid using a compute shader-like effect
    blurred_grid = fast_diffuse_grid(grid, kernel_size=3)
    
    # Blend the blurred and unblurred grids (50% blend)
    grid = lerp_grids(original_grid, blurred_grid, t=0.2)
    
    # Evaporate pixels (after diffusion and lerping)
    grid = evaporate_pixels(grid, evaporation_rate=0.01)
    
    # Draw the grid to the screen
    draw_grid(screen, grid)
    
    pygame.display.flip()

pygame.quit()
sys.exit()