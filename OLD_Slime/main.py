import pygame
import sys
import random
import numpy as np
import threading
import math
from Agent import Agent
from Hash import hash

# Try to import Numba for GPU acceleration
cuda_available = False
try:
    from numba import cuda, float32, int32, uint8
    # Explicitly check for NVVM support by trying to initialize it
    try:
        # Check if the necessary CUDA libraries are available
        test_kernel = cuda.jit("void()")(lambda: None)
        cuda_available = cuda.is_available()
        if cuda_available:
            print("CUDA is fully available with NVVM support! Using GPU acceleration.")
        else:
            print("CUDA is detected but not available. Falling back to CPU threading.")
    except Exception as e:
        print(f"CUDA initialization failed: {str(e)}")
        print("Missing CUDA libraries. Try installing the full CUDA toolkit.")
        print("Falling back to CPU threading.")
        cuda_available = False
except ImportError:
    print("Numba CUDA not available. Falling back to CPU threading.")

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 360, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Slime Wallpaper Proof of Concept")

# CUDA kernel for agent update - only defined if CUDA is available
if cuda_available:
    @cuda.jit
    def update_agents_kernel(agents_x, agents_y, agents_angle, agents_speed, agents_color, 
                             agents_turn_rate, grid, width, height, random_vals):
        # Get the current thread ID
        i = cuda.grid(1)
        
        # Check if the thread ID is valid (less than the number of agents)
        if i < agents_x.size:
            # Extract agent properties
            x = agents_x[i]
            y = agents_y[i]
            angle = agents_angle[i]
            speed = agents_speed[i]
            turn_rate = agents_turn_rate[i]
            random_val = random_vals[i]
            
            # Define the box size for similarity calculation
            box_size = 10
            half_box = box_size // 2
            
            # Calculate the box boundaries, ensuring they stay within grid boundaries
            box_left = max(0, int(x) - half_box)
            box_right = min(width - 1, int(x) + half_box)
            box_top = max(0, int(y) - half_box)
            box_bottom = min(height - 1, int(y) + half_box)
            
            # Variables to track the center of similarity
            total_similarity = 0.0
            weighted_x_sum = 0.0
            weighted_y_sum = 0.0
            
            # Get agent's color
            r = agents_color[i, 0]
            g = agents_color[i, 1]
            b = agents_color[i, 2]
            
            # Iterate through each pixel in the box
            for bx in range(box_left, box_right + 1):
                for by in range(box_top, box_bottom + 1):
                    # Skip the agent's own position
                    if bx == int(x) and by == int(y):
                        continue
                        
                    # Get pixel values - read once to avoid multiple memory accesses
                    pixel_r = grid[bx, by, 0]
                    pixel_g = grid[bx, by, 1]
                    pixel_b = grid[bx, by, 2]
                    
                    # Calculate similarity for all channels together
                    similarity_sum = (255.0 - abs(float(pixel_r) - float(r)) +
                                     255.0 - abs(float(pixel_g) - float(g)) +
                                     255.0 - abs(float(pixel_b) - float(b)))
                    
                    # Normalize similarity (divide by 3*255)
                    pixel_similarity = similarity_sum / (3.0 * 255.0)
                    
                    # Add to weighted sums for center calculation
                    weighted_x_sum += bx * pixel_similarity
                    weighted_y_sum += by * pixel_similarity
                    total_similarity += pixel_similarity
            
            # Only update direction if we found some similarity in the box
            new_angle = angle
            if total_similarity > 0:
                # Calculate the center of similarity
                center_x = weighted_x_sum / total_similarity
                center_y = weighted_y_sum / total_similarity
                
                # Calculate the angle to the center of similarity
                dx = center_x - x
                dy = center_y - y
                
                # Only update the angle if we have a significant vector
                if abs(dx) > 0.001 or abs(dy) > 0.001:
                    new_angle = math.atan2(dy, dx)
                    # Smoothly turn towards the new angle, limited by turn_rate
                    angle_diff = (new_angle - angle + math.pi) % (2.0 * math.pi) - math.pi
                    angle_step = max(-turn_rate, min(turn_rate, angle_diff))
                    new_angle = angle + angle_step
            
            # Calculate new position based on current angle
            new_x = x + math.cos(new_angle) * speed
            new_y = y + math.sin(new_angle) * speed
            
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
            
            # Generate new angle if out of bounds
            if out_of_bounds:
                # Generate new angle based on hash
                max_int = 0x7FFFFFFF  # Maximum value for a signed 32-bit integer
                new_angle = (random_val / max_int) * 2.0 * math.pi
              # Update agent properties
            agents_x[i] = new_x
            agents_y[i] = new_y
            agents_angle[i] = new_angle
            
            # Set pixel to the agent's color - direct assignment instead of atomic operations
            # Convert positions to integers for grid indexing
            grid_x = int(new_x)
            grid_y = int(new_y)
            
            # Ensure we're within bounds
            if 0 <= grid_x < width and 0 <= grid_y < height:
                grid[grid_x, grid_y, 0] = r
                grid[grid_x, grid_y, 1] = g
                grid[grid_x, grid_y, 2] = b

# Function to generate random values for agents
def generate_random_values(agents, width, height):
    random_vals = np.zeros(len(agents), dtype=np.float32)
    for i, agent in enumerate(agents):
        random_vals[i] = hash(int(agent.y * width + agent.x + i)) / 0x7FFFFFFF
    return random_vals

# Update all agents using GPU - only used if CUDA is available
if cuda_available:
    def update_agents_gpu(agents, grid, width, height):
        # Create arrays for agent properties
        n_agents = len(agents)
        agents_x = np.zeros(n_agents, dtype=np.float32)
        agents_y = np.zeros(n_agents, dtype=np.float32)
        agents_angle = np.zeros(n_agents, dtype=np.float32)
        agents_speed = np.zeros(n_agents, dtype=np.float32)
        agents_turn_rate = np.zeros(n_agents, dtype=np.float32)
        agents_color = np.zeros((n_agents, 3), dtype=np.uint8)
        
        # Copy agent properties to arrays
        for i, agent in enumerate(agents):
            agents_x[i] = agent.x
            agents_y[i] = agent.y
            agents_angle[i] = agent.angle
            agents_speed[i] = agent.speed
            agents_turn_rate[i] = agent.turn_rate
            agents_color[i] = agent.color
        
        # Generate random values for agents
        random_vals = generate_random_values(agents, width, height)
        
        # Copy arrays to device
        d_agents_x = cuda.to_device(agents_x)
        d_agents_y = cuda.to_device(agents_y)
        d_agents_angle = cuda.to_device(agents_angle)
        d_agents_speed = cuda.to_device(agents_speed)
        d_agents_turn_rate = cuda.to_device(agents_turn_rate)
        d_agents_color = cuda.to_device(agents_color)
        d_random_vals = cuda.to_device(random_vals)
          # Copy grid to device
        d_grid = cuda.to_device(grid)
        
        # Calculate grid and block dimensions - optimize for GPU utilization
        # Use a smaller block size to accommodate more blocks
        threads_per_block = 128
        blocks_per_grid = min(32, (n_agents + threads_per_block - 1) // threads_per_block)
        
        # Make sure we have at least 8 blocks for better GPU utilization
        if blocks_per_grid < 8 and n_agents > 128:
            threads_per_block = n_agents // 8
            blocks_per_grid = 8
        
        # Launch kernel
        update_agents_kernel[blocks_per_grid, threads_per_block](
            d_agents_x, d_agents_y, d_agents_angle, d_agents_speed, d_agents_color,
            d_agents_turn_rate, d_grid, width, height, d_random_vals
        )
        
        # Copy results back to host
        d_agents_x.copy_to_host(agents_x)
        d_agents_y.copy_to_host(agents_y)
        d_agents_angle.copy_to_host(agents_angle)
        d_grid.copy_to_host(grid)
        
        # Update agent properties
        for i, agent in enumerate(agents):
            agent.x = agents_x[i]
            agent.y = agents_y[i]
            agent.angle = agents_angle[i]
        
        return grid

# Function to update agent position in a separate thread
def update_agent(agent, agent_index, width, height):
    # Generate hash value
    random_val = hash(int(agent.y * width + agent.x + agent_index))
    
    # Define the box size for similarity calculation
    box_size = 16
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

# Create a list of agents
def create_agents(n):
    agents = []
    for i in range(n):
        # Random position and angle
        box_size = 300

        x = random.randint(WIDTH//2-box_size, WIDTH//2+box_size)
        y = random.randint(HEIGHT//2-box_size, HEIGHT//2+box_size)
        angle = random.uniform(0, 2 * math.pi)  # Random angle in radians
        speed = 1
        color = (255, 255, 255)  # Default color (white)
        turn_rate = 0.1  # Rate at which the agent can turn
        agents.append(Agent(x, y, angle, speed, color, turn_rate))
    return agents

# Create a 2D list (grid) of RGB values
def create_grid(width, height):
    # Initialize with zeros (black)
    # Force contiguous memory layout for better performance with CUDA
    grid = np.zeros((width, height, 3), dtype=np.uint8, order='C')
    return grid

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

# Function to display performance metrics
def display_performance_stats(screen, clock, n_agents, is_gpu):
    font = pygame.font.SysFont('Arial', 18)
    fps = int(clock.get_fps())
    mode = "GPU" if is_gpu else "CPU"
    
    # Create text surfaces
    fps_text = font.render(f"FPS: {fps}", True, (255, 255, 255))
    agents_text = font.render(f"Agents: {n_agents}", True, (255, 255, 255))
    mode_text = font.render(f"Mode: {mode}", True, (255, 255, 255))
    
    # Draw text on screen
    screen.blit(fps_text, (10, 10))
    screen.blit(agents_text, (10, 30))
    screen.blit(mode_text, (10, 50))

# Number of agents
n_agents = 100000
agents = create_agents(n_agents)

grid = create_grid(WIDTH, HEIGHT)
# Main loop
running = True
clock = pygame.time.Clock()  # Add a clock to control frame rate
fps = 60  # Target frame rate

# Print instructions
if not cuda_available:
    print("\nTo get GPU acceleration:")
    print("1. Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
    print("2. Install the cudatoolkit package with: pip install cudatoolkit")
    print("3. Ensure nvvm.dll is in your PATH (usually in CUDA Toolkit bin directory)")
    print("4. If using Anaconda, try: conda install numba cudatoolkit")
    print("\nRunning with CPU threading...\n")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Update agent positions using either GPU or CPU
    if cuda_available:
        try:
            grid = update_agents_gpu(agents, grid, WIDTH, HEIGHT)
        except Exception as e:
            print(f"GPU execution failed: {e}")
            print("Falling back to CPU threading...")
            cuda_available = False
            update_agents_threaded(agents, WIDTH, HEIGHT)
    else:
        update_agents_threaded(agents, WIDTH, HEIGHT)
    
    original_grid = grid.copy()  # Keep a copy of the original grid for blending
    # Diffuse the grid using a compute shader-like effect
    blurred_grid = fast_diffuse_grid(grid, kernel_size=3)
    
    # Blend the blurred and unblurred grids (50% blend)
    grid = lerp_grids(original_grid, blurred_grid, t=0.4)
    
    # Evaporate pixels (after diffusion and lerping)
    grid = evaporate_pixels(grid, evaporation_rate=0.05)
    
    # Draw the grid to the screen
    draw_grid(screen, grid)
    
    # Display performance metrics
    display_performance_stats(screen, clock, n_agents, cuda_available)
    
    # Update the display
    pygame.display.flip()
    
    # Control frame rate
    clock.tick(fps)
    clock.tick(fps)  # Control the frame rate

pygame.quit()
sys.exit()