import math

class Agent:
    def __init__(self, x, y, decay, angle, size, velocity=0.0, color=(255, 255, 255), ):
        self.x = x
        self.y = y
        self.decay = decay
        self.angle = angle
        self.size = size
        self.velocity = velocity
        self.color = color
        self.lifetime = 1.0  # Lifetime of the agent, can be used for fading out
        
    def update(self, dt, pixel_grid):
        # Update the position based on velocity
        new_x = self.velocity * dt * math.cos(self.angle)
        new_y = self.velocity * dt * math.sin(self.angle)

        self.draw(pixel_grid, self.x, self.y, self.x + new_x, self.y + new_y)
        
        self.x += new_x
        self.y += new_y
        # Decay the agent's lifetime
        self.lifetime -= self.decay * dt
        
        # If lifetime is less than or equal to 0, mark the agent for removal
        if self.lifetime <= 0:
            return False

        return True

    def draw(self, pixel_grid, old_x, old_y, new_x, new_y):
        dx = new_x - old_x
        dy = new_y - old_y
        steps = int(max(abs(dx), abs(dy))) + 1
        radius = self.size / 2
        for i in range(steps):
            t = i / steps
            x_center = int(old_x + dx * t)
            y_center = int(old_y + dy * t)
            for y in range(int(y_center - radius), int(y_center + radius) + 1):
                for x in range(int(x_center - radius), int(x_center + radius) + 1):
                    if ((x - x_center) ** 2 + (y - y_center) ** 2) <= radius ** 2:
                        if 0 <= x < len(pixel_grid) and 0 <= y < len(pixel_grid[0]):
                            pixel_grid[x][y] = self.color