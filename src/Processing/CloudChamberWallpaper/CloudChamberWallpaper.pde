import android.service.wallpaper.WallpaperService;
import java.util.ArrayList;


// Pixel grid to store color information
int[][][] pixelGrid;

// List to store all active agents
ArrayList<Agent> agentList;



void settings() {
  size(displayWidth, displayHeight, P2D);
}

// Setup function runs once at the beginning
void setup() {
  // Set size to match typical phone screen dimensions
  orientation(PORTRAIT);
  
  // Initialize the pixel grid (x, y, RGB)
  pixelGrid = new int[width][height][3];
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      pixelGrid[x][y][0] = 0; // R
      pixelGrid[x][y][1] = 0; // G
      pixelGrid[x][y][2] = 0; // B
    }
  }
  
  // Initialize agent list
  agentList = new ArrayList<Agent>();
  
  noStroke();
  
  // Set frame rate
  frameRate(10);
}

// Draw function runs continuously
void draw() {
  // Clear the background
  background(0);
  
  // Draw the pixel grid
  drawPixelGrid();
  
  // Apply diffusion effect
  diffuseGrid();
  
  // Apply evaporation effect
  evaporateGrid();
  
  // Update all agents
  for (int i = agentList.size() - 1; i >= 0; i--) {
    Agent agent = agentList.get(i);
    if (!agent.update(1.0/frameRate, pixelGrid)) {
      agentList.remove(i);
    }
  }
  
  // Randomly create new agents
  if (random(1) < 0.5) {
    if (random(1) < 0.1) {
      agentList.add(createAgent("alpha"));
    } else {
      agentList.add(createAgent("beta"));
    }
  }
}

// Create a new agent with random properties
Agent createAgent(String mode) {
  float x, y, decay, angle, velocity;
  int size;
  int[] color_ = {255, 255, 255};
  
  if (mode.equals("alpha")) {
    x = random(width);
    y = random(height);
    decay = random(8, 18);
    angle = random(TWO_PI);
    size = int(random(7, 12));
    velocity = random(1500, 3000);
  } else { // beta
    x = random(width);
    y = random(height);
    decay = random(7, 12);
    angle = random(TWO_PI);
    size = int(random(2, 6));
    velocity = random(2500, 5000);
  }
  
  return new Agent(x, y, decay, angle, size, velocity, color_);
}

// Draw the pixel grid to the screen
void drawPixelGrid() {
  loadPixels();
  int idx = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      pixels[idx++] = color(
        pixelGrid[x][y][0],
        pixelGrid[x][y][1],
        pixelGrid[x][y][2]
      );
    }
  }
  updatePixels();
}

// Apply diffusion effect to the pixel grid
void diffuseGrid() {
  // Optimized diffusion with fixed kernel values
  // Pre-computed kernel: {{0.02, 0.05, 0.02}, {0.05, 0.72, 0.05}, {0.02, 0.05, 0.02}}
  
  // Create a temporary grid to store the result
  int[][][] result = new int[width][height][3];
  
  // Process center area (skip edges to avoid bounds checking)
  for (int x = 1; x < width - 1; x++) {
    for (int y = 1; y < height - 1; y++) {
      for (int c = 0; c < 3; c++) {
        // Apply kernel weights directly
        float sum = pixelGrid[x-1][y-1][c] * 0.08 + pixelGrid[x][y-1][c] * 0.12 + pixelGrid[x+1][y-1][c] * 0.08 +
                    pixelGrid[x-1][y][c] * 0.12 + pixelGrid[x][y][c] * 0.2 + pixelGrid[x+1][y][c] * 0.12 +
                    pixelGrid[x-1][y+1][c] * 0.08 + pixelGrid[x][y+1][c] * 0.12 + pixelGrid[x+1][y+1][c] * 0.08;
        
        result[x][y][c] = int(constrain(sum, 0, 255));
      }
    }
  }
  
  // Copy the result back to the pixel grid (center area only)
  for (int x = 1; x < width - 1; x++) {
    for (int y = 1; y < height - 1; y++) {
      for (int c = 0; c < 3; c++) {
        pixelGrid[x][y][c] = result[x][y][c];
      }
    }
  }
}

// Apply evaporation effect to the pixel grid
void evaporateGrid() {
  // Fixed evaporation rate
  final float evaporationRate = 0.95;
  
  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      pixelGrid[x][y][0] = int(pixelGrid[x][y][0] * evaporationRate);
      pixelGrid[x][y][1] = int(pixelGrid[x][y][1] * evaporationRate);
      pixelGrid[x][y][2] = int(pixelGrid[x][y][2] * evaporationRate);
    }
  }
}

// Agent class for particles
class Agent {
  float x, y;
  float decay;
  float angle;
  int size;
  float velocity;
  int[] color_;
  float lifetime;
  
  Agent(float x, float y, float decay, float angle, int size, float velocity, int[] color_) {
    this.x = x;
    this.y = y;
    this.decay = decay;
    this.angle = angle;
    this.size = size;
    this.velocity = velocity;
    this.color_ = color_;
    this.lifetime = 1.0;
  }
  
  boolean update(float dt, int[][][] pixelGrid) {
    // Update position based on velocity
    float newX = velocity * dt * cos(angle);
    float newY = velocity * dt * sin(angle);
    
    // Draw the agent's trail
    draw(pixelGrid, x, y, x + newX, y + newY);
    
    // Update position
    x += newX;
    y += newY;
    
    // Decay lifetime
    lifetime -= decay * dt;
    
    // Check if agent should be removed
    return lifetime > 0;
  }
    void draw(int[][][] pixelGrid, float oldX, float oldY, float newX, float newY) {
    float dx = newX - oldX;
    float dy = newY - oldY;
    int steps = int(max(abs(dx), abs(dy))) + 1;
    float radius = size / 2.0;
    float radiusSquared = radius * radius;
    
    for (int i = 0; i < steps; i++) {
      float t = i / float(steps);
      int xCenter = int(oldX + dx * t);
      int yCenter = int(oldY + dy * t);
      
      // Calculate bounds of the circle's bounding box
      int startX = max(0, int(xCenter - radius));
      int endX = min(width - 1, int(xCenter + radius));
      int startY = max(0, int(yCenter - radius));
      int endY = min(height - 1, int(yCenter + radius));
      
      // Iterate only through the bounding box
      for (int y = startY; y <= endY; y++) {
        for (int x = startX; x <= endX; x++) {
          float distSquared = (x - xCenter) * (x - xCenter) + (y - yCenter) * (y - yCenter);
          if (distSquared <= radiusSquared) {
            pixelGrid[x][y][0] = color_[0];
            pixelGrid[x][y][1] = color_[1];
            pixelGrid[x][y][2] = color_[2];
          }
        }
      }
    }
  }
}