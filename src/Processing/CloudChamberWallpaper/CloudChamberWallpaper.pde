import android.service.wallpaper.WallpaperService;
import java.util.ArrayList;

// Pixel grid to store color information
int[][][] pixelGrid;

// List to store all active agents
ArrayList<Agent> agentList;

// Shader programs
PShader diffusionShader;
PShader evaporationShader;

// PGraphics objects for GPU processing
PGraphics diffusionInput;
PGraphics diffusionOutput;
PGraphics displayBuffer;



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
  
  // Initialize GPU shader resources
  initShaders();
  
  noStroke();
  
  // Set frame rate
  frameRate(30);
}

// Initialize shader resources
void initShaders() {
  // Create buffer textures for GPU processing
  diffusionInput = createGraphics(width, height, P2D);
  diffusionOutput = createGraphics(width, height, P2D);
  displayBuffer = createGraphics(width, height, P2D);
  
  // Load GLSL shaders
  diffusionShader = loadShader("diffusion.glsl");
  diffusionShader.set("resolution", float(width), float(height));
  
  evaporationShader = loadShader("evaporation.glsl");
  evaporationShader.set("resolution", float(width), float(height));
  evaporationShader.set("evaporationRate", 0.99);
}

// Draw function runs continuously
void draw() {
  // Clear the background
  background(0);
  
  // Apply diffusion effect (GPU accelerated)
  gpuDiffuseGrid();
  
  // Apply evaporation effect (GPU accelerated)
  gpuEvaporateGrid();
  
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
  
  // Draw the final result
  drawPixelGrid();
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
  // Copy pixelGrid data to the screen directly
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

// GPU-accelerated diffusion effect
void gpuDiffuseGrid() {
  // Copy pixelGrid data to the input texture
  diffusionInput.beginDraw();
  diffusionInput.loadPixels();
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int pixelIndex = y * width + x;
      diffusionInput.pixels[pixelIndex] = color(
        pixelGrid[x][y][0],
        pixelGrid[x][y][1],
        pixelGrid[x][y][2]
      );
    }
  }
  diffusionInput.updatePixels();
  diffusionInput.endDraw();
  
  // Apply diffusion shader
  diffusionOutput.beginDraw();
  diffusionOutput.shader(diffusionShader);
  diffusionShader.set("inputTexture", diffusionInput);
  diffusionOutput.image(diffusionInput, 0, 0);
  diffusionOutput.endDraw();
  
  // Copy the results back to pixelGrid
  diffusionOutput.loadPixels();
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int pixelIndex = y * width + x;
      color c = diffusionOutput.pixels[pixelIndex];
      pixelGrid[x][y][0] = (int)red(c);
      pixelGrid[x][y][1] = (int)green(c);
      pixelGrid[x][y][2] = (int)blue(c);
    }
  }
}

// GPU-accelerated evaporation effect
void gpuEvaporateGrid() {
  // Copy pixelGrid data to the display buffer
  displayBuffer.beginDraw();
  displayBuffer.loadPixels();
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int pixelIndex = y * width + x;
      displayBuffer.pixels[pixelIndex] = color(
        pixelGrid[x][y][0],
        pixelGrid[x][y][1],
        pixelGrid[x][y][2]
      );
    }
  }
  displayBuffer.updatePixels();
  
  // Apply evaporation shader
  displayBuffer.shader(evaporationShader);
  displayBuffer.rect(0, 0, width, height);
  displayBuffer.endDraw();
  
  // Copy the results back to pixelGrid
  displayBuffer.loadPixels();
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int pixelIndex = y * width + x;
      color c = displayBuffer.pixels[pixelIndex];
      pixelGrid[x][y][0] = (int)red(c);
      pixelGrid[x][y][1] = (int)green(c);
      pixelGrid[x][y][2] = (int)blue(c);
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