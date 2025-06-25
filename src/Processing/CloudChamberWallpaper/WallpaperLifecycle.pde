// Called by the Android system when the wallpaper is created
void onStart() {
  // Reset the pixel grid and agents
  setup();
}

// Called by the Android system when the wallpaper is destroyed
void onStop() {
  // Clean up resources if needed
}

// Called when the wallpaper is visible
void onVisible() {
  // Resume animation
  loop();
}

// Called when the wallpaper is hidden
void onInvisible() {
  // Pause animation to save battery
  noLoop();
}

// Called when the user's touch events are delivered
void onTouchDown(int x, int y) {
  // Create a new agent at the touch position
  float decay = random(7, 12);
  float angle = random(TWO_PI);
  int size = int(random(2, 6));
  float velocity = random(2500, 5000);
  int[] color_ = {255, 255, 255};
  
  agentList.add(new Agent(x, y, decay, angle, size, velocity, color_));
}

// Called when the wallpaper's offsets change
void onOffsetsChanged(float xOffset, float yOffset) {
  // React to wallpaper scrolling if desired
}
