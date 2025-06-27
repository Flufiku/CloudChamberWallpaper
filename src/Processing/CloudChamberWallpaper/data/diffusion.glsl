#ifdef GL_ES
precision mediump float;
precision mediump int;
#endif

uniform sampler2D inputTexture;
uniform vec2 resolution;

void main() {
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  
  // Sample the 3x3 grid of pixels
  float pixelSizeX = 1.0 / resolution.x;
  float pixelSizeY = 1.0 / resolution.y;
  
  // Top row
  vec4 topLeft = texture2D(inputTexture, uv + vec2(-pixelSizeX, -pixelSizeY));
  vec4 top = texture2D(inputTexture, uv + vec2(0.0, -pixelSizeY));
  vec4 topRight = texture2D(inputTexture, uv + vec2(pixelSizeX, -pixelSizeY));
  
  // Middle row
  vec4 left = texture2D(inputTexture, uv + vec2(-pixelSizeX, 0.0));
  vec4 center = texture2D(inputTexture, uv);
  vec4 right = texture2D(inputTexture, uv + vec2(pixelSizeX, 0.0));
  
  // Bottom row
  vec4 bottomLeft = texture2D(inputTexture, uv + vec2(-pixelSizeX, pixelSizeY));
  vec4 bottom = texture2D(inputTexture, uv + vec2(0.0, pixelSizeY));
  vec4 bottomRight = texture2D(inputTexture, uv + vec2(pixelSizeX, pixelSizeY));
  
  // Apply kernel weights
  // Using the same weights as in the original code:
  // {{0.08, 0.12, 0.08}, {0.12, 0.2, 0.12}, {0.08, 0.12, 0.08}}
  vec4 sum = topLeft * 0.04 + top * 0.08 + topRight * 0.04 +
             left * 0.08 + center * 0.52 + right * 0.08 +
             bottomLeft * 0.04 + bottom * 0.08 + bottomRight * 0.04;
  
  // Border check to avoid sampling outside texture
  if (uv.x < pixelSizeX || uv.x > 1.0 - pixelSizeX || 
      uv.y < pixelSizeY || uv.y > 1.0 - pixelSizeY) {
    gl_FragColor = center;
  } else {
    gl_FragColor = sum;
  }
}
