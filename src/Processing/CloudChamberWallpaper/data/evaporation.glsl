#ifdef GL_ES
precision mediump float;
precision mediump int;
#endif

uniform sampler2D texture;
uniform vec2 resolution;
uniform float evaporationRate;

void main() {
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  vec4 color = texture2D(texture, uv);
  
  // Apply evaporation to RGB channels
  color.rgb *= evaporationRate;
  
  gl_FragColor = color;
}
