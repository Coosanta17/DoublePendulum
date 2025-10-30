@group(0) @binding(0) var resultTexture: texture_2d<f32>;
@group(0) @binding(1) var texSampler: sampler;

@fragment
fn main(@location(0) texCoord: vec2f) -> @location(0) vec4f {
  let data = textureSample(resultTexture, texSampler, texCoord);

  // Use theta1 and theta2 for color
  let r = data.r;
  let g = data.g;
  let b = (data.b + data.a) * 0.5;

  return vec4f(r, g, b, 1.0);
}
