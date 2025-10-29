@group(0) @binding(0) var myTex: texture_2d<f32>;
@group(0) @binding(1) var mySampler: sampler;

@fragment
fn main(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
  let dims = vec2<f32>(textureDimensions(myTex));
  let uv = coord.xy / dims; // sample by pixel coords mapping (canvas vs texture may differ)
  // A better mapping can be implemented to tile/zoom
  return textureSample(myTex, mySampler, uv);
}