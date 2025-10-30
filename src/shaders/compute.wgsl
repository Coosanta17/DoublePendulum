// Compute shader for double pendulum simulation

// I do not care if fp32 can only handle at most 7 digits.
const PI: f32 = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;

struct Params {
  width: u32,
  height: u32,
  steps: u32,
  dt: f32,
  m1: f32,
  m2: f32,
  l1: f32,
  l2: f32,
  g: f32,
  time: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;

// Double pendulum derivatives
fn derivatives(theta1: f32, theta2: f32, omega1: f32, omega2: f32) -> vec4f {
  let m1 = params.m1;
  let m2 = params.m2;
  let l1 = params.l1;
  let l2 = params.l2;
  let g = params.g;

  let delta = theta2 - theta1;
  let den1 = (m1 + m2) * l1 - m2 * l1 * cos(delta) * cos(delta);
  let den2 = (l2 / l1) * den1;

  let dtheta1 = omega1;
  let dtheta2 = omega2;

  let domega1 = (m2 * l1 * omega1 * omega1 * sin(delta) * cos(delta) +
                 m2 * g * sin(theta2) * cos(delta) +
                 m2 * l2 * omega2 * omega2 * sin(delta) -
                 (m1 + m2) * g * sin(theta1)) / den1;

  let domega2 = (-m2 * l2 * omega2 * omega2 * sin(delta) * cos(delta) +
                 (m1 + m2) * g * sin(theta1) * cos(delta) -
                 (m1 + m2) * l1 * omega1 * omega1 * sin(delta) -
                 (m1 + m2) * g * sin(theta2)) / den2;

  return vec4f(dtheta1, dtheta2, domega1, domega2);
}

// RK4 integration step
fn rk4Step(theta1: f32, theta2: f32, omega1: f32, omega2: f32, dt: f32) -> vec4f {
  let k1 = derivatives(theta1, theta2, omega1, omega2);
  let k2 = derivatives(
    theta1 + 0.5 * dt * k1.x,
    theta2 + 0.5 * dt * k1.y,
    omega1 + 0.5 * dt * k1.z,
    omega2 + 0.5 * dt * k1.w
  );
  let k3 = derivatives(
    theta1 + 0.5 * dt * k2.x,
    theta2 + 0.5 * dt * k2.y,
    omega1 + 0.5 * dt * k2.z,
    omega2 + 0.5 * dt * k2.w
  );
  let k4 = derivatives(
    theta1 + dt * k3.x,
    theta2 + dt * k3.y,
    omega1 + dt * k3.z,
    omega2 + dt * k3.w
  );

  let dtheta1 = (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x) / 6.0;
  let dtheta2 = (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y) / 6.0;
  let domega1 = (k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z) / 6.0;
  let domega2 = (k1.w + 2.0 * k2.w + 2.0 * k3.w + k4.w) / 6.0;

  return vec4f(
    theta1 + dt * dtheta1,
    theta2 + dt * dtheta2,
    omega1 + dt * domega1,
    omega2 + dt * domega2
  );
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
  let x = global_id.x;
  let y = global_id.y;

  if (x >= params.width || y >= params.height) {
    return;
  }

  // Map pixel coordinates to initial angles
  // x -> theta1 range: [-PI, PI]
  // y -> theta2 range: [-PI, PI]
  // 0° is vertically down (theta = 0)
  let theta1_init = (f32(x) / f32(params.width) * 2.0 - 1.0) * PI;
  let theta2_init = (f32(y) / f32(params.height) * 2.0 - 1.0) * PI;

  // Initial velocities are zero
  var theta1 = theta1_init;
  var theta2 = theta2_init;
  var omega1 = 0.0;
  var omega2 = 0.0;

  // Simulate from t=0 to current time
  let totalSteps = u32(params.time / params.dt);
  for (var i = 0u; i < totalSteps; i++) {
    let state = rk4Step(theta1, theta2, omega1, omega2, params.dt);
    theta1 = state.x;
    theta2 = state.y;
    omega1 = state.z;
    omega2 = state.w;
  }

  // Normalize angles to [-PI, PI]
  theta1 = theta1 - floor((theta1 + PI) / (2.0 * PI)) * 2.0 * PI;
  theta2 = theta2 - floor((theta2 + PI) / (2.0 * PI)) * 2.0 * PI;

  // Normalize to [0, 1] for visualization
  let norm_theta1 = (theta1 / PI + 1.0) * 0.5;
  let norm_theta2 = (theta2 / PI + 1.0) * 0.5;
  let norm_omega1 = clamp((omega1 / 10.0 + 1.0) * 0.5, 0.0, 1.0);
  let norm_omega2 = clamp((omega2 / 10.0 + 1.0) * 0.5, 0.0, 1.0);

  // Write color to texture
  textureStore(outputTexture, vec2u(x, y), vec4f(norm_theta1, norm_theta2, norm_omega1, norm_omega2));
}
