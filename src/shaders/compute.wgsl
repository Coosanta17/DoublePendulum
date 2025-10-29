// compute.wgsl
// RK4 integrator for independent double pendulums.
// State per-pendulum: vec4<f32> = (theta1, theta2, omega1, omega2)
// Writes updated state into dst[idx] and a color into a storage output texture.
//
// Bindings (group 0):
// @binding(0) src states (read)
// @binding(1) dst states (read_write)
// @binding(2) output storage texture (write)
// @binding(3) uniform params

struct SimParams {
    dt: f32,
    pad0: f32,
    texWidth: u32,
    texHeight: u32,
    g: f32,
    l1: f32,
    l2: f32,
    m1: f32,
    m2: f32,
    nPend: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(3) var<uniform>params: SimParams;
@group(0) @binding(0) var<storage, read > src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write > dst: array<vec4<f32>>;
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba8unorm, write>;

// derivative function: given state (theta1, theta2, omega1, omega2) -> (dtheta1, dtheta2, domega1, domega2)
fn deriv(s: vec4<f32>) -> vec4 < f32 > {
    let theta1 = s.x;
    let theta2 = s.y;
    let omega1 = s.z;
    let omega2 = s.w;

    let delta = theta2 - theta1;

    let m1 = params.m1;
    let m2 = params.m2;
    let l1 = params.l1;
    let l2 = params.l2;
    let g = params.g;

    let denom1 = l1 * (2.0 * m1 + m2 - m2 * cos(2.0 * delta));
    let denom2 = l2 * (2.0 * m1 + m2 - m2 * cos(2.0 * delta));

    // avoid division by zero by small epsilon
    let eps = 1e-6;
    let domega1 = (
        -g * (2.0 * m1 + m2) * sin(theta1)
        - m2 * g * sin(theta1 - 2.0 * theta2)
        - 2.0 * sin(delta) * m2 * (omega2 * omega2 * l2 + omega1 * omega1 * l1 * cos(delta))
    ) / max(denom1, eps);

    let domega2 = (
        2.0 * sin(delta) * (
            omega1 * omega1 * l1 * (m1 + m2)
            + g * (m1 + m2) * cos(theta1)
            + omega2 * omega2 * l2 * m2 * cos(delta)
        )
    ) / max(denom2, eps);

    return vec4<f32>(omega1, omega2, domega1, domega2);
}

// simple HSV->RGB helper: h in [0,1], s,v in [0,1]
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3 < f32 > {
    let i = floor(h * 6.0);
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    let mod6 = i % 6.0;
    if(mod6 == 0.0) {
    return vec3<f32>(v, t, p);
} else if (mod6 == 1.0) {
    return vec3<f32>(q, v, p);
} else if (mod6 == 2.0) {
    return vec3<f32>(p, v, t);
} else if (mod6 == 3.0) {
    return vec3<f32>(p, q, v);
} else if (mod6 == 4.0) {
    return vec3<f32>(t, p, v);
}
return vec3<f32>(v, p, q);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.nPend) {
        return;
    }
    let s = src[idx];

    let dt = params.dt;

    // RK4
    let k1 = deriv(s);
    let s2 = s + 0.5 * dt * k1;
    let k2 = deriv(s2);
    let s3 = s + 0.5 * dt * k2;
    let k3 = deriv(s3);
    let s4 = s + dt * k3;
    let k4 = deriv(s4);

    let newS = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

    dst[idx] = newS;

    // write color mapping into output texture
    // Map index -> pixel coords
    let x = i32(idx % params.texWidth);
    let y = i32(idx / params.texWidth);

    // color mapping: use combined angles for hue, speed for value
    let ang = 0.5 * (newS.x + newS.y); // average angle
    let hue = fract(ang * 0.15915494309189535); // / (2*pi)
    let speed = length(vec2<f32>(newS.z, newS.w));
    let sat = 0.8;
    // scale speed to [0,1] with soft scaling
    let value = clamp(speed * 0.06, 0.05, 1.0);

    let rgb = hsv_to_rgb(hue, sat, value);
    textureStore(outputTex, vec2<i32>(x, y), vec4<f32>(rgb, 1.0));
}
