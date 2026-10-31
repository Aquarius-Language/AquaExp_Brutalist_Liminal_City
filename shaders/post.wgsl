// post: the accumulated linear light to the frame: exposure, filmic curve, a restrained grade, vignette, grain,
// fade, sRGB.
struct UserUniforms {
    uRes: vec2<f32>,
    uExpo: f32,
    uSat: f32,
    uFade: f32,
    uGrainSeed: f32,
    uWarm: f32,
    uVignette: f32,
}

@group(1) @binding(0) var<uniform> U: UserUniforms;
@group(2) @binding(0) var uAcc: texture_2d<f32>;

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}
fn srgb(c: vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3<f32>(0.0031308));
}
fn hash(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

@fragment fn fs_main(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
    let q = vec2<i32>(fc.xy);
    let acc = textureLoad(uAcc, q, 0);
    var c = acc.rgb / max(acc.a, 1e-6);
    // a little halation around the brightest highlights (3x3 is enough for windows and the sun's edge)
    var glow = vec3<f32>(0.0);
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let s = textureLoad(uAcc, clamp(q + vec2<i32>(x * 2, y * 2), vec2<i32>(0), vec2<i32>(U.uRes) - 1), 0);
            glow += max(s.rgb / max(s.a, 1e-6) - 2.5, vec3<f32>(0.0));
        }
    }
    c += glow / 25.0 * 0.35 * vec3<f32>(1.0, 0.85, 0.7);
    c *= U.uExpo;
    let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    c = mix(vec3<f32>(lum), c, U.uSat);
    // cool shadows, a faint warmth in the highlights (uWarm grows it in the sun break and at dusk)
    let sh = 1.0 - smoothstep(0.0, 0.25, lum);
    c *= mix(vec3<f32>(1.0), vec3<f32>(0.94, 0.985, 1.05), sh * 0.6);
    c *= mix(vec3<f32>(1.0), vec3<f32>(1.06, 1.0, 0.92), smoothstep(0.2, 1.2, lum) * U.uWarm);
    c = aces(c);
    let uv = fc.xy / U.uRes - 0.5;
    c *= 1.0 - U.uVignette * dot(uv * vec2<f32>(1.0, 0.75), uv * vec2<f32>(1.0, 0.75)) * 1.6;
    c = srgb(max(c, vec3<f32>(0.0)));
    let g = hash(vec3<f32>(fc.xy, U.uGrainSeed)) + hash(vec3<f32>(fc.xy + 17.0, U.uGrainSeed + 3.0)) - 1.0;
    c += g * 0.018;
    c *= U.uFade;
    return vec4<f32>(c, 1.0);
}
