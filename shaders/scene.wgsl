// scene: one path-traced sample of the city per pixel, added to the accumulation (uAcc + uWeight * sample).
// Joint-core towers (cylindrical shafts carrying stacked beam cribs) over a low city of board-marked concrete,
// overcast sky light, a weak sun through the cloud deck, height fog and a low cloud layer.
struct UserUniforms {
    uRes: vec2<f32>,
    uJitter: vec2<f32>,
    uFrame: f32,
    uWeight: f32,
    uFov: f32,
    uAperture: f32,
    uFocus: f32,
    uSun: f32,
    uSky: f32,
    uFog: f32,
    uFogH: f32,
    uCloudLo: f32,
    uCloudHi: f32,
    uCloudDen: f32,
    uWet: f32,
    uLights: f32,
    uTime: f32,
    uPrimary: f32,
    uCamPos: vec3<f32>,
    uCamR: vec3<f32>,
    uCamU: vec3<f32>,
    uCamF: vec3<f32>,
    uSunDir: vec3<f32>,
    uSunCol: vec3<f32>,
    uSkyCol: vec3<f32>,
    uFogCol: vec3<f32>,
    uWind: vec3<f32>,
}

@group(1) @binding(0) var<uniform> U: UserUniforms;
@group(2) @binding(0) var uAcc: texture_2d<f32>;

const PI: f32 = 3.14159265;

// ---------------------------------------------------------------- random numbers and noise
var<private> seed: u32;
var<private> gFoot: f32;   // metres covered by a pixel at the current hit
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn rnd() -> f32 { seed = pcg(seed); return f32(seed >> 8u) / 16777216.0; }
fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
fn hash13(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash12(i);
    let b = hash12(i + vec2<f32>(1.0, 0.0));
    let c = hash12(i + vec2<f32>(0.0, 1.0));
    let d = hash12(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn fbm2(p0: vec2<f32>) -> f32 {
    var p = p0;
    var a = 0.5;
    var s = 0.0;
    for (var i = 0; i < 4; i++) { s += a * noise2(p); p = vec2<f32>(1.6 * p.x - 1.2 * p.y, 1.2 * p.x + 1.6 * p.y) + 7.1; a *= 0.5; }
    return s;
}
fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash13(i);
    let n100 = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash13(i + vec3<f32>(1.0, 1.0, 1.0));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y), mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

// ---------------------------------------------------------------- distance functions
fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}
fn rot2(v: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}
fn opU(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { if (a.x < b.x) { return a; } return b; }

// Materials: 1 street, 2 board-marked concrete, 3 ribbon glazing, 4 core shaft, 5 crib beam, 6 dwelling unit, 7 paving
// A joint-core cluster: n.x * n.y cylindrical shafts on a grid (spacing sp, radius r, height h), carrying up to two
// beam cribs, each over the local rectangle c (x0, z0, x1, z1) from height y.x with y.y layers.
struct Cl { o: vec2<f32>, a: f32, n: vec2<f32>, sp: vec2<f32>, r: f32, h: f32, c1: vec4<f32>, y1: vec2<f32>, c2: vec4<f32>, y2: vec2<f32>, c3: vec4<f32>, y3: vec2<f32>, id: f32 }

const PITCH: f32 = 2.5;   // layer height of a crib (beams touch the layer above and below)
const BW: f32 = 1.9;      // beam width
const BS: f32 = 5.4;      // beam spacing

// One layer of a crib: beams along x (even layers) or z (odd layers), ragged ends, dwelling units in some gaps.
fn cribLayer(q: vec3<f32>, c0: vec4<f32>, L: f32, nL: f32, seedv: f32) -> vec2<f32> {
    let alongX = (i32(L) % 2) == 0;
    // the crib tapers like a crown: widest a little below its middle, narrow at the foot and the top
    let u = (L + 0.5) / nL;
    let sh = 1.0 - 0.72 * pow(min(abs(u - 0.42) / 0.58, 1.0), 1.4);
    let cm = vec2<f32>((c0.x + c0.z) * 0.5, (c0.y + c0.w) * 0.5);
    let hw = vec2<f32>((c0.z - c0.x) * 0.5, (c0.w - c0.y) * 0.5) * sh + 3.0 * (1.0 - sh);
    let c = vec4<f32>(cm - hw, cm + hw);
    var a = q.x - c.x;
    var b = q.z - c.y;
    var A = c.z - c.x;
    var B = c.w - c.y;
    if (!alongX) { a = q.z - c.y; b = q.x - c.x; A = c.w - c.y; B = c.z - c.x; }
    let n = floor(B / BS) + 1.0;
    let b0 = (B - (n - 1.0) * BS) * 0.5;
    let yc = (L + 0.5) * PITCH;
    let y = q.y - yc;
    let j = clamp(round((b - b0) / BS), 0.0, n - 1.0);
    var res = vec2<f32>(1e9, 5.0);
    for (var k = 0; k < 2; k++) {
        var jj = j;
        if (k == 1) { jj = clamp(j + select(-1.0, 1.0, b > b0 + j * BS), 0.0, n - 1.0); }
        let h1 = hash13(vec3<f32>(jj, L, seedv));
        let h2 = hash13(vec3<f32>(jj + 17.0, L, seedv));
        let lo = -3.0 - 9.0 * h1 * h1 + select(0.0, 6.0, h1 > 0.82);
        let hi = A + 3.0 + 9.0 * h2 * h2 - select(0.0, 6.0, h2 > 0.85);
        let d = sdBox(vec3<f32>(a - (lo + hi) * 0.5, y, b - (b0 + jj * BS)), vec3<f32>((hi - lo) * 0.5, PITCH * 0.5, BW * 0.5));
        res = opU(res, vec2<f32>(d, 5.0));
    }
    // dwelling units filling the gap the point is in, in 7.5 m long segments
    let g = clamp(floor((b - b0) / BS), 0.0, n - 2.0);
    let seg = clamp(floor(a / 7.5), 0.0, floor(A / 7.5) - 1.0);
    if (n > 1.0 && hash13(vec3<f32>(g + 41.0, L * 3.0 + seg, seedv)) > 0.62) {
        let d = sdBox(vec3<f32>(a - (seg + 0.5) * 7.5, y + 0.05, b - (b0 + (g + 0.5) * BS)), vec3<f32>(3.55, PITCH * 0.5 - 0.1, (BS - BW) * 0.5 + 0.02));
        res = opU(res, vec2<f32>(d, 6.0));
    }
    // units of the other cells lie beyond this cell's edges (0.2 m inset at least)
    if (n > 1.0) {
        let e = min(min(abs(a - seg * 7.5), abs((seg + 1.0) * 7.5 - a)), min(abs(b - (b0 + g * BS)), abs(b0 + (g + 1.0) * BS - b)));
        res.x = min(res.x, max(e + 0.2, abs(y) - PITCH * 0.5));
    }
    return res;
}

fn crib(q: vec3<f32>, c: vec4<f32>, y: vec2<f32>, seedv: f32) -> vec2<f32> {
    if (y.y < 0.5) { return vec2<f32>(1e9, 0.0); }
    let qy = vec3<f32>(q.x, q.y - y.x, q.z);
    let bb = sdBox(qy - vec3<f32>((c.x + c.z) * 0.5, y.y * PITCH * 0.5, (c.y + c.w) * 0.5),
        vec3<f32>((c.z - c.x) * 0.5 + 12.0, y.y * PITCH * 0.5, (c.w - c.y) * 0.5 + 12.0));
    if (bb > 3.0) { return vec2<f32>(bb, 5.0); }
    let L = clamp(floor(qy.y / PITCH), 0.0, y.y - 1.0);
    var res = cribLayer(qy, c, L, y.y, seedv);
    let f = qy.y / PITCH - L;
    let L2 = clamp(L + select(-1.0, 1.0, f > 0.5), 0.0, y.y - 1.0);
    if (L2 != L) { res = opU(res, cribLayer(qy, c, L2, y.y, seedv)); }
    return res;
}

fn cluster(p: vec3<f32>, k: Cl) -> vec2<f32> {
    var q = p;
    let xz = rot2(p.xz - k.o, -k.a);
    q = vec3<f32>(xz.x, p.y, xz.y);
    // bounds
    let mn = vec2<f32>(min(min(k.c1.x, k.c2.x), min(k.c3.x, -k.r)) - 14.0, min(min(k.c1.y, k.c2.y), min(k.c3.y, -k.r)) - 14.0);
    let mx = vec2<f32>(max(max(k.c1.z, max(k.c2.z, k.c3.z)), (k.n.x - 1.0) * k.sp.x + k.r) + 14.0,
        max(max(k.c1.w, max(k.c2.w, k.c3.w)), (k.n.y - 1.0) * k.sp.y + k.r) + 14.0);
    let bb = sdBox(q - vec3<f32>((mn.x + mx.x) * 0.5, k.h * 0.5 + 2.0, (mn.y + mx.y) * 0.5), vec3<f32>((mx.x - mn.x) * 0.5, k.h * 0.5 + 2.0, (mx.y - mn.y) * 0.5));
    if (bb > 4.0) { return vec2<f32>(bb, 4.0); }
    // the nearest shaft
    let ix = clamp(round(q.x / k.sp.x), 0.0, k.n.x - 1.0);
    let iz = clamp(round(q.z / k.sp.y), 0.0, k.n.y - 1.0);
    let cxz = q.xz - vec2<f32>(ix * k.sp.x, iz * k.sp.y);
    let hc = k.h * (1.0 - 0.13 * hash13(vec3<f32>(ix, iz, k.id)));
    let ang = atan2(cxz.y, cxz.x);
    let fl = abs(fract(ang * 12.0 / (2.0 * PI)) - 0.5) * 2.0;
    let rr = k.r - 0.2 * smoothstep(0.55, 0.8, fl);
    let rho = length(cxz);
    var res = vec2<f32>(max(rho - rr, abs(q.y - hc * 0.5) - hc * 0.5) * 0.9, 4.0);
    // the cap: a wider drum with a thin roof slab, and the plinth at the foot
    let cap = max(rho - (k.r + 0.9), abs(q.y - (hc + 1.2)) - 1.6);
    let roof = max(rho - (k.r + 1.6), abs(q.y - (hc + 3.0)) - 0.35);
    let plinth = max(rho - (k.r + 2.4), abs(q.y - 1.5) - 1.5);
    res = opU(res, vec2<f32>(min(cap, min(roof, plinth)), 2.0));
    // a lift cabin climbing the outside of every shaft on its guide rail
    let hs = hash13(vec3<f32>(ix + 3.0, iz + 5.0, k.id));
    let lq = rot2(cxz, -hs * 6.2831);
    let ly = 6.0 + (hc - 34.0) * (0.5 + 0.5 * sin(U.uTime * (0.03 + 0.03 * hs) + hs * 20.0));
    let cab = sdBox(vec3<f32>(lq.x - (k.r + 1.7), q.y - ly - 2.2, lq.y), vec3<f32>(1.6, 2.2, 1.7));
    let rail = sdBox(vec3<f32>(lq.x - (k.r + 0.25), q.y - hc * 0.5, lq.y), vec3<f32>(0.35, hc * 0.5, 0.9));
    res = opU(res, vec2<f32>(cab, 6.0));
    res = opU(res, vec2<f32>(rail, 2.0));
    res = opU(res, crib(q, k.c1, k.y1, k.id * 7.0 + 1.0));
    res = opU(res, crib(q, k.c2, k.y2, k.id * 7.0 + 2.0));
    res = opU(res, crib(q, k.c3, k.y3, k.id * 7.0 + 3.0));
    return res;
}

fn cl(o: vec2<f32>, a: f32, n: vec2<f32>, sp: vec2<f32>, r: f32, h: f32, c1: vec4<f32>, y1: vec2<f32>, c2: vec4<f32>, y2: vec2<f32>, c3: vec4<f32>, y3: vec2<f32>, id: f32) -> Cl {
    return Cl(o, a, n, sp, r, h, c1, y1, c2, y2, c3, y3, id);
}
const NO4: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
const NO2: vec2<f32> = vec2<f32>(0.0, 0.0);
fn clusterDef(i: i32) -> Cl {
    switch i {
        case 0: { return cl(vec2<f32>(0.0, 0.0), 0.21, vec2<f32>(2.0, 2.0), vec2<f32>(46.0, 42.0), 8.5, 182.0,
            vec4<f32>(-34.0, -18.0, 80.0, 60.0), vec2<f32>(56.0, 14.0), vec4<f32>(-18.0, -36.0, 64.0, 52.0), vec2<f32>(110.0, 12.0),
            vec4<f32>(-14.0, -14.0, 60.0, 56.0), vec2<f32>(150.0, 8.0), 1.0); }
        case 1: { return cl(vec2<f32>(190.0, -70.0), -0.35, vec2<f32>(2.0, 1.0), vec2<f32>(48.0, 1.0), 8.0, 150.0,
            vec4<f32>(-28.0, -24.0, 76.0, 24.0), vec2<f32>(68.0, 12.0), vec4<f32>(-16.0, -26.0, 44.0, 26.0), vec2<f32>(112.0, 10.0), NO4, NO2, 2.0); }
        case 2: { return cl(vec2<f32>(-165.0, 80.0), 0.6, vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 1.0), 9.0, 140.0,
            vec4<f32>(-30.0, -26.0, 26.0, 28.0), vec2<f32>(52.0, 12.0), vec4<f32>(-22.0, -22.0, 22.0, 22.0), vec2<f32>(94.0, 10.0), NO4, NO2, 3.0); }
        case 3: { return cl(vec2<f32>(90.0, 330.0), -0.1, vec2<f32>(2.0, 2.0), vec2<f32>(44.0, 50.0), 8.0, 168.0,
            vec4<f32>(-26.0, -20.0, 70.0, 70.0), vec2<f32>(74.0, 13.0), vec4<f32>(-18.0, -12.0, 62.0, 62.0), vec2<f32>(124.0, 10.0), NO4, NO2, 4.0); }
        case 4: { return cl(vec2<f32>(-350.0, -230.0), 0.9, vec2<f32>(1.0, 2.0), vec2<f32>(1.0, 44.0), 8.5, 158.0,
            vec4<f32>(-28.0, -20.0, 28.0, 64.0), vec2<f32>(66.0, 12.0), vec4<f32>(-22.0, -10.0, 22.0, 54.0), vec2<f32>(110.0, 10.0), NO4, NO2, 5.0); }
        case 5: { return cl(vec2<f32>(460.0, 180.0), 0.3, vec2<f32>(2.0, 1.0), vec2<f32>(52.0, 1.0), 8.5, 186.0,
            vec4<f32>(-26.0, -26.0, 78.0, 26.0), vec2<f32>(84.0, 13.0), vec4<f32>(-16.0, -22.0, 68.0, 22.0), vec2<f32>(136.0, 10.0), NO4, NO2, 6.0); }
        default: { return cl(vec2<f32>(-130.0, -470.0), -0.5, vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 1.0), 8.0, 150.0,
            vec4<f32>(-24.0, -24.0, 24.0, 24.0), vec2<f32>(62.0, 12.0), vec4<f32>(-18.0, -18.0, 18.0, 18.0), vec2<f32>(104.0, 9.0), NO4, NO2, 7.0); }
    }
}
const NCLUSTERS: i32 = 7;

// Open plazas around the clusters (no city blocks there): centre xz, radius.
fn plaza(c: vec2<f32>) -> bool {
    if (length(c - vec2<f32>(24.0, 30.0)) < 100.0) { return true; }
    if (length(c - vec2<f32>(212.0, -80.0)) < 70.0) { return true; }
    if (length(c - vec2<f32>(-165.0, 80.0)) < 56.0) { return true; }
    if (length(c - vec2<f32>(114.0, 356.0)) < 78.0) { return true; }
    if (length(c - vec2<f32>(-356.0, -206.0)) < 58.0) { return true; }
    if (length(c - vec2<f32>(486.0, 190.0)) < 66.0) { return true; }
    if (length(c - vec2<f32>(-130.0, -470.0)) < 42.0) { return true; }
    return false;
}

const CELL: f32 = 60.0;

// The low city: one building per 60 m cell, streets between; every building stays 6 m inside its cell.
fn city(p: vec3<f32>) -> vec2<f32> {
    if (p.y > 50.0) { return vec2<f32>(p.y - 48.0, 2.0); }
    let id = floor(p.xz / CELL);
    let l = p.xz - (id + 0.5) * CELL;
    let border = CELL * 0.5 - max(abs(l.x), abs(l.y));
    var res = vec2<f32>(border + 5.0, 0.0);
    let cc = (id + 0.5) * CELL;
    // the lot: a 0.15 m kerbed pavement
    res = opU(res, vec2<f32>(sdBox(vec3<f32>(l.x, p.y - 0.075, l.y), vec3<f32>(23.0, 0.075, 23.0)), 7.0));
    if (plaza(cc) || length(cc) > 2600.0) { return res; }
    let h0 = hash12(id + 0.37);
    let h1 = hash12(id * 1.31 + 5.1);
    let h2 = hash12(id * 0.73 + 11.7);
    let h3 = hash12(id * 1.91 + 3.3);
    let bx = 11.0 + 7.0 * h1;
    let bz = 11.0 + 7.0 * h2;
    let lc = l - vec2<f32>((h2 - 0.5) * 3.0, (h1 - 0.5) * 3.0);
    if (h0 < 0.5) {
        // terraces: a glazed volume ringed by cantilevered floor slabs of varying depth
        let fh = 3.6;
        let nF = 2.0 + floor(h3 * h3 * 9.0);
        let H = nF * fh;
        var d = vec2<f32>(sdBox(vec3<f32>(lc.x, p.y - H * 0.5, lc.y), vec3<f32>(bx, H * 0.5, bz)), 3.0);
        let k = clamp(round(p.y / fh), 1.0, nF);
        for (var s = 0; s < 2; s++) {
            var kk = k;
            if (s == 1) { kk = clamp(k + select(-1.0, 1.0, p.y > k * fh), 1.0, nF); }
            let hk = hash13(vec3<f32>(id, kk));
            let pk = 0.6 + 3.3 * hk * hk * step(0.5, kk);
            let th = select(0.55, 1.0, kk == nF);
            let off = (hash13(vec3<f32>(id, kk + 9.0)) - 0.5) * 2.0 * min(pk, 1.5);
            let sd = sdBox(vec3<f32>(lc.x - off, p.y - (kk * fh - th), lc.y), vec3<f32>(bx + pk, th, bz + pk * 0.7));
            d = opU(d, vec2<f32>(sd, 2.0));
        }
        // a stair or lift tower on the side
        let tw = sdBox(vec3<f32>(lc.x - bx + 2.0, p.y - (H + 4.0) * 0.5, lc.y + bz * 0.4), vec3<f32>(3.5, (H + 4.0) * 0.5, 4.5));
        d = opU(d, vec2<f32>(tw, 2.0));
        // plant rooms on the roof
        let pr = sdBox(vec3<f32>(lc.x + bx * 0.3, p.y - H - 1.6, lc.y - bz * 0.2), vec3<f32>(bx * 0.35, 1.6, bz * 0.3));
        d = opU(d, vec2<f32>(pr, 2.0));
        res = opU(res, d);
    } else if (h0 < 0.78) {
        // a blind fly-tower block on a podium
        let H = 16.0 + 28.0 * h3;
        var d = vec2<f32>(sdBox(vec3<f32>(lc.x, p.y - 4.5, lc.y), vec3<f32>(bx, 4.5, bz)), 2.0);
        let t = sdBox(vec3<f32>(lc.x + bx * 0.25, p.y - H * 0.5, lc.y - bz * 0.15), vec3<f32>(bx * 0.55, H * 0.5, bz * 0.62));
        d = opU(d, vec2<f32>(t, 2.0));
        let band = sdBox(vec3<f32>(lc.x, p.y - 9.8, lc.y), vec3<f32>(bx + 1.6, 0.8, bz + 1.6));
        d = opU(d, vec2<f32>(band, 2.0));
        let g = sdBox(vec3<f32>(lc.x, p.y - 6.0, lc.y), vec3<f32>(bx - 1.0, 3.0, bz + 0.3));
        d = opU(d, vec2<f32>(g, 3.0));
        res = opU(res, d);
    } else if (h0 < 0.9) {
        // a low hall under a deep cantilevered roof
        var d = vec2<f32>(sdBox(vec3<f32>(lc.x, p.y - 3.5, lc.y), vec3<f32>(bx - 3.0, 3.5, bz - 3.0)), 3.0);
        let r = sdBox(vec3<f32>(lc.x, p.y - 7.6, lc.y), vec3<f32>(bx + 3.0, 0.9, bz + 2.0));
        d = opU(d, vec2<f32>(r, 2.0));
        let wall = sdBox(vec3<f32>(lc.x + bx - 1.0, p.y - 4.0, lc.y), vec3<f32>(1.0, 4.0, bz));
        d = opU(d, vec2<f32>(wall, 2.0));
        res = opU(res, d);
    }
    return res;
}

// Streets in the sky: 5 m concrete decks at 8 m along some north-south streets, on piers every 18 m.
fn decks(p: vec3<f32>) -> vec2<f32> {
    if (p.y > 12.0) { return vec2<f32>(p.y - 10.0, 2.0); }
    let ix = round(p.x / CELL);
    let lx = p.x - ix * CELL;
    if (hash12(vec2<f32>(ix, 3.7)) < 0.6 || abs(ix * CELL - 30.0) < 160.0) {
        // no deck on this street: the next one is at least a street away
        return vec2<f32>(max(abs(lx) - 3.0, CELL * 0.5 - abs(lx) - 3.0) + 0.5, 2.0);
    }
    let zl = 600.0 + 500.0 * hash12(vec2<f32>(ix, 9.1));
    let zc = (hash12(vec2<f32>(ix, 1.3)) - 0.5) * 400.0;
    let dk = sdBox(vec3<f32>(lx, p.y - 8.0, p.z - zc), vec3<f32>(2.5, 0.55, zl));
    let rail = sdBox(vec3<f32>(abs(lx) - 2.35, p.y - 9.0, p.z - zc), vec3<f32>(0.15, 0.5, zl));
    let pz = p.z - round(p.z / 18.0) * 18.0;
    let pier = max(sdBox(vec3<f32>(lx, p.y - 3.8, pz), vec3<f32>(0.7, 3.8, 0.9)), abs(p.z - zc) - zl);
    let other = CELL - abs(lx) - 3.0;
    var res = vec2<f32>(min(min(dk, rail), min(pier, other)), 2.0);
    // two tram cars shuttling along the deck, one each way
    let spd = 7.0 / (2.0 * zl);
    let z1 = zc + (abs(fract(U.uTime * spd + hash12(vec2<f32>(ix, 4.4))) * 2.0 - 1.0) * 2.0 - 1.0) * (zl - 10.0);
    let z2 = zc + (abs(fract(U.uTime * spd * 0.8 + hash12(vec2<f32>(ix, 6.6))) * 2.0 - 1.0) * 2.0 - 1.0) * (zl - 10.0);
    let car1 = sdBox(vec3<f32>(lx + 1.15, p.y - 10.0, p.z - z1), vec3<f32>(0.95, 1.4, 8.0));
    let car2 = sdBox(vec3<f32>(lx - 1.15, p.y - 10.0, p.z - z2), vec3<f32>(0.95, 1.4, 8.0));
    res = opU(res, vec2<f32>(min(car1, car2), 6.0));
    return res;
}

fn map(p: vec3<f32>) -> vec2<f32> {
    var res = vec2<f32>(p.y, 1.0);
    res = opU(res, city(p));
    res = opU(res, decks(p));
    for (var i = 0; i < NCLUSTERS; i++) { res = opU(res, cluster(p, clusterDef(i))); }
    return res;
}

fn normalAt(p: vec3<f32>, t: f32) -> vec3<f32> {
    let e = max(0.004, 0.0004 * t);
    let k = vec2<f32>(1.0, -1.0);
    return normalize(k.xyy * map(p + k.xyy * e).x + k.yyx * map(p + k.yyx * e).x + k.yxy * map(p + k.yxy * e).x + k.xxx * map(p + k.xxx * e).x);
}

// March a ray; returns (t, material), material 0 when nothing was hit before tmax.
fn march(ro: vec3<f32>, rd: vec3<f32>, tmax: f32, steps: i32, cone: f32) -> vec2<f32> {
    var t = 0.02;
    for (var i = 0; i < steps; i++) {
        let p = ro + rd * t;
        let h = map(p);
        if (h.x < cone * t + 0.003) { return vec2<f32>(t, h.y); }
        t += h.x;
        if (t > tmax) { break; }
    }
    // a ray that runs out of steps close to a surface counts as a hit (grazing streets), otherwise as escaped
    if (t < tmax) { let h = map(ro + rd * t); if (h.x < 0.5 + cone * t * 4.0) { return vec2<f32>(t, h.y); } }
    return vec2<f32>(tmax, 0.0);
}

// ---------------------------------------------------------------- light
fn sky(rd: vec3<f32>) -> vec3<f32> {
    let y = max(rd.y, 0.0);
    // CIE overcast: zenith three times the horizon, slightly cooler overhead
    var c = U.uSky * U.uSkyCol * (1.0 + 2.0 * y) / 3.0 * mix(vec3<f32>(1.04, 1.0, 0.96), vec3<f32>(0.94, 0.98, 1.05), y);
    let mu = max(dot(rd, U.uSunDir), 0.0);
    c += U.uSunCol * U.uSun * (0.06 * pow(mu, 6.0) + 0.25 * pow(mu, 60.0) + 3.0 * pow(mu, 900.0));
    if (rd.y < 0.0) { c = U.uFogCol * U.uSky * 0.35; }
    return c;
}

// Ambient estimate for the surfaces a secondary ray lands on (their own sky light, without further bounces).
fn ambientAt(n: vec3<f32>, alb: vec3<f32>) -> vec3<f32> {
    let skyL = U.uSky * U.uSkyCol * (0.42 + 0.33 * n.y);
    let sunL = U.uSunCol * U.uSun * max(dot(n, U.uSunDir), 0.0) * 0.5;
    return alb * (skyL + sunL);
}

// Height fog between a and b: (transmittance, in-scattered light).
fn fogSeg(ro: vec3<f32>, rd: vec3<f32>, t: f32) -> vec4<f32> {
    let k = 1.0 / U.uFogH;
    let a = ro.y;
    let dy = rd.y * t;
    var od: f32;
    if (abs(dy) < 0.01) { od = U.uFog * exp(-max(a, 0.0) * k) * t; }
    else { od = U.uFog * t * (exp(-max(a, 0.0) * k) - exp(-max(a + dy, 0.0) * k)) / (dy * k); }
    od += U.uFog * 0.08 * t;   // a thin haze everywhere
    let tr = exp(-od);
    let mu = max(dot(rd, U.uSunDir), 0.0);
    let col = U.uFogCol * U.uSky + U.uSunCol * U.uSun * (0.06 * pow(mu, 8.0) + 0.02);
    return vec4<f32>(col * (1.0 - tr), tr);
}

fn cloudDensity(p: vec3<f32>) -> f32 {
    let h = (p.y - U.uCloudLo) / (U.uCloudHi - U.uCloudLo);
    if (h < 0.0 || h > 1.0) { return 0.0; }
    let q = p * 0.012 + U.uWind;
    var n = noise3(q) * 0.55 + noise3(q * 2.3 + 1.7) * 0.28 + noise3(q * 5.1 + 3.1) * 0.12;
    n = n - 0.42 + 0.25 * (1.0 - h);
    let shape = smoothstep(0.0, 0.25, h) * smoothstep(1.0, 0.6, h);
    return max(n, 0.0) * shape * U.uCloudDen;
}

// The cloud layer along a ray segment: (in-scattered light, transmittance).
fn clouds(ro: vec3<f32>, rd: vec3<f32>, tend: f32) -> vec4<f32> {
    if (U.uCloudDen <= 0.0) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    var t0 = 0.0;
    var t1 = tend;
    if (abs(rd.y) < 1e-4) {
        if (ro.y < U.uCloudLo || ro.y > U.uCloudHi) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    } else {
        let ta = (U.uCloudLo - ro.y) / rd.y;
        let tb = (U.uCloudHi - ro.y) / rd.y;
        t0 = max(min(ta, tb), 0.0);
        t1 = min(max(ta, tb), tend);
    }
    t1 = min(t1, t0 + 1600.0);
    if (t1 <= t0) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let N = 28;
    let dt = (t1 - t0) / f32(N);
    var t = t0 + dt * rnd();
    var tr = 1.0;
    var col = vec3<f32>(0.0);
    let mu = dot(rd, U.uSunDir);
    let phase = 0.6 + 1.4 * pow(max(mu, 0.0), 6.0);
    for (var i = 0; i < N; i++) {
        let p = ro + rd * t;
        let d = cloudDensity(p);
        if (d > 0.0) {
            let h = (p.y - U.uCloudLo) / (U.uCloudHi - U.uCloudLo);
            let lum = U.uSky * U.uSkyCol * (0.55 + 0.45 * h) + U.uSunCol * U.uSun * phase * 0.25 * (0.3 + 0.7 * h);
            let a = 1.0 - exp(-d * dt);
            col += tr * a * lum;
            tr *= 1.0 - a;
            if (tr < 0.02) { break; }
        }
        t += dt;
    }
    return vec4<f32>(col, tr);
}

struct Surf { alb: vec3<f32>, rough: f32, n: vec3<f32>, emit: vec3<f32> }

fn tangentCoord(p: vec3<f32>, n: vec3<f32>) -> f32 {
    if (abs(n.x) > abs(n.z)) { return p.z; }
    return p.x;
}

// Board-marked concrete: 0.18 m planks with their own tone, grain along them, tie holes, rain streaks, ground dirt.
fn concrete(p: vec3<f32>, n: vec3<f32>, base: vec3<f32>, plank: f32, face: f32) -> Surf {
    var s: Surf;
    s.n = n;
    s.rough = 0.88;
    s.emit = vec3<f32>(0.0);
    let big = fbm2(p.xz * 0.05 + p.y * 0.03) - 0.5;
    var a = base * 0.8 * (1.0 + 0.18 * big);
    if (abs(n.y) < 0.6) {
        let u = tangentCoord(p, n);
        let v = p.y / plank;
        let pi = floor(v);
        let fv = fract(v);
        // fine detail fades out once a pixel covers it (gFoot: metres per pixel at the hit)
        let df = 1.0 - smoothstep(0.012, 0.05, gFoot);
        let df2 = 1.0 - smoothstep(0.25, 1.2, gFoot);
        a *= 1.0 + (hash12(vec2<f32>(pi, face + floor(u / 2.4))) - 0.5) * 0.09 * df;
        a *= 1.0 + (noise2(vec2<f32>(u * 2.2, p.y * 30.0)) - 0.5) * 0.08 * df;
        let joint = 1.0 - smoothstep(0.0, 0.07, fv) * smoothstep(1.0, 0.93, fv);
        a *= 1.0 - 0.16 * mix(0.13, joint, df);
        s.n = normalize(n + vec3<f32>(0.0, (smoothstep(0.0, 0.07, fv) - smoothstep(0.93, 1.0, fv)) * 0.12 * df, 0.0));
        let th = vec2<f32>(fract(u / 1.2) - 0.5, fract(p.y / 0.9) - 0.5) * vec2<f32>(1.2, 0.9);
        a *= 1.0 - 0.45 * (1.0 - smoothstep(0.018, 0.03, length(th))) * df;
        // rain streaks hanging from every ledge (strongest just under the floor lines) and splash dirt near the ground
        let st = noise2(vec2<f32>(u * 1.7, p.y * 0.06)) * noise2(vec2<f32>(u * 0.31 + 3.0, p.y * 0.02));
        let under = 1.0 - smoothstep(0.0, 2.8, fract(p.y / 3.6) * 3.6);
        a *= 1.0 - mix(0.12, 0.38 * smoothstep(0.18, 0.55, st), df2) - 0.22 * under * mix(0.5, noise2(vec2<f32>(u * 3.0, 1.0)), df2);
        a *= 1.0 - 0.3 * (1.0 - smoothstep(0.0, 1.4, p.y));
    } else if (n.y > 0.0) {
        // roofs and ledges: gravel, tar patches and standing water
        a *= 0.62 * (0.85 + 0.25 * noise2(p.xz * 0.4)) * (0.92 + 0.08 * noise2(p.xz * 3.0));
        a *= 1.0 - 0.35 * U.uWet;
        s.rough = mix(0.85, 0.35, U.uWet);
    } else {
        a *= 0.78;   // soffits stay darker and cleaner
    }
    s.alb = a;
    return s;
}

fn surfaceAt(p: vec3<f32>, n: vec3<f32>, m: f32) -> Surf {
    var s: Surf;
    s.n = n;
    s.emit = vec3<f32>(0.0);
    s.rough = 0.9;
    let mi = i32(m + 0.5);
    let cid = floor(p.xz / CELL);
    if (mi == 1 || mi == 7) {
        // street asphalt with puddles, or paving slabs on the lots and plazas
        let l = p.xz - (cid + 0.5) * CELL;
        var a: vec3<f32>;
        if (mi == 7 || p.y > 0.1) {
            let g = abs(fract(p.xz / 1.5) - 0.5);
            let joint = 1.0 - smoothstep(0.47, 0.49, max(g.x, g.y));
            a = vec3<f32>(0.24, 0.235, 0.225) * (0.85 + 0.25 * hash12(floor(p.xz / 1.5))) * (0.75 + 0.25 * joint);
            if (abs(n.y) < 0.5) { a = vec3<f32>(0.36, 0.35, 0.33); }
        } else {
            a = vec3<f32>(0.085, 0.085, 0.09) * (0.8 + 0.4 * fbm2(p.xz * 0.8));
            // worn lane markings would be text-like noise: keep the asphalt plain, darker along the gutters
            let edge = CELL * 0.5 - max(abs(l.x), abs(l.y));
            a *= 0.9 + 0.1 * smoothstep(0.0, 2.0, abs(edge - 0.0));
        }
        let pud = smoothstep(0.52, 0.6, fbm2(p.xz * 0.11 + 4.0) + 0.25 * (U.uWet - 0.5)) * U.uWet;
        s.alb = a * (1.0 - 0.45 * pud);
        s.rough = mix(mix(0.8, 0.45, U.uWet), 0.03, pud);
        return s;
    }
    if (mi == 3) {
        // ribbon glazing: dark glass between concrete mullions; at dusk a few rooms are lit
        let u = tangentCoord(p, n);
        let mu = abs(fract(u / 1.8) - 0.5);
        if (mu > 0.44 || abs(n.y) > 0.5) { return concrete(p, n, vec3<f32>(0.38, 0.37, 0.35), 0.18, 3.0); }
        s.alb = vec3<f32>(0.03, 0.035, 0.04);
        s.rough = 0.06;
        let room = hash13(vec3<f32>(floor(u / 5.4), floor(p.y / 3.6), cid.x * 13.0 + cid.y));
        if (room > 1.0 - 0.1 * U.uLights) { s.emit = vec3<f32>(1.0, 0.6, 0.3) * (0.25 + 4.0 * (room - 0.9)) * 0.9; }
        return s;
    }
    if (mi == 4) {
        // the shafts: smooth precast rings with a thin joint every 2.4 m and narrow lit slots
        var c = concrete(p, n, vec3<f32>(0.47, 0.46, 0.44), 2.4, 4.0);
        return c;
    }
    if (mi == 5) { return concrete(p, n, vec3<f32>(0.44, 0.43, 0.41), 0.2, 5.0); }
    if (mi == 6) {
        var c = concrete(p, n, vec3<f32>(0.40, 0.395, 0.38), 0.3, 6.0);
        if (abs(n.y) < 0.5) {
            let u = tangentCoord(p, n);
            let fy = fract(p.y / PITCH);
            let fu = fract(u / 3.7);
            if (fy > 0.36 && fy < 0.72 && fu > 0.15 && fu < 0.85) {
                c.alb = vec3<f32>(0.03, 0.035, 0.04);
                c.rough = 0.08;
                let room = hash13(floor(p / vec3<f32>(3.7, PITCH, 3.7)) + 0.5);
                if (room > 1.0 - 0.14 * U.uLights) { c.emit = vec3<f32>(1.0, 0.62, 0.34) * 0.6; }
            }
        }
        return c;
    }
    return concrete(p, n, vec3<f32>(0.42, 0.41, 0.39), 0.18, 2.0);
}

fn cosineDir(n: vec3<f32>) -> vec3<f32> {
    let r1 = rnd();
    let r2 = rnd();
    let ph = 2.0 * PI * r1;
    let sr = sqrt(r2);
    let t = normalize(select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.5));
    let b = normalize(cross(n, t));
    let tt = cross(b, n);
    return normalize(tt * (sr * cos(ph)) + b * (sr * sin(ph)) + n * sqrt(1.0 - r2));
}

// Light reflected towards -rd by a surface (one diffuse sky/bounce sample, the sun through soft shadows, and a
// traced glossy reflection for wet and glazed surfaces).
fn shade(p: vec3<f32>, rd: vec3<f32>, s: Surf, nGeo: vec3<f32>) -> vec3<f32> {
    let n = s.n;
    let po = p + nGeo * 0.02;
    var col = s.emit;
    // diffuse: sky through the hemisphere or one bounce
    let dd = cosineDir(n);
    let hb = march(po, dd, 420.0, 70, 0.002);
    var inc: vec3<f32>;
    if (hb.y < 0.5) { inc = sky(dd); }
    else {
        let p2 = po + dd * hb.x;
        let n2 = normalAt(p2, hb.x);
        gFoot = 1.0;
        let s2 = surfaceAt(p2, n2, hb.y);
        inc = ambientAt(n2, s2.alb) + s2.emit;
        let f = fogSeg(po, dd, hb.x);
        inc = inc * f.w + f.xyz;
    }
    var diff = s.alb * inc;
    // sun through a soft disc
    if (U.uSun > 0.0) {
        let ndl = dot(n, U.uSunDir);
        if (ndl > 0.0) {
            let j = vec3<f32>(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) * 0.05;
            let sd = normalize(U.uSunDir + j);
            let sh = march(po, sd, 700.0, 90, 0.003);
            if (sh.y < 0.5) {
                let cl = clouds(po, sd, 2000.0);
                diff += s.alb / PI * U.uSunCol * U.uSun * 3.2 * ndl * cl.w;
            }
        }
    }
    // glossy reflection (Schlick, F0 0.04)
    let cosv = max(dot(-rd, n), 0.0);
    let F = 0.04 + 0.96 * pow(1.0 - cosv, 5.0) * (1.0 - s.rough) * (1.0 - s.rough);
    var spec = vec3<f32>(0.0);
    if (s.rough < 0.5) {
        var rr = reflect(rd, n);
        rr = normalize(rr + (vec3<f32>(rnd(), rnd(), rnd()) - 0.5) * s.rough * s.rough * 1.6);
        if (dot(rr, nGeo) < 0.0) { rr = reflect(rr, nGeo); }
        let hr = march(po, rr, 1800.0, 110, 0.0015);
        if (hr.y < 0.5) {
            spec = sky(rr);
            let cl = clouds(po, rr, 4000.0);
            spec = spec * cl.w + cl.xyz;
        } else {
            let p2 = po + rr * hr.x;
            let n2 = normalAt(p2, hr.x);
            gFoot = 1.0;
            let s2 = surfaceAt(p2, n2, hr.y);
            spec = ambientAt(n2, s2.alb) + s2.emit;
        }
        let f = fogSeg(po, rr, hr.x);
        spec = spec * f.w + f.xyz;
    } else {
        spec = U.uSky * U.uSkyCol * 0.6;
    }
    return col + diff * (1.0 - F) + spec * F;
}

@fragment fn fs_main(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
    let px = vec2<u32>(fc.xy);
    seed = pcg(px.x * 1973u + px.y * 9277u + u32(U.uFrame) * 26699u) | 1u;
    let prev = textureLoad(uAcc, vec2<i32>(fc.xy), 0);
    // pixel jitter, lens sample (thin lens), primary ray
    let uv = ((fc.xy + U.uJitter) / U.uRes * 2.0 - 1.0) * vec2<f32>(U.uRes.x / U.uRes.y, -1.0);
    var rd = normalize(U.uCamF + (U.uCamR * uv.x + U.uCamU * uv.y) * U.uFov);
    var ro = U.uCamPos;
    if (U.uAperture > 0.0) {
        let fp = ro + rd * (U.uFocus / dot(rd, U.uCamF));
        let a = 2.0 * PI * rnd();
        let r = sqrt(rnd()) * U.uAperture;
        ro = ro + U.uCamR * (r * cos(a)) + U.uCamU * (r * sin(a));
        rd = normalize(fp - ro);
    }
    let cone = U.uFov / U.uRes.y * 0.7;
    let h = march(ro, rd, 6000.0, 260, cone);
    var col: vec3<f32>;
    var tEnd = h.x;
    if (h.y < 0.5) {
        col = sky(rd);
        tEnd = 1e5;
    } else {
        let p = ro + rd * h.x;
        let ng = normalAt(p, h.x);
        gFoot = 2.0 * U.uFov / U.uRes.y * h.x / max(abs(dot(rd, ng)), 0.25);
        let s = surfaceAt(p, ng, h.y);
        col = shade(p, rd, s, ng);
    }
    if (h.y > 0.5) {
        let f = fogSeg(ro, rd, h.x);
        col = col * f.w + f.xyz;
    } else {
        // the sky behind the haze: fog over a long horizontal path
        let f = fogSeg(ro, rd, 9000.0);
        col = col * f.w + f.xyz;
    }
    let cl = clouds(ro, rd, tEnd);
    col = col * cl.w + cl.xyz;
    col = min(col, vec3<f32>(40.0));   // fireflies
    if (!(col.x == col.x && col.y == col.y && col.z == col.z)) { col = vec3<f32>(0.0); }
    return prev + vec4<f32>(col * U.uWeight, U.uWeight);
}
