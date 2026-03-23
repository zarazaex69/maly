struct Params {
    center_x: f32,
    center_y: f32,
    zoom: f32,
    max_iter: u32,
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let hp = h * 6.0;
    let x = c * (1.0 - abs(hp % 2.0 - 1.0));
    let m = v - c;
    var rgb: vec3<f32>;
    if hp < 1.0 {
        rgb = vec3<f32>(c, x, 0.0);
    } else if hp < 2.0 {
        rgb = vec3<f32>(x, c, 0.0);
    } else if hp < 3.0 {
        rgb = vec3<f32>(0.0, c, x);
    } else if hp < 4.0 {
        rgb = vec3<f32>(0.0, x, c);
    } else if hp < 5.0 {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    return rgb + vec3<f32>(m, m, m);
}

fn turbo_colormap(t: f32) -> vec3<f32> {
    let r = vec4<f32>(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    let g = vec4<f32>(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    let b = vec4<f32>(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    let r2 = vec2<f32>(-152.94239396, 59.28637943);
    let g2 = vec2<f32>(4.27729857, 2.82956604);
    let b2 = vec2<f32>(-89.90310912, 27.34824973);

    let v = clamp(t, 0.0, 1.0);
    let v2 = v * v;
    let v3 = v2 * v;
    let v4 = v2 * v2;

    let rv = r.x + r.y * v + r.z * v2 + r.w * v3 + r2.x * v4 + r2.y * v4 * v;
    let gv = g.x + g.y * v + g.z * v2 + g.w * v3 + g2.x * v4 + g2.y * v4 * v;
    let bv = b.x + b.y * v + b.z * v2 + b.w * v3 + b2.x * v4 + b2.y * v4 * v;

    return clamp(vec3<f32>(rv, gv, bv), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if px >= params.width || py >= params.height {
        return;
    }

    let min_dim = f32(min(params.width, params.height));
    let pixel_size = params.zoom / min_dim;
    let half_w = f32(params.width) * 0.5;
    let half_h = f32(params.height) * 0.5;

    let cx = params.center_x + pixel_size * (f32(px) - half_w);
    let cy = params.center_y + pixel_size * (f32(py) - half_h);

    var zx = 0.0f;
    var zy = 0.0f;
    var zx2 = 0.0f;
    var zy2 = 0.0f;
    var iter = 0u;
    let max_i = params.max_iter;

    loop {
        if iter >= max_i {
            break;
        }
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        zx2 = zx * zx;
        zy2 = zy * zy;
        if zx2 + zy2 > 4.0 {
            break;
        }
        iter = iter + 1u;
    }

    var color: vec3<f32>;
    if iter >= max_i {
        color = vec3<f32>(0.0, 0.0, 0.0);
    } else {
        let modulus = sqrt(zx2 + zy2);
        let smooth_val = f32(iter) + 1.0 - log2(log(modulus));
        let t = fract(smooth_val * 0.02);
        color = turbo_colormap(t);
    }

    let idx = py * params.width + px;
    let r = u32(color.x * 255.0);
    let g = u32(color.y * 255.0);
    let b = u32(color.z * 255.0);
    output[idx] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
