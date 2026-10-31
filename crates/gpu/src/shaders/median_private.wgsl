// The same median for large radii, as the CPU computes it: a sliding 512-bin histogram per
// channel (Huang). One invocation per row segment of u[0].y pixels: it fills the histogram for
// the segment's first window, then slides right one column at a time (remove the column
// leaving, add the one entering) and moves each channel's median by the counts below it.
// u[0] = (r, segment length); dispatched over (rows, segments) with 64 × 1 workgroups.
// data = the 512 CPU-rounded level / 511 values.
// The histogram packs two 16-bit counts per u32 (bin i is the (i & 1) half of word i >> 1): a
// count is at most (2r + 1)², which fits for r <= MEDIAN_HUANG_MAX_R (fx_noise.rs), and a plain
// array<u32, 2048> is 8 KB, over WebKit's 8192-byte limit on a function's variables.
@compute @workgroup_size(64, 1)
fn fxn_median_huang(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = out_dims();
    let y = i32(gid.x);
    let seg = i32(P.u[0].y);
    let x0 = i32(gid.y) * seg;
    if (y >= dims.y || x0 >= dims.x) {
        return;
    }
    let x1 = min(x0 + seg, dims.x);
    let r = i32(P.u[0].x);
    let half = u32((2 * r + 1) * (2 * r + 1) / 2);
    var hist: array<u32, 1024>;
    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            let v = vec4<u32>(tex_get_clamped(src, x0 + dx, y + dy));
            for (var c = 0u; c < 4u; c++) {
                let i = c * 512u + v[c];
                hist[i >> 1u] += 1u << ((i & 1u) * 16u);
            }
        }
    }
    // Per channel: the median level and the count of values below it.
    var med = vec4<u32>(0u);
    var lt = vec4<u32>(0u);
    for (var x = x0; x < x1; x++) {
        if (x > x0) {
            for (var dy = -r; dy <= r; dy++) {
                let o = vec4<u32>(tex_get_clamped(src, x - 1 - r, y + dy));
                let n = vec4<u32>(tex_get_clamped(src, x + r, y + dy));
                for (var c = 0u; c < 4u; c++) {
                    let io = c * 512u + o[c];
                    hist[io >> 1u] -= 1u << ((io & 1u) * 16u);
                    if (o[c] < med[c]) {
                        lt[c] -= 1u;
                    }
                    let i_n = c * 512u + n[c];
                    hist[i_n >> 1u] += 1u << ((i_n & 1u) * 16u);
                    if (n[c] < med[c]) {
                        lt[c] += 1u;
                    }
                }
            }
        }
        for (var c = 0u; c < 4u; c++) {
            let base = c * 512u;
            while (lt[c] > half) {
                med[c] -= 1u;
                let i = base + med[c];
                lt[c] -= (hist[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu;
            }
            loop {
                let i = base + med[c];
                let count = (hist[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu;
                if (lt[c] + count > half) {
                    break;
                }
                lt[c] += count;
                med[c] += 1u;
            }
        }
        textureStore(out, vec2<i32>(x, y), vec4<f32>(data[med.x], data[med.y], data[med.z], data[med.w]));
    }
}

