const shDeg3Code = `
    // spherical harmonic coefficients
    const SH_C0 = 0.28209479177387814f;
    const SH_C1 = 0.4886025119029199f;
    const SH_C2 = array(
        1.0925484305920792f,
        -1.0925484305920792f,
        0.31539156525252005f,
        -1.0925484305920792f,
        0.5462742152960396f
    );
    const SH_C3 = array(
        -0.5900435899266435f,
        2.890611442640554f,
        -0.4570457994644658f,
        0.3731763325901154f,
        -0.4570457994644658f,
        1.445305721320277f,
        -0.5900435899266435f
    );

    fn compute_color_from_sh(position: vec3<f32>, sh: array<vec3<f32>, 16>) -> vec3<f32> {
        let dir = normalize(position - uniforms.camera_position);
        var result = SH_C0 * sh[0];

        // if deg > 0
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result = result + SH_C1 * (-y * sh[1] + z * sh[2] - x * sh[3]);

        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;

        // if (sh_degree > 1) {
        result = result +
            SH_C2[0] * xy * sh[4] +
            SH_C2[1] * yz * sh[5] +
            SH_C2[2] * (2. * zz - xx - yy) * sh[6] +
            SH_C2[3] * xz * sh[7] +
            SH_C2[4] * (xx - yy) * sh[8];
        
        // We disable the 3rd degree for now because of a bug causing the entire render
        // to be black. This appears to be a webGPU issue. If uncomment SH_C3[0...3] it works,
        // if you uncomment SH_C3[4...6] it works, if you uncomment the whole and divide by 10
        // it's black.

        // if (sh_degree > 2) {
        //result = result +
        //    SH_C3[0] * y * (3. * xx - yy) * sh[9] +
        //    SH_C3[1] * xy * z * sh[10] +
        //    SH_C3[2] * y * (4. * zz - xx - yy) * sh[11] +
        //    SH_C3[3] * z * (2. * zz - 3. * xx - 3. * yy) * sh[12] +
        //    SH_C3[4] * x * (4. * zz - xx - yy) * sh[13] +
        //    SH_C3[5] * z * (xx - yy) * sh[14] +
        //    SH_C3[6] * x * (xx - 3. * yy) * sh[15];

        // unconditional
        result = result + 0.5;

        return max(result, vec3<f32>(0.));
    }
`;

const shDeg2Code = `
    // spherical harmonic coefficients
    const SH_C0 = 0.28209479177387814f;
    const SH_C1 = 0.4886025119029199f;
    const SH_C2 = array(
        1.0925484305920792f,
        -1.0925484305920792f,
        0.31539156525252005f,
        -1.0925484305920792f,
        0.5462742152960396f
    );

    fn compute_color_from_sh(position: vec3<f32>, sh: array<vec3<f32>, 9>) -> vec3<f32> {
        let dir = normalize(position - uniforms.camera_position);
        var result = SH_C0 * sh[0];

        // if deg > 0
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result = result + SH_C1 * (-y * sh[1] + z * sh[2] - x * sh[3]);

        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;

        // if (sh_degree > 1) {
        result = result +
            SH_C2[0] * xy * sh[4] +
            SH_C2[1] * yz * sh[5] +
            SH_C2[2] * (2. * zz - xx - yy) * sh[6] +
            SH_C2[3] * xz * sh[7] +
            SH_C2[4] * (xx - yy) * sh[8];
        
        // unconditional
        result = result + 0.5;

        return max(result, vec3<f32>(0.));
    }
`;

const shDeg1Code = `
    // spherical harmonic coefficients
    const SH_C0 = 0.28209479177387814f;
    const SH_C1 = 0.4886025119029199f;

    fn compute_color_from_sh(position: vec3<f32>, sh: array<vec3<f32>, 4>) -> vec3<f32> {
        let dir = normalize(position - uniforms.camera_position);
        var result = SH_C0 * sh[0];

        // if deg > 0
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result = result + SH_C1 * (-y * sh[1] + z * sh[2] - x * sh[3]);

        // unconditional
        result = result + 0.5;

        return max(result, vec3<f32>(0.));
    }
`;


export function getShaderCode(canvas: HTMLCanvasElement, shDegree: number, nShCoeffs: number) {
    const shComputeCode = {
        1: shDeg1Code,
        2: shDeg2Code,
        3: shDeg3Code,
    }[shDegree];

    const shaderCode = `
// for some reason passing these as uniform is broken
const canvas_height = ${canvas.height};
const canvas_width = ${canvas.width};
const sh_degree = ${shDegree};
const n_sh_coeffs = ${nShCoeffs};

struct PointInput {
    @location(0) position: vec3<f32>,
    @location(1) log_scale: vec3<f32>,
    @location(2) rot: vec4<f32>,
    @location(3) opacity_logit: f32,
    sh: array<vec3<f32>, n_sh_coeffs>,
};

struct PointOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) conic_and_opacity: vec4<f32>,
};

struct Uniforms {
    viewMatrix: mat4x4<f32>,
    projMatrix: mat4x4<f32>,
    camera_position: vec3<f32>,
    tan_fovx: f32,
    tan_fovy: f32,
    focal_x: f32,
    focal_y: f32,
    scale_modifier: f32,
};

${shComputeCode}

fn sigmoid(x: f32) -> f32 {
    if (x >= 0.) {
        return 1. / (1. + exp(-x));
    } else {
        let z = exp(x);
        return z / (1. + z);
    }
}

fn compute_cov3d(log_scale: vec3<f32>, rot: vec4<f32>) -> array<f32, 6> {
    let modifier = uniforms.scale_modifier;
    let S = mat3x3<f32>(
        exp(log_scale.x) * modifier, 0., 0.,
        0., exp(log_scale.y) * modifier, 0.,
        0., 0., exp(log_scale.z) * modifier,
    );

    let r = rot.x;
    let x = rot.y;
    let y = rot.z;
    let z = rot.w;

    let R = mat3x3<f32>(
        1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y),
        2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
        2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y),
    );

    let M = S * R;
    let Sigma = transpose(M) * M;

    return array<f32, 6>(
        Sigma[0][0],
        Sigma[0][1],
        Sigma[0][2],
        Sigma[1][1],
        Sigma[1][2],
        Sigma[2][2],
    );
} 

fn ndc2pix(v: f32, size: u32) -> f32 {
    return ((v + 1.0) * f32(size) - 1.0) * 0.5;
}

fn compute_cov2d(position: vec3<f32>, log_scale: vec3<f32>, rot: vec4<f32>) -> vec3<f32> {
    let cov3d = compute_cov3d(log_scale, rot);

    var t = uniforms.viewMatrix * vec4<f32>(position, 1.0);

    let limx = 1.3 * uniforms.tan_fovx;
    let limy = 1.3 * uniforms.tan_fovy;
    let txtz = t.x / t.z;
    let tytz = t.y / t.z;

    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    let J = mat4x4(
        uniforms.focal_x / t.z, 0., -(uniforms.focal_x * t.x) / (t.z * t.z), 0.,
        0., uniforms.focal_y / t.z, -(uniforms.focal_y * t.y) / (t.z * t.z), 0.,
        0., 0., 0., 0.,
        0., 0., 0., 0.,
    );

    let W = transpose(uniforms.viewMatrix);

    let T = W * J;

    let Vrk = mat4x4(
        cov3d[0], cov3d[1], cov3d[2], 0.,
        cov3d[1], cov3d[3], cov3d[4], 0.,
        cov3d[2], cov3d[4], cov3d[5], 0.,
        0., 0., 0., 0.,
    );

    var cov = transpose(T) * transpose(Vrk) * T;

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;

    return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
}


@binding(0) @group(0) var<uniform> uniforms: Uniforms;
@binding(1) @group(1) var<storage, read> points: array<PointInput>;

const quadVertices = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
);

@vertex
fn vs_points(@builtin(vertex_index) vertex_index: u32) -> PointOutput {
    var output: PointOutput;
    let pointIndex = vertex_index / 6u;
    let quadIndex = vertex_index % 6u;
    let quadOffset = quadVertices[quadIndex];
    let point = points[pointIndex];

    let cov2d = compute_cov2d(point.position, point.log_scale, point.rot);
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let det_inv = 1.0 / det;
    let conic = vec3<f32>(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
    let mid = 0.5 * (cov2d.x + cov2d.z);
    let lambda_1 = mid + sqrt(max(0.1, mid * mid - det));
    let lambda_2 = mid - sqrt(max(0.1, mid * mid - det));
    let radius_px = ceil(3. * sqrt(max(lambda_1, lambda_2)));
    let radius_ndc = vec2<f32>(
    radius_px / (canvas_height),
    radius_px / (canvas_width),
    );
    output.conic_and_opacity = vec4<f32>(conic, sigmoid(point.opacity_logit));

    var projPosition = uniforms.projMatrix * vec4<f32>(point.position, 1.0);
    projPosition = projPosition / projPosition.w;
    output.position = vec4<f32>(projPosition.xy + 2 * radius_ndc * quadOffset, projPosition.zw);
    output.color = compute_color_from_sh(uniforms.camera_position, point.sh);
    output.uv = radius_px * quadOffset;

    return output;
}

@fragment
fn fs_main(input: PointOutput) -> @location(0) vec4<f32> {
    // we want the distance from the gaussian to the fragment while uv
    // is the reverse
    let d = -input.uv;
    let conic = input.conic_and_opacity.xyz;
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;
    let opacity = input.conic_and_opacity.w;

    if (power > 0.0) {
    discard;
    }

    let alpha = min(0.99, opacity * exp(power));

    return vec4<f32>(input.color * alpha, alpha);
}
`;

    return shaderCode;
}