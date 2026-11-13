struct Status {
    bad: atomic<u32>,
};

@group(0) @binding(0)
var source: texture_storage_2d<rgba32float, read>;
@group(0) @binding(17)
var<storage, read_write> status: Status;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let dimensions = textureDimensions(source);
    let pixel = invocation.x;
    if (pixel >= dimensions.x * dimensions.y) {
        return;
    }
    let position = vec2<i32>(i32(pixel % dimensions.x), i32(pixel / dimensions.x));
    let color = textureLoad(source, position);
    if (any((bitcast<vec4<u32>>(color) & vec4<u32>(0x7f800000u)) == vec4<u32>(0x7f800000u))) {
        atomicOr(&status.bad, 4u);
    }
}
