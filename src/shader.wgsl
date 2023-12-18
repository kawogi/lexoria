struct Globals {
    view_proj: mat4x4<f32>,
    num_lights: vec4<u32>,
};

@group(0)
@binding(0)
var<uniform> u_globals: Globals;

struct Entity {
    world: mat4x4<f32>,
    color: vec4<f32>,
};

@group(1)
@binding(0)
var<uniform> u_entity: Entity;

@vertex
fn vs_bake(@location(0) position: vec4<i32>) -> @builtin(position) vec4<f32> {
    return u_globals.view_proj * u_entity.world * vec4<f32>(position);
}

struct VertexOutput {
    @builtin(position) proj_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec4<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec4<i32>,
    @location(1) normal: vec4<i32>,
    @location(2) color: vec4<f32>,
) -> VertexOutput {
    let w = u_entity.world;
    let world_pos = u_entity.world * vec4<f32>(position);
    var result: VertexOutput;
    result.world_normal = mat3x3<f32>(w[0].xyz, w[1].xyz, w[2].xyz) * vec3<f32>(normal.xyz);
    result.world_position = world_pos;
    result.proj_position = u_globals.view_proj * world_pos;
    result.color = color;
    return result;
}

// fragment shader

struct Light {
    proj: mat4x4<f32>,
    pos: vec4<f32>,
    color: vec4<f32>,
};

@group(0)
@binding(1)
var<storage, read> s_lights: array<Light>;
@group(0)
@binding(1)
var<uniform> u_lights: array<Light, 10>; // Used when storage types are not supported

const c_ambient: vec3<f32> = vec3<f32>(0.05, 0.05, 0.05);
const c_max_lights: u32 = 10u;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(vertex.world_normal);
    // accumulate color
    var color: vec3<f32> = c_ambient;
    for(var i = 0u; i < min(u_globals.num_lights.x, c_max_lights); i += 1u) {
        let light = s_lights[i];
        // project into the light space
        // compute Lambertian diffuse term
        let light_dir = normalize(light.pos.xyz - vertex.world_position.xyz);
        let diffuse = max(0.0, dot(normal, light_dir));
        // add light contribution
        color += diffuse * light.color.xyz;
    }
    // multiply the light by material color
    return vec4<f32>(color, 1.0) * vertex.color;
}

// The fragment entrypoint used when storage buffers are not available for the lights
@fragment
fn fs_main_without_storage(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(vertex.world_normal);
    var color: vec3<f32> = c_ambient;
    for(var i = 0u; i < min(u_globals.num_lights.x, c_max_lights); i += 1u) {
        // This line is the only difference from the entrypoint above. It uses the lights
        // uniform instead of the lights storage buffer
        let light = u_lights[i];
        let light_dir = normalize(light.pos.xyz - vertex.world_position.xyz);
        let diffuse = max(0.0, dot(normal, light_dir));
        color += diffuse * light.color.xyz;
    }
    return vec4<f32>(color, 1.0) * vertex.color;
}
