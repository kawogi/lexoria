#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
// start with xtask:
// cargo xtask run-wasm --bin lexoria

// start manually:
// clear && RUSTFLAGS=--cfg=web_sys_unstable_apis cargo build --target wasm32-unknown-unknown
// wasm-bindgen --out-dir target/generated --web target/wasm32-unknown-unknown/debug/lexoria.wasm
// cd target/generated
// simple-http-server --nocache

mod framework;
mod utils;

use std::{borrow::Cow, f32::consts, iter, mem, ops::Range, sync::Arc};

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use rand::Rng;
use web_time::Instant;
use wgpu::util::{align_to, DeviceExt};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [i16; 4],
    _normal: [i8; 4],
    _color: [f32; 4],
}

fn vertex(pos: [i16; 3], nor: [i8; 3], color: [f32; 3]) -> Vertex {
    Vertex {
        _pos: [pos[0], pos[1], pos[2], 1],
        _normal: [nor[0], nor[1], nor[2], 0],
        _color: [color[0], color[1], color[2], 1.0],
    }
}

#[allow(clippy::too_many_lines)]
fn create_cube() -> (Vec<Vertex>, Vec<u16>) {
    const X: usize = 16;
    const Y: usize = 16;
    const Z: usize = 4;
    const HX: i16 = X as i16 / 2;
    const HY: i16 = Y as i16 / 2;
    const HZ: i16 = Z as i16 / 2;
    let mut rng = rand::thread_rng();
    let mut voxels = [[[false; X]; Y]; Z];
    for (z, xy_plane) in voxels.iter_mut().enumerate() {
        for (y, x_column) in xy_plane.iter_mut().enumerate() {
            for (x, voxel) in x_column.iter_mut().enumerate() {
                //*voxel = (x ^ y ^ z) & 1 == 0;
                *voxel = rng.gen();
            }
        }
    }

    let is_set = |x: i16, y: i16, z: i16| -> bool {
        let Ok(x) = usize::try_from(x + HX) else {
            return false;
        };
        let Ok(y) = usize::try_from(y + HY) else {
            return false;
        };
        let Ok(z) = usize::try_from(z + HZ) else {
            return false;
        };
        let Some(xy_plane) = voxels.get(z) else {
            return false;
        };
        let Some(x_column) = xy_plane.get(y) else {
            return false;
        };
        let Some(&voxel) = x_column.get(x) else {
            return false;
        };

        voxel
    };

    let mut vertex_data = Vec::new();

    for x in -HX..=HX {
        for z in -HZ..HZ {
            for y in -HY..HY {
                let is_a = is_set(x - 1, y, z);
                let is_b = is_set(x, y, z);
                match (is_a, is_b) {
                    (false, true) => {
                        let normal = [-1, 0, 0];
                        let color = [rng.gen(), rng.gen(), rng.gen()];
                        vertex_data.push(vertex([x, y, z], normal, color));
                        vertex_data.push(vertex([x, y + 1, z], normal, color));
                        vertex_data.push(vertex([x, y, z + 1], normal, color));
                        vertex_data.push(vertex([x, y + 1, z + 1], normal, color));
                    }
                    (true, false) => {
                        let normal = [1, 0, 0];
                        let color = [rng.gen(), rng.gen(), rng.gen()];
                        vertex_data.push(vertex([x, y, z], normal, color));
                        vertex_data.push(vertex([x, y, z + 1], normal, color));
                        vertex_data.push(vertex([x, y + 1, z], normal, color));
                        vertex_data.push(vertex([x, y + 1, z + 1], normal, color));
                    }
                    _ => {}
                }
            }
        }
    }

    for y in -HY..=HY {
        for x in -HX..HX {
            for z in -HZ..HZ {
                let is_a = is_set(x, y - 1, z);
                let is_b = is_set(x, y, z);
                match (is_a, is_b) {
                    (false, true) => {
                        let normal = [0, -1, 0];
                        let color = [rng.gen(), rng.gen(), rng.gen()];
                        vertex_data.push(vertex([x, y, z], normal, color));
                        vertex_data.push(vertex([x, y, z + 1], normal, color));
                        vertex_data.push(vertex([x + 1, y, z], normal, color));
                        vertex_data.push(vertex([x + 1, y, z + 1], normal, color));
                    }
                    (true, false) => {
                        let normal = [0, 1, 0];
                        let color = [rng.gen(), rng.gen(), rng.gen()];
                        vertex_data.push(vertex([x, y, z], normal, color));
                        vertex_data.push(vertex([x + 1, y, z], normal, color));
                        vertex_data.push(vertex([x, y, z + 1], normal, color));
                        vertex_data.push(vertex([x + 1, y, z + 1], normal, color));
                    }
                    _ => {}
                }
            }
        }
    }

    for z in -HZ..=HZ {
        for y in -HY..HY {
            for x in -HX..HX {
                let is_a = is_set(x, y, z - 1);
                let is_b = is_set(x, y, z);
                match (is_a, is_b) {
                    (false, true) => {
                        let normal = [0, 0, -1];
                        let color = [rng.gen(), rng.gen(), rng.gen()];
                        vertex_data.push(vertex([x, y, z], normal, color));
                        vertex_data.push(vertex([x + 1, y, z], normal, color));
                        vertex_data.push(vertex([x, y + 1, z], normal, color));
                        vertex_data.push(vertex([x + 1, y + 1, z], normal, color));
                    }
                    (true, false) => {
                        let normal = [0, 0, 1];
                        let color = [rng.gen(), rng.gen(), rng.gen()];
                        vertex_data.push(vertex([x, y, z], normal, color));
                        vertex_data.push(vertex([x, y + 1, z], normal, color));
                        vertex_data.push(vertex([x + 1, y, z], normal, color));
                        vertex_data.push(vertex([x + 1, y + 1, z], normal, color));
                    }
                    _ => {}
                }
            }
        }
    }

    // let quad_count = (vertex_data.len() / 4) as u16;
    let index_data = (0..vertex_data.len())
        .map(|index| index as u16 & !0b11)
        .flat_map(|index| [index, index + 2, index + 1, index + 1, index + 2, index + 3])
        .collect::<Vec<_>>();

    // let vertex_data = [
    //     // top (0, 0, 1)
    //     vertex([-1, -1, 1], [0, 0, 1]),
    //     vertex([1, -1, 1], [0, 0, 1]),
    //     vertex([1, 1, 1], [0, 0, 1]),
    //     vertex([-1, 1, 1], [0, 0, 1]),
    //     // bottom (0, 0, -1)
    //     vertex([-1, 1, -1], [0, 0, -1]),
    //     vertex([1, 1, -1], [0, 0, -1]),
    //     vertex([1, -1, -1], [0, 0, -1]),
    //     vertex([-1, -1, -1], [0, 0, -1]),
    //     // right (1, 0, 0)
    //     vertex([1, -1, -1], [1, 0, 0]),
    //     vertex([1, 1, -1], [1, 0, 0]),
    //     vertex([1, 1, 1], [1, 0, 0]),
    //     vertex([1, -1, 1], [1, 0, 0]),
    //     // left (-1, 0, 0)
    //     vertex([-1, -1, 1], [-1, 0, 0]),
    //     vertex([-1, 1, 1], [-1, 0, 0]),
    //     vertex([-1, 1, -1], [-1, 0, 0]),
    //     vertex([-1, -1, -1], [-1, 0, 0]),
    //     // front (0, 1, 0)
    //     vertex([1, 1, -1], [0, 1, 0]),
    //     vertex([-1, 1, -1], [0, 1, 0]),
    //     vertex([-1, 1, 1], [0, 1, 0]),
    //     vertex([1, 1, 1], [0, 1, 0]),
    //     // back (0, -1, 0)
    //     vertex([1, -1, 1], [0, -1, 0]),
    //     vertex([-1, -1, 1], [0, -1, 0]),
    //     vertex([-1, -1, -1], [0, -1, 0]),
    //     vertex([1, -1, -1], [0, -1, 0]),
    // ];

    // let index_data: &[u16] = &[
    //     0, 1, 2, 2, 3, 0, // top
    //     4, 5, 6, 6, 7, 4, // bottom
    //     8, 9, 10, 10, 11, 8, // right
    //     12, 13, 14, 14, 15, 12, // left
    //     16, 17, 18, 18, 19, 16, // front
    //     20, 21, 22, 22, 23, 20, // back
    // ];

    (vertex_data, index_data)
}

// fn create_plane(size: i8) -> (Vec<Vertex>, Vec<u16>) {
//     let vertex_data = [
//         vertex([size, -size, 0], [0, 0, 1]),
//         vertex([size, size, 0], [0, 0, 1]),
//         vertex([-size, -size, 0], [0, 0, 1]),
//         vertex([-size, size, 0], [0, 0, 1]),
//     ];

//     let index_data: &[u16] = &[0, 1, 2, 2, 1, 3];

//     (vertex_data.to_vec(), index_data.to_vec())
// }

struct Entity {
    mx_world: glam::Mat4,
    rotation_speed: f32,
    color: wgpu::Color,
    vertex_buf: Arc<wgpu::Buffer>,
    index_buf: Arc<wgpu::Buffer>,
    index_format: wgpu::IndexFormat,
    index_count: usize,
    uniform_offset: wgpu::DynamicOffset,
}

struct Light {
    pos: glam::Vec3,
    color: wgpu::Color,
    fov: f32,
    depth: Range<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightRaw {
    proj: [[f32; 4]; 4],
    pos: [f32; 4],
    color: [f32; 4],
}

impl Light {
    fn to_raw(&self) -> LightRaw {
        let view = glam::Mat4::look_at_rh(self.pos, glam::Vec3::ZERO, glam::Vec3::Z);
        let projection = glam::Mat4::perspective_rh(
            self.fov.to_radians(),
            1.0,
            self.depth.start,
            self.depth.end,
        );
        let view_proj = projection * view;
        LightRaw {
            proj: view_proj.to_cols_array_2d(),
            pos: [self.pos.x, self.pos.y, self.pos.z, 1.0],
            color: [
                self.color.r as f32,
                self.color.g as f32,
                self.color.b as f32,
                1.0,
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalUniforms {
    proj: [[f32; 4]; 4],
    num_lights: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EntityUniforms {
    model: [[f32; 4]; 4],
    color: [f32; 4],
}

struct Pass {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
}

struct Example {
    entities: Vec<Entity>,
    lights: Vec<Light>,
    lights_are_dirty: bool,
    forward_pass: Pass,
    forward_depth: wgpu::TextureView,
    entity_bind_group: wgpu::BindGroup,
    light_storage_buf: wgpu::Buffer,
    entity_uniform_buf: wgpu::Buffer,
    time: Instant,
}

impl Example {
    const MAX_LIGHTS: usize = 10;
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::empty(),
            shader_model: wgpu::ShaderModel::Sm5,
            ..wgpu::DownlevelCapabilities::default()
        }
    }
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_webgl2_defaults() // These downlevel limits will allow the code to run on all possible hardware
    }

    fn generate_matrix(aspect_ratio: f32) -> glam::Mat4 {
        let projection = glam::Mat4::perspective_rh(consts::FRAC_PI_4, aspect_ratio, 1.0, 20.0);
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(3.0f32, -10.0, 6.0),
            glam::Vec3::new(0f32, 0.0, 0.0),
            glam::Vec3::Z,
        );
        projection * view
    }

    fn create_depth_texture(
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
    ) -> wgpu::TextureView {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: None,
            view_formats: &[],
        });

        depth_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn optional_features() -> wgpu::Features {
        wgpu::Features::DEPTH_CLIP_CONTROL
    }

    #[allow(clippy::too_many_lines)]
    fn init(
        config: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        struct CubeDesc {
            offset: glam::Vec3,
            angle: f32,
            scale: f32,
            rotation: f32,
        }

        let supports_storage_resources = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::VERTEX_STORAGE)
            && device.limits().max_storage_buffers_per_shader_stage > 0;

        // Create the vertex and index buffers
        let vertex_size = mem::size_of::<Vertex>();
        let (cube_vertex_data, cube_index_data) = create_cube();
        let cube_vertex_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Cubes Vertex Buffer"),
                contents: bytemuck::cast_slice(&cube_vertex_data),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));

        let cube_index_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Cubes Index Buffer"),
                contents: bytemuck::cast_slice(&cube_index_data),
                usage: wgpu::BufferUsages::INDEX,
            },
        ));

        // let (plane_vertex_data, plane_index_data) = create_plane(7);
        // let plane_vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("Plane Vertex Buffer"),
        //     contents: bytemuck::cast_slice(&plane_vertex_data),
        //     usage: wgpu::BufferUsages::VERTEX,
        // });

        // let plane_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("Plane Index Buffer"),
        //     contents: bytemuck::cast_slice(&plane_index_data),
        //     usage: wgpu::BufferUsages::INDEX,
        // });

        let cube_descs = [
            CubeDesc {
                offset: glam::Vec3::new(0.0, 0.0, -2.0),
                angle: 180.0,
                scale: 0.5,
                rotation: 0.2,
            },
            // CubeDesc {
            //     offset: glam::Vec3::new(2.0, -2.0, 2.0),
            //     angle: 50.0,
            //     scale: 1.3,
            //     rotation: 0.2,
            // },
            // CubeDesc {
            //     offset: glam::Vec3::new(-2.0, 2.0, 2.0),
            //     angle: 140.0,
            //     scale: 1.1,
            //     rotation: 0.3,
            // },
            // CubeDesc {
            //     offset: glam::Vec3::new(2.0, 2.0, 2.0),
            //     angle: 210.0,
            //     scale: 0.9,
            //     rotation: 0.4,
            // },
        ];

        let entity_uniform_size = mem::size_of::<EntityUniforms>() as wgpu::BufferAddress;
        let num_entities = 1 + cube_descs.len() as wgpu::BufferAddress;
        // Make the `uniform_alignment` >= `entity_uniform_size` and aligned to `min_uniform_buffer_offset_alignment`.
        let uniform_alignment = {
            let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment);
            align_to(entity_uniform_size, alignment)
        };
        // Note: dynamic uniform offsets also have to be aligned to `Limits::min_uniform_buffer_offset_alignment`.
        let entity_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: num_entities * uniform_alignment,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_format = wgpu::IndexFormat::Uint16;

        let mut entities = vec![
            // {
            // Entity {
            //     mx_world: glam::Mat4::IDENTITY,
            //     rotation_speed: 0.0,
            //     color: wgpu::Color::WHITE,
            //     vertex_buf: Arc::new(plane_vertex_buf),
            //     index_buf: Arc::new(plane_index_buf),
            //     index_format,
            //     index_count: plane_index_data.len(),
            //     uniform_offset: 0,
            // }
        // }
        ];

        for (i, cube) in cube_descs.iter().enumerate() {
            let mx_world = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::splat(cube.scale),
                glam::Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), cube.angle.to_radians()),
                cube.offset,
            );
            entities.push(Entity {
                mx_world,
                rotation_speed: cube.rotation,
                color: wgpu::Color::GREEN,
                vertex_buf: Arc::clone(&cube_vertex_buf),
                index_buf: Arc::clone(&cube_index_buf),
                index_format,
                index_count: cube_index_data.len(),
                uniform_offset: ((i + 1) * uniform_alignment as usize) as _,
            });
        }

        let local_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(entity_uniform_size),
                    },
                    count: None,
                }],
                label: None,
            });
        let entity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &local_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &entity_uniform_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(entity_uniform_size),
                }),
            }],
            label: None,
        });

        let lights = vec![
            Light {
                pos: glam::Vec3::new(7.0, -5.0, 10.0),
                color: wgpu::Color {
                    r: 0.5,
                    g: 1.0,
                    b: 0.5,
                    a: 1.0,
                },
                fov: 60.0,
                depth: 1.0..20.0,
            },
            Light {
                pos: glam::Vec3::new(-5.0, 7.0, 10.0),
                color: wgpu::Color {
                    r: 1.0,
                    g: 0.5,
                    b: 0.5,
                    a: 1.0,
                },
                fov: 45.0,
                depth: 1.0..20.0,
            },
        ];
        let light_uniform_size =
            (Self::MAX_LIGHTS * mem::size_of::<LightRaw>()) as wgpu::BufferAddress;
        let light_storage_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: light_uniform_size,
            usage: if supports_storage_resources {
                wgpu::BufferUsages::STORAGE
            } else {
                wgpu::BufferUsages::UNIFORM
            } | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_attr = wgpu::vertex_attr_array![0 => Sint16x4, 1 => Sint8x4, 2 => Float32x4];
        let vb_desc = wgpu::VertexBufferLayout {
            array_stride: vertex_size as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attr,
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let forward_pass = {
            // Create pipeline layout
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0, // global
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(
                                    mem::size_of::<GlobalUniforms>() as _,
                                ),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1, // lights
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: if supports_storage_resources {
                                    wgpu::BufferBindingType::Storage { read_only: true }
                                } else {
                                    wgpu::BufferBindingType::Uniform
                                },
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(light_uniform_size),
                            },
                            count: None,
                        },
                    ],
                    label: None,
                });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("main"),
                bind_group_layouts: &[&bind_group_layout, &local_bind_group_layout],
                push_constant_ranges: &[],
            });

            let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
            let forward_uniforms = GlobalUniforms {
                proj: mx_total.to_cols_array_2d(),
                num_lights: [lights.len() as u32, 0, 0, 0],
            };
            let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(&forward_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            // Create bind group
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: light_storage_buf.as_entire_binding(),
                    },
                ],
                label: None,
            });

            // Create the render pipeline
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("main"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[vb_desc],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: if supports_storage_resources {
                        "fs_main"
                    } else {
                        "fs_main_without_storage"
                    },
                    targets: &[Some(config.view_formats[0].into())],
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: Self::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

            Pass {
                pipeline,
                bind_group,
                uniform_buf,
            }
        };

        let forward_depth = Self::create_depth_texture(config, device);

        Example {
            entities,
            lights,
            lights_are_dirty: true,
            forward_pass,
            forward_depth,
            light_storage_buf,
            entity_uniform_buf,
            entity_bind_group,
            time: Instant::now(),
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        // update view-projection matrix
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        queue.write_buffer(
            &self.forward_pass.uniform_buf,
            0,
            bytemuck::cast_slice(mx_ref),
        );

        self.forward_depth = Self::create_depth_texture(config, device);
    }

    #[allow(clippy::too_many_lines)]
    fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &framework::Spawner,
    ) {
        let now = Instant::now();
        let dt = now - self.time;
        self.time = now;

        // update uniforms
        for entity in &mut self.entities {
            if entity.rotation_speed != 0.0 {
                let rotation =
                    glam::Mat4::from_rotation_z(entity.rotation_speed * dt.as_secs_f32());
                entity.mx_world *= rotation;
            }
            let data = EntityUniforms {
                model: entity.mx_world.to_cols_array_2d(),
                color: [
                    entity.color.r as f32,
                    entity.color.g as f32,
                    entity.color.b as f32,
                    entity.color.a as f32,
                ],
            };
            queue.write_buffer(
                &self.entity_uniform_buf,
                u64::from(entity.uniform_offset),
                bytemuck::bytes_of(&data),
            );
        }

        if self.lights_are_dirty {
            self.lights_are_dirty = false;
            for (i, light) in self.lights.iter().enumerate() {
                queue.write_buffer(
                    &self.light_storage_buf,
                    (i * mem::size_of::<LightRaw>()) as wgpu::BufferAddress,
                    bytemuck::bytes_of(&light.to_raw()),
                );
            }
        }

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // forward pass
        encoder.push_debug_group("forward rendering pass");
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.forward_depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.forward_pass.pipeline);
            pass.set_bind_group(0, &self.forward_pass.bind_group, &[]);

            for entity in &self.entities {
                pass.set_bind_group(1, &self.entity_bind_group, &[entity.uniform_offset]);
                pass.set_index_buffer(entity.index_buf.slice(..), entity.index_format);
                pass.set_vertex_buffer(0, entity.vertex_buf.slice(..));
                pass.draw_indexed(0..entity.index_count as u32, 0, 0..1);
            }
        }
        encoder.pop_debug_group();

        queue.submit(iter::once(encoder.finish()));
    }
}

//#[cfg(not(test))]
fn main() {
    framework::run("Lexoria");
}
