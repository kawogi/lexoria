#![warn(clippy::pedantic)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
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

mod camera;
mod framework;
mod utils;

use std::{borrow::Cow, f32::consts::TAU, iter, mem, ops::Range, sync::Arc};

use bytemuck::{Pod, Zeroable};
use camera::Camera;
use glam::{Mat4, Vec3};
use rand::Rng;
use web_time::Instant;
use wgpu::{
    util::{align_to, DeviceExt},
    Queue,
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [i16; 4],
    _normal: [i8; 4],
    _color: [f32; 4],
}

struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
}

fn vertex(pos: [i16; 3], nor: [i8; 3], color: Vec3) -> Vertex {
    Vertex {
        _pos: [pos[0], pos[1], pos[2], 1],
        _normal: [nor[0], nor[1], nor[2], 0],
        _color: [color[0], color[1], color[2], 1.0],
    }
}

#[allow(clippy::too_many_lines)]
fn create_world() -> Mesh {
    const X: usize = 32;
    const Y: usize = 32;
    const Z: usize = 8;
    const HX: i16 = X as i16 / 2;
    const HY: i16 = Y as i16 / 2;
    const HZ: i16 = Z as i16 / 2;

    let mut rng = rand::thread_rng();

    let mut height_map = vec![vec![0.0; X]; Y];
    let mut max = 0.0_f32;
    for (y, heights) in height_map.iter_mut().enumerate() {
        for (x, height) in heights.iter_mut().enumerate() {
            let xx = x as f32 / X as f32;
            let yy = y as f32 / Y as f32;
            let h = (xx * 2.0 + 1.6).sin() * (yy * 3.0 + 0.8).sin();
            let h = h + (xx * 7.0 + 0.7).sin() * (yy * 6.6 + 1.3).sin() * 0.5;
            let h = h.max(0.0) + rng.gen::<f32>() * 0.1;
            max = max.max(h);
            *height = h;
        }
    }
    for heights in &mut height_map {
        for height in heights {
            *height *= Z as f32 / max;
        }
    }

    let colors = [
        Vec3::new(0.2, 0.6, 0.8),
        Vec3::new(1.0, 1.0, 0.5),
        Vec3::new(0.6, 0.8, 0.0),
        Vec3::new(0.2, 0.7, 0.1),
        Vec3::new(0.3, 0.7, 0.1),
        Vec3::new(0.7, 0.3, 0.2),
        Vec3::new(0.7, 0.7, 0.7),
        Vec3::new(1.0, 1.0, 1.0),
    ];

    let mut voxels = [[[Option::<Vec3>::None; X]; Y]; Z];
    for (z, xy_plane) in voxels.iter_mut().enumerate() {
        for (y, x_column) in xy_plane.iter_mut().enumerate() {
            for (x, voxel) in x_column.iter_mut().enumerate() {
                *voxel = ((z as f32) < height_map[y][x]).then_some(colors[z]);
            }
        }
    }

    let get_voxel = |x: i16, y: i16, z: i16| -> _ {
        let Ok(x) = usize::try_from(x + HX) else {
            return None;
        };
        let Ok(y) = usize::try_from(y + HY) else {
            return None;
        };
        let Ok(z) = usize::try_from(z + HZ) else {
            return None;
        };
        let Some(xy_plane) = voxels.get(z) else {
            return None;
        };
        let Some(x_column) = xy_plane.get(y) else {
            return None;
        };
        let Some(&voxel) = x_column.get(x) else {
            return None;
        };

        voxel
    };

    let mut vertices = Vec::new();

    for x in -HX..=HX {
        for z in -HZ..HZ {
            for y in -HY..HY {
                let is_a = get_voxel(x - 1, y, z);
                let is_b = get_voxel(x, y, z);
                match (is_a, is_b) {
                    (None, Some(color)) => {
                        let normal = [-1, 0, 0];
                        vertices.push(vertex([x, y, z], normal, color));
                        vertices.push(vertex([x, y + 1, z], normal, color));
                        vertices.push(vertex([x, y, z + 1], normal, color));
                        vertices.push(vertex([x, y + 1, z + 1], normal, color));
                    }
                    (Some(color), None) => {
                        let normal = [1, 0, 0];
                        vertices.push(vertex([x, y, z], normal, color));
                        vertices.push(vertex([x, y, z + 1], normal, color));
                        vertices.push(vertex([x, y + 1, z], normal, color));
                        vertices.push(vertex([x, y + 1, z + 1], normal, color));
                    }
                    _ => {}
                }
            }
        }
    }

    for y in -HY..=HY {
        for x in -HX..HX {
            for z in -HZ..HZ {
                let is_a = get_voxel(x, y - 1, z);
                let is_b = get_voxel(x, y, z);
                match (is_a, is_b) {
                    (None, Some(color)) => {
                        let normal = [0, -1, 0];
                        vertices.push(vertex([x, y, z], normal, color));
                        vertices.push(vertex([x, y, z + 1], normal, color));
                        vertices.push(vertex([x + 1, y, z], normal, color));
                        vertices.push(vertex([x + 1, y, z + 1], normal, color));
                    }
                    (Some(color), None) => {
                        let normal = [0, 1, 0];
                        vertices.push(vertex([x, y, z], normal, color));
                        vertices.push(vertex([x + 1, y, z], normal, color));
                        vertices.push(vertex([x, y, z + 1], normal, color));
                        vertices.push(vertex([x + 1, y, z + 1], normal, color));
                    }
                    _ => {}
                }
            }
        }
    }

    for z in -HZ..=HZ {
        for y in -HY..HY {
            for x in -HX..HX {
                let is_a = get_voxel(x, y, z - 1);
                let is_b = get_voxel(x, y, z);
                match (is_a, is_b) {
                    (None, Some(color)) => {
                        let normal = [0, 0, -1];
                        vertices.push(vertex([x, y, z], normal, color));
                        vertices.push(vertex([x + 1, y, z], normal, color));
                        vertices.push(vertex([x, y + 1, z], normal, color));
                        vertices.push(vertex([x + 1, y + 1, z], normal, color));
                    }
                    (Some(color), None) => {
                        let normal = [0, 0, 1];
                        vertices.push(vertex([x, y, z], normal, color));
                        vertices.push(vertex([x, y + 1, z], normal, color));
                        vertices.push(vertex([x + 1, y, z], normal, color));
                        vertices.push(vertex([x + 1, y + 1, z], normal, color));
                    }
                    _ => {}
                }
            }
        }
    }

    // let quad_count = (vertex_data.len() / 4) as u16;
    let indices = (0..vertices.len())
        .map(|index| index as u16 & !0b11)
        .flat_map(|index| [index, index + 2, index + 1, index + 1, index + 2, index + 3])
        .collect::<Vec<_>>();

    Mesh { vertices, indices }
}

struct Entity {
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
    _start_time: Instant,
    time: Instant,
    phase: u32,
    camera: Camera,
}

impl Example {
    const MAX_LIGHTS: usize = 10;
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    fn compute_camera_pos(phase: u32) -> Vec3 {
        let angle = TAU * (f64::from(phase) / 0x1_0000_0000_u64 as f64) as f32;
        glam::Vec3::new(
            15.0 * angle.cos(),
            15.0 * angle.sin(),
            15.0 + 3.0 * (angle * 2.0).sin(),
        )
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

    #[allow(clippy::too_many_lines)]
    fn init(
        config: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        let supports_storage_resources = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::VERTEX_STORAGE)
            && device.limits().max_storage_buffers_per_shader_stage > 0;

        // Create the vertex and index buffers
        let vertex_size = mem::size_of::<Vertex>();
        let Mesh {
            vertices: cube_vertex_data,
            indices: cube_index_data,
        } = create_world();
        let cube_vertex_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("World Vertex Buffer"),
                contents: bytemuck::cast_slice(&cube_vertex_data),
                usage: wgpu::BufferUsages::VERTEX,
            },
        ));

        let cube_index_buf = Arc::new(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("World Index Buffer"),
                contents: bytemuck::cast_slice(&cube_index_data),
                usage: wgpu::BufferUsages::INDEX,
            },
        ));

        let entity_uniform_size = mem::size_of::<EntityUniforms>() as wgpu::BufferAddress;
        let num_entities = 1 + 1 as wgpu::BufferAddress;
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

        let entities = vec![Entity {
            vertex_buf: Arc::clone(&cube_vertex_buf),
            index_buf: Arc::clone(&cube_index_buf),
            index_format,
            index_count: cube_index_data.len(),
            uniform_offset: ((0 + 1) * uniform_alignment as usize) as _,
        }];

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

        let lights = vec![Light {
            pos: glam::Vec3::new(14.0, -10.0, 20.0),
            color: wgpu::Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            fov: 60.0,
            depth: 1.0..20.0,
        }];
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

        let camera = Camera::new(
            Self::compute_camera_pos(0),
            Vec3::new(0.0, 0.0, 0.0),
            config.width as f32 / config.height as f32,
        );
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

            let forward_uniforms = GlobalUniforms {
                proj: camera.matrix().to_cols_array_2d(),
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
            _start_time: Instant::now(),
            time: Instant::now(),
            phase: 0,
            camera,
        }
    }

    fn update_camera(&self, queue: &Queue) {
        let camera_matrix = self.camera.matrix();
        let mx_ref: &[f32; 16] = camera_matrix.as_ref();
        queue.write_buffer(
            &self.forward_pass.uniform_buf,
            0,
            bytemuck::cast_slice(mx_ref),
        );
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        self.camera.aspect_ratio = config.width as f32 / config.height as f32;
        self.update_camera(queue);

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
        let phase_delta = ((dt.as_secs_f64() / 20.0).fract() * 0x1_0000_0000_u64 as f64) as u32;
        self.phase = self.phase.wrapping_add(phase_delta);

        self.camera.eye = Self::compute_camera_pos(self.phase);
        self.update_camera(queue);

        // update uniforms
        for entity in &mut self.entities {
            let color = wgpu::Color::GREEN;
            let data = EntityUniforms {
                model: Mat4::IDENTITY.to_cols_array_2d(),
                color: [
                    color.r as f32,
                    color.g as f32,
                    color.b as f32,
                    color.a as f32,
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
