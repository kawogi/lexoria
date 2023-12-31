use std::future::Future;
#[cfg(target_arch = "wasm32")]
use std::str::FromStr;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::{ImageBitmapRenderingContext, OffscreenCanvas};
use winit::{
    event::{self, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

use crate::Example;

// #[allow(dead_code)]
// pub fn cast_slice<T>(data: &[T]) -> &[u8] {
//     use std::{mem::size_of_val, slice::from_raw_parts};

//     unsafe { from_raw_parts(data.as_ptr() as *const u8, size_of_val(data)) }
// }

#[allow(dead_code)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

// pub trait Example: 'static + Sized {
//     fn optional_features() -> wgpu::Features {
//         wgpu::Features::empty()
//     }
//     fn required_features() -> wgpu::Features {
//         wgpu::Features::empty()
//     }
//     fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
//         wgpu::DownlevelCapabilities {
//             flags: wgpu::DownlevelFlags::empty(),
//             shader_model: wgpu::ShaderModel::Sm5,
//             ..wgpu::DownlevelCapabilities::default()
//         }
//     }
//     fn required_limits() -> wgpu::Limits {
//         wgpu::Limits::downlevel_webgl2_defaults() // These downlevel limits will allow the code to run on all possible hardware
//     }
//     fn init(
//         config: &wgpu::SurfaceConfiguration,
//         adapter: &wgpu::Adapter,
//         device: &wgpu::Device,
//         queue: &wgpu::Queue,
//     ) -> Self;
//     fn resize(
//         &mut self,
//         config: &wgpu::SurfaceConfiguration,
//         device: &wgpu::Device,
//         queue: &wgpu::Queue,
//     );
//     fn update(&mut self, event: WindowEvent);
//     fn render(
//         &mut self,
//         view: &wgpu::TextureView,
//         device: &wgpu::Device,
//         queue: &wgpu::Queue,
//         spawner: &Spawner,
//     );
// }

// Initialize logging in platform dependant ways.
fn init_logger() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            // As we don't have an environment to pull logging level from, we use the query string.
            let query_string = web_sys::window().unwrap().location().search().unwrap();
            let query_level: Option<log::LevelFilter> = parse_url_query_string(&query_string, "RUST_LOG")
                .and_then(|x| x.parse().ok());

            // We keep wgpu at Error level, as it's very noisy.
            let base_level = query_level.unwrap_or(log::LevelFilter::Info);
            let wgpu_level = query_level.unwrap_or(log::LevelFilter::Error);

            // On web, we use fern, as console_log doesn't have filtering on a per-module level.
            fern::Dispatch::new()
                .level(base_level)
                .level_for("wgpu_core", wgpu_level)
                .level_for("wgpu_hal", wgpu_level)
                .level_for("naga", wgpu_level)
                .chain(fern::Output::call(console_log::log))
                .apply()
                .unwrap();
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        } else {
            // parse_default_env will read the RUST_LOG environment variable and apply it on top
            // of these default filters.
            env_logger::builder()
                .filter_level(log::LevelFilter::Info)
                // We keep wgpu at Error level, as it's very noisy.
                .filter_module("wgpu_core", log::LevelFilter::Error)
                .filter_module("wgpu_hal", log::LevelFilter::Error)
                .filter_module("naga", log::LevelFilter::Error)
                .parse_default_env()
                .init();
        }
    }
}

struct Setup {
    window: winit::window::Window,
    event_loop: EventLoop<()>,
    instance: wgpu::Instance,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    #[cfg(target_arch = "wasm32")]
    offscreen_canvas_setup: Option<OffscreenCanvasSetup>,
}

#[cfg(target_arch = "wasm32")]
struct OffscreenCanvasSetup {
    offscreen_canvas: OffscreenCanvas,
    bitmap_renderer: ImageBitmapRenderingContext,
}

#[allow(clippy::too_many_lines)]
async fn setup(title: &str) -> Setup {
    // #[cfg(not(target_arch = "wasm32"))]
    // {
    //     env_logger::init();
    // };

    let event_loop = EventLoop::new();
    let mut builder = winit::window::WindowBuilder::new();
    builder = builder.with_title(title);
    #[cfg(windows_OFF)] // TODO
    {
        use winit::platform::windows::WindowBuilderExtWindows;
        builder = builder.with_no_redirection_bitmap(true);
    }
    let window = builder.build(&event_loop).unwrap();

    // #[cfg(target_arch = "wasm32")]
    // {
    //     use winit::platform::web::WindowExtWebSys;
    //     let canvas = window.canvas().expect("Couldn't get canvas");
    //     canvas.style().set_css_text("height: 100%; width: 100%;");
    //     // On wasm, append the canvas to the document body
    //     web_sys::window()
    //         .and_then(|win| win.document())
    //         .and_then(|doc| doc.body())
    //         .and_then(|body| body.append_child(&canvas).ok())
    //         .expect("couldn't append canvas to document body");
    // }

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        // let query_string = web_sys::window().unwrap().location().search().unwrap();
        // let level: log::Level = parse_url_query_string(&query_string, "RUST_LOG")
        //     .and_then(|x| x.parse().ok())
        //     .unwrap_or(log::Level::Error);
        // console_log::init_with_level(level).expect("could not initialize logger");
        // std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
    }

    #[cfg(target_arch = "wasm32")]
    let mut offscreen_canvas_setup: Option<OffscreenCanvasSetup> = None;
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;

        let query_string = web_sys::window().unwrap().location().search().unwrap();
        if let Some(offscreen_canvas_param) =
            parse_url_query_string(&query_string, "offscreen_canvas")
        {
            if FromStr::from_str(offscreen_canvas_param) == Ok(true) {
                log::info!("Creating OffscreenCanvasSetup");

                let offscreen_canvas =
                    OffscreenCanvas::new(1024, 768).expect("couldn't create OffscreenCanvas");

                let bitmap_renderer = window
                    .canvas()
                    .get_context("bitmaprenderer")
                    .expect("couldn't create ImageBitmapRenderingContext (Result)")
                    .expect("couldn't create ImageBitmapRenderingContext (Option)")
                    .dyn_into::<ImageBitmapRenderingContext>()
                    .expect("couldn't convert into ImageBitmapRenderingContext");

                offscreen_canvas_setup = Some(OffscreenCanvasSetup {
                    offscreen_canvas,
                    bitmap_renderer,
                })
            }
        }
    };

    log::info!("Initializing the surface...");

    let backends = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
    let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default();
    let gles_minor_version = wgpu::util::gles_minor_version_from_env().unwrap_or_default();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::from_build_config().with_env(),
        dx12_shader_compiler,
        gles_minor_version,
    });
    let (size, surface) = unsafe {
        let size = window.inner_size();

        #[cfg(any(not(target_arch = "wasm32"), target_os = "emscripten"))]
        let surface = instance.create_surface(&window).unwrap();
        #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
        let surface = {
            if let Some(offscreen_canvas_setup) = &offscreen_canvas_setup {
                log::info!("Creating surface from OffscreenCanvas");
                instance.create_surface_from_offscreen_canvas(
                    offscreen_canvas_setup.offscreen_canvas.clone(),
                )
            } else {
                instance.create_surface(&window)
            }
        }
        .unwrap();

        (size, surface)
    };
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, Some(&surface))
        .await
        .expect("No suitable GPU adapters found on the system!");

    #[cfg(not(target_arch = "wasm32"))]
    {
        let adapter_info = adapter.get_info();
        println!("Using {} ({:?})", adapter_info.name, adapter_info.backend);
    }

    let optional_features = wgpu::Features::DEPTH_CLIP_CONTROL;
    let required_features = wgpu::Features::empty();
    let adapter_features = adapter.features();
    assert!(
        adapter_features.contains(required_features),
        "Adapter does not support required features for this example: {:?}",
        required_features - adapter_features
    );

    let required_downlevel_capabilities = wgpu::DownlevelCapabilities {
        flags: wgpu::DownlevelFlags::empty(),
        shader_model: wgpu::ShaderModel::Sm5,
        ..wgpu::DownlevelCapabilities::default()
    };
    let downlevel_capabilities = adapter.get_downlevel_capabilities();
    assert!(
        downlevel_capabilities.shader_model >= required_downlevel_capabilities.shader_model,
        "Adapter does not support the minimum shader model required to run this example: {:?}",
        required_downlevel_capabilities.shader_model
    );
    assert!(
        downlevel_capabilities
            .flags
            .contains(required_downlevel_capabilities.flags),
        "Adapter does not support the downlevel capabilities required to run this example: {:?}",
        required_downlevel_capabilities.flags - downlevel_capabilities.flags
    );

    // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the surface.
    let needed_limits =
        wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());

    let trace_dir = std::env::var("WGPU_TRACE");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: (optional_features & adapter_features) | required_features,
                limits: needed_limits,
            },
            trace_dir.ok().as_ref().map(std::path::Path::new),
        )
        .await
        .expect("Unable to find a suitable GPU adapter!");

    Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
        #[cfg(target_arch = "wasm32")]
        offscreen_canvas_setup,
    }
}

struct FrameCounter {
    // Instant of the last time we printed the frame time.
    last_printed_instant: web_time::Instant,
    // Number of frames since the last time we printed the frame time.
    frame_count: u32,
}

impl FrameCounter {
    fn new() -> Self {
        Self {
            last_printed_instant: web_time::Instant::now(),
            frame_count: 0,
        }
    }

    fn update(&mut self) {
        self.frame_count += 1;
        let new_instant = web_time::Instant::now();
        let elasped_secs = (new_instant - self.last_printed_instant).as_secs_f32();
        if elasped_secs > 1.0 {
            let elapsed_ms = elasped_secs * 1000.0;
            let frame_time = elapsed_ms / self.frame_count as f32;
            let fps = self.frame_count as f32 / elasped_secs;
            log::info!("Frame time {:.2}ms ({:.1} FPS)", frame_time, fps);

            self.last_printed_instant = new_instant;
            self.frame_count = 0;
        }
    }
}

#[allow(clippy::too_many_lines)]
fn start(
    #[cfg(not(target_arch = "wasm32"))] Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
    }: Setup,
    #[cfg(target_arch = "wasm32")] Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
        offscreen_canvas_setup,
    }: Setup,
) {
    init_logger();

    let spawner = Spawner::new();
    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .expect("Surface isn't supported by the adapter.");
    let surface_view_format = config.format.add_srgb_suffix();
    config.view_formats.push(surface_view_format);
    surface.configure(&device, &config);

    log::info!("Initializing the example...");
    let mut example = Example::init(&config, &adapter, &device, &queue);

    let mut frame_counter = FrameCounter::new();

    log::info!("Entering render loop...");
    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter); // force ownership by the closure
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };
        match event {
            event::Event::RedrawEventsCleared => {
                #[cfg(not(target_arch = "wasm32"))]
                spawner.run_until_stalled();

                window.request_redraw();
            }
            event::Event::WindowEvent {
                event:
                    WindowEvent::Resized(size)
                    | WindowEvent::ScaleFactorChanged {
                        new_inner_size: &mut size,
                        ..
                    },
                ..
            } => {
                log::info!("Resizing to {:?}", size);
                config.width = size.width.max(1);
                config.height = size.height.max(1);
                example.resize(&config, &device, &queue);
                surface.configure(&device, &config);
            }
            event::Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::Escape),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                }
                | WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                #[cfg(not(target_arch = "wasm32"))]
                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::R),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    println!("{:#?}", instance.generate_report());
                }
                _ => {}
            },
            event::Event::RedrawRequested(_) => {
                frame_counter.update();

                // //#[cfg(not(target_arch = "wasm32"))]
                // {
                //     accum_time += last_frame_inst.elapsed().as_secs_f32();
                //     last_frame_inst = Instant::now();
                //     frame_count += 1;
                //     if accum_time > 1.0 {
                //         println!(
                //             "Avg frame time {:.2}ms, {:.2}",
                //             accum_time * 1000.0 / frame_count as f32,
                //             frame_count as f32 / accum_time
                //         );
                //         accum_time = 0.0;
                //         frame_count = 0;
                //     }
                // }

                let frame = if let Ok(frame) = surface.get_current_texture() {
                    frame
                } else {
                    surface.configure(&device, &config);
                    surface
                        .get_current_texture()
                        .expect("Failed to acquire next surface texture!")
                };
                let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(surface_view_format),
                    ..wgpu::TextureViewDescriptor::default()
                });

                example.render(&view, &device, &queue, &spawner);

                frame.present();

                #[cfg(target_arch = "wasm32")]
                {
                    if let Some(offscreen_canvas_setup) = &offscreen_canvas_setup {
                        let image_bitmap = offscreen_canvas_setup
                            .offscreen_canvas
                            .transfer_to_image_bitmap()
                            .expect("couldn't transfer offscreen canvas to image bitmap.");
                        offscreen_canvas_setup
                            .bitmap_renderer
                            .transfer_from_image_bitmap(&image_bitmap);

                        log::info!("Transferring OffscreenCanvas to ImageBitmapRenderer");
                    }
                }
            }
            _ => {}
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub struct Spawner<'a> {
    executor: async_executor::LocalExecutor<'a>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<'a> Spawner<'a> {
    fn new() -> Self {
        Self {
            executor: async_executor::LocalExecutor::new(),
        }
    }

    #[allow(dead_code)]
    pub fn spawn_local(&self, future: impl Future<Output = ()> + 'a) {
        self.executor.spawn(future).detach();
    }

    fn run_until_stalled(&self) {
        while self.executor.try_tick() {}
    }
}

#[cfg(target_arch = "wasm32")]
pub struct Spawner {}

#[cfg(target_arch = "wasm32")]
impl Spawner {
    fn new() -> Self {
        Self {}
    }

    #[allow(dead_code)]
    pub fn spawn_local(&self, future: impl Future<Output = ()> + 'static) {
        wasm_bindgen_futures::spawn_local(future);
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run(title: &str) {
    let setup = pollster::block_on(setup(title));
    start(setup);
}

#[cfg(target_arch = "wasm32")]
pub fn run(title: &str) {
    let title = title.to_owned();
    wasm_bindgen_futures::spawn_local(async move {
        let setup = setup(&title).await;
        let start_closure = Closure::once_into_js(move || start(setup));

        // make sure to handle JS exceptions thrown inside start.
        // Otherwise wasm_bindgen_futures Queue would break and never handle any tasks again.
        // This is required, because winit uses JS exception for control flow to escape from `run`.
        if let Err(error) = call_catch(&start_closure) {
            let is_control_flow_exception = error.dyn_ref::<js_sys::Error>().map_or(false, |e| {
                e.message().includes("Using exceptions for control flow", 0)
            });

            if !is_control_flow_exception {
                web_sys::console::error_1(&error);
            }
        }

        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(catch, js_namespace = Function, js_name = "prototype.call.call")]
            fn call_catch(this: &JsValue) -> Result<(), JsValue>;
        }
    });
}

#[cfg(target_arch = "wasm32")]
/// Parse the query string as returned by `web_sys::window()?.location().search()?` and get a
/// specific key out of it.
pub fn parse_url_query_string<'a>(query: &'a str, search_key: &str) -> Option<&'a str> {
    let query_string = query.strip_prefix('?')?;

    for pair in query_string.split('&') {
        let mut pair = pair.split('=');
        let key = pair.next()?;
        let value = pair.next()?;

        if key == search_key {
            return Some(value);
        }
    }

    None
}
