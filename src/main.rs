// https://sotrh.github.io/learn-wgpu/
use winit::{
    event::*,
    event_loop::{EventLoop, ControlFlow},
    window::{Window, WindowBuilder},
};
use wgpu::util::DeviceExt;
use itertools_num::linspace;
use num_integer::binomial;

/* ------- *
 * STRUCTS *
 * ------- */
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

struct BezierCurve {
    base_points: Vec<cgmath::Vector3<f32>>,
    vertices: Vec<Vertex>,
    indicies: Vec<u16>,
    num_points: usize,
    degree: usize,
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    size: winit::dpi::PhysicalSize<u32>,

    // Information about how the render pipeline should be
    line_render_pipeline: wgpu::RenderPipeline,
    point_render_pipeline: wgpu::RenderPipeline,

    // Buffer to store vertex data
    bezier_curve: BezierCurve,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    point_buffer: wgpu::Buffer,
}

/* -------------- *
 * STRUCT METHODS *
 * -------------- */
impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        wgpu::VertexBufferDescriptor {
            stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            // Attributes can be swapped out for &wgpu::vertex_attr_array! macro
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    shader_location: 0, // This is layout(location=0) in the vertex shader
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float2,
                }
            ],
        }
    }
}

impl BezierCurve {
    fn new(base_points: &[cgmath::Vector3<f32>]) -> Self {
        BezierCurve {
            base_points: Vec::from(base_points),
            vertices: Vec::new(),
            indicies: Vec::new(),
            num_points: 10,
            degree: base_points.len() - 1,
        }
    }

    fn calculate_curve(&mut self) {
        self.vertices.clear();
        self.indicies.clear();

        // Calculate vertices for the lines
        for t in linspace(0., 1., self.num_points) {
            let mut point = cgmath::Vector3::new(0.0, 0.0, 0.0);
            for i in 0..self.degree + 1 {
                point += binomial(self.degree, i) as f32 * ((1.0 - t) as f32).powf((self.degree - i) as f32) * (t as f32).powf(i as f32) * self.base_points[i];
            }
            
            let vertex = Vertex{position: [point.x, point.y, point.z], color: [1.0, 1.0, 1.0]};
            self.vertices.push(vertex);
        }

        // Calculate indicies for the lines
        self.indicies.push(0);
        for i in 1..self.num_points as u16 - 1 {
            self.indicies.push(i);
            self.indicies.push(i);
        }
        self.indicies.push(self.num_points as u16 - 1);
    }
    
    fn get_vertex_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Line Vertex Buffer"),
                contents: bytemuck::cast_slice(self.vertices.as_slice()),
                usage: wgpu::BufferUsage::VERTEX,
            }
        )
    }

    fn get_index_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Line Index Buffer"),
                contents: bytemuck::cast_slice(self.indicies.as_slice()),
                usage: wgpu::BufferUsage::INDEX,
            }
        )
    }

    fn get_point_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        let mut points = Vec::new();
        for point in &self.base_points {
            let p = Vertex{position: [point.x, point.y, point.z], color: [0.1, 1.0, 0.1]};
            points.push(p);
        }

        device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Point Vertex Buffer"),
                contents: bytemuck::cast_slice(points.as_slice()),
                usage: wgpu::BufferUsage::VERTEX,
            }
        )
    }
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        // Surface is a place where rendered images can be displayed
        let surface = unsafe { instance.create_surface(window) };
        // Adapter is a handle to a physical graphics and/or compute device
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            },
        ).await.unwrap();

        // Request a connection to a physical device, creating a logical device together with a queue
        // that executes command buffers
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None, // Trace path
        ).await.unwrap();

        // Create a SwapChain which represents the image or series of images to render on a surface
        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo, // VSync
        };
        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        // Load shaders. Shaders are compiled in build.rs only when shaders have changed
        let vs_module = device.create_shader_module(wgpu::include_spirv!("shader.vert.spv"));
        let fs_module = device.create_shader_module(wgpu::include_spirv!("shader.frag.spv"));
 
        // Create a render pipeline that describes what to do with vertices etc. from start to finish
        let line_render_pipeline_layout =
            device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                    label: Some("Line Render Pipeline Layout"),
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
                }
            );

        let line_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Render Pipeline"),
            layout: Some(&line_render_pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main", 
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(
                wgpu::RasterizationStateDescriptor {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: wgpu::CullMode::Back,
                    depth_bias: 0,
                    depth_bias_slope_scale: 0.0,
                    depth_bias_clamp: 0.0,
                    clamp_depth: false,
                }
            ),
            color_states: &[
                wgpu::ColorStateDescriptor {
                    format: sc_desc.format,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
            ],
            primitive_topology: wgpu::PrimitiveTopology::LineStrip,
            depth_stencil_state: None, 
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[
                    Vertex::desc(),
                ],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        let point_render_pipeline_layout =
            device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                    label: Some("Point Render Pipeline Layout"),
                    bind_group_layouts: &[],
                    push_constant_ranges: &[],
                }
            );

        let point_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Render Pipeline"),
            layout: Some(&point_render_pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main", 
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(
                wgpu::RasterizationStateDescriptor {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: wgpu::CullMode::Back,
                    depth_bias: 0,
                    depth_bias_slope_scale: 0.0,
                    depth_bias_clamp: 0.0,
                    clamp_depth: false,
                }
            ),
            color_states: &[
                wgpu::ColorStateDescriptor {
                    format: sc_desc.format,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                },
            ],
            primitive_topology: wgpu::PrimitiveTopology::PointList,
            depth_stencil_state: None, 
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[
                    Vertex::desc(),
                ],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        // Create a bezier curve
        let point0 = cgmath::Vector3::new(-0.5, 0.7, 0.0);
        let point1 = cgmath::Vector3::new(0.5, 0.7, 0.0);
        let point2 = cgmath::Vector3::new(0.5, 0.0, 0.0);
        let point3 = cgmath::Vector3::new(-0.5, 0.0, 0.0);
        let point4 = cgmath::Vector3::new(-0.5, -0.7, 0.0);
        let point5 = cgmath::Vector3::new(0.5, -0.7, 0.0);
        let mut bezier_curve = BezierCurve::new(&[point0, point1, point2, point3, point4, point5]);

        bezier_curve.calculate_curve();
        let vertex_buffer = bezier_curve.get_vertex_buffer(&device);
        let index_buffer = bezier_curve.get_index_buffer(&device);
        let num_indices = bezier_curve.indicies.len() as u32;
        let point_buffer = bezier_curve.get_point_buffer(&device);

        Self {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain,
            size, 

            line_render_pipeline,
            point_render_pipeline,

            bezier_curve,
            vertex_buffer,
            index_buffer,
            num_indices,
            point_buffer,
        }
    }

    // If the window is resized, the swap chain has to be recreated
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
    }

    // Input returns true if the input was processed fully
    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                if *state == ElementState::Pressed {
                    match keycode {
                        VirtualKeyCode::Up => {
                            if self.bezier_curve.num_points < 1_000 {
                                self.bezier_curve.num_points += 1;
                            }
                            self.update_curve();
                            true
                        },
                        VirtualKeyCode::Down => {
                            if self.bezier_curve.num_points > 1 {
                                self.bezier_curve.num_points -= 1;
                            }
                            self.update_curve();
                            true
                        },
                        _ => false,
                    }
                } else {
                    false
                }
            },
            _ => false,
        }
    }

    fn update(&mut self) {

    }

    fn render(&mut self) -> Result<(), wgpu::SwapChainError> {
        let frame = self
            .swap_chain
            .get_current_frame()?
            .output;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &frame.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: true,
                        }
                    }
                ],
                depth_stencil_attachment: None,
            });

            // Draw lines
            render_pass.set_pipeline(&self.line_render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..));
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
            
            // Draw points
            render_pass.set_pipeline(&self.point_render_pipeline);
            render_pass.set_vertex_buffer(0, self.point_buffer.slice(..));
            render_pass.draw_indexed(0..self.bezier_curve.base_points.len() as u32, 0, 0..1);
        }
    
        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
    
        Ok(())
    }

    fn update_curve(&mut self) {
        self.bezier_curve.calculate_curve();
        self.vertex_buffer = self.bezier_curve.get_vertex_buffer(&self.device);
        self.index_buffer = self.bezier_curve.get_index_buffer(&self.device);
        self.num_indices = self.bezier_curve.indicies.len() as u32;
        self.point_buffer = self.bezier_curve.get_point_buffer(&self.device);
    }
}

/* ---- *
 * MAIN *
 * ---- */
fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .unwrap();
    
    // Since main can't be async, we're going to need to block
    use futures::executor::block_on;
    let mut state = block_on(State::new(&window));

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => if !state.input(event) { 
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::KeyboardInput {
                        input,
                        ..
                    } => {
                        match input {
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            } => *control_flow = ControlFlow::Exit,
                            _ => {}
                        }
                    }
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        // new_inner_size is &&mut so we have to dereference it twice
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(_) => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Recreate the swap_chain if lost
                    Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SwapChainError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it.
                window.request_redraw();
            }
            _ => {}
        }
    });
}
