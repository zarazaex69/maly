use eframe::egui;
use rayon::prelude::*;
use rug::Assign;
use rug::Float;
use rug::ops::Pow;
use std::sync::{Arc, Mutex};

const F64_ZOOM_THRESHOLD: f64 = 1e-13;
const GPU_ZOOM_THRESHOLD: f64 = 1e-5;

fn required_precision(zoom_width: f64) -> u32 {
    if zoom_width > F64_ZOOM_THRESHOLD {
        return 0;
    }
    let bits = (-zoom_width.log2() + 32.0).max(128.0) as u32;
    (bits + 31) & !31
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    center_x: f32,
    center_y: f32,
    zoom: f32,
    max_iter: u32,
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuRenderer {
    fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("No GPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("mandelbrot"),
                ..Default::default()
            },
            None,
        ))
        .expect("Failed to create device");

        let shader_src = include_str!("gpu.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mandelbrot_compute"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mandelbrot_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mandelbrot_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mandelbrot_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        }
    }

    fn render(
        &self,
        center_x: f32,
        center_y: f32,
        zoom: f32,
        max_iter: u32,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let params = GpuParams {
            center_x,
            center_y,
            zoom,
            max_iter,
            width,
            height,
            _pad0: 0,
            _pad1: 0,
        };

        let param_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&param_buffer, 0, bytemuck::bytes_of(&params));

        let pixel_count = (width * height) as u64;
        let output_size = pixel_count * 4;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mandelbrot_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mandelbrot_enc"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mandelbrot_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<u8> = data.to_vec();
        drop(data);
        staging_buffer.unmap();
        result
    }
}

fn mandelbrot_f64(cx: f64, cy: f64, max_iter: u32) -> (u32, f64) {
    let mut zx = 0.0f64;
    let mut zy = 0.0f64;
    let mut zx2 = 0.0f64;
    let mut zy2 = 0.0f64;
    let mut i = 0u32;
    while i < max_iter {
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        zx2 = zx * zx;
        zy2 = zy * zy;
        if zx2 + zy2 > 4.0 {
            let modulus = (zx2 + zy2).sqrt();
            let smooth = (i as f64) + 1.0 - modulus.ln().ln() / std::f64::consts::LN_2;
            return (i, smooth);
        }
        i += 1;
    }
    (max_iter, max_iter as f64)
}

fn mandelbrot_bigfloat(cx: &Float, cy: &Float, max_iter: u32, prec: u32) -> (u32, f64) {
    let mut zx = Float::with_val(prec, 0.0);
    let mut zy = Float::with_val(prec, 0.0);
    let mut zx2 = Float::with_val(prec, 0.0);
    let mut zy2 = Float::with_val(prec, 0.0);
    let bailout = Float::with_val(prec, 4.0);
    let mut tmp = Float::with_val(prec, 0.0);

    let mut i = 0u32;
    while i < max_iter {
        tmp.assign(&zx * &zy);
        zy.assign(&tmp * 2.0);
        zy += cy;
        tmp.assign(&zx2 - &zy2);
        zx.assign(&tmp + cx);
        zx2.assign(zx.clone().pow(2));
        zy2.assign(zy.clone().pow(2));
        tmp.assign(&zx2 + &zy2);
        if tmp > bailout {
            let modulus = tmp.to_f64().sqrt();
            let smooth = (i as f64) + 1.0 - modulus.ln().ln() / std::f64::consts::LN_2;
            return (i, smooth);
        }
        i += 1;
    }
    (max_iter, max_iter as f64)
}

fn iteration_color(smooth_iter: f64, max_iter: u32) -> [u8; 3] {
    if smooth_iter >= max_iter as f64 {
        return [0, 0, 0];
    }
    let gradient = colorous::TURBO;
    let t = (smooth_iter * 0.02).fract();
    let c = gradient.eval_continuous(t);
    [c.r, c.g, c.b]
}

enum RenderMode {
    Gpu,
    CpuF64,
    CpuBigFloat(u32),
}

fn choose_render_mode(zoom: f64) -> RenderMode {
    if zoom > GPU_ZOOM_THRESHOLD {
        RenderMode::Gpu
    } else if zoom > F64_ZOOM_THRESHOLD {
        RenderMode::CpuF64
    } else {
        RenderMode::CpuBigFloat(required_precision(zoom))
    }
}

struct RenderResult {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    render_gen: u64,
}

struct MandelbrotApp {
    center_x: Float,
    center_y: Float,
    zoom: Float,
    max_iter: u32,
    render_result: Arc<Mutex<Option<RenderResult>>>,
    current_texture: Option<egui::TextureHandle>,
    generation: u64,
    rendered_generation: u64,
    is_rendering: Arc<Mutex<bool>>,
    gpu: Arc<GpuRenderer>,
}

impl Default for MandelbrotApp {
    fn default() -> Self {
        Self {
            center_x: Float::with_val(128, -0.5),
            center_y: Float::with_val(128, 0.0),
            zoom: Float::with_val(128, 3.0),
            max_iter: 256,
            render_result: Arc::new(Mutex::new(None)),
            current_texture: None,
            generation: 1,
            rendered_generation: 0,
            is_rendering: Arc::new(Mutex::new(false)),
            gpu: Arc::new(GpuRenderer::new()),
        }
    }
}

impl MandelbrotApp {
    fn schedule_render(&mut self, width: usize, height: usize) {
        let is_rendering_arc = Arc::clone(&self.is_rendering);
        {
            let mut rendering = is_rendering_arc.lock().unwrap();
            if *rendering {
                return;
            }
            *rendering = true;
        }

        let zoom_f64 = self.zoom.to_f64();
        let mode = choose_render_mode(zoom_f64);
        let max_iter = self.max_iter;
        let current_gen = self.generation;
        let result_slot = Arc::clone(&self.render_result);
        let cx_f64 = self.center_x.to_f64();
        let cy_f64 = self.center_y.to_f64();

        match mode {
            RenderMode::Gpu => {
                let gpu = Arc::clone(&self.gpu);
                std::thread::spawn(move || {
                    let pixels = gpu.render(
                        cx_f64 as f32,
                        cy_f64 as f32,
                        zoom_f64 as f32,
                        max_iter,
                        width as u32,
                        height as u32,
                    );
                    *result_slot.lock().unwrap() = Some(RenderResult {
                        pixels,
                        width,
                        height,
                        render_gen: current_gen,
                    });
                    *is_rendering_arc.lock().unwrap() = false;
                });
            }
            RenderMode::CpuF64 => {
                std::thread::spawn(move || {
                    let min_dim = width.min(height) as f64;
                    let pixel_size = zoom_f64 / min_dim;
                    let half_w = width as f64 / 2.0;
                    let half_h = height as f64 / 2.0;

                    let rows: Vec<Vec<[u8; 3]>> = (0..height)
                        .into_par_iter()
                        .map(|py| {
                            let world_y = cy_f64 + pixel_size * (py as f64 - half_h);
                            (0..width)
                                .map(|px| {
                                    let world_x = cx_f64 + pixel_size * (px as f64 - half_w);
                                    let (_, smooth) = mandelbrot_f64(world_x, world_y, max_iter);
                                    iteration_color(smooth, max_iter)
                                })
                                .collect()
                        })
                        .collect();

                    let mut buf = Vec::with_capacity(width * height * 4);
                    for row in &rows {
                        for &[r, g, b] in row {
                            buf.extend_from_slice(&[r, g, b, 255]);
                        }
                    }

                    *result_slot.lock().unwrap() = Some(RenderResult {
                        pixels: buf,
                        width,
                        height,
                        render_gen: current_gen,
                    });
                    *is_rendering_arc.lock().unwrap() = false;
                });
            }
            RenderMode::CpuBigFloat(prec) => {
                let cx_big = Float::with_val(prec, &self.center_x);
                let cy_big = Float::with_val(prec, &self.center_y);
                let zoom_big = Float::with_val(prec, &self.zoom);

                std::thread::spawn(move || {
                    let min_dim = width.min(height) as f64;
                    let pixel_size = Float::with_val(prec, &zoom_big / min_dim);
                    let half_w = width as f64 / 2.0;
                    let half_h = height as f64 / 2.0;

                    let rows: Vec<Vec<[u8; 3]>> = (0..height)
                        .into_par_iter()
                        .map(|py| {
                            let dy = Float::with_val(prec, &pixel_size * (py as f64 - half_h));
                            let world_y = Float::with_val(prec, &cy_big + &dy);
                            (0..width)
                                .map(|px| {
                                    let dx =
                                        Float::with_val(prec, &pixel_size * (px as f64 - half_w));
                                    let world_x = Float::with_val(prec, &cx_big + &dx);
                                    let (_, smooth) =
                                        mandelbrot_bigfloat(&world_x, &world_y, max_iter, prec);
                                    iteration_color(smooth, max_iter)
                                })
                                .collect()
                        })
                        .collect();

                    let mut buf = Vec::with_capacity(width * height * 4);
                    for row in &rows {
                        for &[r, g, b] in row {
                            buf.extend_from_slice(&[r, g, b, 255]);
                        }
                    }

                    *result_slot.lock().unwrap() = Some(RenderResult {
                        pixels: buf,
                        width,
                        height,
                        render_gen: current_gen,
                    });
                    *is_rendering_arc.lock().unwrap() = false;
                });
            }
        }
    }

    fn zoom_info(&self) -> String {
        let zoom_f64 = self.zoom.to_f64();
        let magnification = 3.0 / zoom_f64;
        let log10 = magnification.log10();
        let mode = match choose_render_mode(zoom_f64) {
            RenderMode::Gpu => "GPU f32".to_string(),
            RenderMode::CpuF64 => "CPU f64".to_string(),
            RenderMode::CpuBigFloat(p) => format!("CPU BigFloat {}bit", p),
        };
        format!(
            "Zoom: 10^{:.1} | Iter: {} | {} | cx: {:.6e} cy: {:.6e}",
            log10,
            self.max_iter,
            mode,
            self.center_x.to_f64(),
            self.center_y.to_f64()
        )
    }
}

impl eframe::App for MandelbrotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("info_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(self.zoom_info());
                ui.separator();
                ui.label("Drag: pan | Scroll: zoom | +/-: iter | R: reset");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            let w = available.x as usize;
            let h = available.y as usize;
            if w == 0 || h == 0 {
                return;
            }

            let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());

            if response.dragged_by(egui::PointerButton::Primary) {
                let delta = response.drag_delta();
                let prec_bits = required_precision(self.zoom.to_f64());
                if prec_bits > 0 {
                    let p = prec_bits.max(128);
                    let min_dim = w.min(h) as f64;
                    let pixel_size = Float::with_val(p, &self.zoom / min_dim);
                    let dx = Float::with_val(p, &pixel_size * (-(delta.x as f64)));
                    let dy = Float::with_val(p, &pixel_size * (-(delta.y as f64)));
                    self.center_x += &dx;
                    self.center_y += &dy;
                } else {
                    let zoom_f64 = self.zoom.to_f64();
                    let min_dim = w.min(h) as f64;
                    let pixel_size = zoom_f64 / min_dim;
                    self.center_x -= pixel_size * (delta.x as f64);
                    self.center_y -= pixel_size * (delta.y as f64);
                }
                self.generation += 1;
            }

            if let Some(hover_pos) = response.hover_pos() {
                let scroll = ui.input(|i| i.raw_scroll_delta.y);
                if scroll.abs() > 0.0 {
                    let factor = if scroll > 0.0 { 0.75 } else { 1.333 };
                    let min_dim = w.min(h) as f64;
                    let rel_x = (hover_pos.x - rect.center().x) as f64;
                    let rel_y = (hover_pos.y - rect.center().y) as f64;

                    let prec_bits = required_precision(self.zoom.to_f64());
                    if prec_bits > 0 {
                        let p = prec_bits.max(128);
                        let pixel_size = Float::with_val(p, &self.zoom / min_dim);
                        let offset_x = Float::with_val(p, &pixel_size * rel_x);
                        let offset_y = Float::with_val(p, &pixel_size * rel_y);
                        let world_x = Float::with_val(p, &self.center_x + &offset_x);
                        let world_y = Float::with_val(p, &self.center_y + &offset_y);
                        self.zoom *= factor;
                        let new_ps = Float::with_val(p, &self.zoom / min_dim);
                        let new_ox = Float::with_val(p, &new_ps * rel_x);
                        let new_oy = Float::with_val(p, &new_ps * rel_y);
                        self.center_x = Float::with_val(p, &world_x - &new_ox);
                        self.center_y = Float::with_val(p, &world_y - &new_oy);
                    } else {
                        let zoom_f64 = self.zoom.to_f64();
                        let ps = zoom_f64 / min_dim;
                        let wx = self.center_x.to_f64() + ps * rel_x;
                        let wy = self.center_y.to_f64() + ps * rel_y;
                        self.zoom *= factor;
                        let new_ps = self.zoom.to_f64() / min_dim;
                        self.center_x = Float::with_val(128, wx - new_ps * rel_x);
                        self.center_y = Float::with_val(128, wy - new_ps * rel_y);
                    }

                    let magnification = 3.0 / self.zoom.to_f64();
                    let log_mag = magnification.log10();
                    self.max_iter = ((log_mag.abs() * 120.0) as u32 + 256).min(100_000);
                    self.generation += 1;
                }
            }

            ui.input(|i| {
                for event in &i.events {
                    if let egui::Event::Key {
                        key, pressed: true, ..
                    } = event
                    {
                        match key {
                            egui::Key::Plus | egui::Key::Equals => {
                                self.max_iter = (self.max_iter + 100).min(100_000);
                                self.generation += 1;
                            }
                            egui::Key::Minus => {
                                self.max_iter = self.max_iter.saturating_sub(100).max(50);
                                self.generation += 1;
                            }
                            egui::Key::R => {
                                *self = Self::default();
                            }
                            _ => {}
                        }
                    }
                }
            });

            {
                let mut rg = self.render_result.lock().unwrap();
                if let Some(result) = rg.take() {
                    if result.render_gen >= self.rendered_generation {
                        let img = egui::ColorImage::from_rgba_unmultiplied(
                            [result.width, result.height],
                            &result.pixels,
                        );
                        self.current_texture = Some(ctx.load_texture(
                            "mandelbrot",
                            img,
                            egui::TextureOptions {
                                magnification: egui::TextureFilter::Linear,
                                minification: egui::TextureFilter::Linear,
                                ..Default::default()
                            },
                        ));
                        self.rendered_generation = result.render_gen;
                    }
                }
            }

            if let Some(tex) = &self.current_texture {
                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                ui.painter().image(tex.id(), rect, uv, egui::Color32::WHITE);
            }

            if self.generation > self.rendered_generation {
                self.schedule_render(w, h);
            }

            let is_rendering = *self.is_rendering.lock().unwrap();
            if is_rendering || self.generation > self.rendered_generation {
                ctx.request_repaint();
            }
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("Mandelbrot Explorer [GPU]"),
        ..Default::default()
    };
    eframe::run_native(
        "Mandelbrot Explorer",
        options,
        Box::new(|_cc| Ok(Box::new(MandelbrotApp::default()))),
    )
}
