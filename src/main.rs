use eframe::egui;
use rayon::prelude::*;
use rug::Assign;
use rug::Float;
use rug::ops::Pow;
use std::sync::{Arc, Mutex};

const GPU_ZOOM_THRESHOLD: f64 = 1e-5;

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
            label: Some("bgl"),
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
            label: Some("pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
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

    fn render(&self, cx: f32, cy: f32, zoom: f32, max_iter: u32, w: u32, h: u32) -> Vec<u8> {
        let params = GpuParams {
            center_x: cx,
            center_y: cy,
            zoom,
            max_iter,
            width: w,
            height: h,
            _pad0: 0,
            _pad1: 0,
        };
        let param_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&param_buf, 0, bytemuck::bytes_of(&params));

        let out_size = (w * h) as u64 * 4;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((w + 15) / 16, (h + 15) / 16, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
        self.queue.submit(std::iter::once(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        data
    }
}

struct RefOrbit {
    zx: Vec<f64>,
    zy: Vec<f64>,
    escaped_at: u32,
}

fn compute_reference_orbit(cx: &Float, cy: &Float, max_iter: u32, prec: u32) -> RefOrbit {
    let mut zx_big = Float::with_val(prec, 0.0);
    let mut zy_big = Float::with_val(prec, 0.0);
    let mut tmp = Float::with_val(prec, 0.0);
    let mut zx2 = Float::with_val(prec, 0.0);
    let mut zy2 = Float::with_val(prec, 0.0);
    let bailout = Float::with_val(prec, 1e6);

    let mut orbit_x = Vec::with_capacity(max_iter as usize + 1);
    let mut orbit_y = Vec::with_capacity(max_iter as usize + 1);

    orbit_x.push(0.0f64);
    orbit_y.push(0.0f64);

    let mut escaped_at = max_iter;

    for i in 0..max_iter {
        tmp.assign(&zx_big * &zy_big);
        zy_big.assign(&tmp * 2.0);
        zy_big += cy;

        tmp.assign(&zx2 - &zy2);
        zx_big.assign(&tmp + cx);

        zx2.assign(zx_big.clone().pow(2));
        zy2.assign(zy_big.clone().pow(2));

        orbit_x.push(zx_big.to_f64());
        orbit_y.push(zy_big.to_f64());

        tmp.assign(&zx2 + &zy2);
        if tmp > bailout {
            escaped_at = i + 1;
            break;
        }
    }

    RefOrbit {
        zx: orbit_x,
        zy: orbit_y,
        escaped_at,
    }
}

fn perturbation_iterate(
    ref_orbit: &RefOrbit,
    delta_cx: f64,
    delta_cy: f64,
    abs_cx: f64,
    abs_cy: f64,
    max_iter: u32,
) -> (u32, f64) {
    let mut dx = 0.0f64;
    let mut dy = 0.0f64;
    let orbit_len = ref_orbit.zx.len();

    for i in 0..max_iter as usize {
        if i + 1 >= orbit_len {
            break;
        }

        let zx = ref_orbit.zx[i];
        let zy = ref_orbit.zy[i];

        let new_dx = 2.0 * (zx * dx - zy * dy) + dx * dx - dy * dy + delta_cx;
        let new_dy = 2.0 * (zx * dy + zy * dx) + 2.0 * dx * dy + delta_cy;
        dx = new_dx;
        dy = new_dy;

        let full_x = ref_orbit.zx[i + 1] + dx;
        let full_y = ref_orbit.zy[i + 1] + dy;
        let mag2 = full_x * full_x + full_y * full_y;

        if mag2 > 4.0 {
            let smooth = (i as f64) + 1.0 - mag2.sqrt().ln().ln() / std::f64::consts::LN_2;
            return (i as u32, smooth);
        }

        let ref_mag2 = zx * zx + zy * zy;
        if ref_mag2 < dx * dx + dy * dy {
            return mandelbrot_f64_from(full_x, full_y, abs_cx, abs_cy, max_iter, (i + 1) as u32);
        }
    }

    (max_iter, max_iter as f64)
}

fn mandelbrot_f64_from(
    mut zx: f64,
    mut zy: f64,
    cx: f64,
    cy: f64,
    max_iter: u32,
    start: u32,
) -> (u32, f64) {
    let mut zx2 = zx * zx;
    let mut zy2 = zy * zy;
    for i in start..max_iter {
        if zx2 + zy2 > 4.0 {
            let smooth = (i as f64) + 1.0 - (zx2 + zy2).sqrt().ln().ln() / std::f64::consts::LN_2;
            return (i, smooth);
        }
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        zx2 = zx * zx;
        zy2 = zy * zy;
    }
    (max_iter, max_iter as f64)
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
            let smooth = (i as f64) + 1.0 - (zx2 + zy2).sqrt().ln().ln() / std::f64::consts::LN_2;
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

fn required_precision(zoom_width: f64) -> u32 {
    let bits = (-zoom_width.log2() + 32.0).max(128.0) as u32;
    (bits + 31) & !31
}

enum RenderMode {
    Gpu,
    CpuF64,
    Perturbation(u32),
}

fn choose_mode(zoom: f64) -> RenderMode {
    if zoom > GPU_ZOOM_THRESHOLD {
        RenderMode::Gpu
    } else if zoom > 1e-13 {
        RenderMode::CpuF64
    } else {
        RenderMode::Perturbation(required_precision(zoom))
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

impl MandelbrotApp {
    fn new() -> Self {
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

    fn schedule_render(&mut self, width: usize, height: usize) {
        {
            let mut r = self.is_rendering.lock().unwrap();
            if *r {
                return;
            }
            *r = true;
        }

        let zoom_f64 = self.zoom.to_f64();
        let mode = choose_mode(zoom_f64);
        let max_iter = self.max_iter;
        let current_gen = self.generation;
        let result_slot = Arc::clone(&self.render_result);
        let is_rendering_arc = Arc::clone(&self.is_rendering);
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
                    let ps = zoom_f64 / min_dim;
                    let hw = width as f64 / 2.0;
                    let hh = height as f64 / 2.0;
                    let rows: Vec<Vec<[u8; 3]>> = (0..height)
                        .into_par_iter()
                        .map(|py| {
                            let wy = cy_f64 + ps * (py as f64 - hh);
                            (0..width)
                                .map(|px| {
                                    let wx = cx_f64 + ps * (px as f64 - hw);
                                    let (_, s) = mandelbrot_f64(wx, wy, max_iter);
                                    iteration_color(s, max_iter)
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
            RenderMode::Perturbation(prec) => {
                let cx_big = Float::with_val(prec, &self.center_x);
                let cy_big = Float::with_val(prec, &self.center_y);
                let zoom_big = Float::with_val(prec, &self.zoom);

                std::thread::spawn(move || {
                    let ref_orbit = compute_reference_orbit(&cx_big, &cy_big, max_iter, prec);

                    let min_dim = width.min(height) as f64;
                    let ps_big = Float::with_val(prec, &zoom_big / min_dim);
                    let ps_f64 = ps_big.to_f64();
                    let hw = width as f64 / 2.0;
                    let hh = height as f64 / 2.0;

                    let cx_f64_abs = cx_big.to_f64();
                    let cy_f64_abs = cy_big.to_f64();

                    let rows: Vec<Vec<[u8; 3]>> = (0..height)
                        .into_par_iter()
                        .map(|py| {
                            let dcy = ps_f64 * (py as f64 - hh);
                            (0..width)
                                .map(|px| {
                                    let dcx = ps_f64 * (px as f64 - hw);
                                    let (_, s) = perturbation_iterate(
                                        &ref_orbit,
                                        dcx,
                                        dcy,
                                        cx_f64_abs + dcx,
                                        cy_f64_abs + dcy,
                                        max_iter,
                                    );
                                    iteration_color(s, max_iter)
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
        let z = self.zoom.to_f64();
        let mag = 3.0 / z;
        let mode = match choose_mode(z) {
            RenderMode::Gpu => "GPU f32".to_string(),
            RenderMode::CpuF64 => "CPU f64".to_string(),
            RenderMode::Perturbation(p) => format!("Perturbation {}bit", p),
        };
        format!(
            "Zoom: 10^{:.1} | Iter: {} | {} | cx: {:.6e} cy: {:.6e}",
            mag.log10(),
            self.max_iter,
            mode,
            self.center_x.to_f64(),
            self.center_y.to_f64()
        )
    }
}

impl eframe::App for MandelbrotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("info").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(self.zoom_info());
                ui.separator();
                ui.label("Drag: pan | Scroll: zoom | +/-: iter | R: reset");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let av = ui.available_size();
            let w = av.x as usize;
            let h = av.y as usize;
            if w == 0 || h == 0 {
                return;
            }

            let (rect, resp) = ui.allocate_exact_size(av, egui::Sense::click_and_drag());

            if resp.dragged_by(egui::PointerButton::Primary) {
                let d = resp.drag_delta();
                let z = self.zoom.to_f64();
                let prec = if z < 1e-13 {
                    required_precision(z).max(128)
                } else {
                    128
                };
                let min_dim = w.min(h) as f64;
                if prec > 128 {
                    let ps = Float::with_val(prec, &self.zoom / min_dim);
                    let dx = Float::with_val(prec, &ps * (-(d.x as f64)));
                    let dy = Float::with_val(prec, &ps * (-(d.y as f64)));
                    self.center_x += &dx;
                    self.center_y += &dy;
                } else {
                    let ps = z / min_dim;
                    self.center_x -= ps * (d.x as f64);
                    self.center_y -= ps * (d.y as f64);
                }
                self.generation += 1;
            }

            if let Some(hp) = resp.hover_pos() {
                let scroll = ui.input(|i| i.raw_scroll_delta.y);
                if scroll.abs() > 0.0 {
                    let factor = if scroll > 0.0 { 0.75 } else { 1.333 };
                    let min_dim = w.min(h) as f64;
                    let rx = (hp.x - rect.center().x) as f64;
                    let ry = (hp.y - rect.center().y) as f64;
                    let z = self.zoom.to_f64();
                    let prec = if z < 1e-13 {
                        required_precision(z).max(128)
                    } else {
                        128
                    };

                    if prec > 128 {
                        let ps = Float::with_val(prec, &self.zoom / min_dim);
                        let ox = Float::with_val(prec, &ps * rx);
                        let oy = Float::with_val(prec, &ps * ry);
                        let wx = Float::with_val(prec, &self.center_x + &ox);
                        let wy = Float::with_val(prec, &self.center_y + &oy);
                        self.zoom *= factor;
                        let nps = Float::with_val(prec, &self.zoom / min_dim);
                        let nox = Float::with_val(prec, &nps * rx);
                        let noy = Float::with_val(prec, &nps * ry);
                        self.center_x = Float::with_val(prec, &wx - &nox);
                        self.center_y = Float::with_val(prec, &wy - &noy);
                    } else {
                        let ps = z / min_dim;
                        let wx = self.center_x.to_f64() + ps * rx;
                        let wy = self.center_y.to_f64() + ps * ry;
                        self.zoom *= factor;
                        let nps = self.zoom.to_f64() / min_dim;
                        self.center_x = Float::with_val(128, wx - nps * rx);
                        self.center_y = Float::with_val(128, wy - nps * ry);
                    }

                    let mag = 3.0 / self.zoom.to_f64();
                    self.max_iter = ((mag.log10().abs() * 120.0) as u32 + 256).min(100_000);
                    self.generation += 1;
                }
            }

            ui.input(|i| {
                for ev in &i.events {
                    if let egui::Event::Key {
                        key, pressed: true, ..
                    } = ev
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
                                let gpu = Arc::clone(&self.gpu);
                                *self = Self::new();
                                self.gpu = gpu;
                            }
                            _ => {}
                        }
                    }
                }
            });

            {
                let mut rg = self.render_result.lock().unwrap();
                if let Some(r) = rg.take() {
                    if r.render_gen >= self.rendered_generation {
                        let img = egui::ColorImage::from_rgba_unmultiplied(
                            [r.width, r.height],
                            &r.pixels,
                        );
                        self.current_texture = Some(ctx.load_texture(
                            "m",
                            img,
                            egui::TextureOptions {
                                magnification: egui::TextureFilter::Linear,
                                minification: egui::TextureFilter::Linear,
                                ..Default::default()
                            },
                        ));
                        self.rendered_generation = r.render_gen;
                    }
                }
            }

            if let Some(tex) = &self.current_texture {
                ui.painter().image(
                    tex.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            }

            if self.generation > self.rendered_generation {
                self.schedule_render(w, h);
            }
            let busy = *self.is_rendering.lock().unwrap();
            if busy || self.generation > self.rendered_generation {
                ctx.request_repaint();
            }
        });
    }
}

fn main() -> eframe::Result {
    eframe::run_native(
        "Mandelbrot Explorer",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1280.0, 800.0])
                .with_title("Mandelbrot Explorer [GPU+Perturbation]"),
            ..Default::default()
        },
        Box::new(|_| Ok(Box::new(MandelbrotApp::new()))),
    )
}
