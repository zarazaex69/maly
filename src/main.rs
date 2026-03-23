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
        .expect("No GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("mb"),
                ..Default::default()
            },
            None,
        ))
        .expect("No device");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            device,
            queue,
            pipeline,
            bind_group_layout: bgl,
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
        let pb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&pb, 0, bytemuck::bytes_of(&params));
        let sz = (w * h) as u64 * 4;
        let ob = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: sz,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: sz,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pb.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ob.as_entire_binding(),
                },
            ],
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline);
            p.set_bind_group(0, &bg, &[]);
            p.dispatch_workgroups((w + 15) / 16, (h + 15) / 16, 1);
        }
        enc.copy_buffer_to_buffer(&ob, 0, &sb, 0, sz);
        self.queue.submit(std::iter::once(enc.finish()));
        let slice = sb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let d = slice.get_mapped_range().to_vec();
        sb.unmap();
        d
    }
}

struct RefOrbit {
    zx: Vec<f64>,
    zy: Vec<f64>,
}

fn compute_reference_orbit(cx: &Float, cy: &Float, max_iter: u32, prec: u32) -> RefOrbit {
    let mut zx = Float::with_val(prec, 0.0);
    let mut zy = Float::with_val(prec, 0.0);
    let mut tmp = Float::with_val(prec, 0.0);
    let mut zx2 = Float::with_val(prec, 0.0);
    let mut zy2 = Float::with_val(prec, 0.0);
    let bailout = Float::with_val(prec, 1e6);
    let mut ox = Vec::with_capacity(max_iter as usize + 1);
    let mut oy = Vec::with_capacity(max_iter as usize + 1);
    ox.push(0.0f64);
    oy.push(0.0f64);
    for _ in 0..max_iter {
        tmp.assign(&zx * &zy);
        zy.assign(&tmp * 2.0);
        zy += cy;
        tmp.assign(&zx2 - &zy2);
        zx.assign(&tmp + cx);
        zx2.assign(zx.clone().pow(2));
        zy2.assign(zy.clone().pow(2));
        ox.push(zx.to_f64());
        oy.push(zy.to_f64());
        tmp.assign(&zx2 + &zy2);
        if tmp > bailout {
            break;
        }
    }
    RefOrbit { zx: ox, zy: oy }
}

fn perturbation_iterate(ref_orbit: &RefOrbit, dcx: f64, dcy: f64, max_iter: u32) -> (u32, f64) {
    let mut dx = 0.0f64;
    let mut dy = 0.0f64;
    let olen = ref_orbit.zx.len();
    let mut m = 0usize;

    for iter in 0..max_iter {
        if m + 1 >= olen {
            break;
        }
        let zx = ref_orbit.zx[m];
        let zy = ref_orbit.zy[m];
        let ndx = 2.0 * (zx * dx - zy * dy) + dx * dx - dy * dy + dcx;
        let ndy = 2.0 * (zx * dy + zy * dx) + 2.0 * dx * dy + dcy;
        dx = ndx;
        dy = ndy;
        m += 1;
        let fx = ref_orbit.zx[m] + dx;
        let fy = ref_orbit.zy[m] + dy;
        let mag2 = fx * fx + fy * fy;
        if mag2 > 256.0 {
            let smooth = (iter as f64) + 1.0 - mag2.ln().ln() / std::f64::consts::LN_2;
            return (iter, smooth);
        }
        let dmag2 = dx * dx + dy * dy;
        if mag2 < dmag2 {
            dx = fx;
            dy = fy;
            m = 0;
        }
    }
    (max_iter, max_iter as f64)
}

fn mandelbrot_bigfloat(cx: &Float, cy: &Float, max_iter: u32, prec: u32) -> (u32, f64) {
    let mut zx = Float::with_val(prec, 0.0);
    let mut zy = Float::with_val(prec, 0.0);
    let mut zx2 = Float::with_val(prec, 0.0);
    let mut zy2 = Float::with_val(prec, 0.0);
    let mut tmp = Float::with_val(prec, 0.0);
    let bailout = Float::with_val(prec, 256.0);
    for i in 0..max_iter {
        tmp.assign(&zx * &zy);
        zy.assign(&tmp * 2.0);
        zy += cy;
        tmp.assign(&zx2 - &zy2);
        zx.assign(&tmp + cx);
        zx2.assign(zx.clone().pow(2));
        zy2.assign(zy.clone().pow(2));
        tmp.assign(&zx2 + &zy2);
        if tmp > bailout {
            let m2 = tmp.to_f64();
            let smooth = (i as f64) + 1.0 - m2.ln().ln() / std::f64::consts::LN_2;
            return (i, smooth);
        }
    }
    (max_iter, max_iter as f64)
}

fn mandelbrot_f64(cx: f64, cy: f64, max_iter: u32) -> (u32, f64) {
    let mut zx = 0.0f64;
    let mut zy = 0.0f64;
    let mut zx2 = 0.0f64;
    let mut zy2 = 0.0f64;
    for i in 0..max_iter {
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        zx2 = zx * zx;
        zy2 = zy * zy;
        if zx2 + zy2 > 256.0 {
            let smooth = (i as f64) + 1.0 - (zx2 + zy2).ln().ln() / std::f64::consts::LN_2;
            return (i, smooth);
        }
    }
    (max_iter, max_iter as f64)
}

fn iteration_color(smooth_iter: f64, max_iter: u32) -> [u8; 4] {
    if smooth_iter >= max_iter as f64 {
        return [0, 0, 0, 255];
    }
    let t = (smooth_iter * 0.015).fract();
    let c = colorous::TURBO.eval_continuous(t);
    [c.r, c.g, c.b, 255]
}

fn required_precision(zoom_width: f64) -> u32 {
    let bits = (-zoom_width.log2() + 48.0).max(128.0) as u32;
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
            center_x: Float::with_val(256, -0.5),
            center_y: Float::with_val(256, 0.0),
            zoom: Float::with_val(256, 3.0),
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
        let cgen = self.generation;
        let rs = Arc::clone(&self.render_result);
        let ir = Arc::clone(&self.is_rendering);
        let cxf = self.center_x.to_f64();
        let cyf = self.center_y.to_f64();

        match mode {
            RenderMode::Gpu => {
                let gpu = Arc::clone(&self.gpu);
                std::thread::spawn(move || {
                    let px = gpu.render(
                        cxf as f32,
                        cyf as f32,
                        zoom_f64 as f32,
                        max_iter,
                        width as u32,
                        height as u32,
                    );
                    *rs.lock().unwrap() = Some(RenderResult {
                        pixels: px,
                        width,
                        height,
                        render_gen: cgen,
                    });
                    *ir.lock().unwrap() = false;
                });
            }
            RenderMode::CpuF64 => {
                std::thread::spawn(move || {
                    let md = width.min(height) as f64;
                    let ps = zoom_f64 / md;
                    let hw = width as f64 / 2.0;
                    let hh = height as f64 / 2.0;
                    let mut buf = vec![0u8; width * height * 4];
                    let chunks: Vec<(usize, &mut [u8])> =
                        buf.chunks_exact_mut(width * 4).enumerate().collect();
                    chunks.into_par_iter().for_each(|(py, row)| {
                        let wy = cyf + ps * (py as f64 - hh);
                        for px in 0..width {
                            let wx = cxf + ps * (px as f64 - hw);
                            let (_, s) = mandelbrot_f64(wx, wy, max_iter);
                            let c = iteration_color(s, max_iter);
                            row[px * 4..px * 4 + 4].copy_from_slice(&c);
                        }
                    });
                    *rs.lock().unwrap() = Some(RenderResult {
                        pixels: buf,
                        width,
                        height,
                        render_gen: cgen,
                    });
                    *ir.lock().unwrap() = false;
                });
            }
            RenderMode::Perturbation(prec) => {
                let cxb = Float::with_val(prec, &self.center_x);
                let cyb = Float::with_val(prec, &self.center_y);
                let zb = Float::with_val(prec, &self.zoom);
                std::thread::spawn(move || {
                    let refo = compute_reference_orbit(&cxb, &cyb, max_iter, prec);
                    let md = width.min(height) as f64;
                    let psb = Float::with_val(prec, &zb / md);
                    let hw = width as f64 / 2.0;
                    let hh = height as f64 / 2.0;

                    let dcx_arr: Vec<f64> = (0..width)
                        .map(|px| Float::with_val(prec, &psb * (px as f64 - hw)).to_f64())
                        .collect();
                    let dcy_arr: Vec<f64> = (0..height)
                        .map(|py| Float::with_val(prec, &psb * (py as f64 - hh)).to_f64())
                        .collect();

                    let mut buf = vec![0u8; width * height * 4];
                    let chunks: Vec<(usize, &mut [u8])> =
                        buf.chunks_exact_mut(width * 4).enumerate().collect();
                    chunks.into_par_iter().for_each(|(py, row)| {
                        let dy = dcy_arr[py];
                        for px in 0..width {
                            let dx = dcx_arr[px];
                            let (_, s) = perturbation_iterate(&refo, dx, dy, max_iter);
                            let c = iteration_color(s, max_iter);
                            row[px * 4..px * 4 + 4].copy_from_slice(&c);
                        }
                    });

                    *rs.lock().unwrap() = Some(RenderResult {
                        pixels: buf,
                        width,
                        height,
                        render_gen: cgen,
                    });
                    *ir.lock().unwrap() = false;
                });
            }
        }
    }

    fn zoom_info(&self) -> String {
        let z = self.zoom.to_f64();
        let mag = 3.0 / z;
        let mode = match choose_mode(z) {
            RenderMode::Gpu => "GPU f32".into(),
            RenderMode::CpuF64 => "CPU f64".into(),
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
                let prec = required_precision(self.zoom.to_f64()).max(128);
                let md = w.min(h) as f64;
                let ps = Float::with_val(prec, &self.zoom / md);
                let fdx = Float::with_val(prec, &ps * (-(d.x as f64)));
                let fdy = Float::with_val(prec, &ps * (-(d.y as f64)));
                self.center_x += &fdx;
                self.center_y += &fdy;
                self.generation += 1;
            }

            if let Some(hp) = resp.hover_pos() {
                let scroll = ui.input(|i| i.raw_scroll_delta.y);
                if scroll.abs() > 0.0 {
                    let factor = if scroll > 0.0 { 0.75 } else { 1.333 };
                    let md = w.min(h) as f64;
                    let rx = (hp.x - rect.center().x) as f64;
                    let ry = (hp.y - rect.center().y) as f64;
                    let prec = required_precision(self.zoom.to_f64()).max(128);
                    let ps = Float::with_val(prec, &self.zoom / md);
                    let ox = Float::with_val(prec, &ps * rx);
                    let oy = Float::with_val(prec, &ps * ry);
                    let wx = Float::with_val(prec, &self.center_x + &ox);
                    let wy = Float::with_val(prec, &self.center_y + &oy);
                    self.zoom *= factor;
                    let nps = Float::with_val(prec, &self.zoom / md);
                    let nox = Float::with_val(prec, &nps * rx);
                    let noy = Float::with_val(prec, &nps * ry);
                    self.center_x = Float::with_val(prec, &wx - &nox);
                    self.center_y = Float::with_val(prec, &wy - &noy);
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
                                let g = Arc::clone(&self.gpu);
                                *self = Self::new();
                                self.gpu = g;
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
                .with_title("Mandelbrot Explorer"),
            ..Default::default()
        },
        Box::new(|_| Ok(Box::new(MandelbrotApp::new()))),
    )
}
