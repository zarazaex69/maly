use eframe::egui;
use rayon::prelude::*;
use rug::Assign;
use rug::Float;
use rug::ops::Pow;
use std::sync::{Arc, Mutex};

const F64_ZOOM_THRESHOLD: f64 = 1e-13;

fn required_precision(zoom_width: f64) -> u32 {
    if zoom_width > F64_ZOOM_THRESHOLD {
        return 0;
    }
    let bits = (-zoom_width.log2() + 32.0).max(128.0) as u32;
    (bits + 31) & !31
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

struct RenderResult {
    pixels: Vec<u8>,
    width: usize,
    height: usize,
    render_gen: u64,
    pass: u32,
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
    rendered_pass: u32,
    is_rendering: Arc<Mutex<bool>>,
    render_scale: f32,
    target_scale: f32,
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
            rendered_pass: 0,
            is_rendering: Arc::new(Mutex::new(false)),
            render_scale: 0.25,
            target_scale: 1.0,
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

        let scale = if self.rendered_generation < self.generation {
            self.render_scale
        } else {
            self.target_scale
        };

        let pass = if scale < self.target_scale { 0 } else { 1 };

        let zoom_f64 = self.zoom.to_f64();
        let prec = required_precision(zoom_f64);
        let use_bigfloat = prec > 0;

        let cx_f64 = self.center_x.to_f64();
        let cy_f64 = self.center_y.to_f64();
        let cx_big = if use_bigfloat {
            Some(Float::with_val(prec.max(128), &self.center_x))
        } else {
            None
        };
        let cy_big = if use_bigfloat {
            Some(Float::with_val(prec.max(128), &self.center_y))
        } else {
            None
        };
        let zoom_big = if use_bigfloat {
            Some(Float::with_val(prec.max(128), &self.zoom))
        } else {
            None
        };

        let max_iter = self.max_iter;
        let current_gen = self.generation;
        let result_slot = Arc::clone(&self.render_result);

        let rw = ((width as f32) * scale).max(1.0) as usize;
        let rh = ((height as f32) * scale).max(1.0) as usize;

        std::thread::spawn(move || {
            let min_dim = rw.min(rh) as f64;

            let pixels: Vec<u8> = if use_bigfloat {
                let p = prec.max(128);
                let zb = zoom_big.as_ref().unwrap();
                let cxb = cx_big.as_ref().unwrap();
                let cyb = cy_big.as_ref().unwrap();
                let pixel_size = Float::with_val(p, zb / min_dim);
                let half_w = rw as f64 / 2.0;
                let half_h = rh as f64 / 2.0;

                let rows: Vec<Vec<[u8; 3]>> = (0..rh)
                    .into_par_iter()
                    .map(|py| {
                        let dy = Float::with_val(p, &pixel_size * (py as f64 - half_h));
                        let world_y = Float::with_val(p, cyb + &dy);
                        (0..rw)
                            .map(|px| {
                                let dx = Float::with_val(p, &pixel_size * (px as f64 - half_w));
                                let world_x = Float::with_val(p, cxb + &dx);
                                let (_, smooth) =
                                    mandelbrot_bigfloat(&world_x, &world_y, max_iter, p);
                                iteration_color(smooth, max_iter)
                            })
                            .collect()
                    })
                    .collect();

                let mut buf = Vec::with_capacity(rw * rh * 4);
                for row in &rows {
                    for &[r, g, b] in row {
                        buf.extend_from_slice(&[r, g, b, 255]);
                    }
                }
                buf
            } else {
                let pixel_size = zoom_f64 / min_dim;
                let half_w = rw as f64 / 2.0;
                let half_h = rh as f64 / 2.0;

                let rows: Vec<Vec<[u8; 3]>> = (0..rh)
                    .into_par_iter()
                    .map(|py| {
                        let world_y = cy_f64 + pixel_size * (py as f64 - half_h);
                        (0..rw)
                            .map(|px| {
                                let world_x = cx_f64 + pixel_size * (px as f64 - half_w);
                                let (_, smooth) = mandelbrot_f64(world_x, world_y, max_iter);
                                iteration_color(smooth, max_iter)
                            })
                            .collect()
                    })
                    .collect();

                let mut buf = Vec::with_capacity(rw * rh * 4);
                for row in &rows {
                    for &[r, g, b] in row {
                        buf.extend_from_slice(&[r, g, b, 255]);
                    }
                }
                buf
            };

            *result_slot.lock().unwrap() = Some(RenderResult {
                pixels,
                width: rw,
                height: rh,
                render_gen: current_gen,
                pass,
            });
            *is_rendering_arc.lock().unwrap() = false;
        });
    }

    fn zoom_info(&self) -> String {
        let zoom_f64 = self.zoom.to_f64();
        let magnification = 3.0 / zoom_f64;
        let log10 = magnification.log10();
        let prec = required_precision(zoom_f64);
        let mode = if prec > 0 {
            format!("BigFloat {}bit", prec)
        } else {
            "f64 (fast)".to_string()
        };
        format!(
            "Zoom: 10^{:.1} | Iter: {} | {} | Scale: {:.0}%",
            log10,
            self.max_iter,
            mode,
            self.target_scale * 100.0
        )
    }
}

impl eframe::App for MandelbrotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("info_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(self.zoom_info());
                ui.separator();
                ui.label("Drag: pan | Scroll: zoom | +/-: iter | Q/E: quality | R: reset");
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
                    let dx = pixel_size * (-(delta.x as f64));
                    let dy = pixel_size * (-(delta.y as f64));
                    self.center_x += dx;
                    self.center_y += dy;
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
                        let new_pixel_size = Float::with_val(p, &self.zoom / min_dim);
                        let new_off_x = Float::with_val(p, &new_pixel_size * rel_x);
                        let new_off_y = Float::with_val(p, &new_pixel_size * rel_y);
                        self.center_x = Float::with_val(p, &world_x - &new_off_x);
                        self.center_y = Float::with_val(p, &world_y - &new_off_y);
                    } else {
                        let zoom_f64 = self.zoom.to_f64();
                        let pixel_size = zoom_f64 / min_dim;
                        let world_x = self.center_x.to_f64() + pixel_size * rel_x;
                        let world_y = self.center_y.to_f64() + pixel_size * rel_y;
                        self.zoom *= factor;
                        let new_pixel_size = self.zoom.to_f64() / min_dim;
                        let p = 128u32;
                        self.center_x = Float::with_val(p, world_x - new_pixel_size * rel_x);
                        self.center_y = Float::with_val(p, world_y - new_pixel_size * rel_y);
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
                            egui::Key::Q => {
                                self.target_scale = (self.target_scale - 0.25).max(0.25);
                                self.generation += 1;
                            }
                            egui::Key::E => {
                                self.target_scale = (self.target_scale + 0.25).min(2.0);
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
                let mut result_guard = self.render_result.lock().unwrap();
                if let Some(result) = result_guard.take() {
                    if result.render_gen >= self.rendered_generation {
                        let color_image = egui::ColorImage::from_rgba_unmultiplied(
                            [result.width, result.height],
                            &result.pixels,
                        );
                        let texture = ctx.load_texture(
                            "mandelbrot",
                            color_image,
                            egui::TextureOptions {
                                magnification: egui::TextureFilter::Linear,
                                minification: egui::TextureFilter::Linear,
                                ..Default::default()
                            },
                        );
                        self.current_texture = Some(texture);
                        self.rendered_generation = result.render_gen;
                        self.rendered_pass = result.pass;
                    }
                }
            }

            if let Some(tex) = &self.current_texture {
                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                ui.painter().image(tex.id(), rect, uv, egui::Color32::WHITE);
            }

            let needs_render = self.generation > self.rendered_generation
                || (self.rendered_generation == self.generation && self.rendered_pass == 0);
            if needs_render {
                self.schedule_render(w, h);
            }

            let is_rendering = *self.is_rendering.lock().unwrap();
            if is_rendering || needs_render {
                ctx.request_repaint();
            }
        });
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("Mandelbrot Explorer"),
        ..Default::default()
    };
    eframe::run_native(
        "Mandelbrot Explorer",
        options,
        Box::new(|_cc| Ok(Box::new(MandelbrotApp::default()))),
    )
}
