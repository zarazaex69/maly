<div align="center">
<img src="asset/logo.png" width="250">

![License](https://img.shields.io/badge/license-BSD--3--Clause-0D1117?style=flat-square&logo=open-source-initiative&logoColor=green&labelColor=0D1117)
![Rust](https://img.shields.io/badge/-Rust-0D1117?style=flat-square&logo=rust&logoColor=C93C28)
</div>

## About

maly - interactive Mandelbrot set explorer with infinite zoom.

GPU-accelerated rendering via WGPU compute shaders for shallow zooms, CPU f64 with rayon parallelism for mid-range, and perturbation theory with automatic rebasing for deep zooms beyond 10^13 magnification. Arbitrary precision reference orbits powered by GMP/MPFR through the rug crate.

### Rendering Modes

| Zoom | Mode | Description |
|------|------|-------------|
| < 10^5 | GPU f32 | WGSL compute shader |
| 10^5 -- 10^13 | CPU f64 | rayon parallel |
| > 10^13 | CPU f128 | BigFloat reference orbit + f64 deltas with rebasing |

### Controls

| Input | Action |
|-------|--------|
| `LMB drag` | Pan |
| `Scroll wheel` | Zoom to cursor |
| `+` / `-` | Iterations +100 / -100 |
| `R` | Reset view |

### Fast start


Dependent:
```
gmp mpfr base-devel cmake fontconfig pkg-config cargo 
```

Run:
```bash
cargo run --release
```

<div align="center">

---

### Contact

Telegram: [zarazaex](https://t.me/zarazaexe)<br>
Email: [zarazaex@tuta.io](mailto:zarazaex@tuta.io)<br>
Site: [zarazaex.xyz](https://zarazaex.xyz)<br>

</div>
