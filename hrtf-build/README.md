# hrtf-build

Builds synthetic HRTF sets in the layout that `spatial-audio` consumes:
`(DOA, 2, nfft // 2 + 1)` complex spectra, DOA ordered elevation-major.

## Modules

- `grid.py` — `Grid` owns the DOA layout: the azimuth/elevation ranges built
  from `span`/`resolution`, the flattened `elevation_DOA`/`azimuth_DOA`, and
  `nearest_index()`, which maps arbitrary angles back to a DOA index plus a
  left/right-mirror flag for symmetric grids.
- `rigid_sphere.py` — `RigidSphere` is the analytic model: a plane wave
  scattered by a rigid sphere, evaluated at two ear points on its surface.
- `builder.py` — `Builder` wires a grid to a model, casts to the target dtype,
  and writes the raw `.bin`.

## Rigid-sphere model

The total field on a rigid sphere of radius `a` for a plane wave arriving from
direction `-k` is

```
H(ka, gamma) = 1j / (ka)**2 * sum_m (2m + 1) * 1j**m * P_m(cos gamma) / h_m'(ka)
```

where `gamma` is the angle between the propagation direction and the surface
point. The series is truncated per frequency at `ka + extra_orders`; spherical
Hankel functions come from the upward recursion, which is stable because
`h_m` is the growing solution.

The physics convention above is conjugated to the DSP convention (a delay is
`exp(-1j * omega * tau)`).

Two corrections make the result usable as a frequency-domain product in an
`nfft`-point STFT, where the impulse response has to stay short compared to the
window:

- `|H|` tends to 2 at the illuminated pole and does not roll off, so the
  specular arrival is a fractional-sample impulse whose sinc tail wraps around
  the buffer. A raised-cosine taper over the top `taper_fraction` of the band
  brings the response to zero at Nyquist, which makes the tail decay fast.
- The series is referenced to the sphere centre, so the illuminated pole
  arrives `a / c` *before* the reference and the taper adds a pre-ring of about
  `2 / (taper_fraction * sampling_frequency)`. `bulk_delay()` is
  `delay_factor` times the sum of the two, which pushes the whole response past
  `t = 0`.

The tests hold both to their contract: the bulk delay stays inside `nfft // 2`,
and over 99% of the impulse-response energy lands in the first half of the
window.

Both ears sit at `elevation = ear_elevation`, `azimuth = ±ear_azimuth`. The
model therefore gives correct ITD, frequency-dependent ILD and head shadowing,
a weak spectral elevation cue, and no front/back discrimination.

## Configuration

```yaml
dtype: complex64
grid:
  azimuth:
    symmetric: true
    span: 90
    resolution: 10
  elevation:
    span: 30
    resolution: 10
rigid_sphere:
  nfft: 512
  sampling_frequency: 16000
  head_radius: 0.0875
  speed_of_sound: 343.0
  ear_azimuth: 100.0
  ear_elevation: -10.0
  extra_orders: 20
  taper_fraction: 0.1
  delay_factor: 1.5
```

Everything under `rigid_sphere` except `nfft` and `sampling_frequency` is
optional and defaults to the values shown.
