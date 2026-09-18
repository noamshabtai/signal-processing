# equalizer

A graphic equalizer with fixed bands, applied in the STFT domain.

## Model

Each band has a centre frequency and a gain in decibels. The response at an
arbitrary bin is found by interpolating the band gains **in decibels over log
frequency**, which is how the ear hears both axes and gives a smooth curve
rather than the stepped one that per-bin band assignment would produce:

```
gain_dB(f) = interp(log f, log centres, band gains)
response(f) = 10 ** (gain_dB(f) / 20)
```

Below the first centre and above the last, the response is held flat at the end
band's gain. The response is real and positive, so the filter is **zero-phase** —
it scales magnitudes and leaves phase untouched.

The default bands are octave-spaced from 62.5 Hz to 8 kHz.

## Modules

- `equalizer.py` — the bands, the response and the per-frame multiply.
- `system/equalizer.py` — wires `analysis → equalizer → synthesis`.

## Constraints

The constructor rejects a gain list whose length disagrees with the number of
bands, a non-positive centre frequency, and centre frequencies that are not
strictly increasing.
