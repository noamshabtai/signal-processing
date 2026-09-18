import sys

import numpy as np
import scipy.io.wavfile

SAMPLING_FREQUENCY = 16000
CHANNELS = 2
BEATS_PER_MINUTE = 100
BEATS_PER_CHORD = 4

CHORDS = [
    (110.00, [220.00, 261.63, 329.63]),
    (87.31, [174.61, 220.00, 261.63]),
    (130.81, [261.63, 329.63, 392.00]),
    (98.00, [196.00, 246.94, 293.66]),
]

PLUCK_DECAY = 0.45
BASS_DECAY = 0.60
HAT_DECAY = 0.03
DETUNE = 1.004


def sawtooth(frequency, time_N):
    return 2 * (time_N * frequency - np.floor(0.5 + time_N * frequency))


def pluck(frequency, samples, decay):
    time_N = np.arange(samples) / SAMPLING_FREQUENCY
    envelope_N = np.exp(-time_N / decay)
    voice_N = sawtooth(frequency, time_N) + sawtooth(frequency * DETUNE, time_N)
    return envelope_N * voice_N / 2


def hat(samples, generator):
    time_N = np.arange(samples) / SAMPLING_FREQUENCY
    return np.exp(-time_N / HAT_DECAY) * generator.standard_normal(samples)


def add(track_N, start, block_N, gain):
    end = min(start + np.size(block_N), np.size(track_N))
    track_N[start:end] += gain * block_N[: end - start]


def main(path):
    generator = np.random.default_rng(0)
    beat = 60 / BEATS_PER_MINUTE
    beat_samples = int(round(beat * SAMPLING_FREQUENCY))
    chord_samples = beat_samples * BEATS_PER_CHORD
    samples = chord_samples * len(CHORDS)

    track_CHxN = np.zeros((CHANNELS, samples))

    for chord, (bass, notes) in enumerate(CHORDS):
        start = chord * chord_samples
        add(track_CHxN[0], start, pluck(bass, chord_samples, BASS_DECAY), 0.9)
        add(track_CHxN[1], start, pluck(bass, chord_samples, BASS_DECAY), 0.9)

        for half in range(BEATS_PER_CHORD * 2):
            onset = start + half * beat_samples // 2
            for voice, note in enumerate(notes):
                channel = voice % CHANNELS
                add(track_CHxN[channel], onset, pluck(note, beat_samples, PLUCK_DECAY), 0.30)

        for quarter in range(BEATS_PER_CHORD * 4):
            onset = start + quarter * beat_samples // 4
            gain = 0.10 if quarter % 4 else 0.18
            add(track_CHxN[0], onset, hat(beat_samples // 4, generator), gain)
            add(track_CHxN[1], onset, hat(beat_samples // 4, generator), gain)

    fade = np.minimum(np.arange(samples) / beat_samples, 1.0)
    track_CHxN *= np.minimum(fade, fade[::-1])
    track_CHxN /= np.max(np.abs(track_CHxN)) / 0.85

    scipy.io.wavfile.write(path, SAMPLING_FREQUENCY, np.int16(track_CHxN.T * np.iinfo(np.int16).max))
    seconds = samples / SAMPLING_FREQUENCY
    print(f"wrote {path}: {seconds:.1f}s loop, {SAMPLING_FREQUENCY} Hz, {CHANNELS} channels")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "equalizer_input.wav")
