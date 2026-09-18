import numpy as np

DEFAULT_CENTER_FREQUENCIES_HZ = [62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]


class Equalizer:
    def __init__(self, **kwargs):
        self.nfft = kwargs["nfft"]
        self.sampling_frequency = kwargs["sampling_frequency"]
        self.nfrequencies = self.nfft // 2 + 1

        self.center_frequencies_B = np.float64(kwargs.get("center_frequencies_hz", DEFAULT_CENTER_FREQUENCIES_HZ))
        self.nbands = np.size(self.center_frequencies_B)
        self.gains_db_B = np.float64(kwargs.get("gains_db", np.zeros(self.nbands)))

        if np.size(self.gains_db_B) != self.nbands:
            raise ValueError(f"Gains disagree with bands: {np.size(self.gains_db_B)} gains, {self.nbands} bands")

        if not np.all(self.center_frequencies_B > 0):
            raise ValueError(f"Center frequency not positive: {np.min(self.center_frequencies_B)}")

        if np.any(np.diff(self.center_frequencies_B) <= 0):
            raise ValueError("Center frequencies not increasing")

        self.frequency_K = np.fft.rfftfreq(self.nfft, 1 / self.sampling_frequency)

    def frequency_response(self):
        log_frequency_K = np.log(np.maximum(self.frequency_K, self.center_frequencies_B[0]))
        gains_db_K = np.interp(log_frequency_K, np.log(self.center_frequencies_B), self.gains_db_B)
        return 10 ** (gains_db_K / 20)

    def execute(self, frame_fft_CHxK):
        return (frame_fft_CHxK * self.frequency_response()).astype(frame_fft_CHxK.dtype)
