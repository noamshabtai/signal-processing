import numpy as np

import coordinates.coordinates


class RigidSphere:
    def __init__(self, **kwargs):
        self.nfft = kwargs["nfft"]
        self.nfrequencies = self.nfft // 2 + 1
        self.sampling_frequency = kwargs["sampling_frequency"]
        self.head_radius = np.float64(kwargs.get("head_radius", 0.0875))
        self.speed_of_sound = np.float64(kwargs.get("speed_of_sound", 343.0))
        self.ear_azimuth = np.float64(kwargs.get("ear_azimuth", 100.0))
        self.ear_elevation = np.float64(kwargs.get("ear_elevation", -10.0))
        self.extra_orders = np.int32(kwargs.get("extra_orders", 20))
        self.delay_factor = np.float64(kwargs.get("delay_factor", 1.5))
        self.taper_fraction = np.float64(kwargs.get("taper_fraction", 0.1))

        self.frequency_K = np.fft.rfftfreq(self.nfft, 1 / self.sampling_frequency)
        self.ka_K = 2 * np.pi * self.frequency_K * self.head_radius / self.speed_of_sound
        self.norders = np.int32(np.ceil(self.ka_K[-1]) + self.extra_orders)
        self.delay = self.bulk_delay()

        self.ear_azimuth_2 = np.array([-self.ear_azimuth, self.ear_azimuth])
        self.ear_elevation_2 = np.full(2, self.ear_elevation)
        self.coefficient_MxK = self.coefficients()
        self.taper_K = self.taper()

    def bulk_delay(self):
        geometric_advance = self.head_radius / self.speed_of_sound
        taper_pre_ring = 2 / (self.taper_fraction * self.sampling_frequency)
        return self.delay_factor * (geometric_advance + taper_pre_ring)

    def hankel(self, ka_K):
        hankel_MxK = np.empty((self.norders + 1, np.size(ka_K)), dtype=np.complex128)
        exponential_K = np.exp(1j * ka_K) / ka_K
        hankel_MxK[0] = -1j * exponential_K
        hankel_MxK[1] = -(1 + 1j / ka_K) * exponential_K
        with np.errstate(over="ignore", invalid="ignore"):
            for order in range(1, self.norders):
                hankel_MxK[order + 1] = (2 * order + 1) / ka_K * hankel_MxK[order] - hankel_MxK[order - 1]
        return hankel_MxK

    def hankel_derivative(self, ka_K):
        hankel_MxK = self.hankel(ka_K)
        order_M = np.arange(1, self.norders)[:, np.newaxis]
        with np.errstate(over="ignore", invalid="ignore"):
            derivative_MxK = hankel_MxK[:-2] - (order_M + 1) / ka_K * hankel_MxK[1:-1]
        return np.vstack((-hankel_MxK[1:2], derivative_MxK))

    def legendre(self, cosine_A):
        legendre_MxA = np.empty((self.norders, np.size(cosine_A)))
        legendre_MxA[0] = 1
        legendre_MxA[1] = cosine_A
        for order in range(1, self.norders - 1):
            legendre_MxA[order + 1] = (
                (2 * order + 1) * cosine_A * legendre_MxA[order] - order * legendre_MxA[order - 1]
            ) / (order + 1)
        return legendre_MxA

    def coefficients(self):
        order_M = np.arange(self.norders)[:, np.newaxis]
        converged_MxK = order_M <= self.ka_K[1:] + self.extra_orders
        with np.errstate(over="ignore", invalid="ignore"):
            weight_MxK = (2 * order_M + 1) * 1j**order_M / self.hankel_derivative(self.ka_K[1:])
        return np.where(converged_MxK, weight_MxK, 0)

    def taper(self):
        nyquist = self.frequency_K[-1]
        transition_K = (self.frequency_K - (1 - self.taper_fraction) * nyquist) / (self.taper_fraction * nyquist)
        return (1 + np.cos(np.pi * np.clip(transition_K, 0, 1))) / 2

    def surface_pressure(self, cosine_A):
        ka_K = self.ka_K[1:]
        pressure_AxK = 1j / ka_K**2 * (self.legendre(cosine_A).T @ self.coefficient_MxK)
        return np.hstack((np.ones((np.size(cosine_A), 1)), pressure_AxK))

    def transfer_function(self, cosine_A):
        delay_K = np.exp(-2j * np.pi * self.frequency_K * self.delay)
        return np.conj(self.surface_pressure(cosine_A)) * delay_K * self.taper_K

    def cos_incidence(self, elevation_A, azimuth_A):
        source_3xA = np.array(
            coordinates.coordinates.spherical_to_ned(1, np.deg2rad(azimuth_A), np.deg2rad(elevation_A))
        )
        ear_3x2 = np.array(
            coordinates.coordinates.spherical_to_ned(
                1, np.deg2rad(self.ear_azimuth_2), np.deg2rad(self.ear_elevation_2)
            )
        )
        return -(source_3xA.T @ ear_3x2)

    def hrtf(self, grid):
        cosine_DOAx2 = self.cos_incidence(grid.elevation_DOA, grid.azimuth_DOA)
        transfer_DOA2xK = self.transfer_function(np.ravel(cosine_DOAx2))
        return np.reshape(transfer_DOA2xK, (grid.NDOA, 2, self.nfrequencies))
