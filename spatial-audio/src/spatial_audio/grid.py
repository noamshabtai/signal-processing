import numpy as np


class Grid:
    def __init__(self, **kwargs):
        self.azimuth_symmetric = kwargs["azimuth"]["symmetric"]
        self.azimuth_span = np.int32(kwargs["azimuth"]["span"])
        self.azimuth_resolution = np.int32(kwargs["azimuth"]["resolution"])
        self.elevation_span = np.int32(kwargs["elevation"]["span"])
        self.elevation_resolution = np.int32(kwargs["elevation"]["resolution"])

        self.azimuth_range = self.build_azimuth_range()
        self.Nazimuth = np.size(self.azimuth_range)

        self.elevation_range = self.build_elevation_range()
        self.Nelevation = np.size(self.elevation_range)
        self.elevation_min = np.min(self.elevation_range)
        self.elevation_max = np.max(self.elevation_range)

        self.elevation_DOA, self.azimuth_DOA = self.build_doas()
        self.NDOA = np.size(self.azimuth_DOA)

    def build_azimuth_range(self):
        front_range = np.arange(0, self.azimuth_span, self.azimuth_resolution)
        if self.azimuth_symmetric:
            return front_range
        back_range = np.arange(360 - self.azimuth_span, 360, self.azimuth_resolution)
        return np.hstack((front_range, back_range))

    def build_elevation_range(self):
        return np.arange(-self.elevation_span, self.elevation_span, self.elevation_resolution)

    def build_doas(self):
        azimuth_NelevationxNazimuth, elevation_NelevationxNazimuth = np.meshgrid(
            self.azimuth_range, self.elevation_range
        )
        return np.float64(np.ravel(elevation_NelevationxNazimuth)), np.float64(np.ravel(azimuth_NelevationxNazimuth))

    def nearest_index(self, elevation_CH, azimuth_CH):
        elevation_index_CH = np.argmin(np.abs(self.elevation_range - np.expand_dims(elevation_CH, 1)), axis=1)

        wrapped_CH = np.mod(azimuth_CH, 360)
        mirrored_CH = self.azimuth_symmetric & (wrapped_CH > 180)
        folded_CH = np.where(mirrored_CH, 360 - wrapped_CH, wrapped_CH)
        azimuth_index_CH = np.argmin(np.abs(self.azimuth_range - np.expand_dims(folded_CH, 1)), axis=1)

        return np.int64(elevation_index_CH * self.Nazimuth + azimuth_index_CH), mirrored_CH
