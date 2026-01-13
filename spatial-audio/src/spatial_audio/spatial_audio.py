import coordinates.coordinates
import numpy as np
import quaternion


class SpatialAudio:
    def __init__(self, **kwargs):
        self.nfft = kwargs["nfft"]
        self.nfrequencies = self.nfft // 2 + 1
        self.initial_azimuth_CH = np.float64(kwargs["initial_azimuth"])
        self.initial_elevation_CH = np.float64(kwargs["initial_elevation"])
        self.CH = len(self.initial_azimuth_CH)
        self.hrtf_path = kwargs["hrtf"]["path"]
        self.hrtf_dtype = kwargs["hrtf"]["dtype"]
        with open(self.hrtf_path, "rb") as fid:
            self.HRTF_DOAx2xK = (
                np.frombuffer(fid.read(), dtype=self.hrtf_dtype).reshape((-1, 2, self.nfrequencies)) / self.CH
            )
        self.azimuth_CH = self.initial_azimuth_CH.copy()
        self.elevation_CH = self.initial_elevation_CH.copy()

        self.azimuth_symmetric = kwargs["azimuth"]["symmetric"]
        self.azimuth_span = np.int32(kwargs["azimuth"]["span"])
        self.azimuth_resolution = np.int32(kwargs["azimuth"]["resolution"])
        self.azimuth_range = np.arange(0, self.azimuth_span, self.azimuth_resolution)
        if not self.azimuth_symmetric:
            self.azimuth_range = np.hstack(
                (self.azimuth_range, np.arange(360 - self.azimuth_span, 360, self.azimuth_resolution))
            )
        self.Nazimuth = np.int32(np.size(self.azimuth_range))

        self.elevation_span = np.int32(kwargs["elevation"]["span"])
        self.elevation_resolution = np.int32(kwargs["elevation"]["resolution"])
        self.elevation_range = np.arange(-self.elevation_span, self.elevation_span, self.elevation_resolution)
        self.Nelevation = np.int32(np.size(self.elevation_range))
        self.elevation_min = min(self.elevation_range)
        self.elevation_max = max(self.elevation_range)

        self.reset_tracking()
        self.mode = "binaural"  # Default mode
        self.set_doas()

    def tare_head_orientation(self, yaw, pitch, roll):
        self.global_yaw = yaw
        self.global_pitch = pitch
        self.global_roll = roll
        Qx = quaternion.from_rotation_vector(self.xaxis * np.deg2rad(roll))
        Qy = quaternion.from_rotation_vector(self.yaxis * np.deg2rad(pitch))
        Qz = quaternion.from_rotation_vector(self.zaxis * np.deg2rad(yaw))
        self.global_orientation = Qx * Qy * Qz

    def set_head_orientation(self, yaw, pitch, roll):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        Qx = quaternion.from_rotation_vector(self.xaxis * np.deg2rad(roll))
        Qy = quaternion.from_rotation_vector(self.yaxis * np.deg2rad(pitch))
        Qz = quaternion.from_rotation_vector(self.zaxis * np.deg2rad(yaw))
        self.head_orientation = self.global_orientation.conjugate() * Qx * Qy * Qz

    def combine_head_orientation(self):
        x_CH, y_CH, z_CH = coordinates.coordinates.spherical_to_ned(
            1, np.deg2rad(self.azimuth_CH), np.deg2rad(self.elevation_CH)
        )
        location_CHx3 = np.vstack((x_CH, y_CH, z_CH)).T
        quaternion_location_CH = quaternion.from_float_array(np.hstack((np.zeros((self.CH, 1)), location_CHx3)))
        quaternion_rotated_location_CH = (
            self.head_orientation.conjugate() * quaternion_location_CH * self.head_orientation
        )
        rotated_location_CHx3 = quaternion.as_float_array(quaternion_rotated_location_CH)[:, 1:]
        r_CH, azimuth_CH, elevation_CH = coordinates.coordinates.ned_to_spherical(*rotated_location_CHx3.T)
        return np.rad2deg(elevation_CH), np.rad2deg(azimuth_CH)

    def fetch_hrtf(self, elevation_CH, azimuth_CH):
        elevation_index_CH = np.argmin(np.abs(self.elevation_range - np.expand_dims(elevation_CH, 1)), axis=1)
        elevation_offset_CH = elevation_index_CH * self.Nazimuth

        for ch in range(self.CH):
            while azimuth_CH[ch] < 0:
                azimuth_CH[ch] += 360
            while azimuth_CH[ch] > 359:
                azimuth_CH[ch] -= 360

        replace_left_right_CH = np.tile(False, self.CH)
        if self.azimuth_symmetric:
            for ch in range(self.CH):
                if azimuth_CH[ch] > 180:
                    azimuth_CH[ch] = 360 - azimuth_CH[ch]
                    replace_left_right_CH[ch] = True

        azimuth_index_CH = np.argmin(np.abs(self.azimuth_range - np.expand_dims(azimuth_CH, 1)), axis=1)
        index_CH = np.int64(elevation_offset_CH + azimuth_index_CH)

        HRTF_CHx2xK = self.HRTF_DOAx2xK[index_CH]

        for ch in range(self.CH):
            if replace_left_right_CH[ch]:
                HRTF_CHx2xK[ch] = HRTF_CHx2xK[ch, [1, 0], :]

        return HRTF_CHx2xK

    def set_doas(self):
        elevation_CH, azimuth_CH = self.combine_head_orientation()
        self.HRTF_CHx2xK = self.fetch_hrtf(elevation_CH, azimuth_CH)

    def binauralize(self):
        self.mode = "binaural"

    def monify(self):
        self.mode = "mono"

    def stereofy(self):
        self.mode = "stereo"

    def reset_tracking(self):
        self.xaxis = np.array([1, 0, 0])
        self.yaxis = np.array([0, 1, 0])
        self.zaxis = np.array([0, 0, 1])
        self.tare_head_orientation(0, 0, 0)
        self.set_head_orientation(0, 0, 0)

    def execute(self, frame_fft_CHxK):
        if self.mode == "binaural":
            output = np.array(
                [
                    np.sum(
                        np.array([frame_fft_CHxK[ch] * self.HRTF_CHx2xK[ch, ear] for ch in range(self.CH)]),
                        axis=0,
                    )
                    for ear in range(2)
                ]
            )
            return output
        elif self.mode == "stereo":
            pan_angles = (self.azimuth_CH + 90) / 180 * np.pi / 2
            left_gains = np.cos(pan_angles)
            right_gains = np.sin(pan_angles)

            left_output = np.sum(left_gains[:, np.newaxis] * frame_fft_CHxK, axis=0)
            right_output = np.sum(right_gains[:, np.newaxis] * frame_fft_CHxK, axis=0)

            output = np.array([left_output, right_output])
            return output
        else:
            output = np.tile(np.mean(frame_fft_CHxK, axis=0), reps=(2, 1))
            return output
