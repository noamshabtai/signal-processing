import numpy as np
import quaternion

import coordinates.coordinates
import spatial_audio.grid
import spatial_audio.rigid_sphere


class SpatialAudio:
    def __init__(self, **kwargs):
        self.nfft = kwargs["nfft"]
        self.nfrequencies = self.nfft // 2 + 1
        self.initial_azimuth_CH = np.float64(kwargs["initial_azimuth"])
        self.initial_elevation_CH = np.float64(kwargs["initial_elevation"])
        self.CH = len(self.initial_azimuth_CH)
        self.sampling_frequency = kwargs["sampling_frequency"]
        self.azimuth_CH = self.initial_azimuth_CH.copy()
        self.elevation_CH = self.initial_elevation_CH.copy()

        self.grid = spatial_audio.grid.Grid(azimuth=kwargs["azimuth"], elevation=kwargs["elevation"])

        self.hrtf_dtype = kwargs["hrtf"]["dtype"]
        self.hrtf_equalization = kwargs["hrtf"].get("equalization", False)
        self.hrtf_gain_db = np.float64(kwargs["hrtf"].get("gain_db", 0.0))
        self.hrtf_floor_db = np.float64(kwargs["hrtf"].get("floor_db", -40.0))
        self.rigid_sphere_kwargs = {
            "nfft": self.nfft,
            "sampling_frequency": self.sampling_frequency,
        } | kwargs[
            "hrtf"
        ].get("rigid_sphere", {})
        self.HRTF_DOAx2xK = (self.equalize_hrtf(self.synthesize_hrtf()) / self.CH).astype(self.hrtf_dtype)

        self.reset_tracking()
        self.mode = "binaural"
        self.set_doas()

    def synthesize_hrtf(self):
        model = spatial_audio.rigid_sphere.RigidSphere(**self.rigid_sphere_kwargs)
        return model.hrtf(self.grid)

    def equalization(self, HRTF_DOAx2xK):
        magnitude_K = np.full(self.nfrequencies, 10 ** (self.hrtf_gain_db / 20))
        if self.hrtf_equalization:
            diffuse_field_K = np.sqrt(np.mean(np.abs(HRTF_DOAx2xK) ** 2, axis=(0, 1)))
            floor = np.max(diffuse_field_K) * 10 ** (self.hrtf_floor_db / 20)
            magnitude_K = magnitude_K / np.maximum(diffuse_field_K, floor)

        cepstrum_N = np.fft.irfft(np.log(magnitude_K), n=self.nfft)
        causal_N = np.zeros(self.nfft)
        causal_N[0] = 1
        causal_N[1 : self.nfft // 2] = 2
        causal_N[self.nfft // 2] = 1

        return np.exp(np.fft.rfft(cepstrum_N * causal_N))

    def equalize_hrtf(self, HRTF_DOAx2xK):
        return HRTF_DOAx2xK * self.equalization(HRTF_DOAx2xK)

    def tare_head_orientation(self, yaw, pitch, roll):
        self.global_yaw = yaw
        self.global_pitch = pitch
        self.global_roll = roll
        Qx = quaternion.from_rotation_vector(self.xaxis * np.deg2rad(roll))
        Qy = quaternion.from_rotation_vector(self.yaxis * np.deg2rad(pitch))
        Qz = quaternion.from_rotation_vector(self.zaxis * np.deg2rad(yaw))
        self.global_orientation = Qz * Qy * Qx

    def set_head_orientation(self, yaw, pitch, roll):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        Qx = quaternion.from_rotation_vector(self.xaxis * np.deg2rad(roll))
        Qy = quaternion.from_rotation_vector(self.yaxis * np.deg2rad(pitch))
        Qz = quaternion.from_rotation_vector(self.zaxis * np.deg2rad(yaw))
        self.head_orientation = self.global_orientation.conjugate() * Qz * Qy * Qx

    def head_yaw_pitch_roll(self):
        rotation = quaternion.as_rotation_matrix(self.head_orientation)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2]))
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        return np.rad2deg([yaw, pitch, roll])

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
        index_CH, mirrored_CH = self.grid.nearest_index(elevation_CH, azimuth_CH)
        HRTF_CHx2xK = self.HRTF_DOAx2xK[index_CH]
        HRTF_CHx2xK[mirrored_CH] = HRTF_CHx2xK[mirrored_CH][:, [1, 0], :]
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
        match self.mode:
            case "binaural":
                return np.array(
                    [
                        np.sum(
                            np.array([frame_fft_CHxK[ch] * self.HRTF_CHx2xK[ch, ear] for ch in range(self.CH)]),
                            axis=0,
                        )
                        for ear in range(2)
                    ]
                )
            case "stereo":
                pan_angles = (self.azimuth_CH + 90) / 180 * np.pi / 2
                left_output = np.sum(np.cos(pan_angles)[:, np.newaxis] * frame_fft_CHxK, axis=0)
                right_output = np.sum(np.sin(pan_angles)[:, np.newaxis] * frame_fft_CHxK, axis=0)
                return np.array([left_output, right_output])
            case _:
                return np.tile(np.mean(frame_fft_CHxK, axis=0), reps=(2, 1))
