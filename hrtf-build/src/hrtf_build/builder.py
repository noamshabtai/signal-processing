import hrtf_build.grid
import hrtf_build.rigid_sphere


class Builder:
    def __init__(self, **kwargs):
        self.dtype = kwargs["dtype"]
        self.grid = hrtf_build.grid.Grid(**kwargs["grid"])
        self.model = hrtf_build.rigid_sphere.RigidSphere(**kwargs["rigid_sphere"])

    def build(self):
        return self.model.hrtf(self.grid).astype(self.dtype)

    def write(self, path):
        HRTF_DOAx2xK = self.build()
        with open(path, "wb") as fid:
            fid.write(HRTF_DOAx2xK.tobytes())
        return HRTF_DOAx2xK
