import hrtf_build.builder
import numpy as np


def test_init(kwargs_builder):
    kwargs = kwargs_builder
    tested = hrtf_build.builder.Builder(**kwargs["tested"])

    azimuth = kwargs["tested"]["grid"]["azimuth"]
    assert tested.dtype == kwargs["tested"]["dtype"]
    assert tested.grid.Nazimuth == azimuth["span"] // azimuth["resolution"]
    assert tested.model.nfft == kwargs["tested"]["rigid_sphere"]["nfft"]


def test_build(kwargs_builder):
    kwargs = kwargs_builder
    tested = hrtf_build.builder.Builder(**kwargs["tested"])

    HRTF_DOAx2xK = tested.build()

    assert HRTF_DOAx2xK.dtype == np.dtype(tested.dtype)
    assert np.shape(HRTF_DOAx2xK) == (tested.grid.NDOA, 2, tested.model.nfrequencies)
    assert np.all(np.isfinite(HRTF_DOAx2xK))
    assert np.allclose(HRTF_DOAx2xK, tested.model.hrtf(tested.grid), atol=1e-6)


def test_write(kwargs_builder, tmp_path):
    kwargs = kwargs_builder
    tested = hrtf_build.builder.Builder(**kwargs["tested"])
    path = tmp_path / "hrtf.bin"

    HRTF_DOAx2xK = tested.write(path)

    with open(path, "rb") as fid:
        read_DOAx2xK = np.frombuffer(fid.read(), dtype=tested.dtype).reshape((-1, 2, tested.model.nfrequencies))
    assert np.all(read_DOAx2xK == HRTF_DOAx2xK)
    assert np.shape(read_DOAx2xK) == np.shape(HRTF_DOAx2xK)
