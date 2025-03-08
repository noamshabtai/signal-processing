import data_handle.data_handle
import numpy as np


def test_normal_data_file(kwargs_normal_data_file):
    kwargs = kwargs_normal_data_file
    data_handle.data_handle.normal_data_file(**kwargs)
    with open(kwargs["path"], "rb") as file:
        A = np.fromfile(file, dtype=kwargs["dtype"])
    assert (
        A.nbytes
        == np.prod((np.array(kwargs["channel_shape"], dtype=np.int16)))
        * np.int16(kwargs["nsamples"])
        * np.dtype(kwargs["dtype"]).itemsize
    )


def test_make_yaml_safe(kwargs_make_yaml_safe):
    kwargs = kwargs_make_yaml_safe
    input_data = kwargs["input"]
    expected_output = kwargs["expected"]

    assert data_handle.data_handle.make_yaml_safe(input_data) == expected_output
