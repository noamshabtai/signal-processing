import pathlib

import data_handle.utils
import numpy as np


def test_float_dtype_to_complex_dtype():
    assert data_handle.utils.float_dtype_to_complex_dtype(np.float32) == np.complex64
    assert data_handle.utils.float_dtype_to_complex_dtype(np.float64) == np.complex128
    try:
        data_handle.utils.float_dtype_to_complex_dtype(np.int32)
    except ValueError as e:
        assert str(e) == "Unsupported float dtype: <class 'numpy.int32'>"


def test_normal_data_file(kwargs_normal_data_file, tmp_path):
    kwargs = kwargs_normal_data_file
    kwargs["path"] = tmp_path / pathlib.Path(kwargs["path"]).name
    data_handle.utils.normal_data_file(**kwargs)
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

    assert data_handle.utils.make_yaml_safe(input_data) == expected_output


def test_get_int_type_from_nbits():
    assert data_handle.utils.get_int_type_from_nbits(8) == np.int8
    assert data_handle.utils.get_int_type_from_nbits(16) == np.int16
    assert data_handle.utils.get_int_type_from_nbits(32) == np.int32
    assert data_handle.utils.get_int_type_from_nbits(64) == np.int64


def test_get_int_type_from_nbytes():
    assert data_handle.utils.get_int_type_from_nbytes(1) == np.int8
    assert data_handle.utils.get_int_type_from_nbytes(2) == np.int16
    assert data_handle.utils.get_int_type_from_nbytes(4) == np.int32
    assert data_handle.utils.get_int_type_from_nbytes(8) == np.int64


def test_get_float_type_from_nbits():
    assert data_handle.utils.get_float_type_from_nbits(16) == np.float16
    assert data_handle.utils.get_float_type_from_nbits(32) == np.float32
    assert data_handle.utils.get_float_type_from_nbits(64) == np.float64


def test_get_complex_type_from_nbits():
    assert data_handle.utils.get_complex_type_from_nbits(32) == np.complex64
    assert data_handle.utils.get_complex_type_from_nbits(64) == np.complex128
