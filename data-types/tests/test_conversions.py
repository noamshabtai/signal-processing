import data_types.conversions
import numpy as np


def test_float_dtype_to_complex_dtype():
    assert data_types.conversions.float_dtype_to_complex_dtype(np.float32) == np.complex64
    assert data_types.conversions.float_dtype_to_complex_dtype(np.float64) == np.complex128


def test_get_int_type_from_nbits():
    assert data_types.conversions.get_int_type_from_nbits(8) == np.int8
    assert data_types.conversions.get_int_type_from_nbits(16) == np.int16
    assert data_types.conversions.get_int_type_from_nbits(32) == np.int32
    assert data_types.conversions.get_int_type_from_nbits(64) == np.int64


def test_get_int_type_from_nbytes():
    assert data_types.conversions.get_int_type_from_nbytes(1) == np.int8
    assert data_types.conversions.get_int_type_from_nbytes(2) == np.int16
    assert data_types.conversions.get_int_type_from_nbytes(4) == np.int32
    assert data_types.conversions.get_int_type_from_nbytes(8) == np.int64


def test_get_float_type_from_nbits():
    assert data_types.conversions.get_float_type_from_nbits(16) == np.float16
    assert data_types.conversions.get_float_type_from_nbits(32) == np.float32
    assert data_types.conversions.get_float_type_from_nbits(64) == np.float64


def test_get_complex_type_from_nbits():
    assert data_types.conversions.get_complex_type_from_nbits(32) == np.complex64
    assert data_types.conversions.get_complex_type_from_nbits(64) == np.complex128
