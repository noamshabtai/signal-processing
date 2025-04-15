import numpy as np


def float_dtype_to_complex_dtype(float_dtype):
    if float_dtype == np.float32:
        return np.complex64
    elif float_dtype == np.float64:
        return np.complex128
    else:
        raise ValueError("Unsupported float dtype: {}".format(float_dtype))


def normal_data_file(**kwargs):
    A = np.random.normal(
        loc=np.float64(kwargs["mean"]),
        scale=np.float64(kwargs["std"]),
        size=list(kwargs["channel_shape"]) + list([kwargs["nsamples"]]),
    )
    with open(kwargs["path"], "wb") as file:
        A.astype(kwargs["dtype"]).tofile(file)


def make_yaml_safe(data):
    if isinstance(data, dict):
        return {k: make_yaml_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_yaml_safe(v) for v in data]
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    else:
        return str(data)


def get_int_type_from_nbits(nbits):
    return np.int8 if nbits == 8 else np.int16 if nbits == 16 else np.int32 if nbits == 32 else np.int64


def get_int_type_from_nbytes(nbytes):
    return np.int8 if nbytes == 1 else np.int16 if nbytes == 2 else np.int32 if nbytes == 4 else np.int64


def get_float_type_from_nbits(nbits):
    return np.float16 if nbits == 16 else np.float32 if nbits == 32 else np.float64


def get_complex_type_from_nbits(nbits):
    return np.complex64 if nbits == 32 else np.complex128
