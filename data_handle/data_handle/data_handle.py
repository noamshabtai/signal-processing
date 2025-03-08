import numpy as np


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
