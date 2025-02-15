import numpy as np


def normal_data_file(**kwargs):
    A = np.random.normal(
        loc=np.float64(kwargs["mean"]),
        scale=np.float64(kwargs["std"]),
        size=list(kwargs["channel_shape"]) + list([kwargs["nsamples"]]),
    )
    with open(kwargs["path"], "wb") as file:
        A.astype(kwargs["dtype"]).tofile(file)
