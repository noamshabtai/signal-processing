import pathlib
import wave

import numpy as np


def arrange_tmp_path_in_kwargs(kwargs, tmp_path):
    kwargs["tested"]["input"]["path"] = tmp_path / pathlib.Path(kwargs["tested"]["input"]["path"]).name
    if "output" in kwargs["tested"] and "dir" in kwargs["tested"]["output"]:
        kwargs["tested"]["output"]["dir"] = tmp_path / kwargs["tested"]["output"]["dir"]


def create_input_file(kwargs):
    channel_shape = kwargs["tested"]["system"]["input_buffer"]["channel_shape"]
    nsamples = kwargs["test"]["nsamples"]
    dtype = np.dtype(kwargs["tested"]["input"]["dtype"])
    path = kwargs["tested"]["input"]["path"]
    data = np.random.normal(loc=0.0, scale=1.0, size=channel_shape + [nsamples]).astype(dtype)

    if path.suffix.lower() == ".wav":
        nchannels = np.prod(channel_shape)
        fs = kwargs["tested"]["input"].get("fs", kwargs["test"].get("sampling_rate", 44100))
        with wave.open(str(path), "wb") as fid:
            fid.setnchannels(nchannels)
            fid.setsampwidth(dtype.itemsize)
            fid.setframerate(fs)
            fid.writeframes(data.ravel(order="F").tobytes())
    else:
        with path.open("wb") as fid:
            fid.write(data.ravel(order="F").tobytes())


def read_input_chunks(kwargs):
    ib = kwargs["tested"]["system"]["input_buffer"]
    step_size = ib["step_size"]
    step_shape = ib["channel_shape"] + [step_size]
    path = kwargs["tested"]["input"]["path"]
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as fid:
            dtype = np.dtype(f"int{fid.getsampwidth() * 8}")
            read_nbytes = int(np.prod(step_shape)) * dtype.itemsize
            while len(chunk := fid.readframes(step_size)) == read_nbytes:
                yield np.frombuffer(chunk, dtype=dtype).reshape(step_shape, order="F")
    else:
        dtype = np.dtype(kwargs["tested"]["input"]["dtype"])
        read_nbytes = int(np.prod(step_shape)) * dtype.itemsize
        with open(path, "rb") as fid:
            while len(chunk := fid.read(read_nbytes)) == read_nbytes:
                yield np.frombuffer(chunk, dtype=dtype).reshape(step_shape, order="F")


def arrange_kwargs(kwargs, tmp_path):
    arrange_tmp_path_in_kwargs(kwargs, tmp_path)
    create_input_file(kwargs)
    return kwargs
