import wave

import data_types.conversions

import audio_io.conversions


def read_entire_wav_file(path):
    with wave.open(path, "rb") as fid:
        return audio_io.conversions.bytes_to_chunk(
            data_bytes=fid.readframes(fid.getnframes()),
            nchannels=fid.getnchannels(),
            dtype=data_types.conversions.get_int_type_from_nbytes(fid.getsampwidth()),
        )


def read_frame_from_wav_file(fid, nsamples):
    return audio_io.conversions.bytes_to_chunk(
        data_bytes=fid.readframes(nsamples),
        nchannels=fid.getnchannels(),
        dtype=data_types.conversions.get_int_type_from_nbytes(fid.getsampwidth()),
    )


def read_frame_from_wav_file_and_loop(fid, nsamples, nchannels, dtype):
    nbytes = nsamples * nchannels * dtype().itemsize
    input_bytes = fid.readframes(nsamples)
    if len(input_bytes) != nbytes:
        fid.rewind()
        input_bytes = fid.readframes(nsamples)
    return audio_io.conversions.bytes_to_chunk(input_bytes, nchannels=nchannels, dtype=dtype)


def set_wav_file_for_writing(path, fs, nchannels, nbits):
    fid = wave.open(path, "wb")
    fid.setframerate(fs)
    fid.setnchannels(nchannels)
    fid.setsampwidth(int(nbits / 8))
    fid.setcomptype("NONE", compname="not compressed")
    return fid
