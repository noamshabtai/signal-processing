import wave

import audio_io.files
import numpy as np


def test_read_entire_wav_file(tmp_path):
    wav_path = tmp_path / "test.wav"
    sample_rate = 44100
    nchannels = 2
    sample_width = 2
    dtype = np.int16

    audio_data = np.array([[1000, -1000, 500, -500, 0], [2000, -2000, 1000, -1000, 500]], dtype=dtype)
    interleaved = audio_data.T.flatten().tobytes()

    with wave.open(str(wav_path), "wb") as fid:
        fid.setnchannels(nchannels)
        fid.setsampwidth(sample_width)
        fid.setframerate(sample_rate)
        fid.writeframes(interleaved)

    result = audio_io.files.read_entire_wav_file(str(wav_path))

    assert result.shape == (nchannels, 5)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, audio_data)


def test_read_frame_from_wav_file(tmp_path):
    wav_path = tmp_path / "test_read_frame.wav"

    sample_rate = 44100
    nchannels = 2
    sample_width = 2
    dtype = np.int16

    audio_data = np.array([[1000, -1000, 500, -500, 0, 250], [2000, -2000, 1000, -1000, 500, 750]], dtype=dtype)
    interleaved = audio_data.T.flatten().tobytes()

    with wave.open(str(wav_path), "wb") as fid:
        fid.setnchannels(nchannels)
        fid.setsampwidth(sample_width)
        fid.setframerate(sample_rate)
        fid.writeframes(interleaved)

    nsamples_to_read = 4
    with wave.open(str(wav_path), "rb") as fid:
        result = audio_io.files.read_frame_from_wav_file(fid, nsamples_to_read)

    expected_data = audio_data[:, :nsamples_to_read]

    assert result.shape == (nchannels, nsamples_to_read)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, expected_data)


def test_read_frame_from_wav_file_and_loop(tmp_path):
    wav_path = tmp_path / "test_loop.wav"

    sample_rate = 44100
    nchannels = 2
    sample_width = 2
    dtype = np.int16

    audio_data = np.array([[1000, -1000, 500, -500, 0, 250], [2000, -2000, 1000, -1000, 500, 750]], dtype=dtype)
    interleaved = audio_data.T.flatten().tobytes()

    with wave.open(str(wav_path), "wb") as fid:
        fid.setnchannels(nchannels)
        fid.setsampwidth(sample_width)
        fid.setframerate(sample_rate)
        fid.writeframes(interleaved)

    nsamples_to_read = 4
    with wave.open(str(wav_path), "rb") as fid:
        for i in range(audio_data.shape[1] // nsamples_to_read + 1):
            result = audio_io.files.read_frame_from_wav_file_and_loop(fid, nsamples_to_read, nchannels, dtype)
            expected_data = audio_data[:, :nsamples_to_read]
            assert result.shape == (nchannels, nsamples_to_read), f"loop {i}"
            assert result.dtype == dtype
            np.testing.assert_array_equal(result, expected_data)


def test_set_wav_file_for_writing(tmp_path):
    wav_path = tmp_path / "test.wav"
    sample_rate = 44100
    nchannels = 2
    bit_depth = 16
    expected_sampwidth = bit_depth // 8

    fid = audio_io.files.set_wav_file_for_writing(str(wav_path), sample_rate, nchannels, bit_depth)

    assert fid.getframerate() == sample_rate
    assert fid.getnchannels() == nchannels
    assert fid.getsampwidth() == expected_sampwidth
    fid.close()
    assert wav_path.exists()
