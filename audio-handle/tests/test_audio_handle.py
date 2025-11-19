import wave

import audio_handle.utils
import numpy as np
import pyaudio
import pytest


def test_np_dtype_to_pa_format():
    assert audio_handle.utils.np_dtype_to_pa_format(np.int16) == pyaudio.paInt16
    assert audio_handle.utils.np_dtype_to_pa_format(np.int32) == pyaudio.paInt32
    assert audio_handle.utils.np_dtype_to_pa_format(np.float32) == pyaudio.paFloat32
    assert audio_handle.utils.np_dtype_to_pa_format(np.uint8) == pyaudio.paUInt8
    assert audio_handle.utils.np_dtype_to_pa_format(np.int8) == pyaudio.paInt8

    with pytest.raises(ValueError, match="Unsupported dtype: .*"):
        audio_handle.utils.np_dtype_to_pa_format(np.float64)


def test_print_device_indices(mocker):
    mock_pyaudio = mocker.patch("pyaudio.PyAudio")
    mock_instance = mock_pyaudio.return_value
    mock_instance.get_host_api_info_by_index.return_value = {"deviceCount": 2}

    def mock_get_device_info_by_host_api_device_index(api_index, device_index):
        devices = {
            0: {"index": 0, "maxInputChannels": 2, "maxOutputChannels": 2, "name": "Device A"},
            1: {"index": 1, "maxInputChannels": 2, "maxOutputChannels": 2, "name": "Device B"},
        }
        return devices.get(device_index, {})

    mock_instance.get_device_info_by_host_api_device_index.side_effect = mock_get_device_info_by_host_api_device_index
    audio_handle.utils.print_device_indices(mock_instance, "input")
    audio_handle.utils.print_device_indices(mock_instance, "output")


def test_audio_device_index(mocker):
    mock_pyaudio = mocker.patch("pyaudio.PyAudio")
    mock_instance = mock_pyaudio.return_value
    mock_instance.get_host_api_info_by_index.return_value = {"deviceCount": 3}

    def mock_get_device_info_by_host_api_device_index(api_index, device_index):
        devices = {
            0: {"index": 0, "maxOutputChannels": 2, "maxInputChannels": 0, "name": "Speaker A"},
            1: {"index": 1, "maxOutputChannels": 0, "maxInputChannels": 2, "name": "Microphone B"},
            2: {"index": 2, "maxOutputChannels": 2, "maxInputChannels": 2, "name": "USB Audio Interface"},
        }
        return devices.get(device_index, {})

    mock_instance.get_device_info_by_host_api_device_index.side_effect = mock_get_device_info_by_host_api_device_index

    index = audio_handle.utils.audio_device_index(mock_instance, "input", "Microphone B")
    assert index == 1

    index = audio_handle.utils.audio_device_index(mock_instance, "output", "Speaker A")
    assert index == 0

    index = audio_handle.utils.audio_device_index(mock_instance, "output", "USB Audio")
    assert index == 2

    index = audio_handle.utils.audio_device_index(mock_instance, "input", "Nonexistent Device")
    assert index == -1


def test_realtek_output_index(mocker):
    mock_pyaudio = mocker.patch("pyaudio.PyAudio")
    mock_instance = mock_pyaudio.return_value
    mock_instance.get_host_api_info_by_index.return_value = {"deviceCount": 3}

    def mock_get_device_info_by_host_api_device_index(api_index, device_index):
        devices = {
            0: {"index": 0, "maxOutputChannels": 2, "maxInputChannels": 0, "name": "USB Audio Interface"},
            1: {"index": 1, "maxOutputChannels": 2, "maxInputChannels": 0, "name": "Speakers/Headphones (Realtek(R)"},
            2: {"index": 2, "maxOutputChannels": 2, "maxInputChannels": 0, "name": "HDMI Output"},
        }
        return devices.get(device_index, {})

    mock_instance.get_device_info_by_host_api_device_index.side_effect = mock_get_device_info_by_host_api_device_index
    index = audio_handle.utils.realtek_output_index(mock_instance)
    assert index == 1

    mock_instance.get_host_api_info_by_index.return_value = {"deviceCount": 2}

    def mock_get_device_info_by_host_api_device_index(api_index, device_index):
        devices = {
            0: {"index": 0, "maxOutputChannels": 2, "maxInputChannels": 0, "name": "USB Audio Interface"},
            1: {"index": 2, "maxOutputChannels": 2, "maxInputChannels": 0, "name": "HDMI Output"},
        }
        return devices.get(device_index, {})

    mock_instance.get_device_info_by_host_api_device_index.side_effect = mock_get_device_info_by_host_api_device_index
    index = audio_handle.utils.realtek_output_index(mock_instance)
    assert index == -1


def test_vb_cable_input_index(mocker):
    mock_pyaudio = mocker.patch("pyaudio.PyAudio")
    mock_instance = mock_pyaudio.return_value
    mock_instance.get_host_api_info_by_index.return_value = {"deviceCount": 3}

    def mock_get_device_info_by_host_api_device_index(api_index, device_index):
        devices = {
            0: {"index": 0, "maxOutputChannels": 2, "maxInputChannels": 2, "name": "Built-in Microphone"},
            1: {"index": 1, "maxOutputChannels": 2, "maxInputChannels": 2, "name": "VB-Audio Virtual Cable"},
            2: {"index": 2, "maxOutputChannels": 2, "maxInputChannels": 2, "name": "External USB Mic"},
        }
        return devices.get(device_index, {})

    mock_instance.get_device_info_by_host_api_device_index.side_effect = mock_get_device_info_by_host_api_device_index
    index = audio_handle.utils.vb_cable_input_index(mock_instance)
    assert index == 1

    mock_instance.get_host_api_info_by_index.return_value = {"deviceCount": 2}

    def mock_get_device_info_by_host_api_device_index(api_index, device_index):
        devices = {
            0: {"index": 0, "maxOutputChannels": 2, "maxInputChannels": 2, "name": "Built-in Microphone"},
            1: {"index": 2, "maxOutputChannels": 2, "maxInputChannels": 2, "name": "External USB Mic"},
        }
        return devices.get(device_index, {})

    mock_instance.get_device_info_by_host_api_device_index.side_effect = mock_get_device_info_by_host_api_device_index
    index = audio_handle.utils.vb_cable_input_index(mock_instance)
    assert index == -1


def test_find_input_device_index(mocker):
    mock_pyaudio = mocker.patch("pyaudio.PyAudio")
    mock_instance = mock_pyaudio.return_value
    mock_instance.get_device_count.return_value = 2
    mock_instance.get_device_info_by_index.side_effect = [
        {"maxInputChannels": 0},
        {"maxInputChannels": 2, "name": "Test Mic"},
    ]
    assert audio_handle.utils.find_input_device_index() == 1


def test_find_output_device_index(mocker):
    mock_pyaudio = mocker.patch("pyaudio.PyAudio")
    mock_instance = mock_pyaudio.return_value
    mock_instance.get_device_count.return_value = 2
    mock_instance.get_device_info_by_index.side_effect = [
        {"maxOutputChannels": 0},
        {"maxOutputChannels": 2, "name": "Test Speaker"},
    ]
    assert audio_handle.utils.find_output_device_index() == 1


def test_bytes_to_chunk():
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int16).tobytes()
    result = audio_handle.utils.bytes_to_chunk(data, nchannels=2, dtype=np.int16)
    expected = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)


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

    result = audio_handle.utils.read_entire_wav_file(str(wav_path))

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
        result = audio_handle.utils.read_frame_from_wav_file(fid, nsamples_to_read)

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
            result = audio_handle.utils.read_frame_from_wav_file_and_loop(fid, nsamples_to_read, nchannels, dtype)
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

    fid = audio_handle.utils.set_wav_file_for_writing(str(wav_path), sample_rate, nchannels, bit_depth)

    assert fid.getframerate() == sample_rate
    assert fid.getnchannels() == nchannels
    assert fid.getsampwidth() == expected_sampwidth
    fid.close()
    assert wav_path.exists()


def test_read_frame_from_pyaudio(mocker):
    mock_stream = mocker.Mock()
    mock_stream.read.return_value = np.array([1, 2, 3, 4], dtype=np.int16).tobytes()
    result = audio_handle.utils.read_frame_from_pyaudio(mock_stream, nsamples=2, nchannels=2, dtype=np.int16)
    expected = np.array([[1, 3], [2, 4]], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)


def test_read_frame_from_pyaudio_indata():
    indata = np.array([1, 2, 3, 4], dtype=np.int16).tobytes()
    result = audio_handle.utils.read_frame_from_pyaudio_indata(indata, nchannels=2, dtype=np.int16)
    expected = np.array([[1, 3], [2, 4]], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)


def test_sph2cart_ned():
    r, az, el = 1, 0, 0
    x, y, z = audio_handle.utils.sph2cart_ned(r, az, el)
    expected_x = 1
    expected_y = 0
    expected_z = 0
    assert np.isclose(x, expected_x, atol=1e-3)
    assert np.isclose(y, expected_y, atol=1e-3)
    assert np.isclose(z, expected_z, atol=1e-3)

    r, az, el = 1, np.pi / 2, 0
    x, y, z = audio_handle.utils.sph2cart_ned(r, az, el)
    expected_x = 0
    expected_y = 1
    expected_z = 0
    assert np.isclose(x, expected_x, atol=1e-3)
    assert np.isclose(y, expected_y, atol=1e-3)
    assert np.isclose(z, expected_z, atol=1e-3)

    r, az, el = 1, np.pi / 2, np.pi / 2
    x, y, z = audio_handle.utils.sph2cart_ned(r, az, el)
    expected_x = 0
    expected_y = 0
    expected_z = -1
    assert np.isclose(x, expected_x, atol=1e-3)
    assert np.isclose(y, expected_y, atol=1e-3)
    assert np.isclose(z, expected_z, atol=1e-3)


def test_cart2sph_ned():
    r, az, el = 0.1, 0.2, 0.3
    assert np.all(
        np.isclose((r, az, el), audio_handle.utils.sph2cart_ned(*audio_handle.utils.cart2sph_ned(r, az, el)), atol=1e-3)
    )


def test_sph2cart_enu():
    r, az, inc = 1, 0, np.pi / 2
    x, y, z = audio_handle.utils.sph2cart_enu(r, az, inc)
    expected_x = 1
    expected_y = 0
    expected_z = 0
    assert np.isclose(x, expected_x, atol=1e-3)
    assert np.isclose(y, expected_y, atol=1e-3)
    assert np.isclose(z, expected_z, atol=1e-3)

    r, az, inc = 1, np.pi / 2, np.pi / 2
    x, y, z = audio_handle.utils.sph2cart_enu(r, az, inc)
    expected_x = 0
    expected_y = 1
    expected_z = 0
    assert np.isclose(x, expected_x, atol=1e-3)
    assert np.isclose(y, expected_y, atol=1e-3)
    assert np.isclose(z, expected_z, atol=1e-3)

    r, az, inc = 1, 0, 0
    x, y, z = audio_handle.utils.sph2cart_enu(r, az, inc)
    expected_x = 0
    expected_y = 0
    expected_z = 1
    assert np.isclose(x, expected_x, atol=1e-3)
    assert np.isclose(y, expected_y, atol=1e-3)
    assert np.isclose(z, expected_z, atol=1e-3)


def test_cart2sph_enu():
    r, az, inc = 0.1, 0.2, 0.3
    assert np.all(
        np.isclose(
            (r, az, inc), audio_handle.utils.sph2cart_ned(*audio_handle.utils.cart2sph_ned(r, az, inc)), atol=1e-3
        )
    )


def test_distance_to():
    v = np.array([[3, 4, 0], [6, 8, 0]])
    result = audio_handle.utils.distance_to(v)
    expected = np.array([5, 10])
    np.testing.assert_array_equal(result, expected)


def test_freq_index():
    assert audio_handle.utils.freq_index(1000, 1024, 44100) == 23


def test_lin2db():
    assert np.isclose(audio_handle.utils.lin2db(1), 0)
    assert np.isclose(audio_handle.utils.lin2db(10), 20)


def test_db2lin():
    assert np.isclose(audio_handle.utils.db2lin(0), 1)
    assert np.isclose(audio_handle.utils.db2lin(20), 10)
