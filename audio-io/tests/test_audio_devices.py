import audio_io.devices
import numpy as np


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
    audio_io.devices.print_device_indices(mock_instance, "input")
    audio_io.devices.print_device_indices(mock_instance, "output")


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

    index = audio_io.devices.audio_device_index(mock_instance, "input", "Microphone B")
    assert index == 1

    index = audio_io.devices.audio_device_index(mock_instance, "output", "Speaker A")
    assert index == 0

    index = audio_io.devices.audio_device_index(mock_instance, "output", "USB Audio")
    assert index == 2

    index = audio_io.devices.audio_device_index(mock_instance, "input", "Nonexistent Device")
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
    index = audio_io.devices.realtek_output_index(mock_instance)
    assert index == 1

    mock_instance.get_host_api_info_by_index.return_value = {"deviceCount": 2}

    def mock_get_device_info_by_host_api_device_index(api_index, device_index):
        devices = {
            0: {"index": 0, "maxOutputChannels": 2, "maxInputChannels": 0, "name": "USB Audio Interface"},
            1: {"index": 2, "maxOutputChannels": 2, "maxInputChannels": 0, "name": "HDMI Output"},
        }
        return devices.get(device_index, {})

    mock_instance.get_device_info_by_host_api_device_index.side_effect = mock_get_device_info_by_host_api_device_index
    index = audio_io.devices.realtek_output_index(mock_instance)
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
    index = audio_io.devices.vb_cable_input_index(mock_instance)
    assert index == 1

    mock_instance.get_host_api_info_by_index.return_value = {"deviceCount": 2}

    def mock_get_device_info_by_host_api_device_index(api_index, device_index):
        devices = {
            0: {"index": 0, "maxOutputChannels": 2, "maxInputChannels": 2, "name": "Built-in Microphone"},
            1: {"index": 2, "maxOutputChannels": 2, "maxInputChannels": 2, "name": "External USB Mic"},
        }
        return devices.get(device_index, {})

    mock_instance.get_device_info_by_host_api_device_index.side_effect = mock_get_device_info_by_host_api_device_index
    index = audio_io.devices.vb_cable_input_index(mock_instance)
    assert index == -1


def test_find_input_device_index(mocker):
    mock_pyaudio = mocker.patch("pyaudio.PyAudio")
    mock_instance = mock_pyaudio.return_value
    mock_instance.get_device_count.return_value = 2
    mock_instance.get_device_info_by_index.side_effect = [
        {"maxInputChannels": 0},
        {"maxInputChannels": 2, "name": "Test Mic"},
    ]
    assert audio_io.devices.find_input_device_index() == 1


def test_find_output_device_index(mocker):
    mock_pyaudio = mocker.patch("pyaudio.PyAudio")
    mock_instance = mock_pyaudio.return_value
    mock_instance.get_device_count.return_value = 2
    mock_instance.get_device_info_by_index.side_effect = [
        {"maxOutputChannels": 0},
        {"maxOutputChannels": 2, "name": "Test Speaker"},
    ]
    assert audio_io.devices.find_output_device_index() == 1


def test_read_frame_from_pyaudio(mocker):
    mock_stream = mocker.Mock()
    mock_stream.read.return_value = np.array([1, 2, 3, 4], dtype=np.int16).tobytes()
    result = audio_io.devices.read_frame_from_pyaudio(mock_stream, nsamples=2, nchannels=2, dtype=np.int16)
    expected = np.array([[1, 3], [2, 4]], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)


def test_read_frame_from_pyaudio_indata():
    indata = np.array([1, 2, 3, 4], dtype=np.int16).tobytes()
    result = audio_io.devices.read_frame_from_pyaudio_indata(indata, nchannels=2, dtype=np.int16)
    expected = np.array([[1, 3], [2, 4]], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)
