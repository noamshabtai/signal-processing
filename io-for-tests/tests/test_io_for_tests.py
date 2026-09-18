import io_for_tests.io_for_tests


def test_arrange_tmp_path_in_kwargs(kwargs_io_for_tests, tmp_path):
    kwargs = kwargs_io_for_tests
    io_for_tests.io_for_tests.arrange_tmp_path_in_kwargs(kwargs, tmp_path)

    assert kwargs["tested"]["input"]["path"].parent == tmp_path
    assert kwargs["tested"]["output"]["dir"].parent == tmp_path


def test_create_input_file(kwargs_io_for_tests, tmp_path):
    kwargs = kwargs_io_for_tests
    io_for_tests.io_for_tests.arrange_tmp_path_in_kwargs(kwargs, tmp_path)
    io_for_tests.io_for_tests.create_input_file(kwargs)

    assert kwargs["tested"]["input"]["path"].exists()


def test_read_input_chunks(kwargs_io_for_tests, tmp_path):
    kwargs = kwargs_io_for_tests
    io_for_tests.io_for_tests.arrange_tmp_path_in_kwargs(kwargs, tmp_path)
    io_for_tests.io_for_tests.create_input_file(kwargs)

    ib = kwargs["tested"]["system"]["input_buffer"]
    step_shape = ib["channel_shape"] + [ib["step_size"]]
    chunks = list(io_for_tests.io_for_tests.read_input_chunks(kwargs))

    assert len(chunks) == kwargs["test"]["nsamples"] // ib["step_size"]
    for chunk in chunks:
        assert list(chunk.shape) == step_shape


def test_arrange_kwargs(kwargs_io_for_tests, tmp_path):
    kwargs = kwargs_io_for_tests
    result = io_for_tests.io_for_tests.arrange_kwargs(kwargs, tmp_path)

    assert result is kwargs
    assert kwargs["tested"]["input"]["path"].parent == tmp_path
    assert kwargs["tested"]["input"]["path"].exists()
