def test_parse_sweeps_config(check_parse):
    check_parse("config.yaml", 6)


def test_parse_sweeps_activator_config(check_parse):
    check_parse("activator_config.yaml", 7)
