import parametrize_tests.yaml_sweep_parser
import yaml


def test_yaml_sweep_parser(config_dir):
    cases = parametrize_tests.yaml_sweep_parser.parse(config_dir / "yaml_sweep_parser.yaml")
    with open(config_dir / "yaml_sweep_parser_expected.yaml") as f:
        expected = yaml.safe_load(f)
    assert cases == expected
