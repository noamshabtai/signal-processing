import conftest
import parametrize_tests.yaml_sweep_parser
import yaml


def test_yaml_sweep_parser():
    cases = parametrize_tests.yaml_sweep_parser.parse(conftest.config_dir / "yaml_sweep_parser.yaml")
    with open(conftest.config_dir / "yaml_sweep_parser_expected.yaml") as f:
        expected = yaml.safe_load(f)
    assert cases == expected
