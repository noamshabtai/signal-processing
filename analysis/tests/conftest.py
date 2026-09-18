import pathlib
import sys

import parametrize_tests.kwargs
import pytest

import analysis.analysis


class MockedActivator:
    def __init__(self, **kwargs):
        self.output_dir = pathlib.Path(kwargs["output"]["dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.completed = False

    def execute(self):
        self.completed = True

    def cleanup(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if not self.completed:
            self.cleanup()


class AnalysisWithMockedActivator(analysis.analysis.Analysis):
    def __init__(self, **kwargs):
        super().__init__(activator=MockedActivator, **kwargs)

    def extract_results(self, activator, activator_kwargs):
        for key in self.results:
            self.results[key].append(activator_kwargs["simulation"][key])


@pytest.fixture
def Analysis():
    return AnalysisWithMockedActivator


@pytest.fixture(scope="session")
def project_dir():
    return pathlib.Path(__file__).parent.parent


tests_dir = pathlib.Path(__file__).parent
config_dir = tests_dir / "config"
module = sys.modules[__name__]
for fixture in [
    "analysis",
]:
    parametrize_tests.kwargs.setattr_kwargs(fixture, config_dir, module)
