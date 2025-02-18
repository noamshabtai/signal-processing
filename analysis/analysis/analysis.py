import pathlib

import numpy as np
import parse_sweeps.parse_sweeps


class Analysis:
    def __init__(self, activator, yaml_path, **kwargs):
        self.activator_class = activator
        self.activator_kwargs_list = parse_sweeps.parse_sweeps.parse_sweeps(yaml_path)
        self.cases = kwargs["cases"] if "cases" in kwargs else np.arange(len(self.activator_kwargs_list))
        self.activator_kwargs_list = [self.activator_kwargs_list[ind] for ind in self.cases]
        self.results = {key: [] for key in kwargs["results"]}

    def extract_results(self):
        pass

    def execute(self):
        pathlib.Path("outputs").mkdir(exist_ok=True)
        for i, kwargs in zip(self.cases, self.activator_kwargs_list):
            self.activator = self.activator_class(**kwargs)
            self.activator.execute()
            self.activator.close()
            self.extract_results(**kwargs)
            pathlib.Path(f"outputs/output{i}").mkdir(exist_ok=True)
            for output_path in self.activator.output_path:
                output_path.rename(f"outputs/output{i}/{output_path.name}")
            for png_path in self.activator.png_path:
                png_path.rename(f"outputs/output{i}/{png_path.name}")
            self.activator.params_path.rename(f"outputs/output{i}/params.yaml")
