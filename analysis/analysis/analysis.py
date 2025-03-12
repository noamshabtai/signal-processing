import pathlib
import time

import numpy as np
import parse_sweeps.parse_sweeps
import wraplogging.wraplogging


class Analysis:
    def __init__(self, activator, yaml_path, **kwargs):
        self.logger = wraplogging.wraplogging.create_logger(__name__)

        self.activator_class = activator
        self.activator_kwargs_list = parse_sweeps.parse_sweeps.parse_sweeps(yaml_path)
        self.cases = kwargs["cases"] if "cases" in kwargs else np.arange(len(self.activator_kwargs_list))
        self.activator_kwargs_list = [self.activator_kwargs_list[ind] for ind in self.cases]
        self.activation = 0
        self.nactivations = len(self.activator_kwargs_list)
        self.results = {key: [] for key in kwargs["results"]}

    def extract_results(self, activator, **kwargs):
        pass

    def log_output(self):
        self.activation += 1
        ellapsed = time.time() - self.start_time
        eta = ellapsed * (self.nactivations - self.activation) / self.activation
        self.logger.info(
            f"Step {self.activation}/{self.nactivations} ({100*self.activation/self.nactivations:.2f}%) | "
            f"Elapsed: {ellapsed:.2f}s | ETA:"
            f"{eta:.2f}s"
        )

    def execute(self):
        self.start_time = time.time()
        pathlib.Path("outputs").mkdir(exist_ok=True)
        for i, kwargs in zip(self.cases, self.activator_kwargs_list):
            with self.activator_class(**kwargs) as act:
                act.log_rate *= 10
                act.execute()
                self.extract_results(activator=act, **kwargs)
                pathlib.Path(f"outputs/output{i}").mkdir(exist_ok=True)
                for output_path in act.output_path:
                    output_path.rename(f"outputs/output{i}/{output_path.name}")
                for png_path in act.png_path:
                    png_path.rename(f"outputs/output{i}/{png_path.name}")
                act.params_path.rename(f"outputs/output{i}/params.yaml")
            self.log_output()
