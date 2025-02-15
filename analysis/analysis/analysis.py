import pathlib

import parse_sweeps.parse_sweeps


class Analysis:
    def __init__(self, activator, yaml_path, **kwargs):
        self.root_dir = kwargs["root_dir"] if "root_dir" in kwargs else "."
        self.activator_class = activator
        self.activator_kwargs_list = parse_sweeps.parse_sweeps.parse_sweeps(yaml_path)
        self.results = {key: [] for key in kwargs["results"]}

    def extract_results(self):
        pass

    def execute(self):
        pathlib.Path("outputs").mkdir(exist_ok=True)
        pathlib.Path("pngs").mkdir(exist_ok=True)
        for i, kwargs in enumerate(self.activator_kwargs_list):

            self.activator = self.activator_class(**(kwargs | {"root_dir": self.root_dir}))
            self.activator.execute()
            self.activator.close()
            self.extract_results(**kwargs)
            pathlib.Path(f"outputs/output{i}").mkdir(exist_ok=True)
            for output_path in self.activator.output_path:
                output_path.rename(f"outputs/output{i}/{output_path.name}")
            pathlib.Path(f"pngs/png{i}").mkdir(exist_ok=True)
            for png_path in self.activator.png_path:
                png_path.rename(f"pngs/png{i}/{png_path.name}")
