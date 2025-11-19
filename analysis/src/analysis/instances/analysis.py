import activator.instances.activator
import numpy as np

import analysis.analysis


class Analysis(analysis.analysis.Analysis):
    def __init__(self, cliargs):
        cliargs.results = ["step_size", "output_mean", "nsamples"]
        super().__init__(activator=activator.instances.activator.Activator, cliargs=cliargs)

    def extract_results(self, activator, **kwargs):
        self.results["step_size"].append(activator.system.input_buffer.step_size)
        with open(activator.output_path[-1], "rb") as fid:
            output_mean = np.mean(np.fromfile(fid, dtype=np.float64))
        self.results["output_mean"].append(output_mean)

        self.results["nsamples"].append(kwargs["simulation"]["nsamples"])
