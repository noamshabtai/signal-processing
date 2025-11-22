import argparse
import multiprocessing as mp
import pathlib
import time

import numpy as np
import parametrize_tests.yaml_sweep_parser

mp.set_start_method("spawn", force=True)


def get_parser():
    parser = argparse.ArgumentParser(
        prog="Regression",
        description="Execute tests according to a csv lists of arguments.",
        epilog="Noam Shabtai",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-y", "--yaml-path", dest="yaml_path", type=str, default="config.yaml", help="path of yaml file."
    )
    parser.add_argument(
        "-i",
        "--indices",
        dest="indices",
        nargs="*",
        type=int,
        help="list of case indices, e.g. 0 4 8 9, or blank for all entries.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default="outputs",
        help="directory to place bin and the png files.",
    )
    parser.add_argument(
        "-r",
        "--results",
        dest="results",
        nargs="*",
        type=str,
        help="list of case indices, e.g. 0 4 8 9, or -1 for all entries.",
    )

    return parser


def get_cliargs(parser):
    return parser.parse_known_args()[0]


class Analysis:
    def __init__(self, activator, cliargs):
        self.activator_class = activator
        self.activator_kwargs_list = parametrize_tests.yaml_sweep_parser.parse(cliargs.yaml_path)
        self.total_number_of_cases = len(self.activator_kwargs_list)
        self.case_ndigits = np.int16(np.log10(self.total_number_of_cases - 1)) + 1
        self.nactivations = len(self.activator_kwargs_list)
        self.cases = cliargs.indices if cliargs.indices else np.arange(self.nactivations)
        self.activator_kwargs_list = [self.activator_kwargs_list[ind] for ind in self.cases]
        self.results = {key: [] for key in cliargs.results}
        self.output_dir = pathlib.Path(cliargs.output_dir)

    def extract_results(self, activator, **kwargs):
        pass

    def log_output(self, activation_index):
        ellapsed = time.time() - self.start_time
        eta = ellapsed * (self.nactivations - activation_index) / activation_index
        print(
            f"Activation {activation_index}/{self.nactivations} ({100*activation_index/self.nactivations:.2f}%) | ",
            f"Elapsed: {ellapsed:.2f}s | ETA:",
            f"{eta:.2f}s",
        )

    def single_activation(self, kwargs):
        kwargs["output"]["dir"] = self.output_dir / f"output{kwargs['current_case']:0{self.case_ndigits}}"
        with self.activator_class(**kwargs) as act:
            act.log_rate *= 10
            act.execute()
            self.extract_results(activator=act, **kwargs)
        self.log_output(kwargs["activation_index"] + 1)

    def execute(self):
        self.start_time = time.time()
        self.output_dir.mkdir(exist_ok=True)

        kwargs_list = [
            kwargs | {"activation_index": i, "current_case": j}
            for i, j, kwargs in zip(range(len(self.cases)), self.cases, self.activator_kwargs_list)
        ]
        for kwargs in kwargs_list:
            self.single_activation(kwargs)


if __name__ == "__main__":
    parser = get_parser()
    cliargs = get_cliargs(parser)
    analysis = Analysis(cliargs)
    analysis.execute()
    print(analysis.results)
