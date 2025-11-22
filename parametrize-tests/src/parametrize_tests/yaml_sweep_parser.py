import copy
import itertools

import yaml


def parse(yaml_file):
    def expand_sweep(node):
        if isinstance(node, dict):
            if "sweep" in node:
                static_dictionary = {key: value for key, value in node.items() if key != "sweep"}
                dictionary_permutations = collect_dictionary_permutations(node["sweep"])
                dictionary_combinations = [
                    {**static_dictionary, **dictionary} for dictionary in dictionary_permutations
                ]
                return dictionary_combinations

            expanded_children = {key: expand_sweep(value) for key, value in node.items()}
            return collect_dictionary_permutations(expanded_children)
        else:
            return [node]

    def collect_dictionary_permutations(node):
        keys, values = zip(*node.items())
        return (
            [dict(zip(keys, permutation_of_values)) for permutation_of_values in itertools.product(*values)]
            if node
            else ([], [])
        )

    def has_sweep(node):
        if isinstance(node, dict):
            if "sweep" in node:
                return True
            return any(has_sweep(value) for value in node.values())
        return False

    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    cases = copy.deepcopy(config["cases"])
    while any(has_sweep(case) for case in cases):
        built_cases = []
        for case in cases:
            built_cases.extend(expand_sweep(case))
        cases = built_cases

    if "base" in config:
        import deepmerge

        merger = deepmerge.Merger([(dict, ["merge"])], ["override"], ["override"])
        cases = [merger.merge(copy.deepcopy(config["base"]), case) for case in cases]

    return cases
