import copy
import itertools

import deepmerge
import yaml


def _expand_sweep(node):
    if isinstance(node, dict):
        if "sweep" in node:
            static_dictionary = {key: value for key, value in node.items() if key != "sweep"}
            dictionary_permutations = _collect_dictionary_permutations(node["sweep"])
            return [{**static_dictionary, **dictionary} for dictionary in dictionary_permutations]

        expanded_children = {key: _expand_sweep(value) for key, value in node.items()}
        return _collect_dictionary_permutations(expanded_children)
    else:
        return [node]


def _collect_dictionary_permutations(node):
    keys, values = zip(*node.items())
    return [dict(zip(keys, permutation)) for permutation in itertools.product(*values)] if node else ([], [])


def _has_sweep(node):
    if isinstance(node, dict):
        if "sweep" in node:
            return True
        return any(_has_sweep(value) for value in node.values())
    return False


def parse(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    cases = copy.deepcopy(config["cases"])
    while any(_has_sweep(case) for case in cases):
        built_cases = []
        for case in cases:
            built_cases.extend(_expand_sweep(case))
        cases = built_cases

    if "base" in config:
        merger = deepmerge.Merger([(dict, ["merge"])], ["override"], ["override"])
        cases = [merger.merge(copy.deepcopy(config["base"]), case) for case in cases]

    return cases
