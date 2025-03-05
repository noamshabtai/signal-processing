import copy
import itertools

import yaml


def parse_sweeps(yaml_file):
    def expand_node(node, path="root"):
        if isinstance(node, dict):
            if "sweeps" in node:
                static = {k: v for k, v in node.items() if k != "sweeps"}
                sweeps = node["sweeps"]
                combinations = generate_combinations(sweeps)
                return [{**static, **combo} for combo in combinations]

            expanded_children = {key: expand_node(value, path=f"{path}.{key}") for key, value in node.items()}
            keys, values = zip(*expanded_children.items()) if expanded_children else ([], [])
            return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

        elif isinstance(node, list):
            return [node]

        return [node]

    def generate_combinations(sweeps):
        keys, values = zip(*sweeps.items())
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        return combinations

    def has_sweeps(node):
        if isinstance(node, dict):
            if "sweeps" in node:
                return True
            return any(has_sweeps(value) for value in node.values())
        if isinstance(node, list):
            return any(has_sweeps(item) for item in node)
        return False

    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    expanded_cases = copy.deepcopy(config["cases"])
    iteration = 0

    while any(has_sweeps(case) for case in expanded_cases):
        iteration += 1
        new_cases = []
        for i, case in enumerate(expanded_cases):
            expanded = expand_node(case, path=f"case[{i}]")
            new_cases.extend(expanded)
        expanded_cases = new_cases

        if iteration > 100:  # Failsafe for infinite loops
            raise RuntimeError("Expansion stuck in an infinite loop.")

    if "base" in config:
        import deepmerge

        merger = deepmerge.Merger([(dict, ["merge"])], ["override"], ["override"])
        expanded_cases = [merger.merge(case, config["base"]) for case in expanded_cases]

    return expanded_cases


# Example usage
if __name__ == "__main__":
    yaml_path = "tests_parse_sweeps/config.yaml"  # Replace with your file path
    cases = parse_sweeps(yaml_path)

    # Print all cases and count
    print(f"Total Cases: {len(cases)}")
    for i, case in enumerate(cases):
        print(f"Case {i + 1}: {case}")

    yaml_path = "tests_parse_sweeps/siso_config.yaml"  # Replace with your file path
    cases = parse_sweeps(yaml_path)

    # Print all cases and count
    print(f"Total Cases: {len(cases)}")
    for i, case in enumerate(cases):
        print(f"Case {i + 0}: {case}")
