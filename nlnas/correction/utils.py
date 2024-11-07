"""Some random utilities"""

Matching = (
    dict[int, set[int]]
    | dict[str, set[int]]
    | dict[int, set[str]]
    | dict[str, set[str]]
)


def to_int_matching(matching: Matching) -> dict[int, set[int]]:
    """
    Sometimes, when a matching is loaded from a JSON file, the keys are
    strings rather than ints. This function converts the keys to ints.
    """
    return {int(k): set(int(x) for x in v) for k, v in matching.items()}
