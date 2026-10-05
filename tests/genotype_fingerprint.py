"""Reference-value fingerprints of the genotype's random, mutate and crossover.

Every one of these operations consumes a seeded random generator, so with a
fixed seed the results are exact numbers rather than distributions. This module
records them for a genotype module, so two implementations can be compared
value by value.

`tests/genotype_reference.json` holds the reference values;
`tests/test_genotype.py` checks the current module against them.
"""

import json
from pathlib import Path
from typing import Any

REFERENCE_PATH = Path(__file__).resolve().parent / "genotype_reference.json"

SEED = 20260920
SIZES = (4, 22, 33, 44)
# (mutate_std, mutation_probability) pairs: the baseline configs' values, the
# heavier test-variant values, and the degenerate all/none cases.
MUTATION_SETTINGS = ((0.15, 0.01), (0.15, 0.25), (0.5, 1.0), (0.1, 0.0))
DECIMALS = 12


def _round_all(values: Any) -> Any:
    """Round a nested sequence of numbers so float noise cannot change a fingerprint."""
    if hasattr(values, "__len__"):
        return [_round_all(v) for v in values]
    rounded = round(float(values), DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def fingerprint(genotype_module: Any) -> dict[str, Any]:
    """
    Record what the genotype operations produce under a fixed seed.

    :param genotype_module: The module exposing `Genotype`.
    :returns: A JSON-serializable fingerprint.
    """
    import numpy as np

    genotype_class = genotype_module.Genotype

    random_genotypes = {}
    for size in SIZES:
        rng = np.random.default_rng(SEED)
        random_genotypes[str(size)] = _round_all(genotype_class.random(size, rng).parameters)

    mutations = {}
    for size in SIZES:
        for mutate_std, probability in MUTATION_SETTINGS:
            rng = np.random.default_rng(SEED)
            original = genotype_class.random(size, rng)
            mutated = original.mutate(rng, mutate_std, probability)
            key = f"size={size},std={mutate_std},p={probability}"
            mutations[key] = {
                "before": _round_all(original.parameters),
                "after": _round_all(mutated.parameters),
                "genes_changed": int(np.count_nonzero(original.parameters != mutated.parameters)),
                "original_unchanged": bool(
                    np.array_equal(original.parameters, np.asarray(random_genotypes[str(size)]))
                ),
            }

    # Mutation must clip back into [-1, 1]: start every gene near the upper
    # bound and mutate hard, so unclipped noise would escape the range.
    rng = np.random.default_rng(SEED)
    near_bound = genotype_class(np.full(SIZES[-1], 0.95))
    clipped = near_bound.mutate(rng, 0.5, 1.0)
    clipping = {
        "after": _round_all(clipped.parameters),
        "max": _round_all(np.max(clipped.parameters)),
        "min": _round_all(np.min(clipped.parameters)),
        "within_bounds": bool(np.all(clipped.parameters >= -1.0) and np.all(clipped.parameters <= 1.0)),
    }

    crossovers = {}
    for size in SIZES:
        rng = np.random.default_rng(SEED)
        parent1 = genotype_class.random(size, rng)
        parent2 = genotype_class.random(size, rng)
        child1, child2 = genotype_class.one_point_crossover(parent1, parent2, rng)
        combined_parents = sorted(
            _round_all(np.concatenate([parent1.parameters, parent2.parameters]))
        )
        combined_children = sorted(
            _round_all(np.concatenate([child1.parameters, child2.parameters]))
        )
        crossovers[str(size)] = {
            "parent1": _round_all(parent1.parameters),
            "parent2": _round_all(parent2.parameters),
            "child1": _round_all(child1.parameters),
            "child2": _round_all(child2.parameters),
            # One-point crossover only swaps tails, so the two children together
            # must hold exactly the same genes as the two parents together.
            "genes_preserved": combined_parents == combined_children,
        }

    errors = {}
    rng = np.random.default_rng(SEED)
    base = genotype_class.random(8, rng)
    for probability in (-0.1, 1.1):
        try:
            base.mutate(rng, 0.15, probability)
        except ValueError:
            errors[f"mutate_probability={probability}"] = "ValueError"
        else:
            errors[f"mutate_probability={probability}"] = "ACCEPTED"
    try:
        genotype_class.one_point_crossover(base, genotype_class.random(9, rng), rng)
    except ValueError:
        errors["crossover_length_mismatch"] = "ValueError"
    else:
        errors["crossover_length_mismatch"] = "ACCEPTED"

    # A single-gene genotype has no valid crossover point; the parents are
    # returned unchanged rather than an error being raised.
    single1, single2 = genotype_class(np.array([0.5])), genotype_class(np.array([-0.5]))
    short1, short2 = genotype_class.one_point_crossover(single1, single2, rng)
    errors["crossover_single_gene"] = _round_all([short1.parameters, short2.parameters])

    # copy() must be deep: mutating the copy must not touch the original.
    source = genotype_class(np.zeros(8))
    duplicate = source.copy()
    duplicate.parameters[0] = 1.0
    copy_behaviour = {
        "original_after_editing_copy": _round_all(source.parameters),
        "is_deep_copy": bool(source.parameters[0] == 0.0),
    }

    return {
        "settings": {
            "seed": SEED,
            "sizes": list(SIZES),
            "mutation_settings": [list(pair) for pair in MUTATION_SETTINGS],
            "decimals": DECIMALS,
        },
        "random": random_genotypes,
        "mutate": mutations,
        "mutate_clipping": clipping,
        "crossover": crossovers,
        "edge_cases": errors,
        "copy": copy_behaviour,
    }


def load_reference() -> dict[str, Any]:
    """
    Load the stored reference fingerprint.

    :returns: The reference values of the genotype module.
    """
    return json.loads(REFERENCE_PATH.read_text())
