"""Check the genotype operations against their reference values.

Run with `make check-genotype`, or `python tests/test_genotype.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402

import genotype_fingerprint  # noqa: E402
from compare import differences  # noqa: E402

ROOT = paths.ROOT


def check_invariants(genotype_module) -> list[str]:
    """
    Check genotype properties the fixed-seed recording does not pin down.

    :param genotype_module: The genotype module.
    :returns: A list of problems; empty when all hold.
    """
    import numpy as np

    genotype_class = genotype_module.Genotype
    problems: list[str] = []
    rng = np.random.default_rng(7)

    # Random genotypes must start inside [-1, 1].
    for _ in range(200):
        parameters = genotype_class.random(32, rng).parameters
        if parameters.min() < -1.0 or parameters.max() > 1.0:
            problems.append("random() produced a gene outside [-1, 1]")
            break

    # Mutation must respect the per-gene probability. With p=0 nothing changes;
    # with p=1 the chance of any gene landing on its exact previous value is nil.
    base = genotype_class(np.zeros(64))
    if not np.array_equal(base.mutate(rng, 0.15, 0.0).parameters, base.parameters):
        problems.append("mutate() changed genes at probability 0.0")
    if np.count_nonzero(base.mutate(rng, 0.15, 1.0).parameters) != 64:
        problems.append("mutate() left genes untouched at probability 1.0")

    # Over many draws the fraction of changed genes should track the probability.
    changed = sum(
        int(np.count_nonzero(genotype_class(np.zeros(100)).mutate(rng, 0.15, 0.25).parameters))
        for _ in range(40)
    )
    observed = changed / (40 * 100)
    if not 0.20 <= observed <= 0.30:
        problems.append(f"mutate() changed {observed:.1%} of genes at probability 0.25")

    # Mutation must never escape [-1, 1], from either bound.
    for start in (0.99, -0.99):
        mutated = genotype_class(np.full(128, start)).mutate(rng, 2.0, 1.0).parameters
        if mutated.min() < -1.0 or mutated.max() > 1.0:
            problems.append(f"mutate() escaped [-1, 1] starting from {start}")

    # Mutation must not modify the genotype it was called on.
    source = genotype_class(np.zeros(16))
    source.mutate(rng, 0.5, 1.0)
    if np.count_nonzero(source.parameters) != 0:
        problems.append("mutate() modified the original genotype in place")

    # Crossover must only swap tails: each child gene comes from one parent at
    # the same position, and the two children together hold all parent genes.
    for _ in range(50):
        parent1 = genotype_class.random(24, rng)
        parent2 = genotype_class.random(24, rng)
        child1, child2 = genotype_class.one_point_crossover(parent1, parent2, rng)
        for index in range(24):
            pair = {parent1.parameters[index], parent2.parameters[index]}
            if child1.parameters[index] not in pair or child2.parameters[index] not in pair:
                problems.append("one_point_crossover() invented a gene value")
                break
        if sorted(np.concatenate([child1.parameters, child2.parameters])) != sorted(
            np.concatenate([parent1.parameters, parent2.parameters])
        ):
            problems.append("one_point_crossover() did not preserve the parents' genes")
            break

    return problems


def main() -> None:
    """Compare the genotype with the reference and report differences."""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import genotype
    except ImportError as error:
        raise SystemExit(f"could not import `genotype` as a top-level module: {error}")

    reference = genotype_fingerprint.load_reference()
    current = genotype_fingerprint.fingerprint(genotype)

    problems = differences(reference, current)
    problems.extend(check_invariants(genotype))

    if problems:
        print("GENOTYPE CHECK FAILED")
        for problem in problems[:25]:
            print(f"  - {problem}")
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        raise SystemExit(1)

    print("GENOTYPE CHECK PASSED")
    print(f"  random/mutate/crossover identical to the reference at seed {genotype_fingerprint.SEED}")
    print(f"  sizes {list(genotype_fingerprint.SIZES)}, mutation settings {[list(p) for p in genotype_fingerprint.MUTATION_SETTINGS]}")
    print("  invariants: gene bounds, per-gene mutation rate, gene preservation under crossover, deep copy")


if __name__ == "__main__":
    main()
