"""Comparing a recorded fingerprint against its reference.

Shared by the checks that prove a module still behaves exactly as its recorded
reference. Keys beginning with `_` are provenance notes about the reference
file itself and are not compared.
"""

from typing import Any


def differences(reference: dict[str, Any], current: dict[str, Any], label: str = "") -> list[str]:
    """
    Compare two fingerprints and describe every value that differs.

    :param reference: The expected values.
    :param current: The values produced by the module under test.
    :param label: Optional prefix naming what is being compared.
    :returns: Human-readable differences, empty when identical.
    """
    prefix = f"{label}: " if label else ""
    problems: list[str] = []

    def walk(expected: Any, actual: Any, path: str) -> None:
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key in sorted(set(expected) | set(actual)):
                where = f"{path}.{key}" if path else str(key)
                if key not in actual:
                    problems.append(f"{prefix}{where} missing")
                elif key not in expected:
                    problems.append(f"{prefix}{where} unexpected")
                else:
                    walk(expected[key], actual[key], where)
        elif expected != actual:
            summary = "" if isinstance(expected, list) else f": expected {expected!r}, got {actual!r}"
            problems.append(f"{prefix}{path} differs{summary}")

    walk(
        {key: value for key, value in reference.items() if not key.startswith("_")},
        {key: value for key, value in current.items() if not key.startswith("_")},
        "",
    )
    return problems
