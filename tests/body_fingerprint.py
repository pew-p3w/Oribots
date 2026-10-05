"""Structural fingerprints of robot bodies.

A fingerprint describes a body's actual module structure: the tree of modules
with their types, attachment indices and orientations, the spatial grid the
body occupies, and the resulting controller size. Two bodies with the same
fingerprint are the same robot.

It guards the body catalogue (`config/bodies.py`) against accidental changes:
`tests/body_reference.json` holds the reference fingerprint of every body, and
`tests/test_bodies.py` checks the current catalogue against it.

Hinge counts alone are not enough: two bodies can have the same number of
hinges and a different layout, which would keep the parameter count intact
while silently changing the robot.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# src/ holds the shared path bootstrap, but is not importable until it is on
# the path itself - hence these two lines before `import paths`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402

ROOT = paths.ROOT
ENGINE = paths.ENGINE
REFERENCE_PATH = Path(__file__).resolve().parent / "body_reference.json"

# Feedback inputs per active hinge in the steering layer of the brain. Only used
# here to report the controller parameter count a body implies.
FEEDBACK_NUM_INPUTS = 4


def _round(value: float) -> float:
    """Round a float so tiny floating-point differences do not change a fingerprint."""
    rounded = round(float(value), 6)
    return 0.0 if rounded == 0.0 else rounded  # normalize -0.0


def _quaternion(orientation: Any) -> list[float]:
    """Represent an orientation as a rounded, sign-normalized xyzw list."""
    values = [_round(v) for v in (orientation.x, orientation.y, orientation.z, orientation.w)]
    # q and -q are the same rotation; pick a canonical sign.
    for value in values:
        if value != 0.0:
            if value < 0.0:
                values = [_round(-v) for v in values]
            break
    return values


def _attachment_geometry(module: Any) -> list[dict[str, Any]]:
    """Describe the attachment points a module offers."""
    return [
        {
            "index": index,
            "orientation": _quaternion(point.orientation),
            "offset": [_round(v) for v in point.offset],
        }
        for index, point in sorted(module.attachment_points.items())
    ]


def _module_node(module: Any, geometry_table: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """
    Describe one module and everything attached below it.

    The same attachment-point geometry repeats across thousands of modules, so
    each distinct geometry is stored once in `geometry_table` and referenced by
    a content-derived key. Variants of one type are kept apart: a core's four
    attachment faces are all `AttachmentFaceCoreV2` but each is rotated
    differently, and collapsing them would hide a real difference.

    :param module: The module to describe.
    :param geometry_table: Accumulated distinct geometries, keyed by type and content.
    :returns: A JSON-serializable description of the module subtree.
    """
    type_name = type(module).__name__
    geometry = _attachment_geometry(module)
    digest = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()[:8]
    key = f"{type_name}:{digest}"
    geometry_table.setdefault(key, geometry)

    node: dict[str, Any] = {
        "type": type_name,
        "attachment_points": key,
        "orientation": _quaternion(module.orientation),
        "children": {
            str(index): _module_node(child, geometry_table)
            for index, child in sorted(module.children.items())
            if child is not None
        },
    }
    mass = getattr(module, "mass", None)
    if mass is not None:
        node["mass"] = _round(mass)
    return node


def _grid(body: Any) -> dict[str, Any]:
    """
    Describe where each module physically sits, as integer grid coordinates.

    This repeats the walk that `Body.to_grid()` performs instead of calling it.
    `_GridMaker` in the vendored engine keeps its coordinate lists in class-level
    mutable defaults and never clears them, so consecutive `to_grid()` calls in
    one process accumulate each other's modules. Walking the tree here keeps the
    fingerprint independent of that bug and of call order.

    :param body: A BodyV2 instance.
    :returns: Grid extent and one entry per occupied cell.
    """
    from pyrr import Quaternion, Vector3

    placements: list[tuple[tuple[int, int, int], str]] = []

    def walk(module: Any, position: Any, orientation: Any) -> None:
        coordinate = tuple(int(round(float(v))) for v in position)
        placements.append((coordinate, type(module).__name__))
        for child_index, attachment_point in sorted(module.attachment_points.items()):
            child = module.children.get(child_index)
            if child is None:
                continue
            rotation = orientation * attachment_point.orientation * child.orientation
            walk(child, position + rotation * Vector3([1.0, 0.0, 0.0]), rotation)

    walk(body.core, Vector3(), Quaternion())

    xs, ys, zs = (sorted({p[0][axis] for p in placements}) for axis in range(3))
    origin = (xs[0], ys[0], zs[0])
    # Sorted, but not deduplicated: several modules can share a grid cell (the
    # core's attachment faces sit on the core), and losing that multiplicity
    # would hide a real structural difference.
    cells = sorted(
        (
            coordinate[0] - origin[0],
            coordinate[1] - origin[1],
            coordinate[2] - origin[2],
            type_name,
        )
        for coordinate, type_name in placements
    )
    return {
        "shape": [xs[-1] - xs[0] + 1, ys[-1] - ys[0] + 1, zs[-1] - zs[0] + 1],
        "core_at": [-origin[0], -origin[1], -origin[2]],
        "cells": [{"at": [x, y, z], "type": name} for x, y, z, name in cells],
        "placements": len(cells),
    }


def fingerprint_body(body: Any, geometry_table: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """
    Build the structural fingerprint of one body.

    :param body: A BodyV2 instance.
    :param geometry_table: Accumulated distinct attachment geometries.
    :returns: A JSON-serializable description of the body's structure.
    """
    from revolve2.modular_robot.body.base import ActiveHinge, Brick
    from revolve2.modular_robot.brain.cpg import (
        active_hinges_to_cpg_network_structure_neighbor,
    )

    active_hinges = body.find_modules_of_type(ActiveHinge)
    cpg_network_structure, output_mapping = active_hinges_to_cpg_network_structure_neighbor(
        active_hinges
    )
    num_cpg_connections = cpg_network_structure.num_connections
    return {
        "counts": {
            "active_hinges": len(active_hinges),
            "bricks": len(body.find_modules_of_type(Brick)),
            "cpg_connections": num_cpg_connections,
            "controller_parameters": num_cpg_connections
            + len(output_mapping) * FEEDBACK_NUM_INPUTS,
        },
        "grid": _grid(body),
        "tree": _module_node(body.core, geometry_table if geometry_table is not None else {}),
    }


def fingerprint_catalogue(catalogue: Any) -> dict[str, Any]:
    """
    Fingerprint every body constructor in a catalogue module.

    :param catalogue: A module exposing `<name>_v2()` body constructors.
    :returns: Mapping of constructor name to fingerprint, plus the catalogue's own lists.
    :raises SystemExit: If the module exposes no body constructors.
    """
    names = sorted(name for name in dir(catalogue) if name.endswith("_v2") and callable(getattr(catalogue, name)))
    if not names:
        raise SystemExit("catalogue module exposes no *_v2 body constructors")

    listing_function = getattr(catalogue, "all_bodies", None) or getattr(catalogue, "all", None)
    geometry_table: dict[str, list[dict[str, Any]]] = {}
    bodies = {name: fingerprint_body(getattr(catalogue, name)(), geometry_table) for name in names}
    return {
        "constructors": names,
        "listed_by_all": len(listing_function()) if listing_function else None,
        "module_geometry": geometry_table,
        "bodies": bodies,
    }


def load_reference() -> dict[str, Any]:
    """
    Load the stored reference fingerprints.

    :returns: The reference fingerprints of the catalogue.
    """
    return json.loads(REFERENCE_PATH.read_text())


def compare(reference: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """
    Compare two catalogue fingerprints.

    :param reference: The expected fingerprints.
    :param current: The fingerprints of the catalogue under test.
    :returns: A list of human-readable differences; empty when identical.
    """
    problems: list[str] = []

    missing = sorted(set(reference["bodies"]) - set(current["bodies"]))
    added = sorted(set(current["bodies"]) - set(reference["bodies"]))
    if missing:
        problems.append(f"missing bodies: {', '.join(missing)}")
    if added:
        problems.append(f"unexpected new bodies: {', '.join(added)}")

    if reference.get("module_geometry") != current.get("module_geometry"):
        expected_types = reference.get("module_geometry") or {}
        actual_types = current.get("module_geometry") or {}
        for type_name in sorted(set(expected_types) | set(actual_types)):
            if expected_types.get(type_name) != actual_types.get(type_name):
                problems.append(f"attachment geometry differs for {type_name}")

    for name in sorted(set(reference["bodies"]) & set(current["bodies"])):
        expected, actual = reference["bodies"][name], current["bodies"][name]
        if expected == actual:
            continue
        for section in ("counts", "grid", "tree"):
            if expected[section] != actual[section]:
                detail = ""
                if section == "counts":
                    detail = f" (expected {expected['counts']}, got {actual['counts']})"
                problems.append(f"{name}: {section} differs{detail}")

    if reference.get("listed_by_all") != current.get("listed_by_all"):
        problems.append(
            f"catalogue listing length changed: expected {reference.get('listed_by_all')}, "
            f"got {current.get('listed_by_all')}"
        )
    return problems
