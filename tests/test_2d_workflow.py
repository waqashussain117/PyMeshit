import numpy as np
import pytest

pytest.importorskip("triangle")
pytest.importorskip("shapely")

try:
    from Pymeshit.two_d_workflow import TwoDInputFeature, TwoDRunConfig, run_two_d_one_click
except ModuleNotFoundError:
    import pathlib
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    for module_name in list(sys.modules):
        if module_name == "Pymeshit" or module_name.startswith("Pymeshit."):
            del sys.modules[module_name]
    from Pymeshit.two_d_workflow import TwoDInputFeature, TwoDRunConfig, run_two_d_one_click


def _feature(name, coords, is_closed):
    return TwoDInputFeature(name=name, coords=np.asarray(coords, dtype=float), is_closed=is_closed)


def test_nested_closed_polygons_create_hole_domain():
    outer = _feature(
        "outer",
        [[0.0, 0.0], [12.0, 0.0], [12.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
        True,
    )
    inner = _feature(
        "inner",
        [[3.0, 3.0], [9.0, 3.0], [9.0, 7.0], [3.0, 7.0], [3.0, 3.0]],
        True,
    )

    result = run_two_d_one_click([outer, inner], TwoDRunConfig())

    assert result.vertices.shape[0] > 0
    assert result.triangles.shape[0] > 0
    assert result.plc_holes.shape[0] >= 1


def test_overlapping_closed_polygons_produce_meshable_regions():
    a = _feature(
        "A",
        [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0], [0.0, 0.0]],
        True,
    )
    b = _feature(
        "B",
        [[4.0, 1.0], [12.0, 1.0], [12.0, 9.0], [4.0, 9.0], [4.0, 1.0]],
        True,
    )

    result = run_two_d_one_click([a, b], TwoDRunConfig())

    assert result.vertices.shape[0] > 0
    assert result.triangles.shape[0] > 0
    assert len(result.regions) >= 2


def test_open_lines_are_kept_as_constraints():
    outer = _feature(
        "outer",
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
        True,
    )
    fault = _feature(
        "fault",
        [[1.0, 5.0], [9.0, 5.5]],
        False,
    )

    result = run_two_d_one_click([outer, fault], TwoDRunConfig())

    assert result.triangles.shape[0] > 0
    assert len(result.constraint_lines) >= 1


def test_invalid_polygon_is_warned_and_mesh_still_runs_when_possible():
    outer = _feature(
        "outer",
        [[0.0, 0.0], [12.0, 0.0], [12.0, 12.0], [0.0, 12.0], [0.0, 0.0]],
        True,
    )
    bow_tie = _feature(
        "bow_tie",
        [[3.0, 3.0], [9.0, 9.0], [9.0, 3.0], [3.0, 9.0], [3.0, 3.0]],
        True,
    )

    result = run_two_d_one_click([outer, bow_tie], TwoDRunConfig())

    assert result.triangles.shape[0] > 0
    assert any("repaired" in w.lower() or "skipped" in w.lower() for w in result.warnings)
