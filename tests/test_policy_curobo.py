# SPDX-License-Identifier: Apache-2.0
"""cuRobo world loading -- the TSDF's frame, a TSDF with every mesh filtered, sphere paths --
a move too short to have a direction planned free, and a plan's rows cut from cuRobo's
fixed buffer and resampled to the stream rate.

cuRobo is not a test dependency, so the few names these paths import are stubbed."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robosimstudio_policy_agent.planners.curobo import MotionBackendCuRobo
from tests.fixtures.policy import pose

TIER = "fast"
"""Backends built bare with stubbed cuRobo modules: no GPU, no cuRobo."""


class _Feature(np.ndarray):
    def copy_(self, other) -> None:
        self[...] = other

    def clone(self):
        return self.copy()


class _SceneCfg:
    def __init__(self, **kw) -> None:
        self.kw = kw


class _Planner:
    def __init__(self) -> None:
        self.loaded = None

    def clear_scene_cache(self) -> None:
        self.loaded = None

    def update_world(self, cfg) -> None:
        self.loaded = cfg.kw


def _yaml(path):
    """cuRobo's ``load_yaml``, for the JSON these tests write: JSON is YAML, and needs no ``pyyaml``."""
    with open(path) as f:
        return json.load(f)


def _curobo_stubs(configs_dir: str = "/nonexistent") -> dict:
    modules = {
        name: types.ModuleType(name)
        for name in (
            "curobo",
            "curobo._src",
            "curobo._src.geom",
            "curobo._src.geom.types",
            "curobo.config_io",
            "curobo.content",
        )
    }
    modules["curobo._src.geom.types"].SceneCfg = _SceneCfg
    modules["curobo._src.geom.types"].Mesh = lambda **kw: kw
    modules["curobo.config_io"].load_yaml = _yaml
    modules["curobo.config_io"].join_path = lambda a, b: str(Path(a) / b)
    modules["curobo.content"].get_robot_configs_path = lambda: configs_dir
    return modules


def _backend_with_grid(build_arm: int) -> MotionBackendCuRobo:
    backend = MotionBackendCuRobo.__new__(MotionBackendCuRobo)
    eye = np.eye(3)
    bases = {0: (eye, np.array([0.0, -0.3, 0.0])), 1: (eye, np.array([0.0, 0.3, 0.0]))}
    backend._base_cache = {-1: bases[0], **bases}
    n = 5
    feature = np.ones((n, n, n)).view(_Feature)
    backend._tsdf_grid = types.SimpleNamespace(feature_tensor=feature)
    backend._tsdf_pristine = feature.clone()
    backend._tsdf_axes = [(np.arange(n) - (n - 1) / 2) * 0.01 for _ in range(3)]
    backend._tsdf_base = bases[build_arm]
    backend._robot_spheres_world = lambda: np.zeros((0, 4))
    return backend


class TsdfWorldTest(unittest.TestCase):
    def test_tsdf_loads_when_every_mesh_is_filtered(self) -> None:
        backend = _backend_with_grid(0)
        planner = _Planner()
        world = {"use_tsdf": True, "meshes": [{"name": "empty", "vertices": [], "faces": []}]}
        with mock.patch.dict(sys.modules, _curobo_stubs()):
            backend._update_world(planner, world, 0)
        self.assertIsNotNone(planner.loaded, "the cleared world was never reloaded")
        self.assertIs(planner.loaded["voxel"][0], backend._tsdf_grid)

    def test_another_arms_grid_is_refused(self) -> None:
        backend = _backend_with_grid(0)
        with mock.patch.dict(sys.modules, _curobo_stubs()):
            backend._update_world(_Planner(), {"use_tsdf": True}, 0)
            with self.assertRaisesRegex(ValueError, "arm_id=1"):
                backend._update_world(_Planner(), {"use_tsdf": True}, 1)

    def test_carving_uses_the_grids_own_base(self) -> None:
        # Built for arm 1 (base y=+0.3); a contact sphere at world y=+0.3 is the
        # grid's centre voxel in that frame, and arm 0's frame would miss it.
        backend = _backend_with_grid(1)
        backend._tsdf_pristine[2, 2, 2] = 0.0
        backend._tsdf_grid.feature_tensor[...] = backend._tsdf_pristine
        backend._carve_tsdf_contact_spheres(backend._tsdf_grid, np.array([[0.0, 0.3, 0.0, 0.001]]))
        self.assertEqual(float(backend._tsdf_grid.feature_tensor[2, 2, 2]), 1.0)


class CollisionSpheresTest(unittest.TestCase):
    def _backend(self, config_path) -> MotionBackendCuRobo:
        backend = MotionBackendCuRobo.__new__(MotionBackendCuRobo)
        backend.config_path = config_path
        return backend

    def test_none_becomes_a_mapping(self) -> None:
        kinematics = {"collision_spheres": None}
        spheres = self._backend("franka.yml")._inline_collision_spheres(kinematics)
        self.assertEqual(spheres, {})
        self.assertIs(kinematics["collision_spheres"], spheres)

    def test_a_path_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "spheres").mkdir()
            (Path(tmp) / "spheres" / "arm.yml").write_text(
                json.dumps({"collision_spheres": {"link0": [{"center": [0, 0, 0], "radius": 0.05}]}})
            )
            config = Path(tmp) / "robot.yml"
            config.write_text(json.dumps({"robot_cfg": {}}))
            kinematics = {"collision_spheres": "spheres/arm.yml"}
            with mock.patch.dict(sys.modules, _curobo_stubs()):
                spheres = self._backend(str(config))._inline_collision_spheres(kinematics)
        self.assertEqual(list(spheres), ["link0"])
        self.assertIs(kinematics["collision_spheres"], spheres)

    def test_a_path_resolves_in_curobos_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "spheres").mkdir()
            (Path(tmp) / "spheres" / "franka.yml").write_text(json.dumps({"collision_spheres": {"hand": []}}))
            kinematics = {"collision_spheres": "spheres/franka.yml"}
            with mock.patch.dict(sys.modules, _curobo_stubs(tmp)):
                spheres = self._backend("franka.yml")._inline_collision_spheres(kinematics)
        self.assertEqual(spheres, {"hand": []})


class TestCuroboShortMove(unittest.TestCase):
    def test_a_rotation_in_place_is_planned_free(self):
        planner = object.__new__(MotionBackendCuRobo)
        calls = []
        planner._import = lambda: (None, None, None, None, None, None)
        planner._orientation_held = lambda *a, **k: False

        def plan(position, quat, arm_id, seed, criteria, num_waypoints, **kwargs):
            calls.append(criteria)
            return {"waypoints": ["free"]}

        planner._plan = plan
        self.assertEqual(planner.plan_linear(pose(0.0), pose(0.0)), {"waypoints": ["free"]})
        self.assertEqual(calls, [None])


class _Tensor:
    """Just enough of a torch tensor for ``.detach().cpu().numpy()``."""

    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float64)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def _plan(moving: int, *, buffer: int = 5000, dt: float = 0.025, last=True):
    """cuRobo's answer for a plan of ``moving`` rows from 0 to 1 on two joints, padded to ``buffer`` with the goal.

    Returns ``(planner, result)`` as ``_rows`` reads them: the buffer through the
    planner's kinematics, ``interpolated_last_tstep`` and ``dt`` as the ``(1, 1)``
    tensors cuRobo hands back.
    """
    path = np.linspace(0.0, 1.0, moving)
    rows = np.ones((buffer, 2))
    rows[:moving] = np.stack([path, 2.0 * path], axis=1)
    planner = types.SimpleNamespace(
        kinematics=types.SimpleNamespace(
            get_active_js=lambda trajectory: types.SimpleNamespace(position=_Tensor(trajectory.rows))
        )
    )
    result = types.SimpleNamespace(
        interpolated_trajectory=types.SimpleNamespace(rows=rows, dt=np.array([[dt]])),
        interpolated_last_tstep=np.array([[moving - 1]]) if last else None,
    )
    return planner, result


class TestAPlansRowsAreItsOwnLength(unittest.TestCase):
    """``_rows``: cuRobo's fixed buffer cut at the plan's arrival, then resampled to the stream rate.

    A static method over the result, so nothing here imports cuRobo or torch.
    """

    def test_the_buffer_is_cut_at_the_last_step_inclusive(self):
        rows = MotionBackendCuRobo._rows(*_plan(42), num_waypoints=0)
        self.assertEqual(rows.shape, (42, 2))
        np.testing.assert_allclose(rows[-1], [1.0, 2.0])

    def test_a_result_that_does_not_say_where_it_arrived_keeps_the_buffer(self):
        self.assertEqual(len(MotionBackendCuRobo._rows(*_plan(42, buffer=300, last=False), num_waypoints=0)), 300)

    def test_streamed_at_the_cells_rate_it_lasts_as_long_as_the_plan(self):
        # 42 rows of 25 ms is 1.05 s, which at 60 Hz is 63 control steps.
        rows = MotionBackendCuRobo._rows(*_plan(42), num_waypoints=0, stream_hz=60.0)
        self.assertEqual(len(rows), 63)
        np.testing.assert_allclose(rows[[0, -1]], [[0.0, 0.0], [1.0, 2.0]])

    def test_an_explicit_count_wins_over_the_rate(self):
        self.assertEqual(len(MotionBackendCuRobo._rows(*_plan(42), num_waypoints=10, stream_hz=60.0)), 10)

    def test_lengthening_interpolates_along_the_path_rather_than_repeating_rows(self):
        rows = MotionBackendCuRobo._rows(*_plan(5), num_waypoints=17)
        self.assertTrue(np.all(np.diff(rows[:, 0]) > 0), "a repeated row is a stationary control step")
        np.testing.assert_allclose(rows[:, 1], 2.0 * rows[:, 0])


if __name__ == "__main__":
    unittest.main()
