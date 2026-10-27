# SPDX-License-Identifier: Apache-2.0
"""The shared fixtures behave as the objects they stand in for, and clean up after a failing test."""

from __future__ import annotations

import os
import sys
import threading
import time
import types
import unittest
from unittest import mock

import numpy as np

from robosimstudio.registry import Registry
from tests.fixtures.arrays import host
from tests.fixtures.registries import registered, restored
from tests.fixtures.sockets import chdir, short_dir
from tests.fixtures.waits import fake_clock, wait_until

TIER = "fast"
"""No GPU: numpy, a throwaway registry, a thread and the temp directory."""


class TestHostArrays(unittest.TestCase):
    def test_a_cpu_array_hands_out_its_own_memory(self) -> None:
        array = host([1.0, 2.0])
        array.numpy()[0] = 9.0
        self.assertEqual(array.numpy()[0], 9.0, "a write through numpy() is seen, as on Warp's CPU device")

    def test_a_device_array_hands_out_copies(self) -> None:
        array = host([1.0, 2.0], copy=True)
        array.numpy()[0] = 9.0
        self.assertEqual(array.numpy()[0], 1.0, "only assign() writes, as on a CUDA device")
        array.assign([3.0, 4.0])
        np.testing.assert_array_equal(array.numpy(), [3.0, 4.0])

    def test_an_index_array_stays_integer_and_a_mask_stays_bool(self) -> None:
        self.assertEqual(host(np.array([0, 3, 5], dtype=np.int32)).dtype, np.int32)
        self.assertEqual(host(np.array([True, False])).dtype, np.bool_)
        self.assertEqual(host([1, 2]).dtype, np.float64, "a plain list is positions, as most stand-ins held")

    def test_an_array_handed_in_is_the_one_read_back(self) -> None:
        q = np.zeros(3)
        array = host(q)
        q[1] = 4.0
        self.assertEqual(array.numpy()[1], 4.0, "as SimpleNamespace(numpy=lambda: q) did")

    def test_the_device_says_which_rule_it_follows(self) -> None:
        self.assertEqual((host([0.0]).device, host([0.0], copy=True).device), ("cpu", "cuda:0"))

    def test_shape_dtype_length_and_fill(self) -> None:
        array = host(np.zeros((4, 3)), dtype=np.float32)
        array.fill_(2.0)
        self.assertEqual((array.shape, array.dtype, len(array)), ((4, 3), np.float32, 4))
        self.assertTrue((array.numpy() == 2.0).all())


class TestRegistries(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = Registry("probe")
        self.registry.register("kept", 1, origin="setup")

    def test_a_name_registered_for_a_block_is_gone_after_it(self) -> None:
        with registered(self.registry, "probe", 2) as value:
            self.assertEqual((value, self.registry.get("probe")), (2, 2))
        self.assertNotIn("probe", self.registry.names())
        self.assertEqual(self.registry.get("kept"), 1)

    def test_a_registry_that_loads_lazily_keeps_what_it_loaded(self) -> None:
        lazy = Registry("lazy")
        lazy.add_discover(lambda: lazy.register("found", 5))
        with registered(lazy, "probe", 1):
            self.assertEqual(lazy.get("probe"), 1)
        self.assertEqual(lazy.get("found"), 5, "what the first lookup loaded survives the block")
        self.assertNotIn("probe", lazy.names())

    def test_a_populator_added_in_the_block_is_taken_back(self) -> None:
        with restored(self.registry):
            self.registry.add_discover(lambda: None)
        self.assertEqual(self.registry.discovers, ())

    def test_a_block_that_raises_still_restores_names_and_origins(self) -> None:
        with self.assertRaises(RuntimeError), restored(self.registry):
            self.registry.register("added", 3)
            self.registry.replace("kept", 4, origin="test")
            raise RuntimeError("a failing test")
        self.assertEqual(self.registry.names(), ["kept"])
        self.assertEqual(self.registry.get("kept"), 1)
        self.assertEqual(self.registry._origins["kept"], "setup")


class TestWaits(unittest.TestCase):
    def _module(self, name: str, source: str):
        """A throwaway module in ``sys.modules``, removed when the test ends."""
        module = types.ModuleType(name)
        exec(source, module.__dict__)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        return module

    def test_a_wait_returns_what_the_predicate_found(self) -> None:
        box: list[int] = []
        threading.Timer(0.01, box.append, args=(7,)).start()
        self.assertEqual(wait_until(lambda: box and box[0], timeout=5.0), 7)

    def test_a_wait_that_never_ends_fails_by_name_instead_of_hanging(self) -> None:
        start = time.monotonic()
        with self.assertRaisesRegex(AssertionError, "the worker's reply"):
            wait_until(lambda: False, timeout=0.05, what="the worker's reply")
        self.assertLess(time.monotonic() - start, 2.0)

    def test_a_fake_clock_moves_only_the_module_it_was_given(self) -> None:
        probe = self._module(
            "probe_clock_module", "import time\ndef now(): return time.monotonic()\ndef nap(s): time.sleep(s)\n"
        )
        with fake_clock("probe_clock_module") as clock:
            before = probe.now()
            probe.nap(30.0)
            self.assertEqual(probe.now() - before, 30.0)
            clock.advance(0.5)
            self.assertEqual(probe.now() - before, 30.5)
            self.assertNotEqual(time.monotonic(), clock.now, "the time module itself keeps real time")
        self.assertNotEqual(probe.now(), clock.now)

    def test_a_function_a_module_imported_by_name_is_patched_by_name(self) -> None:
        probe = self._module("probe_clock_names", "from time import monotonic\ndef now(): return monotonic()\n")
        with fake_clock("probe_clock_names.monotonic", start=5.0):
            self.assertEqual(probe.now(), 5.0)

    def test_a_module_with_no_time_of_its_own_is_refused_by_name(self) -> None:
        self._module("probe_clock_none", "x = 1\n")
        with self.assertRaisesRegex(AttributeError, "probe_clock_none"), fake_clock("probe_clock_none"):
            pass


class TestShortDirectories(unittest.TestCase):
    def test_a_socket_fits_under_a_short_dir_whatever_tmpdir_is(self) -> None:
        deep = "/tmp/" + "d" * 150
        with mock.patch.dict(os.environ, {"TMPDIR": deep}), short_dir() as path:
            self.assertLess(len(os.fsencode(path / "run" / "ctl.sock")), 100)
            self.assertTrue(path.is_dir())
        self.assertFalse(path.exists())

    def test_chdir_comes_back_when_the_block_raises(self) -> None:
        here = os.getcwd()
        with short_dir() as path, self.assertRaises(ValueError), chdir(path):
            self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(path))
            raise ValueError
        self.assertEqual(os.getcwd(), here)


if __name__ == "__main__":
    unittest.main()
