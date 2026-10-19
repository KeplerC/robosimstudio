# SPDX-License-Identifier: MIT
"""cagecut: fast cutting of cage based deformation with walk on spheres (2D and 3D).

Vendored, not written here. This is the ``cagecut`` package of "Fast Cutting of Cage Based
Deformation with Walk on Spheres" (Saeed, Lambiri, Schneider, Nowrouzezahrai and Kry, ACM SIGGRAPH
2026 Conference Papers, doi:10.1145/3799902.3811051), from https://github.com/hoskillua/wos_cutting_siggraph26
at commit 74190cf31be04b05e5b242130c64830b041e538b, under the MIT licence in ``LICENSE`` beside this
file (Copyright (c) 2026 Hossam Mohamed Saeed).

What changed from upstream, and nothing else:

* the polyscope viewer (``cagecut/app``), the paper's cases (``data``, ``data2d``) and the project
  page are left out;
* every ``from cagecut.`` import names this package by its full path,
  ``robosimstudio_materials.cutting.cagecut``;
* this docstring, and an SPDX line on top of each file.

The code keeps its own style (tabs), so ``pyproject.toml`` excludes this folder from ruff. The cutting
kind in ``robosimstudio_materials/cutting`` drives it through :class:`~.pipeline.Body` and the
:class:`~.scene.Scene` dataclass, and imports it only when a block is built.
"""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)
