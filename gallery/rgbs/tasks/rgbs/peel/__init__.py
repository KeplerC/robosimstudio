# SPDX-License-Identifier: Apache-2.0
"""What the peel variants peel: a tape strip off a box and a protective film off a phone.

Both are the library's tape (:mod:`robosimstudio_materials.tape`), placed by ``layout``, which is
what the documents type: each record is a box, a phone or a sheet with the factory that lays it on
this suite's bench, under a name that says which layout it is.
"""

from __future__ import annotations

from . import layout

__all__ = ["layout"]
