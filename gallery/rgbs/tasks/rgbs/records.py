# SPDX-License-Identifier: Apache-2.0
"""RGBS's assets: five sub-packages, one per family of variants, and the look.

Sub-packages rather than one flat directory because four of the five ship a
module called ``records.py`` and a ``layout.py``, and three of them stand a
work mat somewhere different. A record that builds a thing builds it in one
place, so each family's mat is its own record (``decant_mat``, ``polish_mat``,
``prep_mat``) rather than one ``work_mat`` placed three ways; keeping the
families apart is what stops that becoming *confusing* as well.

Importing this module registers every record and entity class in all five,
which is what a document wants: ``part - carton_contents`` has to find that
record, and its ``build`` factory, whichever variant is being lowered. The
generic loader walks every module under here, so nothing has to be listed.

``look`` is here rather than beside ``staging.py`` because a documents-only
suite's top-level Python is its staging, its cell and its solutions; how the
bench is drawn is an asset's business, and ``(:render "assets.look:RGBS_LOOK")``
is how a document reaches it.
"""

from __future__ import annotations

from . import decant, look, peel, polish, prep, tunnel

__all__ = ["decant", "look", "peel", "polish", "prep", "tunnel"]
