# SPDX-License-Identifier: Apache-2.0
"""How a particle body is drawn: as points, or as a surface regrown every frame.

A heap of rice, a pool, a lump of dough is particles with no triangle and no
order. Its :class:`~robosimstudio.scene.handles.DeformableHandle` says so with
a positive ``point_radius`` and may carry a ``surface`` callable; this module is
the one place that reads those two fields, so the studio page and the path
tracer agree about what the handle means:

* :func:`draws_points` is the test both apply -- no triangles, no tube, a
  positive point radius -- and the points are the particles at that radius.
* :func:`surface_mesh` calls a handle's ``surface`` and hands back host arrays
  of one shape whatever the callable returned, because the two renderers push
  vertices through different channels and neither should have to know whether
  a kind returned warp or numpy, two arrays or three.
* :class:`ParticleSurfacer` is the ``surface`` a kind hands its handle: the
  isosurface Newton's :class:`~newton.geometry.ParticleSurface` wraps round the
  particles, from the positions it is given and nothing else -- which is what
  lets an offline replay of a recorded ``particle_q`` regrow it. The MPM kinds
  hand it their solver's voxel and kernel; a kind with no grid of its own
  takes the defaults, sized by the particle radius.
* :func:`extract_shifted` is the extraction it goes through, because Newton's
  needs the particles moved into the positive octant first.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["ParticleSurfacer", "draws_points", "extract_shifted", "surface_mesh"]

EMPTY_VERTICES = np.zeros((0, 3), dtype=np.float32)
EMPTY_TRIANGLES = np.zeros((0, 3), dtype=np.int32)


def draws_points(handle: Any) -> bool:
    """Whether a deformable handle is a particle body: no triangles, no tube, a positive ``point_radius``.

    A sheet has triangles, a centerline has a tube radius, and both keep their
    own draw; this is the remaining case, and it is decided off the handle
    alone so that every renderer makes the same call.
    """
    triangles = getattr(handle, "triangles", None)
    if triangles is not None and len(triangles):
        return False
    if float(getattr(handle, "tube_radius", 0.0) or 0.0) > 0.0:
        return False
    return float(getattr(handle, "point_radius", 0.0) or 0.0) > 0.0


def _host(array: Any, dtype) -> np.ndarray:
    if hasattr(array, "numpy"):
        array = array.numpy()
    return np.asarray(array, dtype=dtype)


def surface_mesh(
    surface: Callable[[np.ndarray], Any], positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """A handle's ``surface`` over ``positions``, as ``(vertices, triangles, normals)`` on the host.

    ``surface(positions)`` may return ``(vertices, triangles)`` or
    ``(vertices, triangles, normals)``, each a warp array or anything numpy
    reads, the triangles flat ``[3k]`` or ``[k, 3]``; ``None`` in place of the
    tuple or of its vertices means there is nothing to draw this frame, which
    an isosurface over a few scattered grains legitimately produces. Vertices
    come back ``float32 [m, 3]``, triangles ``int32 [k, 3]``, and normals
    ``float32 [m, 3]`` or ``None`` when the callable computed none -- the
    caller then grows its own, as it does for a sheet.

    Takes the callable itself rather than the handle, deliberately: the
    studio's :class:`~robosimstudio_studio.garments.GarmentVisual` has a
    ``surface`` *method* of its own (a sheet's live vertices and normals), and
    a helper that reached for ``.surface`` on whatever it was given once drew
    the raw particles as a mesh and called it a surface.

    Args:
        surface: The handle's ``surface`` callable.
        positions: The handle's particles' world coordinates, ``[n, 3]``.
    """
    result = surface(np.asarray(positions, dtype=np.float32))
    if result is None:
        return EMPTY_VERTICES, EMPTY_TRIANGLES, None
    vertices, triangles, *rest = result
    normals = rest[0] if rest else None
    if vertices is None or triangles is None:
        return EMPTY_VERTICES, EMPTY_TRIANGLES, None
    vertices = _host(vertices, np.float32).reshape(-1, 3)
    triangles = _host(triangles, np.int32).reshape(-1, 3)
    if not len(vertices) or not len(triangles):
        return EMPTY_VERTICES, EMPTY_TRIANGLES, None
    if normals is not None:
        normals = _host(normals, np.float32).reshape(-1, 3)
        if len(normals) != len(vertices):
            normals = None
    return vertices, triangles, normals


def extract_shifted(surface: Any, positions: np.ndarray, radii: Any, margin: float) -> tuple[Any, Any, Any]:
    """``surface.extract`` over ``positions`` moved into the positive octant, and the vertices moved back.

    Newton 1.6.0's extraction is wrong wherever a particle's support reaches a
    negative coordinate (:class:`ParticleSurfacer` has the measurements), so
    every extraction here goes through this: the particles are shifted by
    their minimum less ``margin``, which puts the nearest one ``margin`` inside
    the octant, and the vertices are shifted back. The vertices and normals are
    exact under it.

    Args:
        surface: A :class:`~newton.geometry.ParticleSurface`.
        positions: World coordinates, ``[n, 3]``.
        radii: One radius per particle, a warp ``float32`` array on the
            device the mesh is extracted on.
        margin: How far inside the octant the nearest particle goes [m]: the
            kernel's reach plus a leaf tile of voxels.

    Returns:
        ``(vertices, triangles, normals)``: the vertices on the host,
        ``float32 [m, 3]`` in world coordinates, the triangles and normals as
        Newton returned them; ``(None, None, None)`` when there is nothing to
        draw.
    """
    import warp as wp

    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    if not len(positions):
        return None, None, None
    shift = (positions.min(axis=0) - float(margin)).astype(np.float32)
    points = wp.array(np.ascontiguousarray(positions - shift), dtype=wp.vec3, device=radii.device)
    vertices, triangles, normals = surface.extract(points, radii, compute_normals=True).to_arrays()
    if vertices is None:
        return None, None, None
    # The vertices come back on the host, shifted home: every renderer reads
    # them there anyway (:func:`surface_mesh`), so a second device round trip
    # for the shift would buy nothing.
    return vertices.numpy() + shift, triangles, normals


class ParticleSurfacer:
    """A ``DeformableHandle.surface``: the isosurface Newton wraps round the particles.

    One :class:`~newton.geometry.ParticleSurface` kept for the body's life,
    sized by the particle radius unless told otherwise: voxels of one radius,
    so the mesh resolves a single grain's bump, and a kernel of three spacings
    (six radii), which is what Newton's own dam-break example uses relative to
    its particle spacing. An MPM kind states its own: 0.45 of the solver's
    voxel and a kernel of three of those, Newton's defaults for the solver's
    ``create_particle_surface``.
    Each call allocates the mesh exactly (no ``max_grid_cells``), which is a
    host synchronisation per frame and fine for a renderer; a graph-captured
    extraction is a different object.

    **The particles are shifted into the positive octant before extraction and
    the mesh shifted back.** Newton 1.6.0's extraction never returns when a
    particle's support reaches a negative coordinate: its support-voxel kernel
    advances through a leaf tile by a segment length that comes out
    non-positive there and spins at 100 % of the GPU (measured on GPU 5, an
    RTX PRO 6000 Blackwell, warp 1.18.0: an 8 x 8 x 10 block at ``+0.02`` or
    ``+0.5`` m extracts in 43 ms cold and 3 ms warm; the same block at
    ``-0.5`` m, or astride the origin as every bench scene is, never comes
    back). The shift is by the particles' minimum less :attr:`margin`, so the
    whole support stays positive; the vertices and the normals are exact under
    it. A body astride only a vertical plane does come
    back, with the wrong mesh: the MPM solver's own context over the dough puck
    on ``bench_075``, whose particles all sit at negative y, returned 3,480
    triangles where the same 7,168 particles moved into the positive octant
    give 9,940, a closed surface (GPU 1, same card and versions). Moved to
    straddle z = 0 they never came back.

    Measured on that GPU over 7,290 particles at 3 mm spacing (a 27 x 27 x 10
    block): 10,802 vertices and 21,600 triangles in 43 ms on the first call
    and 3.6 ms after.

    Args:
        radius: The particle radius [m], the handle's own ``point_radius``.
        voxel_size: The surface's voxel [m], or ``None`` for one radius.
        device: Where the particles live; the mesh is extracted there.
        options: Anything else :class:`~newton.geometry.ParticleSurface` takes
            -- ``kernel_radius`` (six radii unless stated), ``anisotropic=True``
            for a liquid's flat free surface, ``mesh_smooth_iterations`` for a
            heap, ``world_count``.
    """

    def __init__(self, radius: float, *, voxel_size: float | None = None, device: Any = None, **options: Any) -> None:
        if not radius > 0.0:
            raise ValueError(f"a particle surface needs a positive radius, got {radius!r}")
        self.radius = float(radius)
        self.voxel_size = self.radius if voxel_size is None else float(voxel_size)
        if not self.voxel_size > 0.0:
            raise ValueError(f"a particle surface needs a positive voxel, got {voxel_size!r}")
        self.device = device
        self.options = dict(options)
        self.options.setdefault("kernel_radius", 6.0 * self.radius)
        self._surface: Any = None
        self._radii: Any = None

    @property
    def margin(self) -> float:
        """How far inside the positive octant the nearest particle is placed [m].

        Twice the kernel radius, Newton's grid padding, and a leaf tile of
        eight voxels: the isotropic density kernel reaches one kernel radius,
        an anisotropic one further along its long axis, the field is padded
        past that, and the extraction walks whole tiles. Sixteen voxels for an
        MPM kind (a kernel of three, a padding of two), 22 radii by default.
        """
        padding = int(self.options.get("padding", 2))
        return 2.0 * float(self.options["kernel_radius"]) + (padding + 8) * self.voxel_size

    def __call__(self, positions: np.ndarray) -> tuple[Any, Any, Any]:
        """``(vertices, triangles, normals)`` as warp arrays, or ``(None, None, None)`` for nothing to draw."""
        import warp as wp
        from newton.geometry import ParticleSurface

        positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        if not len(positions):
            return None, None, None
        if self._surface is None:
            self._surface = ParticleSurface(self.voxel_size, device=self.device, **self.options)
        if self._radii is None or len(self._radii) != len(positions):
            self._radii = wp.full(len(positions), self.radius, dtype=wp.float32, device=self.device)
        return extract_shifted(self._surface, positions, self._radii, self.margin)
