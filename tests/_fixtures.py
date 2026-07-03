"""Tiny deterministic meshes for cleanup/bake tests (no GPU, no network)."""
import numpy as np
import trimesh


def icosphere(subdivisions=3):
    return trimesh.creation.icosphere(subdivisions=subdivisions)


def box_dense(count=5):
    m = trimesh.creation.box(extents=(1, 1, 1))
    for _ in range(count):
        m = m.subdivide()
    return m


def cube_with_bump():
    """A subdivided top face with a single raised region — a known 'detail'."""
    m = trimesh.creation.box(extents=(2, 2, 0.2))
    for _ in range(4):
        m = m.subdivide()
    v = m.vertices.copy()
    top = v[:, 2] > 0.09
    cx, cy = v[:, 0], v[:, 1]
    r = np.sqrt(cx ** 2 + cy ** 2)
    bump = np.clip(0.25 - r, 0, None) * top
    v[:, 2] += bump
    return trimesh.Trimesh(vertices=v, faces=m.faces, process=False)


def unit_uv_quad():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    return verts, faces, uv
