import os
import igl
import struct
import subprocess
import numpy as np
import torch
# from utils.common import ndarray, filter_unused_vertices

filter_unused_vertices = None
ndarray = np.array

# use this function to generate *3D* tet meshes.
# vertices: n x 3 numpy array.
# faces: m x 4 numpy array.
def write_mesh(vertices, faces, bin_file_name):
    with open(bin_file_name, 'wb') as f:
        f.write(struct.pack('i', 3))
        f.write(struct.pack('i', 4))
        # Vertices.
        vert_num, _ = ndarray(vertices).shape
        f.write(struct.pack('i', 3))
        f.write(struct.pack('i', vert_num))
        for v in vertices:
            f.write(struct.pack('d', v[0]))
        for v in vertices:
            f.write(struct.pack('d', v[1]))
        for v in vertices:
            f.write(struct.pack('d', v[2]))

        # Faces.
        faces = ndarray(faces).astype(np.int)
        face_num, nv = faces.shape
        f.write(struct.pack('i', nv))
        f.write(struct.pack('i', face_num))
        for j in range(nv):
            for i in range(face_num):
                f.write(struct.pack('i', faces[i, j]))


def read_mesh(bin_file_name):
    if not os.path.exists(bin_file_name):
        return None

    with open(bin_file_name, 'rb') as mesh:
        struct.unpack('i', mesh.read(4))
        struct.unpack('i', mesh.read(4))
        # Vertices.
        nd = struct.unpack('i', mesh.read(4))[0]
        vert_num = struct.unpack('i', mesh.read(4))[0]
        vertices = np.zeros(shape=[vert_num, nd]).astype(np.float64)
        for j in range(nd):
            for i in range(vert_num):
                vertices[i, j] = struct.unpack('d', mesh.read(8))[0]

        # Faces.
        nv = struct.unpack('i', mesh.read(4))[0]
        face_num = struct.unpack('i', mesh.read(4))[0]
        faces = np.zeros(shape=[face_num, nv]).astype(np.int32)
        for j in range(nv):
            for i in range(face_num):
                faces[i, j] = struct.unpack('i', mesh.read(4))[0]

        return vertices, faces


def write_w(w, bin_file_name):
    n = w.shape[0]
    with open(bin_file_name, 'wb') as f:
        f.write(struct.pack('i', n))
        for i in range(n):
            f.write(struct.pack('f', w[i]))


def read_w(bin_file_name):
    if not os.path.exists(bin_file_name):
        return None

    with open(bin_file_name, 'rb') as weights:
        n = struct.unpack('i', weights.read(4))[0]
        # Vertices.
        w = np.zeros(shape=[n]).astype(np.float32)
        for i in range(n):
            w[i] = struct.unpack('f', weights.read(4))[0]
        return w


def write_d(d, bin_file_name):
    n = d.shape[0]
    with open(bin_file_name, 'wb') as f:
        f.write(struct.pack('i', n))
        for i in range(n):
            f.write(struct.pack('i', d[i]))


def read_d(bin_file_name):
    if not os.path.exists(bin_file_name):
        return None

    with open(bin_file_name, 'rb') as dirichilet:
        n = struct.unpack('i', dirichilet.read(4))[0]
        # Vertices.
        d = np.zeros(shape=[n]).astype(np.int64)
        for i in range(n):
            d[i] = struct.unpack('i', dirichilet.read(4))[0]
        return d

# Given four vertices of a tet, return a 4 x 3 int arrays of 0, 1, 2, and 3. Each row describes
# a surface triangle whose normal is pointing outward if you follow the vertices by the righ-hand rule.
def fix_tet_faces(verts):
    verts = ndarray(verts)
    v0, v1, v2, v3 = verts
    f = []
    if np.cross(v1 - v0, v2 - v1).dot(v3 - v0) < 0:
        f = [
            (0, 1, 2),
            (2, 1, 3),
            (1, 0, 3),
            (0, 2, 3),
        ]
    else:
        f = [
            (1, 0, 2),
            (1, 2, 3),
            (0, 1, 3),
            (2, 0, 3),
        ]

    return ndarray(f).astype(np.int)

# Given a tet mesh, save it as an obj file with texture coordinates.
def tet2obj_with_textures(tet_mesh, obj_file_name=None, pbrt_file_name=None):
    vertex_num = tet_mesh.NumOfVertices()
    element_num = tet_mesh.NumOfElements()

    v = []
    for i in range(vertex_num):
        v.append(tet_mesh.py_vertex(i))
    v = ndarray(v)

    face_dict = {}
    for i in range(element_num):
        fi = list(tet_mesh.py_element(i))
        element_vert = []
        for vi in fi:
            element_vert.append(tet_mesh.py_vertex(vi))
        element_vert = ndarray(element_vert)
        face_idx = fix_tet_faces(element_vert)
        for f in face_idx:
            vidx = [int(fi[fij]) for fij in f]
            vidx_key = tuple(sorted(vidx))
            if vidx_key in face_dict:
                del face_dict[vidx_key]
            else:
                face_dict[vidx_key] = vidx

    f = []
    for _, vidx in face_dict.items():
        f.append(vidx)
    f = ndarray(f).astype(int)

    v, f = filter_unused_vertices(v, f)

    v_out = []
    f_out = []
    v_cnt = 0
    for fi in f:
        fi_out = [v_cnt, v_cnt + 1, v_cnt + 2]
        f_out.append(fi_out)
        v_cnt += 3
        for vi in fi:
            v_out.append(ndarray(v[vi]))

    texture_map = [[0, 0], [1, 0], [0, 1]]
    if obj_file_name is not None:
        with open(obj_file_name, 'w') as f_obj:
            for vv in v_out:
                f_obj.write('v {:6f} {:6f} {:6f}\n'.format(vv[0], vv[1], vv[2]))
            for u, v in texture_map:
                f_obj.write('vt {:6f} {:6f}\n'.format(u, v))
            for ff in f_out:
                f_obj.write('f {:d}/1 {:d}/2 {:d}/3\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1))

    if pbrt_file_name is not None:
        with open(pbrt_file_name, 'w') as f_pbrt:
            f_pbrt.write('AttributeBegin\n')
            f_pbrt.write('Shape "trianglemesh"\n')

            # Log point data.
            f_pbrt.write('  "point3 P" [\n')
            for vv in v_out:
                f_pbrt.write('  {:6f} {:6f} {:6f}\n'.format(vv[0], vv[1], vv[2]))
            f_pbrt.write(']\n')

            # Log texture data.
            f_pbrt.write('  "float uv" [\n')
            for _ in range(int(len(v_out) / 3)):
                f_pbrt.write('  0 0\n')
                f_pbrt.write('  1 0\n')
                f_pbrt.write('  0 1\n')
            f_pbrt.write(']\n')

            # Log face data.
            f_pbrt.write('  "integer indices" [\n')
            for ff in f_out:
                f_pbrt.write('  {:d} {:d} {:d}\n'.format(ff[0], ff[1], ff[2]))
            f_pbrt.write(']\n')
            f_pbrt.write('AttributeEnd\n')

# Given tet_mesh, return vert and faces that describes the surface mesh as a triangle mesh.
# You should use this function mostly for rendering.
# Output:
# - vertices: an n x 3 double array.
# - faces: an m x 3 integer array.
def tet2obj(tet_mesh, obj_file_name=None):
    vertex_num = tet_mesh.NumOfVertices()
    element_num = tet_mesh.NumOfElements()

    v = []
    for i in range(vertex_num):
        v.append(tet_mesh.py_vertex(i))
    v = ndarray(v)

    face_dict = {}
    for i in range(element_num):
        fi = list(tet_mesh.py_element(i))
        element_vert = []
        for vi in fi:
            element_vert.append(tet_mesh.py_vertex(vi))
        element_vert = ndarray(element_vert)
        face_idx = fix_tet_faces(element_vert)
        for f in face_idx:
            vidx = [int(fi[fij]) for fij in f]
            vidx_key = tuple(sorted(vidx))
            if vidx_key in face_dict:
                del face_dict[vidx_key]
            else:
                face_dict[vidx_key] = vidx

    f = []
    for _, vidx in face_dict.items():
        f.append(vidx)
    f = ndarray(f).astype(int)

    v, f = filter_unused_vertices(v, f)

    if obj_file_name is not None:
        with open(obj_file_name, 'w') as f_obj:
            for vv in v:
                f_obj.write('v {} {} {}\n'.format(vv[0], vv[1], vv[2]))
            for ff in f:
                f_obj.write('f {} {} {}\n'.format(ff[0] + 1, ff[1] + 1, ff[2] + 1))

    return v, f

# Extract boundary faces from a 3D mesh.
def get_boundary_face(tet_mesh):
    _, f = tet2obj(tet_mesh)
    return f

# Input:
# - verts: a 4 x 3 matrix.
# Output:
# - solid_angles: a 4d array where each element corresponds to the solid angle spanned by the other three vertices.
def compute_tet_angles(verts):
    partition = [
        ([1, 2, 3], 0),
        ([0, 2, 3], 1),
        ([0, 1, 3], 2),
        ([0, 1, 2], 3)
    ]
    verts = ndarray(verts)
    solid_angles = np.zeros(4)
    for (i0, i1, i2), apex_idx in partition:
        apex = verts[apex_idx]
        v0 = verts[i0]
        v1 = verts[i1]
        v2 = verts[i2]
        # https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron.
        a_vec = v0 - apex
        b_vec = v1 - apex
        c_vec = v2 - apex
        a = np.linalg.norm(a_vec)
        b = np.linalg.norm(b_vec)
        c = np.linalg.norm(c_vec)
        tan_half_omega = a_vec.dot(np.cross(b_vec, c_vec)) / (
            a * b * c + a_vec.dot(b_vec) * c + a_vec.dot(c_vec) * b + b_vec.dot(c_vec) * a
        )
        solid_angles[apex_idx] = np.arctan(np.abs(tan_half_omega)) * 2
    return solid_angles

# Return a heuristic set of vertices that could be used for contact handling.
# - threshold: a vertex is considered to be a contact vertex if its solid angle < threshold.
def get_contact_vertex(tet_mesh, threshold=2 * np.pi):
    vertex_num = tet_mesh.NumOfVertices()
    element_num = tet_mesh.NumOfElements()

    v_solid_angle = np.zeros(vertex_num)
    for e in range(element_num):
        vertex_indices = list(tet_mesh.py_element(e))
        verts = []
        for vi in vertex_indices:
            verts.append(tet_mesh.py_vertex(vi))
        solid_angles = compute_tet_angles(verts)
        for vi, ai in zip(vertex_indices, solid_angles):
            v_solid_angle[vi] += ai

    contact_nodes = []
    for i, val in enumerate(v_solid_angle):
        if val < threshold:
            contact_nodes.append(i)
    return contact_nodes

# Input:
# - node_file and ele_file: generated by tetgen.
# Output:
# - verts: n x 3.
# - elements: m x 4.
def read_tetgen_file(node_file, ele_file):
    with open(node_file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip().split() for l in lines]
        node_num = int(lines[0][0])
        nodes = ndarray([[float(v) for v in lines[i + 1][1:4]] for i in range(node_num)])

    with open(ele_file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip().split() for l in lines]
        ele_num = int(lines[0][0])
        elements = np.asarray([[int(e) for e in lines[i + 1][1:5]] for i in range(ele_num)], dtype=int)
        elements = elements - 1

    # nodes is an n x 3 matrix.
    # elements is an m x 4 or m x 10 matrix. See this doc for details.
    # http://wias-berlin.de/software/tetgen/1.5/doc/manual/manual006.html#ff_ele.
    # In both cases, the first four columns of elements are the tets.
    # Fix the sign of elements if necessary.
    # elements_unsigned = elements.copy()
    # elements = []
    # for e in elements_unsigned:
    #     v = ndarray([nodes[ei] for ei in e])
    #     v0, v1, v2, v3 = v
    #     if np.cross(v1 - v0, v2 - v1).dot(v3 - v0) < 0:
    #         elements.append(e)
    #     else:
    #         elements.append([e[0], e[2], e[1], e[3]])
    # elements = ndarray(elements).astype(np.int)
    # nodes, elements = filter_unused_vertices(nodes, elements)
    return nodes, elements


def write_poly_file(vertices, faces, holes, file_name):
    polyfile = open(file_name, 'w')

    # Part 1 - node list
        # First line: <# of points> <dimension (must be 3)> <# of attributes> <# of boundary markers (0 or 1)>
        # Remaining lines list # of points:
        # <point #> <x> <y> <z>[attributes] [boundary marker]
    polyfile.write("{} {} {} {}\n".format(vertices.shape[0], int(3), int(0), int(0)))  # no attributes, no boundary markers
    for i, item in enumerate(vertices):
        polyfile.write("{} {} {} {}\n".format(i+1, item[0], item[1], item[2]))

    # Part 2 - facet list
        # One line: <# of facets> <boundary markers (0 or 1)>
        # Following lines list # of facets:
        # <facet #>
            # where each <facet #> has the following format:
            # One line: <# of polygons> [# of holes] [boundary marker]
            # Following lines list # of polygons:
            # <# of corners> <corner 1> <corner 2> ... <corner #>
            # ...
            # Following lines list # of holes:
            # <hole #> <x> <y> <z>
    polyfile.write("{} {}\n".format(faces.shape[0], int(0)))  # no boundary markers
    for item in faces:
        item = item + 1   # cpp tetgen starts at 1 not 0
        polyfile.write("{}\n".format(int(1)))    # 1 polygon
        polyfile.write("{} {} {} {}\n".format(int(3), item[0], item[1], item[2]))    # 3 corners

    # Part 3 - hole list
        # One line: <# of holes>
        # Following lines list # of holes:
        # <hole #> <x> <y> <z>
    if holes is None:
        holes = np.array([])

    polyfile.write("{}\n".format(holes.shape[0]))
    for i, item in enumerate(holes):
        polyfile.write("{} {} {} {}\n".format(i+1, item[0], item[1], item[2]))

    # Part 4 - region list
    polyfile.write("{}\n".format(0))    # no regions

    polyfile.close()


# ========================= voxel related ============================= #
def initialize_grid(vertices, step):
    bb_max = vertices.max(0)
    bb_min = vertices.min(0)
    radius = np.linalg.norm(bb_max - bb_min) * 0.5
    step = step * radius

    bb_max = vertices.max(0)
    bb_min = vertices.min(0)
    center = (bb_max + bb_min) * 0.5
    bb_min = center + (bb_min - center) * 1.2
    bb_max = center + (bb_max - center) * 1.2

    origin = bb_min
    grid_size = np.ceil((bb_max - bb_min) / step).astype(np.int64)
    occupancy = np.zeros(shape=grid_size[::-1], dtype=np.bool)

    return occupancy, origin, step, radius


def voxelize_sur_mesh(vertices, faces, step):
    # vertices m * d
    # 1. define the region of the voxelization
    occupancy, origin, step, radius = initialize_grid(vertices, step)

    # Voxelization.
    d, h, w = occupancy.shape
    z = np.arange(0, d)[:, None, None]
    y = np.arange(0, h)[None, :, None]
    x = np.arange(0, w)[None, None, :]
    z = np.tile(z, [1, h, w]).reshape(-1, 1)
    y = np.tile(y, [d, 1, w]).reshape(-1, 1)
    x = np.tile(x, [d, h, 1]).reshape(-1, 1)
    p = np.concatenate([x, y, z], axis=-1)
    center = (p + 0.5) * step + origin
    signed_distance, _, _ = igl.signed_distance(center, vertices, faces)
    indexing = signed_distance < 0
    x = x[indexing]
    y = y[indexing]
    z = z[indexing]
    occupancy[z, y, x] = 1

    return occupancy, origin, step


def voxelize_tet_mesh(vertices, elements, surfaces, step=0.01, add_flesh_constraint=True):
    # vertices m * d
    # 1. define the region of the voxelization
    occupancy, origin, step, radius = initialize_grid(vertices, step)
    occupancy = rasterize_tets(vertices[elements], occupancy, origin, step)

    for surface in surfaces:
        s_vertices, s_faces = surface
        occupancy = np.logical_or(occupancy, rasterize_faces(s_vertices[s_faces], occupancy.copy(), origin, step))

    # add flesh width constraints
    if add_flesh_constraint:
        grid_z, grid_y, grid_x = occupancy.nonzero()
        grid = np.concatenate([grid_x[:, None], grid_y[:, None], grid_z[:, None]], axis=-1)
        grid = (grid + 0.5) * step + origin

        m = 0
        v = []
        f = []
        for i, surface in enumerate(surfaces):
            v.append(surface[0])
            f.append(surface[1] + m)
            m += v[-1].shape[0]
        v = np.concatenate(v, axis=0)
        f = np.concatenate(f, axis=0)

        dists, _, _ = igl.point_mesh_squared_distance(grid, v, f)
        index = dists > (radius * 0.05) ** 2
        grid_z = grid_z[index]
        grid_y = grid_y[index]
        grid_x = grid_x[index]

        occupancy[grid_z, grid_y, grid_x] = 0

    return occupancy, origin, step


def occupancy_to_tet_mesh(occupancy, origin, surfaces, step=0.001):
    # extract surface point interpolation weights and control points
    # occupancy d * h * w * c, c being the material parameters, suitable for non-manifold meshing
    vertices = []
    elements = []

    d, h, w = occupancy.shape
    vertex_indices = np.ones([d*2+1, h*2+1, w*2+1], dtype=np.int64) * -1

    def add_object(x):
        vertices.append(x)
        return len(vertices) - 1

    def add_tet_point(i, j, k):
        ii = int(i * 2)
        jj = int(j * 2)
        kk = int(k * 2)

        if vertex_indices[kk, jj, ii] < 0:
            id = add_object((i * step, j * step, k * step))
            vertex_indices[kk, jj, ii] = id
        else:
            id = vertex_indices[kk, jj, ii]

        return id

    def add_tet_cubic(i, j, k):
        a = add_tet_point(i, j, k)
        b = add_tet_point(i, j + 1, k)
        c = add_tet_point(i + 1, j + 1, k)
        d = add_tet_point(i + 1, j, k)

        e = add_tet_point(i, j, k + 1)
        f = add_tet_point(i, j + 1, k + 1)
        g = add_tet_point(i + 1, j + 1, k + 1)
        h = add_tet_point(i + 1, j, k + 1)

        faces = [[a, b, c, d],
                 [d, c, g, h],
                 [h, g, f, e],
                 [e, f, b, a],
                 [f, b, c, g],
                 [e, a, d, h]]

        C = add_tet_point(i + 0.5, j + 0.5, k + 0.5)

        centers = [add_tet_point(i + 0.5, j + 0.5, k),
                   add_tet_point(i + 1, j + 0.5, k + 0.5),
                   add_tet_point(i + 0.5, j + 0.5, k + 1),
                   add_tet_point(i, j + 0.5, k + 0.5),
                   add_tet_point(i + 0.5, j + 1, k + 0.5),
                   add_tet_point(i + 0.5, j, k + 0.5)]

        for c, f in zip(centers, faces):
            ff = f + [f[0]]
            for i in range(4):
                elements.append((ff[i], ff[i + 1], c, C))

    d, h, w = occupancy.shape
    for k in range(d):
        for j in range(h):
            for i in range(w):
                if occupancy[k, j, i]:
                    add_tet_cubic(i, j, k)

    vertices = np.array(vertices, dtype=np.float32) + origin
    elements = np.array(elements, dtype=np.float32)

    # surface points
    control_indices = []
    control_weights = []
    grid_indices = vertex_indices[::2, ::2, ::2]
    for surface in surfaces:
        s = (surface - origin) / step
        x, y, z = np.split(s, 3, axis=-1)

        x0 = x.astype(np.int64)
        y0 = y.astype(np.int64)
        z0 = z.astype(np.int64)

        z1 = np.clip(z0 + 1, 0, d)
        y1 = np.clip(y0 + 1, 0, h)
        x1 = np.clip(x0 + 1, 0, w)

        wz = z - z0.astype(np.float32)
        wy = y - y0.astype(np.float32)
        wx = x - x0.astype(np.float32)

        wzInv = 1. - wz
        wyInv = 1. - wy
        wxInv = 1. - wx

        v000 = grid_indices[z0, y0, x0]
        v001 = grid_indices[z0, y0, x1]
        v010 = grid_indices[z0, y1, x0]
        v011 = grid_indices[z0, y1, x1]

        v100 = grid_indices[z1, y0, x0]
        v101 = grid_indices[z1, y0, x1]
        v110 = grid_indices[z1, y1, x0]
        v111 = grid_indices[z1, y1, x1]

        w000 = wzInv * wyInv * wxInv
        w001 = wzInv * wyInv * wx
        w010 = wzInv * wy * wxInv
        w011 = wzInv * wy * wx

        w100 = wz * wyInv * wxInv
        w101 = wz * wyInv * wx
        w110 = wz * wy * wxInv
        w111 = wz * wy * wx

        control_indices.append(np.concatenate([v000, v001, v010, v011,
                                               v100, v101, v110, v111], axis=-1))
        control_weights.append(np.concatenate([w000, w001, w010, w011,
                                               w100, w101, w110, w111], axis=-1))

        # m, v = control_indices[-1].shape
        # control_vertices = vertices[control_indices[-1].reshape(-1)].reshape(m, v, -1)
        # control_v = control_vertices * control_weights[-1][..., None]
        # control_v = control_v.sum(-2)

    return vertices, elements, control_indices, control_weights


def occupancy_to_tet_mesh_with_flags(occupancy, origin, surfaces, surfaces_states, step=0.001):
    # extract surface point interpolation weights and control points
    # occupancy d * h * w * c, c being the material parameters, suitable for non-manifold meshing
    vertices = []
    elements = []

    d, h, w, nf = occupancy.shape
    vertex_indices = np.ones([d*2+1, h*2+1, w*2+1, nf], dtype=np.int64) * -1
    grid_indices = vertex_indices[::2, ::2, ::2] # a view

    def add_object(x):
        vertices.append(x)
        return len(vertices) - 1

    def add_tet_point(i, j, k, flag):
        ii = int(i * 2)
        jj = int(j * 2)
        kk = int(k * 2)

        id = None
        if vertex_indices[kk, jj, ii, 0] < 0:
            # second check the flag
            if vertex_indices[kk, jj, ii, flag] < 0:
                id = add_object((i * step, j * step, k * step))
                vertex_indices[kk, jj, ii, flag] = id
            else:
                id = vertex_indices[kk, jj, ii, flag]
        else:
            id = vertex_indices[kk, jj, ii, 0]

        return id

    def add_tet_cubic(i, j, k, flag):
        a = add_tet_point(i, j, k, flag)
        b = add_tet_point(i, j + 1, k, flag)
        c = add_tet_point(i + 1, j + 1, k, flag)
        d = add_tet_point(i + 1, j, k, flag)

        e = add_tet_point(i, j, k + 1, flag)
        f = add_tet_point(i, j + 1, k + 1, flag)
        g = add_tet_point(i + 1, j + 1, k + 1, flag)
        h = add_tet_point(i + 1, j, k + 1, flag)

        faces = [[a, b, c, d],
                 [d, c, g, h],
                 [h, g, f, e],
                 [e, f, b, a],
                 [f, b, c, g],
                 [e, a, d, h]]

        C = add_tet_point(i + 0.5, j + 0.5, k + 0.5, flag)

        centers = [add_tet_point(i + 0.5, j + 0.5, k, flag),
                   add_tet_point(i + 1, j + 0.5, k + 0.5, flag),
                   add_tet_point(i + 0.5, j + 0.5, k + 1, flag),
                   add_tet_point(i, j + 0.5, k + 0.5, flag),
                   add_tet_point(i + 0.5, j + 1, k + 0.5, flag),
                   add_tet_point(i + 0.5, j, k + 0.5, flag)]

        for c, f in zip(centers, faces):
            ff = f + [f[0]]
            for i in range(4):
                elements.append((ff[i], ff[i + 1], c, C))

    def get_grid_index(z, y, x, flag):
        flag = flag.copy()
        exist_flags = (grid_indices[z, y, x] >= 0).astype(np.int64)
        condition = grid_indices[z, y, x, flag] < 0
        # condition = np.sum(exist_flags, axis=-1)
        # assert np.all(condition > 0)
        # index = condition == 1
        # flag = exist_flags.argmax(-1)
        flag[condition] = exist_flags.argmax(-1)[condition]
        return grid_indices[z, y, x, flag]

    for flag in range(nf):
        for k in range(d):
            for j in range(h):
                for i in range(w):
                    if occupancy[k, j, i, flag]:
                        add_tet_cubic(i, j, k, flag)

    vertices = np.array(vertices, dtype=np.float32) + origin
    elements = np.array(elements, dtype=np.float32)

    # surface points
    control_indices = []
    control_weights = []

    for surface, surface_states in zip(surfaces, surfaces_states):
        s = (surface - origin) / step
        x, y, z = np.split(s, 3, axis=-1)

        x0 = x.astype(np.int64)
        y0 = y.astype(np.int64)
        z0 = z.astype(np.int64)

        z1 = np.clip(z0 + 1, 0, d)
        y1 = np.clip(y0 + 1, 0, h)
        x1 = np.clip(x0 + 1, 0, w)

        wz = z - z0.astype(np.float32)
        wy = y - y0.astype(np.float32)
        wx = x - x0.astype(np.float32)

        wzInv = 1. - wz
        wyInv = 1. - wy
        wxInv = 1. - wx

        v000 = get_grid_index(z0, y0, x0, surface_states)
        v001 = get_grid_index(z0, y0, x1, surface_states)
        v010 = get_grid_index(z0, y1, x0, surface_states)
        v011 = get_grid_index(z0, y1, x1, surface_states)

        v100 = get_grid_index(z1, y0, x0, surface_states)
        v101 = get_grid_index(z1, y0, x1, surface_states)
        v110 = get_grid_index(z1, y1, x0, surface_states)
        v111 = get_grid_index(z1, y1, x1, surface_states)

        w000 = wzInv * wyInv * wxInv
        w001 = wzInv * wyInv * wx
        w010 = wzInv * wy * wxInv
        w011 = wzInv * wy * wx

        w100 = wz * wyInv * wxInv
        w101 = wz * wyInv * wx
        w110 = wz * wy * wxInv
        w111 = wz * wy * wx

        control_indices.append(np.concatenate([v000, v001, v010, v011,
                                               v100, v101, v110, v111], axis=-1))
        control_weights.append(np.concatenate([w000, w001, w010, w011,
                                               w100, w101, w110, w111], axis=-1))

        m, v = control_indices[-1].shape
        control_vertices = vertices[control_indices[-1].reshape(-1)].reshape(m, v, -1)
        control_v = control_vertices * control_weights[-1][..., None]
        control_v = control_v.sum(-2)
        print(np.max(np.abs(control_v - surface)))

    return vertices, elements, control_indices, control_weights


def rasterize_points(points, occupancy, origin, step):
    p = points
    g = (p - origin) / step
    g = g.astype(np.int64)
    occupancy[g[:, 2], g[:, 1], g[:, 0]] = 1

    return occupancy


def rasterize_lines(lines, occupancy, origin, step):
    st_pt = lines[:, 0]
    en_pt = lines[:, 1]
    occupancy[:] = 0
    length = np.linalg.norm(st_pt - en_pt, axis=-1)
    steps = max(int(np.ceil(length.max() / step * 1.1)), 2)
    for w in np.linspace(0, 1, steps):
        p = st_pt * (1 - w) + en_pt * w
        g = (p - origin) / step
        g = g.astype(np.int64)
        occupancy[g[:, 2], g[:, 1], g[:, 0]] = 1

    return occupancy


def rasterize_faces(faces, occupancy, origin, step):

    occupancy[:] = 0  # bool
    grid_size = np.array(occupancy.shape[::-1])
    for v in faces:

        grid_min = (v.min(0) - origin) / step
        grid_max = (v.max(0) - origin) / step
        grid_min = np.clip(grid_min.astype(np.int) - 1, 0, grid_size - 1)   # included
        grid_max = np.clip(grid_max.astype(np.int) + 2, 0, grid_size)

        w, h, d = (grid_max - grid_min)[:]
        z = np.arange(grid_min[2], grid_max[2])[:, None, None]
        y = np.arange(grid_min[1], grid_max[1])[None, :, None]
        x = np.arange(grid_min[0], grid_max[0])[None, None, :]
        z = np.tile(z, [1, h, w])[..., None]
        y = np.tile(y, [d, 1, w])[..., None]
        x = np.tile(x, [d, h, 1])[..., None]
        p = np.concatenate([x, y, z], axis=-1)
        p = (p + 0.5) * step + origin  # center of the grid
        p = p.reshape(-1, 3)

        dists, facets, points = igl.point_mesh_squared_distance(p, v, np.arange(v.shape[0])[None])
        occ = dists < (step * 0.5) ** 2

        occupancy[grid_min[2]:grid_max[2], grid_min[1]:grid_max[1], grid_min[0]:grid_max[0]] |= occ.reshape((d, h, w))

    g = (faces.reshape(-1, 3) - origin) / step
    g = g.astype(np.int64)
    occupancy[g[:, 2], g[:, 1], g[:, 0]] = 1

    return occupancy


def rasterize_tets(tets, occupancy, origin, step):
    # different to rasterize faces/lines, only tet whose center lies in the tet is considered
    occupancy[:] = 0  # bool
    grid_size = np.array(occupancy.shape[::-1])
    for v in tets:
        grid_min = (v.min(0) - origin) / step
        grid_max = (v.max(0) - origin) / step
        grid_min = np.clip(grid_min.astype(np.int) - 1, 0, grid_size - 1)   # included
        grid_max = np.clip(grid_max.astype(np.int) + 2, 0, grid_size)

        w, h, d = (grid_max - grid_min)[:]

        z = np.arange(grid_min[2], grid_max[2])[:, None, None]
        y = np.arange(grid_min[1], grid_max[1])[None, :, None]
        x = np.arange(grid_min[0], grid_max[0])[None, None, :]

        z = np.tile(z, [1, h, w])[..., None]
        y = np.tile(y, [d, 1, w])[..., None]
        x = np.tile(x, [d, h, 1])[..., None]

        p = np.concatenate([x, y, z], axis=-1)
        p = (p + 0.5) * step + origin
        p = p.reshape(-1, 3)

        l = get_barycentric_weights_N_1(p, v)

        index = np.all(l >= 0, axis=0)
        index = np.logical_and(index, l.sum(0) <= 1)
        occ = np.zeros(shape=[d*h*w], dtype=np.bool)
        occ[index] = True

        occupancy[grid_min[2]:grid_max[2], grid_min[1]:grid_max[1], grid_min[0]:grid_max[0]] |= occ.reshape((d, h, w))

    return occupancy


def cut_voxel_with_boundary(surface_vertices_half, surface_faces_half, occupancy, origin, step):
    boundary_edges = igl.boundary_facets(surface_faces_half)
    boundary_edges = surface_vertices_half[boundary_edges].copy()
    boundary = rasterize_lines(boundary_edges, occupancy.copy(), origin, step)

    # ad-hoc part
    z, y, x = boundary.nonzero()
    for i in range(z.shape[0]):
        boundary[:, y[i], x[i]:] = 1

    # rasterize the surfaces, make sure every vertices locate inside the voxels
    boundary = np.logical_or(boundary, rasterize_faces(surface_vertices_half[surface_faces_half], boundary.copy(), origin, step))
    occupancy = np.logical_and(occupancy, boundary)

    return occupancy


# def dilate_occ_with_guidance(surface, vertices, elements, occ, coorespondence):
#     # occ_source could not contain occ
#     surface_indices = get_indices_by_distance(surface, vertices)
#     one_jump_neigbors = get_neigbors_with_one_jump(vertices, elements)
#     _, neighbors = get_neigbors_with_multiple_jumps(surface_indices, one_jump_neigbors, jump=0, elements=elements)
#
#     write_mesh(vertices, elements[neighbors], "sss.bin")
#     # dilate the occupancy with guidance
#     d, h, w = occ.shape
#     grid = occ.reshape(-1).nonzero()[0]
#     grid_z = grid // (h*w)
#     grid_y = grid % (h*w) // w
#     grid_x = grid % w
#     assert(np.all(grid_x + grid_y * w + grid_z * h * w == grid))
#
#     # coords = np.concatenate([grid_x[:, None], grid_y[:, None], grid_z[:, None]], axis=-1)
#     dilated = np.zeros_like(occ)
#     for k in range(-1, 2):
#         for i in range(-1, 2):
#             for j in range(-1, 2):
#                 if i == 0 and j == 0 and k == 0:
#                     continue
#                 z = grid_z + k
#                 y = grid_y + j
#                 x = grid_x + i
#
#                 tets = coorespondence[z, y, x]
#                 flag = np.min(np.abs(tets[:, None] - neighbors[None]), axis=-1) == 0
#
#                 z = z[flag]
#                 y = y[flag]
#                 x = x[flag]
#
#                 dilated[z, y, x] = True
#
#     return dilated


def dilate_occ_with_guidance_simple(surface_vertices, surface_faces, occ, origin, step):

    # dilate the occupancy with guidance
    grid_z, grid_y, grid_x = occ.nonzero()
    normals = igl.per_face_normals(surface_vertices, surface_faces, np.zeros(3))

    # coords = np.concatenate([grid_x[:, None], grid_y[:, None], grid_z[:, None]], axis=-1)
    dilated = np.zeros_like(occ)
    for k in range(-1, 2):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0 and k == 0:
                    continue

                z = grid_z + k
                y = grid_y + j
                x = grid_x + i

                grid = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=-1)
                grid = (grid + 0.5) * step + origin
                _, facets, points = igl.point_mesh_squared_distance(grid, surface_vertices, surface_faces)

                ratio = normals[facets].reshape(-1, 1, 3) @ (grid - points).reshape(-1, 3, 1)
                ratio = ratio.reshape(-1)
                index = ratio < 0

                z = z[index]
                y = y[index]
                x = x[index]

                dilated[z, y, x] = True

    return np.logical_xor(occ, dilated)


def tetrahedralize_surface_mesh(vertices, faces, holes):

    write_poly_file(vertices, faces, holes, f"utils/tetgen/test.poly")
    subprocess.call(f"utils/tetgen/TetGen.exe -pq2 -Y utils/tetgen/test.poly")
    vertices, elements = read_tetgen_file(f"utils/tetgen/test.1.node", f"utils/tetgen/test.1.ele")

    return vertices, elements


def find_closet_points_on_faces(queries, vertices, faces, step_size, e_indices=None):
    dists, facets, points = igl.point_mesh_squared_distance(queries, vertices, faces)
    indices = (dists < step_size**2).nonzero()[0]

    if e_indices is not None:
        indices = remove_redundant_indices(indices, e_indices)

    facets = facets[indices]
    points = points[indices]

    n = facets.shape[0]
    f = faces[facets]
    v = vertices[f, :]
    p = points.reshape(n, 3, 1)

    A = np.zeros(shape=[n, 4, 4])
    b = np.zeros(shape=[n, 4, 1])
    A[:, :3, :3] = v @ v.transpose(0, 2, 1)
    A[:, 3, :3] = 1
    A[:, :3, 3] = 1
    b[:, :3, :] = v @ p
    b[:, 3, :] = 1

    w = torch.linalg.solve(torch.from_numpy(A), torch.from_numpy(b)).numpy()[:, :3]
    assert(np.all(w >= -1e-5))
    assert(np.all(np.abs(w.sum(-2) - 1) < 1e-5))
    assert(np.max(np.abs((v * w).sum(-2) - points)) < 1e-5)

    return indices, f, w


def change_topology(vertices_old, faces_old, indices):
    m, d = vertices_old.shape
    n = indices.shape[0]
    indices = np.sort(indices)

    hash_table = np.ones(shape=[m], dtype=indices.dtype) * -1
    hash_table[indices] = np.arange(n)

    faces_new = hash_table[faces_old]
    vertices_new = vertices_old[indices]

    return vertices_new, faces_new


def delete_faces(vertices, faces, vertices_deleted, faces_deleted):
    indices = get_indices_by_distance(vertices_deleted, vertices)
    faces_deleted = indices[faces_deleted]

    diff = np.sum(np.abs(faces[:, None] - faces_deleted[None]), axis=-1)
    index = diff.min(-1) > 0
    faces_new = faces[index]

    indices_new = np.unique(faces_new.reshape(-1))

    _, faces_new = change_topology(vertices, faces_new, indices_new)

    return indices_new, faces_new


def get_indices_by_distance(source, target):
    return np.argmin(np.sum(np.abs(source[:, None] - target[None, :]), axis=-1), axis=-1)


def remove_redundant_indices(source, reference):
    diff = np.abs(source[:, None] - reference[None])
    unique = diff[np.arange(diff.shape[0]), np.argmin(diff, axis=-1)]
    unique = unique > 0
    source = source[unique]
    return source


def clean_facets(facets):
    # n * 2/3/4
    I = np.lexsort(facets.transpose(1, 0)[::-1])
    facets = facets[I]
    M = np.any(facets[1:] - facets[:-1], axis=-1)  # is there any element > 0
    M = np.append([True], M)
    facets = facets[M, :]
    return facets


def get_neigbors_with_one_jump(vertices, elements):
    m, d = vertices.shape

    v_neighbors = []
    for i in range(m):
        v_neighbors.append([])

    for element in elements:
        for v_i in list(element):
            for v_j in list(element):
                if v_i != v_j:
                    v_neighbors[v_i].append(v_j)

    for i in range(m):
        v_neighbors[i] = list(np.unique(np.array(v_neighbors[i])))

    return v_neighbors


def get_neigbors_with_multiple_jumps(indices, one_jump_neighbors, jump=3, elements=None):
    # initialize neigbors
    neighbors = indices.copy()

    # expand the neigbors
    for i in range(jump):
        neighbors = list(neighbors)
        for v_id in neighbors.copy():
            neighbors = neighbors + one_jump_neighbors[v_id].copy()
        neighbors = np.unique(neighbors)

    e_neighbors = None
    if elements is not None:
        e_neighbors = np.any(neighbors[:, None, None] - elements == 0, axis=-1)
        e_neighbors = np.sum(e_neighbors, axis=0)
        e_neighbors = np.nonzero(e_neighbors)[0]

    return neighbors, e_neighbors


def get_barycentric_weights_N_1(p, v):
    # v - the vertices of the simplex
    # p - the points to be calculated
    v = v.transpose()
    T = v[:, :3] - v[:, 3:]
    T = np.linalg.inv(T)

    p = p.reshape(-1, 3).transpose()
    p = p - v[:, 3:]  # centers of the grid
    w = T @ p

    return w


def get_barycentric_weights_N_N(p, v):
    # v - the vertices of the simplex
    # p - the points to be calculated
    v = v.transpose(0, 2, 1)    # N * 3 * 4
    T = v[..., :3] - v[..., 3:]
    T = np.linalg.inv(T)        # N * 3 * 3

    p = p.reshape(-1, 1, 3, 1)  # B * 1 * 3 * 1
    p = p - v[..., 3:]  # centers of the grid B * N * 3 * 1
    w = T @ p

    return w[..., 0]    # B * N * 3


def get_voxel_tet_correspondence(vertices, elements, occupancy, origin, step):
    # shape of occupancy
    d, h, w = occupancy.shape
    grid = occupancy.reshape(-1).nonzero()[0]
    grid_z = grid // (h*w)
    grid_y = grid % (h*w) // w
    grid_x = grid % w
    assert(np.all(grid_x + grid_y * w + grid_z * h * w == grid))

    correspondence = np.ones_like(occupancy) * -1
    p = np.concatenate([grid_x[:, None], grid_y[:, None], grid_z[:, None]], axis=-1)
    p = (p + 0.5) * step + origin
    w = get_barycentric_weights_N_N(p, vertices[elements])  # B * N * 3
    loss = np.clip(-w, 0, None).sum(-1) + np.clip(w.sum(-1), 1, None)
    tets = loss.argmin(-1)
    print(loss[np.arange(p.shape[0]), tets])
    print(w[np.arange(p.shape[0]), tets])

    correspondence[occupancy] = tets

    return correspondence

"""
starfish_array = read_mesh('tetmesh/starfish.bin')
print(starfish_array[0].shape)  # (1045, 3)
print(starfish_array[1].shape)  # (3910, 4)

starfish_bone = read_d('tetmesh/bone.bin')
print(starfish_bone)  # (31, )

starfish_surface = read_d('tetmesh/surface.bin')
print(starfish_surface)  # (762, 0)

A = read_mesh('surmesh/target/00000.bin')
print(len(A))  # 2
print(A[0].shape, A[1].shape)  # (762, 3) (1520, 3)
"""