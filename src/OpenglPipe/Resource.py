import numpy as np
import copy
import os
from ..utils.utils import _read_img_2d, MessageAttribute

class Resource:
    def __init__(self, resource_component, file_path=None):
        self.resource_component = resource_component

        self.initialized = False

    def initialization(self):
        self.initialized = True

    def load_data(self, file_path):
        pass


class MaterialResource(Resource):
    def __init__(self, resource_component, material_name, vert_path=None, frag_path=None):
        Resource.__init__(self, resource_component=resource_component)
        self.vert_path = vert_path
        self.frag_path = frag_path
        self.material_name = material_name
        self.uniform_dict = {}

    def initialization(self):
        if self.material_name == "Simple":
            self.uniform_dict['material.ambient'] = np.array([0.2, 0.2, 0.2], dtype=np.float32)
            self.uniform_dict['material.diffuse'] = np.array([1, 0.5, 0.31], dtype=np.float32)
            self.uniform_dict['material.specular'] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self.uniform_dict['material.shininess'] = float(32.0)
        elif self.material_name == "Wood":
            self.uniform_dict['sampler0'] = _read_img_2d(os.path.join(self.resource_component.scene_path, "textures/png/wood_box.png"), np.float32) / 255.0
            self.uniform_dict['material.ambient'] = np.array([0.2, 0.2, 0.2], dtype=np.float32)
            self.uniform_dict['material.diffuse'] = np.array([1, 0.5, 0.31], dtype=np.float32)
            self.uniform_dict['material.specular'] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            self.uniform_dict['material.shininess'] = float(32.0)


class MeshResource(Resource):
    def __init__(self, resource_component, file_path=None):
        Resource.__init__(self, resource_component=resource_component)
        self.file_path = file_path
        self.data = []
        self.faces = []
        self.Pointer_info = []
        self.model_mat = np.eye(4, dtype=np.float32)
        self.mtl = None

    def initialization(self, scale=None, translate=None):
        self.load_data(file_path=self.file_path, scale=scale, translate=translate)
        self.data = np.array(self.data, dtype=np.float32)
        self.faces = np.array(self.faces, dtype=np.int)
        self.initialized = True

    def set_modelmat(self, scale=None, rotation=None, translate=None):
        self.model_mat = np.eye(4, dtype=np.float32)
        if translate.shape == (4, 4):
            self.model_mat = translate
        elif translate.shape == (3, 1):
            self.model_mat[0, 3], self.model_mat[1, 3], self.model_mat[2, 3] = translate[0], translate[1], translate[2]
        elif translate is not None:
            raise ValueError("The form of translation can not be recognized")

        if rotation.shape == (4, 4):
            self.model_mat = self.model_mat.dot(rotation)
        elif rotation is not None:
            raise ValueError("The form of rotation can not be recognized")

        if scale.shape == (4, 4):
            self.model_mat = self.model_mat.dot(scale)
        elif scale.shape == (3, 1):
            self.model_mat = self.model_mat.dot(np.diag([scale[0], scale[1], scale[2], 1]))
        elif scale is not None:
            raise ValueError("The form of translation can not be recognized")


    def load_data(self, file_path, scale=None, translate=None):
        swapyz = False
        index_vert = 0
        scale = np.array([1, 1, 1]) if scale is None else scale
        translate = np.array([0, 0, 0]) if translate is None else translate
        normals = []
        texcoords = []
        f_texcoords_idx = []
        f_normals_idx = []
        for line in open(file_path, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values:
                continue
            vert = []
            normal = []
            texcoord = []
            if values[0] == 'v':
                # v = map(float, values[1:4])
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                v[0], v[1], v[2] = v[0] * scale[0], v[1] * scale[1], v[2] * scale[2]
                v[0], v[1], v[2] = v[0] + translate[0], v[1] + translate[1], v[2] + translate[2]
                vert = v
            elif values[0] == 'vn':
                # v = map(float, values[1:4])
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], v[2], v[1]
                normal = v
            elif values[0] == 'vt':
                v = [float(x) for x in values[1:3]]
                texcoord = v
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = values[1]
            elif values[0] == 'f':
                for v in values[1:]:
                    w = v.split('/')
                    self.faces.append(int(w[0]) - 1)  # 大坑！索引要从0开始，但是obj文件的索引从1开始！
                    if len(w) >= 2 and len(w[1]) > 0:
                        f_texcoords_idx.append(int(w[1]) - 1)
                    if len(w) >= 3 and len(w[2]) > 0:
                        f_normals_idx.append(int(w[2]) - 1)
            if 'vertices' not in self.Pointer_info: # and len(vert) != 0:
                self.Pointer_info.append('vertices')
            if 'normals' not in self.Pointer_info: # and len(normal) != 0:
                self.Pointer_info.append('normals')
            if 'texcoords' not in self.Pointer_info: # and len(texcoord) != 0:
                self.Pointer_info.append('texcoords')

            if len(vert) != 0:
                self.data.append([vert[0], vert[1], vert[2],
                                  0, 1, 0,
                                  0, 0])
                index_vert += 1
            if len(normal) != 0:
                normals.append([normal[0], normal[1], normal[2]])
            if len(texcoord) != 0:
                texcoords.append([texcoord[0], texcoord[1]])
        for i in range(int(len(self.faces) / 3)):
            vert_idx_1, vert_idx_2, vert_idx_3 = self.faces[3 * i], self.faces[3 * i + 1], self.faces[3 * i + 2]
            # normals
            if len(f_normals_idx) != 0:
                self.data[vert_idx_1][3:6] = normals[f_normals_idx[3 * i]][0:3]
                self.data[vert_idx_2][3:6] = normals[f_normals_idx[3 * i + 1]][0:3]
                self.data[vert_idx_3][3:6] = normals[f_normals_idx[3 * i + 2]][0:3]
            else:
                # 通过自己计算来补足法线
                pass

            if len(f_texcoords_idx) != 0:
                self.data[vert_idx_1][6:8] = texcoords[f_texcoords_idx[3 * i]][0:2]
                self.data[vert_idx_2][6:8] = texcoords[f_texcoords_idx[3 * i + 1]][0:2]
                self.data[vert_idx_3][6:8] = texcoords[f_texcoords_idx[3 * i + 2]][0:2]
        self.resource_component.core_component.log_component.Slog(MessageAttribute.EInfo,
            "Newly read an object with all the vert nums = {}, face nums = {}, texcoord nums = {}, normal nums = {}".
                                                                  format(index_vert, len(self.faces), len(f_texcoords_idx),
                                                                         len(f_normals_idx)))
