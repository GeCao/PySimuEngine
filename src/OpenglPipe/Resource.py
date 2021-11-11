import numpy as np
import copy
import os
from ..utils.utils import _read_img_2d, MessageAttribute, normalize

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
    def __init__(self, resource_component, file_path=None, data=None, faces=None):
        Resource.__init__(self, resource_component=resource_component)
        self.file_path = file_path
        self.data = [] if data is None else data
        self.faces = [] if faces is None else faces
        self.Pointer_info = []
        self.model_mat = np.eye(4, dtype=np.float32)
        self.mtl = None
        self.need_update_to_pipe = False

    def initialization(self, scale=None, translate=None):
        if len(self.data) == 0:
            self.load_data(file_path=self.file_path, scale=scale, translate=translate)
            self.data = np.array(self.data, dtype=np.float32)
            self.faces = np.array(self.faces, dtype=np.int)
            self.generate_normals_automatically()
            if 'normals' not in self.Pointer_info:
                self.generate_normals_automatically()
                self.Pointer_info = ['vertices', 'normals']
        else:
            if len(self.faces) == 0:
                self.generate_faces_automatically()

            for i in range(len(self.data)):
                self.data[i][0], self.data[i][1], self.data[i][2] = \
                    self.data[i][0] * scale[0], self.data[i][1] * scale[1], self.data[i][2] * scale[2]
                self.data[i][0], self.data[i][1], self.data[i][2] = \
                    self.data[i][0] + translate[0], self.data[i][1] + translate[1], self.data[i][2] + translate[2]

            if len(self.data[0]) == 8:
                self.Pointer_info = ['vertices', 'normals', 'texcoords']
            elif len(self.data[0]) == 6:
                self.Pointer_info = ['vertices', 'normals']
            elif len(self.data[0]) == 3:
                self.generate_normals_automatically()
                self.Pointer_info = ['vertices', 'normals']
        self.data = np.array(self.data, dtype=np.float32)
        self.faces = np.array(self.faces, dtype=np.int)

        self.need_update_to_pipe = True
        self.initialized = True

    def generate_faces_automatically(self):
        if len(self.faces) > 0:
            self.resource_component.core_component.log_component.WarnLog(
                "Your original faces might be replaced, please inform yourself this change in your mind")
            self.faces = []

        self.faces = [i for i in range(len(self.data))]

    def update(self, data):
        if data.shape[1] == 3:
            self.data[:, 0:3] = data
            self.generate_normals_automatically()
        elif data.shape[1] == 6:
            self.data[:, 0:6] = data
        elif data.shape[1] == 8:
            self.data = data
        else:
            self.resource_component.core_component.log_component.WarnLog("Error for updated data format")
        self.need_update_to_pipe = True

    def generate_normals_automatically(self):
        if len(self.data) == 0:
            self.resource_component.core_component.log_component.WarnLog(
                "Can not generate normals if we have no any information about this mesh")
            return
        if len(self.data[0]) == 3:
            self.data = np.pad(self.data, pad_width=((0, 0), (0, 3)), mode='constant', constant_values=0)
        elif len(self.data[0]) < 6:
            self.resource_component.core_component.log_component.ErrorLog(
                "Can not read this Mesh Resource data format!")
        else:
            for i in range(len(self.data)):
                self.data[i][3], self.data[i][4], self.data[i][5] = 0, 0, 0
        if len(self.faces) == 0:
            self.resource_component.core_component.log_component.WarnLog(
                "Can not generate normals since you did not indicate faces, we will generate faces for you automatically")
            self.generate_faces_automatically()
        used_count = [0 for i in range(len(self.data))]
        for i in range(len(self.faces) // 3):
            idx1, idx2, idx3 = self.faces[3*i], self.faces[3*i + 1], self.faces[3*i + 2]
            vec1 = self.data[idx2, 0:3] - self.data[idx1, 0:3]
            vec2 = self.data[idx3, 0:3] - self.data[idx2, 0:3]
            normal = np.cross(vec1, vec2)
            normal_len = np.sqrt(normal.dot(normal))
            if normal_len > 0:
                normal = normal / normal_len
            else:
                normal = [0, 1, 0]
                self.resource_component.core_component.log_component.ErrorLog(
                    "The normal has been generated wrongly, has already set it as [0, 1, 0]")
            self.data[idx1][3:6] += normal
            self.data[idx2][3:6] += normal
            self.data[idx3][3:6] += normal
            used_count[idx1] += 1
            used_count[idx2] += 1
            used_count[idx3] += 1
        for i in range(len(self.data)):
            if used_count[i] != 0:
                self.data[i][3:6] = normalize(self.data[i][3:6])
            else:
                self.data[i][3], self.data[i][4], self.data[i][5] = 0, 1, 0


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
        self.resource_component.core_component.log_component.InfoLog(
            "Newly read an object with all the vert nums = {}, face nums = {}, texcoord nums = {}, normal nums = {}".
                                                                  format(index_vert, len(self.faces), len(f_texcoords_idx),
                                                                         len(f_normals_idx)))
