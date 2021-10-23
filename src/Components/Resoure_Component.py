import os

import numpy as np

from ..OpenglPipe.Resource import MeshResource, MaterialResource

class ResourceComponent:
    def __init__(self, core_component):
        self.core_component = core_component
        self.root_path = self.core_component.root_path
        self.scene_path = os.path.join(self.root_path, 'scene')
        self.mesh_path = os.path.join(self.scene_path, 'mesh')
        self.material_path = os.path.join(self.root_path, "shaders")

        self.mesh_dict = {}
        self.material_dict = {}  # material_name -> Material instance
        self.material_list = {}  # mesh_name -> material_name

        self.initialized = False

    def initialization(self):
        self.register_resource('mesh', 'sphere', 'obj', scale=np.array([0.15, 0.15, 0.15]), translate=np.array([1.2, 1.2, 2.0]))
        self.register_resource('mesh', 'camelhead', 'obj')
        self.register_resource('mesh', 'plane', 'obj', scale=np.array([5, 0.5, 5]), translate=np.array([0, -0.38, 0]))

        # self.get_mesh_resource("plane").set_modelmat(scale=np.array([0.2, 0.2, 0.2]))
        self.initialized = True

    def load_materials(self):
        self.register_resource('material', 'sphere', material_name='normal')
        self.register_resource('material', 'camelhead', material_name='Wood')
        self.register_resource('material', 'plane', material_name='Wood')

    def get_material_list(self):
        return self.material_list

    def register_single_material(self, material_name):
        if material_name not in self.material_list:
            vert_path = os.path.join(self.material_path, material_name + ".vert")
            frag_path = os.path.join(self.material_path, material_name + ".frag")
            material_resource = MaterialResource(self, material_name, vert_path, frag_path)
            material_resource.initialization()
            self.material_list[material_name] = material_resource

    def register_resource(self, resource_type, resource_name,
                          filename_extension=None, scale=None, translate=None,
                          material_name=None):
        if resource_type == "mesh":
            file_path = os.path.join(self.mesh_path, filename_extension, resource_name + "." + filename_extension)
            mesh_resource = MeshResource(self, file_path)
            mesh_resource.initialization(scale, translate)
            self.mesh_dict[resource_name] = mesh_resource
        elif resource_type == 'material':
            self.register_single_material(material_name)
            self.material_dict[resource_name] = material_name

    def get_mesh_resource(self, resource_name):
        if resource_name in self.mesh_dict:
            return self.mesh_dict[resource_name]
        else:
            self.register_resource('mesh', 'obj', resource_name)
            return self.mesh_dict[resource_name]

    def get_material_resource(self, material_name):
        if material_name in self.material_list:
            return self.material_list[material_name]
        else:
            return None