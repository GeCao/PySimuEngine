import os

import numpy as np

from ..OpenglPipe.AnimationActor import AnimationActor
from ..OpenglPipe.Resource import MeshResource, MaterialResource
from src.utils.tet_mesh import read_mesh
from ..Simulation.deformation.projective_dynamics import ProjectiveDynamics

class ResourceComponent:
    def __init__(self, core_component):
        self.core_component = core_component
        self.root_path = self.core_component.root_path
        self.scene_path = os.path.join(self.root_path, 'scene')
        self.mesh_path = os.path.join(self.scene_path, 'mesh')
        self.animation_path = os.path.join(self.scene_path, 'animation')
        self.material_path = os.path.join(self.root_path, "shaders")

        self.mesh_dict = {}
        self.animation_dict = {}
        self.material_dict = {}  # material_name -> Material instance
        self.material_list = {}  # mesh_name -> material_name

        self.initialized = False

    def initialization(self):
        # self.register_resource('mesh', 'sphere', 'obj', scale=np.array([0.45, 0.45, 0.45]), translate=np.array([0.6, -0.05, 0.6]))
        self.register_resource('mesh', 'camelhead', 'obj')
        self.register_resource('mesh', 'plane', 'obj', scale=np.array([5, 0.5, 5]), translate=np.array([0, -0.38, 0]))

        self.register_animator('precomputed', 'starfish', 'obj',
                               scale=np.array([5, 5, 5]), translate=np.array([0, -2, -2]))

        self.initialized = True

    def load_materials(self):
        # self.register_resource('material', 'sphere', material_name='glossy')
        self.register_resource('material', 'camelhead', material_name='glossy')
        self.register_resource('material', 'plane', material_name='Wood')
        self.register_resource('material', 'starfish', material_name='glossy')

    def get_material_list(self):
        return self.material_list

    def register_single_material(self, material_name):
        if material_name not in self.material_list:
            vert_path = os.path.join(self.material_path, material_name + ".vert")
            frag_path = os.path.join(self.material_path, material_name + ".frag")
            material_resource = MaterialResource(self, material_name, vert_path, frag_path)
            material_resource.initialization()
            self.material_list[material_name] = material_resource

    def register_animator(self, resource_type, resource_name,
                          filename_extension=None,
                          scale=np.array([1, 1, 1]), translate=np.array([0, 0, 0])):
        if resource_type == 'precomputed':
            all_data = []
            bin_path = os.path.join(self.animation_path, resource_name + '/surmesh/target')
            all_bin_files = os.listdir(bin_path)
            self.core_component.log_component.InfoLog(
                "{} frames has been collected for {}".format(len(all_bin_files), resource_name))
            for bin_file in all_bin_files:
                all_data.append(read_mesh(os.path.join(bin_path, bin_file))[0])
            all_data = np.array(all_data, dtype=np.float32)
            all_data = np.array(all_data * scale + translate, dtype=np.float32)

            self.register_resource('mesh', resource_name, filename_extension, scale=scale, translate=translate)
            mesh_resource = self.get_mesh_resource(resource_name)
            animation_actor = AnimationActor(self)
            animation_actor.initialization(mesh=mesh_resource, all_data=all_data, source_from='precomputed')
            self.animation_dict[resource_name] = animation_actor
        elif resource_type == 'deformation':
            # Please note that in most of cases, deformation is based on a tet mesh, with a surface mesh
            simulator = self.core_component.simulator_component.get_simulator(
                simulator_name=resource_name, simulator_type=resource_type)
            self.register_resource('mesh', resource_name, filename_extension, scale=scale, translate=translate)
            animation_actor = AnimationActor(self)
            animation_actor.initialization(simulator=simulator, mesh=self.get_mesh_resource(resource_name),
                                           source_from='deformation')
            self.animation_dict[resource_name] = animation_actor
            # initialization is always the last step, as we will use animation_actor we created before.
            simulator.initialization(resource_name=resource_name, resource_type='tet')

    def register_resource(self, resource_type, resource_name=None,
                          filename_extension=None, scale=np.array([1, 1, 1]), translate=np.array([0, 0, 0]),
                          material_name=None):
        if resource_type == "mesh":
            file_path = os.path.join(self.mesh_path, filename_extension, resource_name + "." + filename_extension)
            mesh_resource = MeshResource(self, file_path)
            mesh_resource.initialization(scale, translate)
            self.mesh_dict[resource_name] = mesh_resource
        elif resource_type == 'material':
            self.register_single_material(material_name)
            if resource_name is not None:
                self.material_dict[resource_name] = material_name

    def get_mesh_resource(self, resource_name):
        if resource_name in self.mesh_dict:
            return self.mesh_dict[resource_name]
        else:
            self.register_resource('mesh', 'obj', resource_name)
            return self.mesh_dict[resource_name]

    def get_animation_resource(self, resource_name):
        if resource_name in self.animation_dict:
            return self.animation_dict[resource_name]
        else:
            self.core_component.log_component.ErrorLog(
                "Try to get animation resource {}, but only got None instead since we did not find it.".format(resource_name))
            return None

    def get_material_resource(self, material_name):
        if material_name in self.material_list:
            return self.material_list[material_name]
        else:
            return None

    def update(self):
        for resource_name in self.animation_dict:
            self.animation_dict[resource_name].update()