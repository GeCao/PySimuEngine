import os
from ..Simulation.deformation.projective_dynamics.ProjectiveDynamics import ProjectiveDynamics


class SimulatorComponent:
    def __init__(self, core_component):
        self.core_component = core_component
        self.root_path = self.core_component.root_path
        self.simulation_path = self.core_component.simulation_path
        self.scene_path = os.path.join(self.root_path, 'scene')
        self.mesh_path = os.path.join(self.scene_path, 'mesh')
        self.animation_path = os.path.join(self.scene_path, 'animation')
        self.material_path = os.path.join(self.root_path, "shaders")

        self.fluid_dict = {}
        self.deformation_dict = {}
        self.cloth_dict = {}
        self.rigid_dict = {}

        self.initialized = False

    def initialization(self):
        self.initialized = True

    def register_simulator(self, resource_type, resource_name):
        if resource_type == 'fluid':
            pass
        elif resource_type == 'deformation':
            # generally, this need to load a mesh resource from resource_component,
            # so please refer to this componnet for later initialization
            simulator = ProjectiveDynamics(self)
            if resource_name not in self.deformation_dict:
                self.deformation_dict[resource_name] = simulator
        elif resource_type == 'cloth':
            pass
        elif resource_type == 'rigid':
            pass

    def get_simulator_dict(self, simulator_type):
        if simulator_type == 'fluid':
            return self.fluid_dict
        elif simulator_type == 'deformation':
            return self.deformation_dict
        elif simulator_type == 'cloth':
            return self.cloth_dict
        elif simulator_type == 'rigid':
            return self.rigid_dict
        else:
            self.core_component.log_component.ErrorLog("Try to get a wrong simulator dict")
            return None

    def get_all_simulator_dict(self):
        return self.fluid_dict, self.deformation_dict, self.cloth_dict, self.rigid_dict

    def get_simulator(self, simulator_name, simulator_type=None):
        if simulator_type is not None:
            simulator_dict = self.get_simulator_dict(simulator_type)
            if simulator_name not in simulator_dict:
                self.register_simulator(simulator_type, simulator_name)
            return simulator_dict[simulator_name]
        else:
            for simulator_dict in self.get_all_simulator_dict():
                if simulator_name in simulator_dict:
                    return simulator_dict[simulator_name]
            self.core_component.log_component.ErrorLog(
                "Try to get a simulator named {}, but did not find it in all the known simulators," +
                " and since you did not refer the corresponding simulator type, can not generate a new one for you")
            return None

    def update(self):
        for simulator_dict in self.get_all_simulator_dict():
            for simulator in simulator_dict.values():
                simulator.update()
