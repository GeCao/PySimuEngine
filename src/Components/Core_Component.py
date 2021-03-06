import numpy as np
import os

from ..PyGame.PyGame import MyPygame
from ..OpenglPipe.OpenglPipe import OpenglPipe
from .Simulator_Component import SimulatorComponent
from .Resoure_Component import ResourceComponent
from .Log_Component import LogComponent
from .Scene_Component import SceneComponent
from ..OpenglPipe.Camera import Camera
from ..Simulation.Fluid_LBM.LBM_D2Q9 import LBM_D2Q9
from ..Simulation.Fluid_FVM.FVM_AdvectionEquation import FVM_AdvectionEquation
from ..Simulation.Fluid_FVM.FVM_NS import FVM_NS


class CoreComponent:
    def __init__(self):
        self.root_path = os.path.abspath(os.curdir)
        self.src_path = os.path.join(self.root_path, 'src')
        self.simulation_path = os.path.join(self.src_path, 'Simulation')
        self.log_component = None
        self.simulator_component = None
        self.resource_component = None
        self.scene_component = None
        self.my_pygame = None
        self.opengl_pipe = None
        self.fluid_solver = None

        self.initialized = False

    def initialization(self):
        self.camera = Camera(self)
        self.camera.initialization()

        self.log_component = LogComponent(self, log_to_disk=True)
        self.log_component.initialization()

        self.simulator_component = SimulatorComponent(self)
        self.simulator_component.initialization()

        self.resource_component = ResourceComponent(self)
        self.resource_component.initialization()

        self.my_pygame = MyPygame(self)
        self.opengl_pipe = OpenglPipe(self)
        self.my_pygame.initialization()
        self.opengl_pipe.initialization()

        self.scene_component = SceneComponent(self)
        self.scene_component.initialization()

        # self.fluid_solver = LBM_D2Q9(self)
        # self.fluid_solver.initialization()

        self.initialized = True

    def update(self):
        self.camera.update()
        self.scene_component.update()
        self.resource_component.update()
        self.simulator_component.update()
        self.opengl_pipe.update()
        # self.fluid_solver.update()

    def run(self):
        while True:
            self.my_pygame.run()
            self.update()
