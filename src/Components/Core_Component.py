import numpy as np
import os

from ..PyGame.PyGame import MyPygame
from ..OpenglPipe.OpenglPipe import OpenglPipe
from .Resoure_Component import ResourceComponent
from ..OpenglPipe.Camera import Camera
from ..Simulation.Fluid_LBM.LBM_D2Q9 import LBM_D2Q9
from ..Simulation.Fluid_FVM.FVM_AdvectionEquation import FVM_AdvectionEquation
from ..Simulation.Fluid_FVM.FVM_NS import FVM_NS


class CoreComponent:
    def __init__(self):
        self.root_path = os.path.abspath(os.curdir)
        self.src_path = os.path.join(self.root_path, 'src')
        self.resource_component = None
        self.my_pygame = None
        self.opengl_pipe = None
        self.fluid_solver = None

        self.initialized = False

    def initialization(self):
        self.camera = Camera(self)
        self.camera.initialization()

        self.resource_component = ResourceComponent(self)
        self.resource_component.initialization()

        self.my_pygame = MyPygame(self)
        self.opengl_pipe = OpenglPipe(self)
        self.my_pygame.initialization()
        self.opengl_pipe.initialization()

        # self.fluid_solver = LBM_D2Q9(self)
        # self.fluid_solver.initialization()

        self.initialized = True

    def update(self):
        self.camera.update()
        self.opengl_pipe.update()
        # self.fluid_solver.update()

    def run(self):
        while True:
            self.my_pygame.run()
            self.update()
