import torch
import numpy as np
import os
from enum import Enum
import imageio


class Flag(Enum):
    Fluid = 0
    Wall = 1
    Von_Neumann = 2
    Dirichlet = 3


class LBM:
    def __init__(self, core_component, boundary_band_="PPPPPP"):
        self.core_component = core_component
        self.dtype = torch.float
        self.np_dtype = np.float32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("What we use as our device: ", self.device)

        """ The basic parameters which never should be changed"""
        self.dt = torch.Tensor([1.0]).to(self.dtype).to(self.device)
        self.c = torch.Tensor([1.0]).to(self.dtype).to(self.device)
        self.c2 = self.c * self.c
        self.cs2 = self.c * self.c / 3.0
        self.p0 = torch.Tensor([0.0]).to(self.dtype).to(self.device)  # 初始压强 | The initial pressure of particles.

        """ The parameters should always be re-defined in every following class """
        self.rho_w = torch.Tensor([1.0]).to(self.dtype).to(self.device)
        self.Ma = None  # Vmax / c
        self.beta = None
        self.tau = None  # tau = 1 / beta
        self.visc = None  # 初始粘度 | The initial viscosity of particles.
        self.Re = None  # Re number
        self.Vmax = None

        ''' Boundary '''
        self.boundary_band = boundary_band_  # means: left, right, down, up, forward, behind
        self.left_boundary = boundary_band_[0]  # 左边界为Periodic / Bounceback / Von Neumann(Flux) / Dirichlet boundaries? choose P, B, V, D for your prefer.
        self.right_boundary = boundary_band_[1]  # 右边界为Periodic / Bounceback / Von Neumann(Flux) / Dirichlet boundaries? choose P, B, V, D for yo
        self.down_boundary = boundary_band_[2]  # 下边界为Periodic / Bounceback / Von Neumann(Flux) / Dirichlet boundaries? choose P, B, V, D for yo
        self.up_boundary = boundary_band_[3]  # 上边界为Periodic / Bounceback / Von Neumann(Flux) / Dirichlet boundaries? choose P, B, V, D for yo
        self.forward_boundary = boundary_band_[4]  # 前边界为Periodic / Bounceback / Von Neumann(Flux) / Dirichlet boundaries? choose P, B, V, D for yo
        self.behind_boundary = boundary_band_[5]  # 后边界为Periodic / Bounceback / Von Neumann(Flux) / Dirichlet boundaries? choose P, B, V, D for yo

        ''' Visualization '''
        self.t_max = 60000  # 最大的迭代计算步数 | The max steps this programm run.
        self.if_dump = False  # 是否进行文件的输出 | Whether do we need to dump files.
        self.Nwri = 100  # 每隔Nwri步，进行一次文件的输出 | Every "Nwri" steps, dump a file.
        self.radius = 0.2  # 在以小球的形式绘制流场时，小球的半径 | while fluid particle was seen as a little sphere, hereby defined its radius.

        self.current_time = 0  # 时间步 | time step.

        self.f = None
        self.feq = None
        self.velocity = None
        self.density = None
        self.force = None

        self.initialized = False

    def initialization(self):
        pass

    def Bounceback_Boundary(self):
        pass

    def calculate_feq(self):
        pass

    def Cal_error(self):
        pass

    def Record_velocity(self):
        pass

    def calcu_Fxy(self):
        pass

    def rebounce_f(self):
        pass

    def Streaming(self):
        pass

    def macro_process(self):
        pass

    def Collision(self):
        pass

    def drawing_liquid(self):
        pass

    def dump_file(self, filepath):
        pass

    def read_file(self, filepath):
        pass

    def compute(self):
        self.Record_velocity()
        self.Streaming()
        self.macro_process()
        # self.rebounce_f()
        self.Collision()

    def marching_cube(self):
        Exception("Marching cube algorithm: Did not implemented!")

    def get_tmax(self):
        return self.t_max

    def ifdumpfile(self):
        return self.if_dump

    def get_time(self):
        return self.current_time

    def time_tick(self):
        self.current_time += 1

    def get_Lz(self):
        return 0

    def get_resolution(self):
        print("Error! did not define resolution in LBM.h!")
        return 0

    def update(self):
        self.compute()
        self.time_tick()
        print("====================================================")
        print("Simulation step: ", self.get_time(), "; Error: ", self.Cal_error())
        if self.if_dump and self.get_time() % self.Nwri == 0:
            imageio.imwrite(os.path.join(self.core_component.root_path, "data", "LBM_" + str(self.get_time()) + ".png"),
                            self.density.permute(1, 2, 0).numpy(), "png")