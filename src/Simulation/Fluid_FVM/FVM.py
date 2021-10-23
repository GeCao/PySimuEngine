import torch
import numpy as np
from enum import Enum


class Flag(Enum):
    Fluid = 0
    Wall = 1
    Von_Neumann = 2
    Dirichlet = 3


class FVM:
    def __init__(self, core_component, boundary_band_="PPPPPP"):
        self.core_component = core_component
        self.dtype = torch.float
        self.np_dtype = np.float32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("What we use as our device: ", self.device)

        self.dt = torch.Tensor([1.0]).to(self.dtype).to(self.device)

        """ The parameters should always be re-defined in every following class """
        self.rho_w = torch.Tensor([1.0]).to(self.dtype).to(self.device)
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

        self.current_time = 0  # 时间步 | time step.

        self.velocity = None
        self.velocity_before = None
        self.density = None
        self.p = None
        self.C = None
        self.force = None

        self.initialized = False

    def initialization(self):
        pass

    def solve_Advection_step(self):
        pass

    def visualization(self):
        pass

    def Record_velocity(self):
        pass

    def time_tick(self):
        self.current_time += 1

    def compute(self):
        # self.Record_velocity()
        self.velocity_before = self.get_analytical_sol()
        self.solve_Advection_step()
        if self.current_time % self.Nwri == 0:
            self.visualization()

    def Cal_error(self):
        pass

    def get_time(self):
        return self.current_time

    def update(self):
        pass

