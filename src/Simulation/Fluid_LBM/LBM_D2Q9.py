from .LBM import LBM, Flag
import numpy as np
import math
import torch.nn.functional as F
import torch


class LBM_D2Q9(LBM):
    def __init__(self, core_component, boundary_band_="PPPPPP"):
        super(LBM_D2Q9, self).__init__(core_component, boundary_band_)
        self.D = 2
        self.Q = 9

        self.resolution = torch.Tensor([64, 64]).to(torch.int).to(self.device)

        self.real_range = self.resolution.clone().to(self.dtype)
        """
        // The lattice velocities are defined according to the following scheme:
        // index:   0  1  2  3  4  5  6  7  8
        // ----------------------------------
        // x:       0 +1  0 -1  0 +1 -1 -1 +1
        // y:       0  0 +1  0 -1 +1 +1 -1 -1
        //
        // 6 2 5  ^y
        //  \|/   |   x
        // 3-0-1   --->
        //  /|\
        // 7 4 8
        """
        self.e = torch.Tensor([[0, 0],
                               [1, 0], [0, 1], [-1, 0], [0, -1],
                               [1, 1], [-1, 1], [-1, -1], [1, -1]]).to(self.dtype).to(self.device)
        self.w = torch.Tensor([4.0 / 9.0,
                               1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                               1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0]).to(self.dtype).to(self.device)
        # self.opp_index = torch.Tensor([0, 3, 4, 1, 2, 7, 8, 5, 6]).to(torch.int).to(self.device)

        self.Re = torch.Tensor([200.0]).to(self.dtype).to(self.device)
        self.tau = torch.Tensor([0.8]).to(self.dtype).to(self.device)
        self.beta = 0.5 / self.tau
        self.visc = self.cs2 * (self.tau - 0.5)
        self.Vmax = self.visc * self.Re / (self.resolution[0])

        self.Ma = self.Vmax / math.sqrt(self.cs2)

    def initialization(self):
        print("[Re Number]: ", self.Re)

        self.t_max = int(5.0 * (self.resolution[0]) / self.Vmax)
        print("tmax: ", self.t_max)
        lambda_alpha = 1.0
        K_x = 2.0 * math.pi / lambda_alpha / float(self.resolution[0])  # [K_x] -> m ** (-1)
        K_y = 2.0 * math.pi / lambda_alpha / float(self.resolution[1])  # [K_y] -> m ** (-1)
        K = math.sqrt(K_x * K_x + K_y * K_y)

        self.Vonneumann_Boundary(1, 1, 1, 1)
        self.Dirichlet_Boundary(1, 1, 1, 1)
        self.Bounceback_Boundary()

        self.f = torch.zeros(torch.Size([9, self.resolution[1], self.resolution[0]])).to(self.dtype).to(self.device)
        self.feq = torch.zeros(torch.Size([9, self.resolution[1], self.resolution[0]])).to(self.dtype).to(self.device)
        self.velocity = torch.zeros(torch.Size([2, self.resolution[1], self.resolution[0]])).to(self.dtype).to(
            self.device)
        self.density = torch.ones(torch.Size([1, self.resolution[1], self.resolution[0]])).to(self.dtype).to(
            self.device)
        self.velocity_before = torch.zeros(torch.Size([2, self.resolution[1], self.resolution[0]])).to(self.dtype).to(
            self.device)
        self.force = torch.zeros(torch.Size([2, self.resolution[1], self.resolution[0]])).to(self.dtype).to(self.device)
        self.status = torch.zeros(torch.Size([1, self.resolution[1], self.resolution[0]])).to(torch.int).to(self.device)

        x_linspace = torch.linspace(0, self.resolution[0] - 1, self.resolution[0]).to(self.device) * self.density
        y_linspace = (torch.linspace(0, self.resolution[1] - 1, self.resolution[1]).to(self.device) * self.density.permute(0, 2, 1)).permute(0, 2, 1)
        self.density = 1.0 - self.Ma * self.Ma / 2.0 / K / K * \
                       (K_y * K_y * torch.cos(2 * K_x * x_linspace) + K_x * K_x * torch.cos(2 * K_y * y_linspace))
        self.velocity[0:1, ...] = -self.Vmax * K_y / K * torch.sin(K_y * y_linspace) * torch.cos(K_x * x_linspace)
        self.velocity[1:2, ...] = self.Vmax * K_x / K * torch.sin(K_x * x_linspace) * torch.cos(K_y * y_linspace)

        if self.get_time() == 0:
            # print("Time 0 at init step, check f: ", self.f.norm())
            print("Time 0 at init step, check density: ", self.density.norm())
            print("Time 0 at init step, check velo: ", self.velocity.norm())
            print("K_x, K_y = ", K_x, K_y)
            print("res_x = ", self.resolution[0])

        self.calculate_feq()
        self.f = self.feq.clone()

        self.initialized = True

    def Bounceback_Boundary(self):
        if self.left_boundary == 'B':
            self.status[..., 0:1] = Flag.Wall
            self.density[..., 0:1] = self.rho_w[0]
        if self.right_boundary == 'B':
            self.status[..., -1:] = Flag.Wall
            self.density[..., -1:] = self.rho_w[0]

        if self.down_boundary == 'B':
            self.status[..., 0:1, :] = Flag.Wall
            self.density[..., 0:1, :] = self.rho_w[0]
        if self.up_boundary == 'B':
            self.status[..., -1, :] = Flag.Wall
            self.density[..., -1:, :] = self.rho_w[0]

    def Vonneumann_Boundary(self, u_left, u_right, v_down, v_up):
        if self.left_boundary == 'V':
            self.status[..., 0:1] = Flag.Von_Neumann
            self.velocity[0:1, ..., 0:1] = u_left
        if self.right_boundary == 'V':
            self.status[..., -1:] = Flag.Von_Neumann
            self.velocity[0:1, ..., -1:] = u_right

        if self.down_boundary == 'V':
            self.status[..., 0:1, :] = Flag.Von_Neumann
            self.velocity[1:2, ..., 0:1, :] = v_down
        if self.up_boundary == 'V':
            self.status[..., -1, :] = Flag.Von_Neumann
            self.velocity[1:2, ..., -1:, :] = v_up

    def Dirichlet_Boundary(self, rh_left, rh_right, rh_down, rh_up):
        if self.left_boundary == 'D':
            self.status[..., 0:1] = Flag.Dirichlet
            self.density[..., 0:1] = rh_left
        if self.right_boundary == 'D':
            self.status[..., -1:] = Flag.Dirichlet
            self.density[..., -1:] = rh_right

        if self.down_boundary == 'D':
            self.status[..., 0:1, :] = Flag.Dirichlet
            self.density[..., 0:1, :] = rh_down
        if self.up_boundary == 'D':
            self.status[..., -1, :] = Flag.Dirichlet
            self.density[..., -1:, :] = rh_up

    def calculate_feq(self):
        u = self.velocity[0:1, ...]
        v = self.velocity[1:2, ...]
        temp_1 = torch.sqrt(1.0 + 3.0 * u * u)
        temp_2 = torch.sqrt(1.0 + 3.0 * v * v)
        ones_feq = torch.ones(torch.Size([9, self.resolution[1], self.resolution[0]])).to(self.dtype).to(self.device)
        self.feq = (2.0 - temp_1) * \
                   (2.0 - temp_2) * \
                   (torch.pow(((2.0 * u + temp_1) / (1 - u) * ones_feq).permute(2, 1, 0), self.e[:, 0]) * \
                   torch.pow(((2.0 * v + temp_2) / (1 - v) * ones_feq).permute(2, 1, 0), self.e[:, 1])).permute(2, 1, 0)
        self.feq = self.density * self.feq
        self.feq = (self.w * self.feq.permute(2, 1, 0)).permute(2, 1, 0)

    def Cal_error(self):
        lambda_alpha = 1.0
        K_x = 2.0 * math.pi / lambda_alpha / float(self.resolution[0])  # [K_x] -> m ** (-1)
        K_y = 2.0 * math.pi / lambda_alpha / float(self.resolution[1])  # [K_y] -> m ** (-1)
        K = math.sqrt(K_x * K_x + K_y * K_y)
        analy_velo = torch.zeros(torch.Size([2, self.resolution[1], self.resolution[0]])).to(self.dtype).to(self.device)
        ones = torch.ones(torch.Size([1, self.resolution[1], self.resolution[0]])).to(self.dtype).to(self.device)
        x_linspace = torch.linspace(0, self.resolution[0] - 1, self.resolution[0]).to(self.device) * ones
        y_linspace = (torch.linspace(0, self.resolution[1] - 1, self.resolution[1]).to(self.device) *
                      ones.permute(0, 2, 1)).permute(0, 2, 1)
        analy_velo[0:1, ...] = -self.Vmax * K_y / K * torch.sin(K_y * y_linspace) * torch.cos(K_x * x_linspace) * \
                               math.exp(-self.visc * K * K * self.get_time())
        analy_velo[1:2, ...] = self.Vmax * K_x / K * torch.sin(K_x * x_linspace) * torch.cos(K_y * y_linspace) * \
                               math.exp(-self.visc * K * K * self.get_time())
        return (self.velocity - analy_velo).norm()

    def Record_velocity(self):
        self.velocity_before = self.velocity.clone()

    def rebounce_f(self):
        self.f[1, ...], self.f[3, ...] = torch.where(self.status == Flag.Wall, self.f[3, ...], self.f[1, ...]), \
                                         torch.where(self.status == Flag.Wall, self.f[1, ...], self.f[3, ...])
        self.f[2, ...], self.f[4, ...] = torch.where(self.status == Flag.Wall, self.f[4, ...], self.f[2, ...]), \
                                         torch.where(self.status == Flag.Wall, self.f[2, ...], self.f[4, ...])
        self.f[5, ...], self.f[7, ...] = torch.where(self.status == Flag.Wall, self.f[7, ...], self.f[5, ...]), \
                                         torch.where(self.status == Flag.Wall, self.f[5, ...], self.f[7, ...])
        self.f[6, ...], self.f[8, ...] = torch.where(self.status == Flag.Wall, self.f[8, ...], self.f[6, ...]), \
                                         torch.where(self.status == Flag.Wall, self.f[6, ...], self.f[8, ...])

    def Vonneumann_f(self, u_left, u_right, v_down, v_up):
        if self.left_boundary == 'V':
            self.velocity[0:1, ..., 0:1] = u_left
            self.velocity[1:2, ..., 0:1] = 0.0
            self.density[..., 0:1] = (self.f[0, :, 0:1] + self.f[2, :, 0:1] + self.f[4, :, 0:1] + \
                           2.0 * (self.f[3, :, 0:1] + self.f[6, :, 0:1] + self.f[7, :, 0:1])) / \
                           (1 - self.velocity[0:1, :, 0:1] / self.c)
            self.f[1, :, 0:1] = self.f[3, :, 0:1] + 2.0 / 3.0 / self.c * self.density[..., 0:1] * self.velocity[0:1, :, 0:1]
            self.f[8, :, 0:1] = self.f[6, :, 0:1] + 0.5 * (self.f[2, :, 0:1] - self.f[4, :, 0:1]) + \
                                self.density[..., 0:1] * self.velocity[0:1, :, 0:1] / (6.0 * self.c)
            self.f[5, :, 0:1] = self.f[7, :, 0:1] - 0.5 * (self.f[2, :, 0:1] - self.f[4, :, 0:1]) + \
                                self.density[..., 0:1] * self.velocity[0:1, :, 0:1] / (6.0 * self.c)
        if self.right_boundary == 'V':
            self.velocity[0:1, ..., -1:] = u_right
            self.velocity[1:2, ..., -1:] = 0.0
            self.density[..., -1:] = (self.f[0, :, -1:] + self.f[2, :, -1:] + self.f[4, :, -1:] + \
                                      2.0 * (self.f[1, :, -1:] + self.f[5, :, -1:] + self.f[8, :, -1:])) / \
                                     (1 - self.velocity[0:1, :, -1:] / self.c)
            self.f[3, :, -1:] = self.f[1, :, -1:] - 2.0 / 3.0 / self.c * self.density[..., -1:] * self.velocity[0:1, :,-1:]
            self.f[6, :, -1:] = self.f[8, :, -1:] - 0.5 * (self.f[2, :, -1:] - self.f[4, :, -1:]) + \
                                self.density[..., -1:] * self.velocity[0:1, :, -1:] / (6.0 * self.c)
            self.f[7, :, -1:] = self.f[5, :, -1:] + 0.5 * (self.f[2, :, -1:] - self.f[4, :, -1:]) + \
                                self.density[..., -1:] * self.velocity[0:1, :, -1:] / (6.0 * self.c)

        if self.down_boundary == 'V':
            pass
        if self.up_boundary == 'V':
            pass

    def outlet_f(self):
        if self.left_boundary == 'O':
            self.density[..., 0] = self.density[..., 1]
            self.velocity[..., 0] = self.velocity[..., 1]
            self.f[..., 0] = self.f[..., 1]
        if self.right_boundary == 'O':
            self.density[..., -1] = self.density[..., -2]
            self.velocity[..., -1] = self.velocity[..., -2]
            self.f[..., -1] = self.f[..., -2]

        if self.down_boundary == 'O':
            self.density[..., 0, :] = self.density[..., 1, :]
            self.velocity[..., 0, :] = self.velocity[..., 1, :]
            self.f[..., 0, :] = self.f[..., 1, :]
        if self.up_boundary == 'O':
            self.density[..., -1, :] = self.density[..., -2, :]
            self.velocity[..., -1, :] = self.velocity[..., -2, :]
            self.f[..., -1, :] = self.f[..., -2, :]

    def Streaming(self):
        fnew = torch.zeros(torch.Size([9, self.resolution[1], self.resolution[0]])).to(self.dtype).to(self.device)
        fnew[1, :, 1:], fnew[1, :, 0] = self.f[1, :, :-1], self.f[1, :, -1]  # ->right
        fnew[2, 1:, :], fnew[2, 0, :] = self.f[2, :-1, :], self.f[2, -1, :]  # ->up
        fnew[3, :, :-1], fnew[3, :, -1] = self.f[3, :, 1:], self.f[3, :, 0]  # ->left
        fnew[4, :-1, :], fnew[4, -1, :] = self.f[4, 1:, :], self.f[4, 0, :]  # ->down

        fnew[5, 1:, 1:], fnew[5, 1:, 0], fnew[5, 0, 1:], fnew[5, 0, 0] = \
            self.f[5, :-1, :-1], self.f[5, :-1, -1], self.f[5, -1, :-1], self.f[5, -1, -1]  # ->right, up

        fnew[6, 1:, :-1], fnew[6, 1:, -1], fnew[6, 0, :-1], fnew[6, 0, -1] = \
            self.f[6, :-1, 1:], self.f[6, :-1, 0], self.f[6, -1, 1:], self.f[6, -1, 0]  # ->left, up

        fnew[7, :-1, :-1], fnew[7, :-1, -1], fnew[7, -1, :-1], fnew[7, -1, -1] = \
            self.f[7, 1:, 1:], self.f[7, 1:, 0], self.f[7, 0, 1:], self.f[7, 0, 0]  # ->left, down

        fnew[8, :-1, 1:], fnew[8, :-1, 0], fnew[8, -1, 1:], fnew[8, -1, 0] = \
            self.f[8, 1:, :-1], self.f[8, 1:, -1], self.f[8, 0, :-1], self.f[8, 0, -1]  # ->right, down

        self.f[1:, ...] = fnew[1:, ...]

    def macro_process(self):
        self.density = self.f.sum(0).unsqueeze(0)
        self.velocity[0, ...] = (self.e[:, 0] * self.f.permute(2, 1, 0)).permute(2, 1, 0).sum(0).unsqueeze(0) / self.density
        self.velocity[1, ...] = (self.e[:, 1] * self.f.permute(2, 1, 0)).permute(2, 1, 0).sum(0).unsqueeze(0) / self.density

    def Collision(self):
        u_up = self.velocity + self.force
        self.calculate_feq()
        self.f = self.f * (1.0 - 1.0 / self.tau) + self.feq / self.tau
