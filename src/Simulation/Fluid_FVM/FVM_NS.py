import numpy as np
import math
import torch.nn.functional as F
import torch
from .FVM import FVM, Flag
from enum import Enum
from ..grid_fractory import *
from src.utils.utils import _save_img_2d
import os
import matplotlib.pyplot as plt


class FVM_NS(FVM):
    def __init__(self, core_component, boundary_band_="BBBBPP"):
        super(FVM_NS, self).__init__(core_component, boundary_band_)
        self.dimension = 2

        self.resolution = torch.Tensor([90, 30]).to(torch.int).to(self.device)
        self.bound_x = torch.Tensor([0, 0.3]).to(self.dtype).to(self.device)
        self.bound_y = torch.Tensor([0, 0.1]).to(self.dtype).to(self.device)
        self.dx_dy = 1.0 / self.resolution.to(self.dtype)
        self.dx_dy[0] *= (self.bound_x[1] - self.bound_x[0])
        self.dx_dy[1] *= (self.bound_y[1] - self.bound_y[0])

        self.real_range = self.resolution.clone().to(self.dtype)
        self.eps = 1e-3
        """
        This class is been implemented to solve an NS Equation.

        It is obvious, no diffusion term, no pressure projection term.
        """

    def get_MAC_grid(self, Fill_values=0):
        if self.dimension == 2:
            return Fill_values + torch.zeros(torch.Size([2, self.resolution[1] + 1, self.resolution[0] + 1])).to(
                self.dtype).to(self.device)
        elif self.dimension == 2:
            return Fill_values + torch.zeros(
                torch.Size([3, self.resolution[2] + 1, self.resolution[1] + 1, self.resolution[0] + 1])).to(
                self.dtype).to(self.device)

    def get_Central_grid(self, Fill_values=0):
        if self.dimension == 2:
            return Fill_values + torch.zeros(torch.Size([1, self.resolution[1], self.resolution[0]])).to(self.dtype).to(
                self.device)
        elif self.dimension == 3:
            return Fill_values + torch.zeros(
                torch.Size([1, self.resolution[2], self.resolution[1], self.resolution[0]])).to(self.dtype).to(self.device)

    def get_Central_Velocity_grid(self, Fill_values=0):
        if self.dimension == 2:
            return Fill_values + torch.zeros(torch.Size([2, self.resolution[1], self.resolution[0]])).to(self.dtype).to(
                self.device)
        elif self.dimension == 3:
            return Fill_values + torch.zeros(
                torch.Size([3, self.resolution[2], self.resolution[1], self.resolution[0]])).to(self.dtype).to(self.device)

    def get_velocity_center(self):
        """
        :return: Velocity: from MAC grid to Central grid
        """
        if self.dimension == 2:
            u = 0.5 * (self.velocity[0:1, :-1, :-1] + self.velocity[0:1, :-1, 1:])
            v = 0.5 * (self.velocity[1:2, :-1, :-1] + self.velocity[1:2, 1:, :-1])
            return torch.cat((u, v), dim=0)
        elif self.dimension == 3:
            u1 = 0.5 * (self.velocity[0:1, :-1, :-1, :-1] + self.velocity[0:1, :-1, :-1, 1:])
            u2 = 0.5 * (self.velocity[1:2, :-1, :-1, :-1] + self.velocity[1:2, :-1, 1:, :-1])
            u3 = 0.5 * (self.velocity[2:3, :-1, :-1, :-1] + self.velocity[2:3, 1:, :-1, :-1])
            return torch.cat((u1, u2, u3), dim=0)

    def initialization(self):
        self.visc = torch.Tensor([0.001]).to(self.dtype).to(self.device)

        self.t_max = 400
        self.dt = torch.Tensor([0.05]).to(self.dtype).to(self.device)

        self.velocity = self.get_MAC_grid(Fill_values=0)
        self.density = self.get_Central_grid(Fill_values=1)
        self.velocity_before = self.get_MAC_grid(Fill_values=0)
        self.p = self.get_Central_grid(Fill_values=0)
        self.C = self.get_Central_grid(Fill_values=0)
        self.C[..., 1:int(self.resolution[1]/2), :] = \
            torch.ones(torch.Size([1, int(self.resolution[1]/2) - 1, self.resolution[0]])).to(self.device).to(self.dtype)

        self.RK_order = 3
        self.Errors = []

        self.initialized = True

    def Cal_error(self):
        return (self.velocity - self.velocity_before).norm() * math.sqrt(self.dx_dy[0] * self.dx_dy[1])

    def Record_velocity(self):
        self.velocity_before = self.velocity.clone().detach()

    def reset_dt(self):
        pass

    def sharp_C(self):
        print("Step 0.5: Sharp C")
        eps = torch.max(self.dx_dy)
        ddt = 0.15 / eps * self.dx_dy[0] * self.dx_dy[1]

        n_x = self.get_Central_grid()
        n_y = self.get_Central_grid()
        n = self.get_Central_grid()

        fx_d = self.get_Central_grid()
        fy_d = self.get_Central_grid()
        fx_c = self.get_Central_grid()
        fy_c = self.get_Central_grid()

        dx, dy = self.dx_dy[0], self.dx_dy[1]

        for ic in range(20):
            if self.dimension == 2:
                # Diffusive
                extend_C = F.pad(self.C.unsqueeze(0), pad=(1, 1, 1, 1), mode="replicate").squeeze(0)
                fx_d = (extend_C[:, 1:-1, :-2] - 2 * self.C + extend_C[:, 1:-1, 2:]) / dx / dx
                fy_d = (extend_C[:, :-2, 1:-1] - 2 * self.C + extend_C[:, 2:, 1:-1]) / dy / dy
                self.C += ddt * (fx_d + fy_d)

                # Convective
                extend_C = F.pad(self.C.unsqueeze(0), pad=(1, 1, 1, 1), mode="replicate").squeeze(0)
                n_x = 0.5 * (extend_C[:, 1:-1, :-2] - extend_C[:, 1:-1, 2:])
                n_y = 0.5 * (extend_C[:, :-2, 1:-1] - extend_C[:, 2:, 1:-1])
                n_norm = torch.sqrt(n_x * n_x + n_y * n_y)
                n_x = n_x / n_norm
                n_y = n_y / n_norm
                fx = self.C * (1 - self.C) * n_x
                fy = self.C * (1 - self.C) * n_y
                extend_fx = F.pad(fx.unsqueeze(0), pad=(1, 1, 0, 0), mode="replicate").squeeze(0)
                extend_fy = F.pad(fy.unsqueeze(0), pad=(0, 0, 1, 1), mode="replicate").squeeze(0)
                fx_c = 0.5 * (extend_fx[:, :, :-2] - extend_fx[:, :, 2:])
                fy_c = 0.5 * (extend_fy[:, :-2, :] - extend_fy[:, 2:, :])
                self.C -= ddt * (fx_c + fy_c)
            elif self.dimension == 3:
                pass

    def solve_Color_step(self):
        print("Step 0: Solve Color")
        velocity_center = self.get_velocity_center()
        if self.dimension == 2:
            extend_C = F.pad(self.C.unsqueeze(0), pad=(1, 1, 1, 1), mode="replicate").squeeze(0)
            dc_dx = torch.where(velocity_center[0:1, ...] > 0,
                                extend_C[:, 1:-1, 1:-1] - extend_C[:, 1:-1, :-2],
                                extend_C[:, 1:-1, 2:] - extend_C[:, 1:-1, 1:-1])
            dc_dy = torch.where(velocity_center[1:2, ...] > 0,
                                extend_C[:, 1:-1, 1:-1] - extend_C[:, :-2, 1:-1],
                                extend_C[:, 2:, 1:-1] - extend_C[:, 1:-1, 1:-1])
            delta_C = torch.cat((dc_dx, dc_dy), dim=0)
        elif self.dimension == 3:
            extend_C = F.pad(self.C.unsqueeze(0), pad=(1, 1, 1, 1, 1, 1), mode="replicate").squeeze(0)
            dc_dx = torch.where(velocity_center[0:1, ...] > 0,
                                extend_C[:, 1:-1, 1:-1, 1:-1] - extend_C[:, 1:-1,  1:-1, :-2],
                                extend_C[:, 1:-1, 1:-1, 2:] - extend_C[:, 1:-1, 1:-1, 1:-1])
            dc_dy = torch.where(velocity_center[1:2, ...] > 0,
                                extend_C[:, 1:-1, 1:-1, 1:-1] - extend_C[:, 1:-1, :-2, 1:-1],
                                extend_C[:, 1:-1, 2:, 1:-1] - extend_C[:, 1:-1, 1:-1, 1:-1])
            dc_dz = torch.where(velocity_center[2:3, ...] > 0,
                                extend_C[:, 1:-1, 1:-1, 1:-1] - extend_C[:, :-2, 1:-1, 1:-1],
                                extend_C[:, 2:, 1:-1, 1:-1] - extend_C[:, 1:-1, 1:-1, 1:-1])
            delta_C = torch.cat((dc_dx, dc_dy, dc_dz), dim=0)
        C_C = (velocity_center * delta_C).sum(dim=0)
        self.C = self.C -self.dt * C_C
        self.sharp_C()

    def solve_linear_system(self, y_min, x_min, diag, x_plus, y_plus, b):
        # TODO: GS method
        N = len(diag)
        N_y = len(y_plus)
        fD = 1.0 / diag
        x = 0 * diag
        y_min = -fD[N - N_y:] * y_min
        x_min = -fD[1:] * x_min
        x_plus = -fD[:-1] * x_plus
        y_plus = -fD[:N_y] * y_plus
        b = fD * b
        err = 1.0
        k = 0
        iteration_T = 6000
        while err > self.eps and k < iteration_T:
            k += 1
            x_before = x.clone().detach()
            Bx = 0 * diag
            Bx[0:N - 1] += x_plus * x[1:N]
            Bx[1:] += x_min * x[0:N - 1]
            Bx[0:N_y] += y_plus * x[N - N_y:]
            Bx[N - N_y:] += y_min * x[0:N_y]
            x = Bx + b
            err = (x - x_before).norm()
        if k >= iteration_T:
            print("Solving linear systems: The iteration time has out the bound with error: ", err)
        return x

    def solve_u_star(self):
        print("Step 1: Solve u^star")
        velocity_center = self.get_velocity_center()
        if self.dimension == 2:
            dx, dy = self.dx_dy[0], self.dx_dy[1]

            du_dx = self.velocity[0:1, :-1, 1:] - self.velocity[0:1, :-1, :-1]
            u_pad = F.pad(velocity_center[0:1, ...].unsqueeze(0), pad=(1, 1, 0, 0), mode="constant").squeeze(0)
            du_dy = 0.5 * (u_pad[..., 2:] - u_pad[..., :-2])
            delta_u = torch.cat((du_dx, du_dy), dim=0)
            c_u = self.density * (velocity_center * delta_u).sum(dim=0)

            v_pad = F.pad(velocity_center[1:2, ...].unsqueeze(0), pad=(0, 0, 1, 1), mode="constant").squeeze(0)
            dv_dx = 0.5 * (v_pad[..., 2:, :] - v_pad[..., :-2, :])
            dv_dy = self.velocity[1:2, 1:, :-1] - self.velocity[1:2, :-1, :-1]
            delta_v = torch.cat((dv_dx, dv_dy), dim=0)
            c_v = self.density * (velocity_center * delta_v).sum(dim=0)

            map_A = self.density.view(-1) *(1.0 / self.dt + 2.0 * self.visc * (1.0 / dx / dx + 1.0 / dy / dy))

            # TODO: Here density is constant, so it does not matter, fix it later.
            map_A_x_plus = (-F.pad(self.density[:, :, 1:].unsqueeze(0), pad=(0, 1, 0, 0), mode="constant").squeeze(0) *
                            (self.visc / dx / dx)).view(-1)[:-1]
            map_A_x_minus = (-F.pad(self.density[:, :, :-1].unsqueeze(0), pad=(1, 0, 0, 0), mode="constant").squeeze(0) *
                             (self.visc / dx / dx)).view(-1)[1:]

            map_A_y_plus = (-F.pad(self.density[:, 1:, :].unsqueeze(0), pad=(0, 0, 0, 1), mode="constant").squeeze(0) *
                            (self.visc / dy / dy)).view(-1)[:-self.resolution[1]]
            map_A_y_minus = (-F.pad(self.density[:, :-1, :].unsqueeze(0), pad=(0, 0, 1, 0), mode="constant").squeeze(0) *
                             (self.visc / dy / dy)).view(-1)[self.resolution[1]:]

            map_bu = (-c_u + self.density * velocity_center[0:1, ...] / self.dt).view(-1)
            map_bv = (-c_v + self.density * velocity_center[1:2, ...] / self.dt + self.C).view(-1)

            u_star = self.solve_linear_system(map_A_y_minus, map_A_x_minus, map_A, map_A_x_plus, map_A_y_plus, map_bu)
            v_star = self.solve_linear_system(map_A_y_minus, map_A_x_minus, map_A, map_A_x_plus, map_A_y_plus, map_bv)
            u_star = u_star.view(1, self.resolution[1], self.resolution[0])
            v_star = v_star.view(1, self.resolution[1], self.resolution[0])
            u_star = F.pad(u_star.unsqueeze(0), pad=(1, 1, 0, 0), mode="constant").squeeze(0)
            v_star = F.pad(v_star.unsqueeze(0), pad=(0, 0, 1, 1), mode="constant").squeeze(0)
            u_star = 0.5 * (u_star[..., :-1] + u_star[..., 1:])
            v_star = 0.5 * (v_star[..., :-1, :] + v_star[..., 1:, :])
            u_star = F.pad(u_star.unsqueeze(0), pad=(0, 0, 0, 1), mode="constant").squeeze(0)
            v_star = F.pad(v_star.unsqueeze(0), pad=(0, 1, 0, 0), mode="constant").squeeze(0)
            self.velocity = torch.cat((u_star, v_star), dim=0)

        elif self.dimension == 3:
            pass

    def solve_poisson(self):
        print("Step 2: Solving poisson equation for pressure")
        dx, dy = self.dx_dy[0], self.dx_dy[1]
        map_bp = self.density / self.dt * (
            (self.velocity[0:1, :-1, 1:] - self.velocity[0:1, :-1, :-1]) / dx +
            (self.velocity[1:2, 1:, :-1] - self.velocity[1:2, :-1, :-1]) / dy)
        map_bp = map_bp.view(-1)
        ones = self.get_Central_grid(Fill_values=1)
        map_p = -4 * ones.view(-1)

        map_p_x_plus = F.pad(ones[:, :, 1:].unsqueeze(0), pad=(0, 1, 0, 0), mode="constant").squeeze(0).view(-1)[:-1]
        map_p_x_minus = F.pad(ones[:, :, :-1].unsqueeze(0), pad=(1, 0, 0, 0), mode="constant").squeeze(0).view(-1)[1:]
        map_p_y_plus = F.pad(ones[:, 1:, :].unsqueeze(0), pad=(0, 0, 0, 1), mode="constant").squeeze(0).view(-1)[:-self.resolution[1]]
        map_p_y_minus = F.pad(ones[:, :-1, :].unsqueeze(0), pad=(0, 0, 1, 0), mode="constant").squeeze(0).view(-1)[self.resolution[1]:]

        self.p = self.solve_linear_system(map_p_y_minus, map_p_x_minus, map_p, map_p_x_plus, map_p_y_plus, map_bp)
        self.p =self.p.view(1, self.resolution[1], self.resolution[0])

    def solve_u_next(self):
        print("Step 3: project pressure onto velocity")
        dx, dy = self.dx_dy[0], self.dx_dy[1]
        extend_p = F.pad(self.p.unsqueeze(0), pad=(1, 1, 1, 1), mode="constant").squeeze(0)
        self.velocity[0:1, :-1, :] += (self.dt / dx) * (
                extend_p[..., 1:-1, 1:] - extend_p[..., 1:-1, :-1]) / \
                                      F.pad(self.density.unsqueeze(0), pad=(0, 1, 0, 0), mode="replicate").squeeze(0)
        self.velocity[1:2, :, :-1] += (self.dt / dy) * (
                extend_p[..., 1:, 1:-1] - extend_p[..., :-1, 1:-1]) / \
                                      F.pad(self.density.unsqueeze(0), pad=(0, 0, 0, 1), mode="replicate").squeeze(0)

    def solve_p_next(self):
        pass

    def visualization(self):
        _save_img_2d(self.velocity.permute(2, 1, 0).cpu().numpy(),
                     os.path.join(self.core_component.root_path, "data", "ACFD_homework", "Assignment1", "Velocity",
                                  str(
                                      int(self.current_time)) + ".png"))

    def compute(self):
        self.Record_velocity()

        self.reset_dt()
        self.solve_Color_step()
        self.solve_u_star()
        self.solve_poisson()
        self.solve_u_next()
        self.solve_p_next()

        if self.current_time % self.Nwri == 0:
            self.visualization()

    def update(self):
        if self.initialized:
            self.compute()
            self.time_tick()
            print("====================================================")
            print("Simulation step: ", self.get_time(), "; Error: ", self.Cal_error())
            self.Errors.append(self.Cal_error())
            if self.if_dump:
                pass