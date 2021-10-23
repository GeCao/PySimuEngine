import numpy as np
import math
import torch.nn.functional as F
import torch
from .FVM import FVM, Flag
from enum import Enum
from src.utils.utils import _save_img_2d
import os
import matplotlib.pyplot as plt


class Init_Velocity(Enum):
    smooth_sinusoidal_initial_condition = 0
    discontinuous_top_hat_initial_condition = 1


class FVM_AdvectionEquation(FVM):
    def __init__(self, core_component, boundary_band_="PPPPPP"):
        super(FVM_AdvectionEquation, self).__init__(core_component, boundary_band_)
        self.dimension = 2

        self.resolution = torch.Tensor([50, 50]).to(torch.int).to(self.device)
        self.bound_x = torch.Tensor([0, 1.0]).to(self.dtype).to(self.device)
        self.bound_y = torch.Tensor([0, 1.0]).to(self.dtype).to(self.device)
        self.dx_dy = 1.0 / self.resolution.to(self.dtype)
        self.dx_dy[0] *= (self.bound_x[1] - self.bound_x[0])
        self.dx_dy[1] *= (self.bound_y[1] - self.bound_y[0])

        self.real_range = self.resolution.clone().to(self.dtype)
        """
        This class is been implemented to solve an Advection Equation.
        
        \frac{u}{t} + \frac{v_1 u}{x} + \frac{v_2 u}{y} = 0
        u(x, t = 0) = u_0(x)
        
        It is obvious, no diffusion term, no pressure projection term.
        """

    def initialization(self):
        self.advection_velocity = torch.Tensor([1.0, 0.5]).to(self.dtype).to(self.device)  # [u, v]
        self.velocity = torch.zeros(torch.Size([2, self.resolution[1], self.resolution[0]])).to(self.dtype).to(
            self.device)
        self.density = torch.ones(torch.Size([1, self.resolution[1], self.resolution[0]])).to(self.dtype).to(
            self.device)
        self.velocity_before = torch.zeros(torch.Size([2, self.resolution[1], self.resolution[0]])).to(self.dtype).to(
            self.device)

        self.init_velo_choice = Init_Velocity.discontinuous_top_hat_initial_condition
        self.RK_order = 3
        self.dt = 0.5 * torch.min(self.dx_dy) / torch.max(self.advection_velocity)

        x_linspace = torch.linspace(self.bound_x[0] + 0.5 * self.dx_dy[0], self.bound_x[1] - 0.5 * self.dx_dy[0],
                                    self.resolution[0]).to(
            self.device) * self.density
        y_linspace = (torch.linspace(self.bound_y[0] + 0.5 * self.dx_dy[1], self.bound_y[1] - 0.5 * self.dx_dy[1],
                                     self.resolution[1]).to(
            self.device) * self.density.permute(0, 2, 1)).permute(0, 2, 1)
        self.advect_particle_grid = torch.cat((x_linspace, y_linspace), dim=0)
        if self.init_velo_choice == Init_Velocity.smooth_sinusoidal_initial_condition:
            # smooth sinusoidal initial condition
            self.velocity[0:1, ...] = torch.sin(2.0 * math.pi * x_linspace) * torch.sin(2.0 * math.pi * y_linspace)
        elif self.init_velo_choice == Init_Velocity.discontinuous_top_hat_initial_condition:
            #  discontinuous top-hat initial condition
            self.velocity = torch.where(torch.abs(self.advect_particle_grid - 0.5) < 0.25,
                                        torch.ones(torch.Size([2, self.resolution[1], self.resolution[0]])).to(
                                            self.dtype).to(self.device),
                                        self.velocity.detach())
        self.Errors = []

        self.initialized = True

    def get_analytical_sol(self):
        # Get our grid
        grid_one_time_x = self.advect_particle_grid[0:1, ...] - self.advection_velocity[0] * self.dt
        grid_one_time_y = self.advect_particle_grid[1:2, ...] - self.advection_velocity[1] * self.dt

        # Apply periodical condition
        box_len_x, box_len_y = self.bound_x[1] - self.bound_x[0], self.bound_y[1] - self.bound_y[0]
        grid_one_time_x = torch.where(grid_one_time_x < self.bound_x[0], grid_one_time_x + box_len_x, grid_one_time_x)
        grid_one_time_x = torch.where(grid_one_time_x > self.bound_x[1], grid_one_time_x - box_len_x, grid_one_time_x)
        grid_one_time_y = torch.where(grid_one_time_y < self.bound_y[0], grid_one_time_y + box_len_y, grid_one_time_y)
        grid_one_time_y = torch.where(grid_one_time_y > self.bound_y[1], grid_one_time_y - box_len_y, grid_one_time_y)

        self.advect_particle_grid = torch.cat((grid_one_time_x, grid_one_time_y), dim=0)
        velo = torch.zeros(torch.Size([2, self.resolution[1], self.resolution[0]])).to(self.dtype).to(self.device)
        if self.init_velo_choice == Init_Velocity.smooth_sinusoidal_initial_condition:
            # smooth sinusoidal initial condition
            velo[0:1, ...] = torch.sin(2.0 * math.pi * grid_one_time_x) * torch.sin(2.0 * math.pi * grid_one_time_y)
        elif self.init_velo_choice == Init_Velocity.discontinuous_top_hat_initial_condition:
            #  discontinuous top-hat initial condition
            velo = torch.where(torch.abs(self.advect_particle_grid - 0.5) < 0.25,
                               torch.ones(torch.Size([2, self.resolution[1], self.resolution[0]])).to(self.dtype).to(
                                   self.device),
                               torch.zeros(torch.Size([2, self.resolution[1], self.resolution[0]])).to(self.dtype).to(
                                   self.device))
        return velo

    def get_increment(self, velocity_):
        F_x = self.advection_velocity[0] * velocity_
        F_y = self.advection_velocity[1] * velocity_

        F_right = F.pad(F_x.unsqueeze(0), pad=(0, 1, 0, 0), mode='constant').squeeze(0)
        F_left = F.pad(F_x.unsqueeze(0), pad=(1, 0, 0, 0), mode='constant').squeeze(0)
        F_up = F.pad(F_y.unsqueeze(0), pad=(0, 0, 0, 1), mode='constant').squeeze(0)
        F_down = F.pad(F_y.unsqueeze(0), pad=(0, 0, 1, 0), mode='constant').squeeze(0)

        if self.left_boundary == 'P':
            F_left[..., 0:1] = F_x[..., -1:]
        if self.right_boundary == 'P':
            F_right[..., -1:] = F_x[..., 0:1]
        if self.down_boundary == 'P':
            F_down[..., 0:1, :] = F_y[..., -1:, :]
        if self.up_boundary == 'P':
            F_up[..., -1:, :] = F_y[..., 0:1, :]

        # upwind scheme
        if self.advection_velocity[0] > 0:
            # back diff
            F_x_upwind = F_left  # torch.Size([1, 50, 51])
        else:
            F_x_upwind = F_right  # torch.Size([1, 50, 51])
        if self.advection_velocity[1] > 0:
            # back diff
            F_y_upwind = F_down  # torch.Size([1, 51, 50])
        else:
            F_y_upwind = F_up  # torch.Size([1, 51, 50])

        increment = -((F_x_upwind[..., 1:] - F_x_upwind[..., :-1]) / self.dx_dy[0] +
                      (F_y_upwind[..., 1:, :] - F_y_upwind[..., :-1, :]) / self.dx_dy[1])
        return increment

    def solve_Advection_step(self):
        if self.RK_order == 1:
            k1_ = self.get_increment(velocity_=self.velocity)
            increment = k1_
            self.velocity += self.dt * increment
        if self.RK_order == 2:
            k1_ = self.get_increment(velocity_=self.velocity)
            velocity_half_time = self.velocity + 0.5 * self.dt * k1_
            k2_ = self.get_increment(velocity_=velocity_half_time)
            increment = k2_
            self.velocity += self.dt * increment
        elif self.RK_order == 3:
            velo = self.velocity + self.get_increment(velocity_=self.velocity) * (self.dt / 3.0)
            velo = self.velocity + self.get_increment(velocity_=velo) * (self.dt / 2.0)
            self.velocity = self.velocity + self.get_increment(velocity_=velo) * (self.dt / 1.0)
        elif self.RK_order == 4:
            velo = self.velocity + self.get_increment(velocity_=self.velocity) * (self.dt / 4.0)
            velo = self.velocity + self.get_increment(velocity_=velo) * (self.dt / 3.0)
            velo = self.velocity + self.get_increment(velocity_=velo) * (self.dt / 2.0)
            self.velocity = self.velocity + self.get_increment(velocity_=velo) * (self.dt / 1.0)

    def Record_velocity(self):
        self.velocity_before = self.velocity.clone()

    def visualization(self):
        _save_img_2d(self.velocity.permute(2, 1, 0).cpu().numpy(),
                     os.path.join(self.core_component.root_path, "data", "ACFD_homework", "Assignment1", "Velocity",
                                  str(
                                      int(self.current_time)) + ".png"))
        _save_img_2d(self.velocity_before.permute(2, 1, 0).cpu().numpy(),
                     os.path.join(self.core_component.root_path, "data",
                                  "ACFD_homework", "Assignment1", "Analysis_Velocity", str(
                             int(self.current_time)) + ".png"))

    def Cal_error(self):
        return (self.velocity - self.velocity_before).norm() * math.sqrt(self.dx_dy[0] * self.dx_dy[1])

    def update(self):
        if self.initialized:
            self.compute()
            self.time_tick()
            print("====================================================")
            print("Simulation step: ", self.get_time(), "; Error: ", self.Cal_error())
            self.Errors.append(self.Cal_error())
            if self.if_dump:
                pass
            if self.current_time == 3000:
                N_steps = len(self.Errors)
                plt.figure(1)
                plt.title("Convergence Analysis")
                plt.xlabel("log(Steps)")
                plt.ylabel("log(Errors)")
                plt.plot(np.log([i + 1 for i in range(N_steps)]), -0.72 * np.log([i + 1 for i in range(N_steps)]),
                         'g', label="Reference: Algebraic convergence with rate 0.72")
                plt.plot(np.log([i + 1 for i in range(N_steps)]), [(math.log(self.Errors[i]) if self.Errors[i] > 0 else 0) for i in range(N_steps)],
                         'r--', label="Convergence of errors")
                plt.legend()
                plt.show()
