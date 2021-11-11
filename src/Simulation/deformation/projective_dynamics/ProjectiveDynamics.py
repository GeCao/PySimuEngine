import numpy as np
import torch
import time, os
from enum import Enum
from src.utils.tet_mesh import read_mesh, read_d


class ConstraintType(Enum):
    CAttachment = 0
    CSpring = 1
    CTet = 2


class Constraint:
    def __init__(self, weight):
        self.p = None
        self.weight = weight

    def get_constraint_vert_index(self):
        return self.p

    def get_constraint_weight(self):
        return self.weight

    def get_fixed_point(self):
        pass


class ConstraintAttachment(Constraint):
    def __init__(self, cell, weight, pos):
        super(ConstraintAttachment, self).__init__(weight)
        self.p = cell
        self.pos = pos

    def get_fixed_point(self):
        return self.pos


class ConstraintSpring(Constraint):
    def __init__(self, cell, weight, rest_length):
        super(ConstraintSpring, self).__init__(weight)
        self.p = (cell[0], cell[1], cell[2])
        self.rest_length = rest_length

    def get_rest_length(self):
        return self.rest_length


class ConstraintTet(Constraint):
    def __init__(self, cell, weight):
        super(ConstraintTet, self).__init__(weight)
        self.p = (cell[0], cell[1], cell[2], cell[3])
        self.Dr = None
        self.Dr_inv = None

    def set_Dr_and_its_inv(self, pos):
        self.Dr = torch.stack((pos[3 * self.p[0]:3 * (self.p[0] + 1)] - pos[3 * self.p[3]:3 * (self.p[3] + 1)],
                               pos[3 * self.p[1]:3 * (self.p[1] + 1)] - pos[3 * self.p[3]:3 * (self.p[3] + 1)],
                               pos[3 * self.p[2]:3 * (self.p[2] + 1)] - pos[3 * self.p[3]:3 * (self.p[3] + 1)]), dim=1)
        self.Dr_inv = torch.linalg.inv(self.Dr)

    def get_deformation_gradient(self, curr_pos_vec):
        Ds = torch.stack((curr_pos_vec[0:3] - curr_pos_vec[9:12],
                          curr_pos_vec[3:6] - curr_pos_vec[9:12],
                          curr_pos_vec[6:9] - curr_pos_vec[9:12]), dim=1)
        return torch.mm(self.Dr_inv, Ds)

    def update_volume_preserving_vertex_positions(self, curr_pos_vec):
        has_collide = False
        F_ = self.get_deformation_gradient(curr_pos_vec)
        U_, SIGMA_, V_ = torch.svd(F_)
        det_F = torch.linalg.det(F_)
        if det_F < 0.0:
            has_collide = True
            SIGMA_[2] *= -1.0
            SIGMA_, _ = torch.sort(SIGMA_, dim=0, descending=True)
        SIGMA_ = torch.clamp(SIGMA_, min=0.95, max=1.05)

        F_ = torch.mm(torch.mm(U_, torch.diag(SIGMA_)), V_.t())
        deformed_basis = torch.mm(F_, self.Dr)
        tet_centroid = curr_pos_vec.reshape(4, 3).sum(dim=0) / 4.0
        curr_pos_vec[9:12] = tet_centroid - deformed_basis.sum(dim=0) / 4.0
        curr_pos_vec[0:3] = curr_pos_vec[9:12] + deformed_basis[:, 0]
        curr_pos_vec[3:6] = curr_pos_vec[9:12] + deformed_basis[:, 1]
        curr_pos_vec[6:9] = curr_pos_vec[9:12] + deformed_basis[:, 2]


class ProjectiveDynamics:
    def __init__(self, simulator_component):
        self.simulator_component = simulator_component
        self.animation_actor = None  # Every simulator need at least an actor so that to show them on the screen
        self.dtype = torch.float32
        self.device = "cpu"
        self.pos = None  # Position data, usually [m*3]
        self.cells = None  # faces data, usually [e*3]
        self.surface_ind = None

        self.velocity = None  # [3*m,]
        self.mass_mat = None  # [3*m , 3*m]
        self.f_ext = [0, -9.8, 0]
        self.A_attachment = None  # [3, 3]
        self.A_spring = None  # [6, 6]
        self.A_tet = None  # [12, 12]
        self.B_attachment = None  # = A_attachment
        self.B_spring = None  # = A_spring
        self.B_tet = None  # = A_tet
        # self.S_mat = None  # [3/6/12, 3*m]
        self.LU_factored = None
        self.weights = []

        # Parameters
        self.h_step = 0.01
        self.max_iter = 8
        self.max_steps = 25
        self.step = 0
        self.constraint_type = None
        self.constraints = []

        self.data_path = None

        self.initialized = False

    def initialization(self, resource_name, resource_type="tet"):
        # Only do this AFTER resource_component has loaded every thing.
        self.animation_actor = self.simulator_component.core_component.resource_component.get_animation_resource(
            resource_name)
        if self.animation_actor is None:
            self.simulator_component.core_component.log_component.ErrorLog("PD Method: , No surface mesh info loaded!")
        self.data_path = self.simulator_component.animation_path
        self.pos, self.cells = read_mesh(
            os.path.join(self.data_path, resource_name + "/tetmesh/" + resource_name + ".bin"))
        self.surface_ind = read_d(os.path.join(self.data_path, resource_name + "/tetmesh/surface.bin"))
        self.data_path = os.path.join(self.simulator_component.animation_path, resource_name + "/surmesh")
        if not os.path.exists(os.path.join(self.data_path, "simulated")):
            os.mkdir(os.path.join(self.data_path, "simulated"))
        self.data_path = os.path.join(self.data_path, "simulated")

        if self.cells.shape[1] == 3:
            self.constraint_type = ConstraintType.CSpring
        elif self.cells.shape[1] == 4:
            self.constraint_type = ConstraintType.CTet
            all_vol = 0.0
            self.weights = np.zeros((self.cells.shape[0],), dtype=np.float32)
            for i in range(len(self.cells)):
                p0, p1, p2, p3 = self.cells[i, 0], self.cells[i, 1], self.cells[i, 2], self.cells[i, 3]
                vec1 = self.pos[p1, :] - self.pos[p0, :]
                vec2 = self.pos[p2, :] - self.pos[p0, :]
                normal = np.cross(vec1, vec2)
                area = np.sqrt(normal.dot(normal)) / 2.0
                normal = normal / (2.0 * area)
                vec3 = self.pos[p3, :] - self.pos[p0, :]
                dist = np.abs(vec3.dot(normal))
                self.weights[i] = dist * area / 3.0
                all_vol += self.weights[i]
            self.weights = self.weights / all_vol
        else:
            self.simulator_component.core_component.log_component.ErrorLog(
                "Read ambiguous cell information for deformation simulator {}'s initialization".format(resource_name))
        self.pos = self.pos.reshape(self.pos.shape[0] * self.pos.shape[1])
        self.pos = torch.from_numpy(self.pos).to(self.dtype).to(self.device)
        self.velocity = torch.zeros(self.pos.shape).to(self.dtype).to(self.device)
        self.f_ext = torch.Tensor(self.f_ext).to(self.dtype).to(self.device).repeat(self.pos.shape[0] // 3)

        self.set_constraints_from_faces(resource_type)
        self.mass_mat = np.eye(self.pos.shape[0])
        self.mass_mat = torch.from_numpy(self.mass_mat).to(self.dtype).to(self.device)
        self.set_A_B_matrix()
        self.set_LHS_matrix()

        self.initialized = True

    def is_obj_based(self):
        return True

    def set_constraints_from_faces(self, resource_type):
        if resource_type == 'spring':
            # TODO: Not implemented completed
            weight = 0.1
            self.constraints = [ConstraintSpring(self.cells[i], weight, rest_length=0.1) for i in range(self.cells.shape[0])]
        elif resource_type == 'tet':
            self.constraints = [ConstraintTet(self.cells[i], self.weights[i]) for i in range(self.cells.shape[0])]
            for i in range(len(self.constraints)):
                self.constraints[i].set_Dr_and_its_inv(self.pos)

    def set_A_B_matrix(self):
        self.A_attachment = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        sv1, sv2 = 0.5, -0.5
        self.A_spring = np.array([[sv1, 0, 0, sv2, 0, 0],
                                  [0, sv1, 0, 0, sv2, 0],
                                  [0, 0, sv1, 0, 0, sv2],
                                  [sv2, 0, 0, sv1, 0, 0],
                                  [0, sv2, 0, 0, sv1, 0],
                                  [0, 0, sv2, 0, 0, sv1]])

        v1, v2 = 2.0 / 3.0, -1.0 / 3.0
        self.A_tet = np.zeros((12, 12))
        self.A_tet[0, 0] = self.A_tet[1, 1] = self.A_tet[2, 2] = self.A_tet[3, 3] = v1
        self.A_tet[0, 4] = self.A_tet[1, 5] = self.A_tet[2, 6] = self.A_tet[3, 7] = v2
        self.A_tet[0, 8] = self.A_tet[1, 9] = self.A_tet[2, 10] = self.A_tet[3, 11] = v2

        self.A_tet[4, 0] = self.A_tet[5, 1] = self.A_tet[6, 2] = self.A_tet[7, 3] = v2
        self.A_tet[4, 4] = self.A_tet[5, 5] = self.A_tet[6, 6] = self.A_tet[7, 7] = v1
        self.A_tet[4, 8] = self.A_tet[5, 9] = self.A_tet[6, 10] = self.A_tet[7, 11] = v2

        self.A_tet[8, 0] = self.A_tet[9, 1] = self.A_tet[10, 2] = self.A_tet[11, 3] = v2
        self.A_tet[8, 4] = self.A_tet[9, 5] = self.A_tet[10, 6] = self.A_tet[11, 7] = v2
        self.A_tet[8, 8] = self.A_tet[9, 9] = self.A_tet[10, 10] = self.A_tet[11, 11] = v1

        self.B_attachment = 1.0 * self.A_attachment
        self.B_spring = 1.0 * self.A_spring
        self.B_tet = 1.0 * self.A_tet

        self.A_attachment = torch.from_numpy(self.A_attachment).to(self.dtype).to(self.device)
        self.A_spring = torch.from_numpy(self.A_spring).to(self.dtype).to(self.device)
        self.A_tet = torch.from_numpy(self.A_tet).to(self.dtype).to(self.device)
        self.B_attachment = torch.from_numpy(self.B_attachment).to(self.dtype).to(self.device)
        self.B_spring = torch.from_numpy(self.B_spring).to(self.dtype).to(self.device)
        self.B_tet = torch.from_numpy(self.B_tet).to(self.dtype).to(self.device)

    def get_S_matrix(self, constraint_):
        if self.constraint_type.value == ConstraintType.CAttachment.value:
            S_mat = torch.zeros((3, self.pos.shape[0])).to(self.dtype).to(self.device)
            p0 = constraint_.get_constraint_vert_index()
            S_mat[0, 3 * p0 + 0] = 1
            S_mat[1, 3 * p0 + 1] = 1
            S_mat[2, 3 * p0 + 2] = 1
            return S_mat
        elif self.constraint_type.value == ConstraintType.CSpring.value:
            S_mat = torch.zeros((6, self.pos.shape[0])).to(self.dtype).to(self.device)
            p0, p1 = constraint_.get_constraint_vert_index()
            S_mat[0, 3 * p0 + 0] = 1
            S_mat[1, 3 * p0 + 1] = 1
            S_mat[2, 3 * p0 + 2] = 1
            S_mat[3, 3 * p1 + 0] = 1
            S_mat[4, 3 * p1 + 1] = 1
            S_mat[5, 3 * p1 + 2] = 1
            return S_mat
        elif self.constraint_type.value == ConstraintType.CTet.value:
            S_mat = torch.zeros((12, self.pos.shape[0])).to(self.dtype).to(self.device)
            p0, p1, p2, p3 = constraint_.get_constraint_vert_index()
            S_mat[0, 3 * p0 + 0] = 1
            S_mat[1, 3 * p0 + 1] = 1
            S_mat[2, 3 * p0 + 2] = 1
            S_mat[3, 3 * p1 + 0] = 1
            S_mat[4, 3 * p1 + 1] = 1
            S_mat[5, 3 * p1 + 2] = 1
            S_mat[6, 3 * p2 + 0] = 1
            S_mat[7, 3 * p2 + 1] = 1
            S_mat[8, 3 * p2 + 2] = 1
            S_mat[9, 3 * p3 + 0] = 1
            S_mat[10, 3 * p3 + 1] = 1
            S_mat[11, 3 * p3 + 2] = 1
            return S_mat
        else:
            return None

    def get_A_matrix(self, constraint_):
        if self.constraint_type.value == ConstraintType.CAttachment.value:
            return self.A_attachment
        elif self.constraint_type.value == ConstraintType.CSpring.value:
            return self.A_spring
        elif self.constraint_type.value == ConstraintType.CTet.value:
            return self.A_tet
        else:
            return None

    def get_B_matrix(self, constraint_):
        if self.constraint_type.value == ConstraintType.CAttachment.value:
            return self.B_attachment
        elif self.constraint_type.value == ConstraintType.CSpring.value:
            return self.B_spring
        elif self.constraint_type.value == ConstraintType.CTet.value:
            return self.B_tet
        else:
            return None

    def set_LHS_matrix(self):
        # you only need to do it once in the initialization step
        lhs_mat = self.mass_mat / (self.h_step * self.h_step)
        for i, constraint_ in enumerate(self.constraints):
            weight = constraint_.get_constraint_weight()
            S_mat = self.get_S_matrix(constraint_)
            A_mat = self.get_A_matrix(constraint_)
            AS_mat = torch.mm(A_mat, S_mat)
            lhs_mat = lhs_mat + weight * torch.mm(torch.transpose(AS_mat, 1, 0), AS_mat)
        self.LU_factored = torch.lu(
            lhs_mat)  # every time you wanna solve LSE, just use: torch.lu_solve(b, *self.LU_factored)

    def get_RHS_vec_in_sum_part(self, constraint_):
        # Unlike the LHS Matrix computed before, you have to do this in every iteration
        weight = constraint_.get_constraint_weight()
        # S_mat = self.get_S_matrix(constraint_)  # [12, 3m]
        A_mat = self.get_A_matrix(constraint_)  # [12, 12]
        B_mat = self.get_B_matrix(constraint_)  # [12, 12]
        At_B_mat = torch.mm(A_mat.t(), B_mat)  # [12, 12]

        return weight * At_B_mat

    def apply_external_force(self, s_n):
        # TODO: Did not implement this external forces
        if self.step % 10 == 0:
            self.f_ext = -self.f_ext
        return s_n + self.h_step * self.h_step * self.f_ext

    def update(self):
        if self.initialized:
            s_n = self.pos + self.h_step * self.velocity
            s_n = self.apply_external_force(s_n)
            self.velocity = -self.pos / self.h_step
            self.pos = s_n  # Warm start
            # Local solve step
            for iter in range(self.max_iter):
                print(iter)
                t1 = time.time()
                # rhs_vec = torch.mm(self.mass_mat, s_n.unsqueeze(1)).squeeze(1) / (self.h_step * self.h_step)
                # TODO: give up generalization for run faster
                rhs_vec = s_n / (self.h_step * self.h_step)
                for constraint_ in self.constraints:
                    pi = None
                    if self.constraint_type.value == ConstraintType.CAttachment.value:
                        pi = torch.zeros((3,)).to(self.dtype).to(self.device)
                        pi[0:3] = constraint_.get_fixed_point()

                        p0 = constraint_.get_constraint_vert_index()
                        w_At_B = torch.mm(self.get_RHS_vec_in_sum_part(constraint_), pi.unsqueeze(1)).squeeze(1)
                        rhs_vec[3 * p0:3 * (p0 + 1)] += w_At_B[0:3]
                    elif self.constraint_type.value == ConstraintType.CSpring.value:
                        p0, p1 = constraint_.get_constraint_vert_index()
                        curr_vec = self.pos[3 * p0:3 * (p0 + 1)] - self.pos[3 * p1:3 * (p1 + 1)]
                        curr_stretch = torch.norm(curr_vec, p=2) - constraint_.get_rest_length()
                        curr_vec = (0.5 * curr_stretch) * (curr_vec / curr_vec.norm(p=2))

                        pi = torch.zeros((6,)).to(self.dtype).to(self.device)
                        pi[0:3] = self.pos[3 * p0:3 * (p0 + 1)] - curr_vec
                        pi[3:6] = self.pos[3 * p1:3 * (p1 + 1)] + curr_vec

                        w_At_B = torch.mm(self.get_RHS_vec_in_sum_part(constraint_), pi.unsqueeze(1)).squeeze(1)
                        rhs_vec[3 * p0:3 * (p0 + 1)] += w_At_B[0:3]
                        rhs_vec[3 * p1:3 * (p1 + 1)] += w_At_B[3:6]
                    elif self.constraint_type.value == ConstraintType.CTet.value:
                        curr_vec = torch.zeros((12,)).to(self.dtype).to(self.device)
                        p0, p1, p2, p3 = constraint_.get_constraint_vert_index()
                        curr_vec[0:3] = self.pos[3 * p0:3 * (p0 + 1)]
                        curr_vec[3:6] = self.pos[3 * p1:3 * (p1 + 1)]
                        curr_vec[6:9] = self.pos[3 * p2:3 * (p2 + 1)]
                        curr_vec[9:12] = self.pos[3 * p3:3 * (p3 + 1)]
                        constraint_.update_volume_preserving_vertex_positions(curr_vec)
                        pi = curr_vec

                        w_At_B = torch.mm(self.get_RHS_vec_in_sum_part(constraint_), pi.unsqueeze(1)).squeeze(1)
                        rhs_vec[3 * p0:3 * (p0 + 1)] += w_At_B[0:3]
                        rhs_vec[3 * p1:3 * (p1 + 1)] += w_At_B[3:6]
                        rhs_vec[3 * p2:3 * (p2 + 1)] += w_At_B[6:9]
                        rhs_vec[3 * p3:3 * (p3 + 1)] += w_At_B[9:12]
                    else:
                        pass
                self.pos = torch.lu_solve(rhs_vec.unsqueeze(1), *self.LU_factored).squeeze(1)
            self.velocity = self.velocity + self.pos / self.h_step

            new_data = self.pos.reshape((-1, 3))[self.surface_ind, :].cpu().numpy() * 5
            new_data = new_data + np.array([0, -2, -2]).repeat(new_data.shape[0]).reshape(3, -1).T
            self.animation_actor.update(new_data)

            self.step += 1
            self.simulator_component.core_component.log_component.InfoLog("deformation simulator: at its iteration {}".format(self.step))

    def dump_bin_file(self, itr):
        itr = itr % self.max_steps
        file_path = os.path.join(self.data_path, "{:0>5d}.bin".format(itr))
