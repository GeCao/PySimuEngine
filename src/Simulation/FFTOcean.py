"""
Create by Ge Cao
# We do not need to implement optimization for this module
"""
import sys
import numpy as np
import math

from OpenGL.GL import *

cm = 0.23
km = 370.0

WIND = 5.0
OMEGA = 0.84
AMPLITUDE = 0.5

CHOPPY_FACTOR = np.array([2.3, 2.1, 1.3, 0.9], dtype=np.float32)

PASSES = 8  # number of passes needed for the FFT 6 -> 64, 7 -> 128, 8 -> 256, etc
FFT_SIZE = 1 << PASSES  # size of the textures storing the waves in frequency and spatial domains

N_SLOPE_VARIANCE = 10
GRID1_SIZE = 5488.0
GRID2_SIZE = 392.0
GRID3_SIZE = 28.0
GRID4_SIZE = 2.0
GRID_SIZES = np.array([GRID1_SIZE, GRID2_SIZE, GRID3_SIZE, GRID4_SIZE], dtype=np.float32)
INVERSE_GRID_SIZES = np.array([2.0 * math.pi * FFT_SIZE / GRID1_SIZE,
                               2.0 * math.pi * FFT_SIZE / GRID2_SIZE,
                               2.0 * math.pi * FFT_SIZE / GRID3_SIZE,
                               2.0 * math.pi * FFT_SIZE / GRID4_SIZE], dtype=np.float32)

GRID_VERTEX_COUNT = 200
GRID_CELL_SIZE = np.array([1.0 / float(GRID_VERTEX_COUNT), 1.0 / float(GRID_VERTEX_COUNT)], dtype=np.float32)


class FFTOcean:
    def __init__(self, core_component):
        self.core_component = core_component
        self.random_seed = 0
        self.need_render = True

        self.gravity = 9.8
        self.wind_speed_V = 1.0
        self.wind_dir = np.array([1.0, 0.0])
        self.Phillips_L = self.wind_speed_V * self.wind_speed_V / self.gravity
        self.amplitude = 3.0
        self.time = 0
        self.delta_time = 0.01

        self.grid_resolution = 64
        self.grid_size = 64
        self.height_grid = None
        self.normals = None

        self.VAO = None
        self.VBO = None

        self.initialized = False

    def initialization(self):
        np.random.seed(self.random_seed)

        if self.need_render:
            self.height_grid = np.zeros(shape=(self.grid_resolution, self.grid_resolution))
            self.height_grid = np.array(self.height_grid, dtype=np.complex)

            self.normals = np.zeros(shape=(self.grid_resolution, self.grid_resolution))
            self.normals = np.array(self.height_grid, dtype=np.complex)
        self.initialized = True

    def get_height0(self, k):
        random_number_list = np.random.normal(0, 1, 2)  # Mean, Standard deviation, Output shape
        complex_ran_num = complex(random_number_list[0], random_number_list[1])
        return complex_ran_num * math.sqrt(self.Phillips_spectrum(k)) / math.sqrt(2.0)

    def Phillips_spectrum(self, k):
        abs_k = math.sqrt(k[0] * k[0] + k[1] * k[1])
        if abs_k < 1e-6:
            return 0
        return self.amplitude * math.exp(-1.0 / (abs_k * self.Phillips_L) ** 2) / (abs_k ** 4) * (
                (k.dot(self.wind_dir)) ** 2)

    def get_height(self, k, time):
        abs_k = math.sqrt(k[0] * k[0] + k[1] * k[1])
        omega = math.sqrt(abs_k * self.gravity)
        exp_pos_jwk = complex(math.cos(omega * time), math.sin(omega * time))
        exp_neg_jwk = exp_pos_jwk.conjugate()
        return self.get_height0(k) * exp_pos_jwk + self.get_height0(-k).conjugate() * exp_neg_jwk

    def update(self):
        if self.need_render:
            for i in range(self.grid_resolution):
                kx = (2 * math.pi / self.grid_size) * (-self.grid_resolution / 2 + i)
                for j in range(self.grid_resolution):
                    kz = (2 * math.pi / self.grid_size) * (-self.grid_resolution / 2 + j)
                    height = self.get_height(k=np.array([kx, kz]), time=self.time)
                    self.height_grid[i][j] = height
                    if i % 2 == 1:
                        self.height_grid[i][j] = -self.height_grid[i][j]
                    self.normals[i][j] = complex(-kz, kx) * self.height_grid[i][j]

            for j in range(self.grid_resolution):
                self.height_grid[:, j] = np.fft.ifft(self.height_grid[:, j])
                self.normals[:, j] = np.fft.ifft(self.normals[:, j])

            for i in range(self.grid_resolution):
                for j in range(self.grid_resolution):
                    if (int(-self.grid_resolution / 2) + i + j) % 2 == 1:
                        self.height_grid[i][j] = -self.height_grid[i][j]
                        self.normals[i][j] = -self.normals[i][j]

            for i in range(self.grid_resolution):
                self.height_grid[i, :] = np.fft.ifft(self.height_grid[i, :])
                self.normals[i, :] = np.fft.ifft(self.normals[i, :])

            for i in range(self.grid_resolution):
                for j in range(self.grid_resolution):
                    if (int(-self.grid_resolution / 2) + j) % 2 == 1:
                        self.height_grid[i][j] = -self.height_grid[i][j]
                        self.normals[i][j] = -self.normals[i][j]
        self.time += self.delta_time
        self.update_VBO()

    def update_VBO(self):
        if self.need_render:
            print("The max height of ocean: ", self.height_grid[12][34])
            dx, dy = 10.0 / self.grid_resolution, 10.0 / self.grid_resolution
            data = []
            for i in range(self.grid_resolution):
                pos_x = -5.0 + dx * i
                for j in range(self.grid_resolution):
                    pos_y = -5.0 + dy * j
                    normal = np.array([self.normals[i][j].real / abs(self.normals[i][j]),
                                       1.0,
                                       self.normals[i][j].imag / abs(self.normals[i][j])], dtype=np.float32)
                    normal = normal / math.sqrt(normal.dot(normal))
                    data.append([pos_x, self.height_grid[i][j].real * 200, pos_y,
                                 normal[0], normal[1], normal[2]])
            data = np.array(data, dtype=np.float32)
            if self.VAO is None:
                indices = []
                for i in range(self.grid_resolution - 1):
                    for j in range(self.grid_resolution - 1):
                        indices.append(int(i * self.grid_resolution + j))
                        indices.append(int((i + 1) * self.grid_resolution + j))
                        indices.append(int(i * self.grid_resolution + (j + 1)))

                        indices.append(int((i + 1) * self.grid_resolution + j))
                        indices.append(int(i * self.grid_resolution + (j + 1)))
                        indices.append(int((i + 1) * self.grid_resolution + (j + 1)))
                indices = np.array(indices, dtype=np.int)
                self.core_component.resource_component.register_resource('material', material_name="ocean")
                self.VAO, self.VBO = self.core_component.opengl_pipe.bind_buffer.bind_VAO("ocean",
                                                                                          data=data,
                                                                                          indices=indices,
                                                                                          Pointer_info=['vertices',
                                                                                                        'normals'])
            else:
                self.core_component.opengl_pipe.bind_buffer.sub_change_buffer(VAO=self.VAO, VBO=self.VBO,
                                                                              format=GL_ARRAY_BUFFER, data=data)
