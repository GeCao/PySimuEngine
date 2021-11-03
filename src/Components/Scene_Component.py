import numpy as np
import os
import sys

from OpenGL.GL import *
from OpenGL.GLUT import *

from ..Simulation.FFTOcean import FFTOcean
from ..utils.utils import _read_img_2d, MessageAttribute


class SceneComponent:
    def __init__(self, core_component):
        self.core_component = core_component
        self.root_path = self.core_component.root_path
        self.scene_path = os.path.join(self.root_path, 'scene')
        self.cubemap_path = os.path.join(self.scene_path, 'textures\\cubemap')
        self.cubemap_dirs = []
        self.cubemap_files = []
        self.cubemap_mapping = {'negx': GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
                                'negy': GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
                                'negz': GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
                                'posx': GL_TEXTURE_CUBE_MAP_POSITIVE_X,
                                'posy': GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
                                'posz': GL_TEXTURE_CUBE_MAP_POSITIVE_Z}
        for parent, dirnames, filenames in os.walk(self.cubemap_path):
            self.cubemap_dirs = dirnames
            break
        for parent, dirnames, filenames in os.walk(os.path.join(self.cubemap_path, self.cubemap_dirs[0])):
            self.cubemap_files = filenames
            break
        self.cubemap_files = ['posx.jpg', 'negx.jpg', 'posy.jpg', 'negy.jpg', 'posz.jpg', 'negz.jpg']

        self.textureID = None

        self.skyboxVAO = None
        self.skyboxVBO = None

        self.ocean = FFTOcean(self.core_component)

        self.initialized = False

    def initialization(self):
        self.load_cube_map()
        self.cubeVertices = [-0.5, -0.5, -0.5, 0.0, 0.0, -1.0,
                             0.5, -0.5, -0.5, 0.0, 0.0, -1.0,
                             0.5, 0.5, -0.5, 0.0, 0.0, -1.0,
                             0.5, 0.5, -0.5, 0.0, 0.0, -1.0,
                             -0.5, 0.5, -0.5, 0.0, 0.0, -1.0,
                             -0.5, -0.5, -0.5, 0.0, 0.0, -1.0,

                             -0.5, -0.5, 0.5, 0.0, 0.0, 1.0,
                             0.5, -0.5, 0.5, 0.0, 0.0, 1.0,
                             0.5, 0.5, 0.5, 0.0, 0.0, 1.0,
                             0.5, 0.5, 0.5, 0.0, 0.0, 1.0,
                             -0.5, 0.5, 0.5, 0.0, 0.0, 1.0,
                             -0.5, -0.5, 0.5, 0.0, 0.0, 1.0,

                             -0.5, 0.5, 0.5, -1.0, 0.0, 0.0,
                             -0.5, 0.5, -0.5, -1.0, 0.0, 0.0,
                             -0.5, -0.5, -0.5, -1.0, 0.0, 0.0,
                             -0.5, -0.5, -0.5, -1.0, 0.0, 0.0,
                             -0.5, -0.5, 0.5, -1.0, 0.0, 0.0,
                             -0.5, 0.5, 0.5, -1.0, 0.0, 0.0,

                             0.5, 0.5, 0.5, 1.0, 0.0, 0.0,
                             0.5, 0.5, -0.5, 1.0, 0.0, 0.0,
                             0.5, -0.5, -0.5, 1.0, 0.0, 0.0,
                             0.5, -0.5, -0.5, 1.0, 0.0, 0.0,
                             0.5, -0.5, 0.5, 1.0, 0.0, 0.0,
                             0.5, 0.5, 0.5, 1.0, 0.0, 0.0,

                             -0.5, -0.5, -0.5, 0.0, -1.0, 0.0,
                             0.5, -0.5, -0.5, 0.0, -1.0, 0.0,
                             0.5, -0.5, 0.5, 0.0, -1.0, 0.0,
                             0.5, -0.5, 0.5, 0.0, -1.0, 0.0,
                             -0.5, -0.5, 0.5, 0.0, -1.0, 0.0,
                             -0.5, -0.5, -0.5, 0.0, -1.0, 0.0,

                             -0.5, 0.5, -0.5, 0.0, 1.0, 0.0,
                             0.5, 0.5, -0.5, 0.0, 1.0, 0.0,
                             0.5, 0.5, 0.5, 0.0, 1.0, 0.0,
                             0.5, 0.5, 0.5, 0.0, 1.0, 0.0,
                             -0.5, 0.5, 0.5, 0.0, 1.0, 0.0,
                             -0.5, 0.5, -0.5, 0.0, 1.0, 0.0]
        self.skyboxVertices = [-1.0, 1.0, -1.0,
                               -1.0, -1.0, -1.0,
                               1.0, -1.0, -1.0,
                               1.0, -1.0, -1.0,
                               1.0, 1.0, -1.0,
                               -1.0, 1.0, -1.0,

                               -1.0, -1.0, 1.0,
                               -1.0, -1.0, -1.0,
                               -1.0, 1.0, -1.0,
                               -1.0, 1.0, -1.0,
                               -1.0, 1.0, 1.0,
                               -1.0, -1.0, 1.0,

                               1.0, -1.0, -1.0,
                               1.0, -1.0, 1.0,
                               1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0,
                               1.0, 1.0, -1.0,
                               1.0, -1.0, -1.0,

                               -1.0, -1.0, 1.0,
                               -1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0,
                               1.0, -1.0, 1.0,
                               -1.0, -1.0, 1.0,

                               -1.0, 1.0, -1.0,
                               1.0, 1.0, -1.0,
                               1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0,
                               -1.0, 1.0, 1.0,
                               -1.0, 1.0, -1.0,

                               -1.0, -1.0, -1.0,
                               -1.0, -1.0, 1.0,
                               1.0, -1.0, -1.0,
                               1.0, -1.0, -1.0,
                               -1.0, -1.0, 1.0,
                               1.0, -1.0, 1.0]
        self.skyboxVertices = 5.0 * np.array(self.skyboxVertices, dtype=np.float32)
        # self.skyboxVertices = [float(x) for x in self.skyboxVertices]

        self.setup()

        self.ocean.initialization()

        self.initialized = True

    def load_cube_map(self):
        self.textureID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.textureID)
        self.core_component.log_component.Slog(MessageAttribute.EInfo,
            "A message from PySimuEngine.SceneComponent: A cube map {} has been established".format(self.textureID))

        for i, filename in enumerate(self.cubemap_files):
            image_data = _read_img_2d(os.path.join(self.cubemap_path, self.cubemap_dirs[2], filename), np.float32) / 255.0
            width, height, channel = image_data.shape
            # print("data shape of ", filename, ": ", width, height, channel)  # (512, 512, 3)
            glTexImage2D(self.cubemap_mapping[filename[0:4]], 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT,
                         image_data.data)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    def setup(self):
        self.skyboxVAO = glGenVertexArrays(1)
        self.skyboxVBO = glGenBuffers(1)
        glBindVertexArray(self.skyboxVAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.skyboxVBO)
        glBufferData(GL_ARRAY_BUFFER, int(len(self.skyboxVertices) * 4), self.skyboxVertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, c_void_p(0))

    def hlp_activate_cube_map(self, shaderProgram):
        glActiveTexture(GL_TEXTURE0 + self.textureID)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.textureID)
        CM_loc = glGetUniformLocation(shaderProgram, 'skybox')
        glUniform1i(CM_loc, self.textureID)

    def update(self):
        self.ocean.update()

    def render(self):
        glDepthFunc(GL_LEQUAL)
        # glDepthMask(GL_FALSE)
        shader_name = "skybox"
        shaderProgram = self.core_component.opengl_pipe.shader.get_shaderProgram(shader_name)
        glUseProgram(shaderProgram)
        self.core_component.opengl_pipe.shader.bind_shader_uniform.bind_uniform(shaderProgram,
                                                                                'ModelViewProjectionMatrix',
                                                                                self.core_component.camera.get_modelviewprojection_mat())

        glBindVertexArray(self.skyboxVAO)
        glActiveTexture(GL_TEXTURE0 + self.textureID)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.textureID)
        CM_loc = glGetUniformLocation(shaderProgram, 'skybox')
        glUniform1i(CM_loc, self.textureID)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)

        glDepthFunc(GL_LESS)
        # glDepthMask(GL_TRUE)
