import os, random

import numpy as np
from OpenGL.GL import *
from ..utils.utils import compute_perspective_mat, compute_lookat_mat, compute_ortho_mat, MessageAttribute

from .Bind_Shader_Uniform import BindShaderUniform

class Shader:
    def __init__(self, core_component):
        self.core_component = core_component
        self.shaders_dict = {}  # shader_name -> shaderProgram
        self.shaders_uniform_dict = {}  # shader_name -> list of uniform_names of this shader
        self.uniform_dict = {'ModelViewProjectionMatrix' : self.core_component.camera.get_modelviewprojection_mat()}  # uniform_name -> uniform value
        self.shaders_root_path = os.path.join(self.core_component.root_path, 'shaders')

        self.bind_shader_uniform = BindShaderUniform(core_component)

        # Shadow Map related
        self.depthTex = None
        self.shadowFBO = None
        self.shadowMapWidth = 1024
        self.shadowMapHeight = 1024

        # Light related
        self.lightPos = np.array([1.2, 1.0, 2.0], dtype=np.float32)
        self.lightColor = np.array([1, 1, 1], dtype=np.float32)

        # deferred light related
        self.deferred_lightPos = (np.random.rand(32, 3) * 6.0 - 3.0).astype(np.float32)
        self.deferred_lightColor = (np.random.rand(32, 3) * 0.5 + 0.5).astype(np.float32)
        # self.deferred_lightPos = np.array([[1.2, 1.0, 2.0]], dtype=np.float32)
        # self.deferred_lightColor = np.array([[1.2, 1.0, 2.0]], dtype=np.float32)

        self.initialized = False

    def initialization(self):
        self.bind_shader_uniform.initialization()
        self.core_component.resource_component.load_materials()
        material_list = self.core_component.resource_component.get_material_list()

        for material_name in material_list:
            self.load_shader(material_name)

        self.setup_FBO()

        self.initialized = True

    def load_shader(self, shader_name, vertexShaderSource=None, fragmentShaderSource=None):
        try:
            if vertexShaderSource is None:
                vert_file_path = os.path.join(self.shaders_root_path, shader_name + '.vert')
                vertexShaderSource = self.read_file_as_str(vert_file_path)
            if fragmentShaderSource is None:
                frag_file_path = os.path.join(self.shaders_root_path, shader_name + '.frag')
                fragmentShaderSource = self.read_file_as_str(frag_file_path)

            vertexShader = glCreateShader(GL_VERTEX_SHADER)
            glShaderSource(vertexShader, vertexShaderSource)
            glCompileShader(vertexShader)

            fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
            glShaderSource(fragmentShader, fragmentShaderSource)
            glCompileShader(fragmentShader)

            shaderProgram = glCreateProgram()  # ????????????GLSL?????????
            glAttachShader(shaderProgram, vertexShader)
            glAttachShader(shaderProgram, fragmentShader)  # ?????????shader?????????????????????
            glLinkProgram(shaderProgram)  # ??????shader???????????????shader????????????????????????????????????????????????

            glDeleteShader(vertexShader)
            glDeleteShader(fragmentShader)

            self.shaders_dict[shader_name] = shaderProgram

            vert_frag_source_split = (vertexShaderSource + '\n' + fragmentShaderSource).split('uniform ')
        except:
            glsl_file_path = os.path.join(self.shaders_root_path, shader_name + '.glsl')
            glslShaderSource = self.read_file_as_str(glsl_file_path)

            glslShader = glCreateShader(GL_FRAGMENT_SHADER)
            glShaderSource(glslShader, glslShaderSource)
            glCompileShader(glslShader)

            shaderProgram = glCreateProgram()  # ????????????GLSL?????????
            glAttachShader(shaderProgram, glslShader)  # ???shader?????????????????????
            glLinkProgram(shaderProgram)  # ??????shader???????????????shader????????????????????????????????????????????????

            glDeleteShader(glslShader)

            self.shaders_dict[shader_name] = shaderProgram

            vert_frag_source_split = (glslShaderSource).split('uniform ')

        self.shaders_uniform_dict[shader_name] = []
        for i, str_piece in enumerate(vert_frag_source_split):
            if i > 0:
                if vert_frag_source_split[i - 1][-1] == '/':
                    continue
                str_piece_split = str_piece.split()
                uniform_type = str_piece_split[0]
                uniform_name = str_piece_split[1][:-1]  # There might exist a ';'
                self.shaders_uniform_dict[shader_name].append(uniform_name)

    def get_shadowMap_size(self):
        return self.shadowMapWidth, self.shadowMapHeight

    def setup_FBO(self):
        glClearColor(0, 0, 0, 1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)

        self.depthTex = glGenTextures(1)
        self.shadowFBO = glGenFramebuffers(1)
        self.core_component.log_component.Slog(MessageAttribute.EInfo,
            "A message from PySimuEngine.Shader: A depth map {} has been established".format(self.depthTex))

        glBindTexture(GL_TEXTURE_2D, self.depthTex)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.shadowMapWidth, self.shadowMapHeight, 0,
                     GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        glBindFramebuffer(GL_FRAMEBUFFER, self.shadowFBO)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depthTex, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)

        result = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if result != GL_FRAMEBUFFER_COMPLETE:
            self.core_component.log_component.Slog(MessageAttribute.EError,
                                                   "A message from PySimuEngine.Shader: Framebuffer is not complete.")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def get_shaderProgram(self, shader_name):
        if shader_name not in self.shaders_dict:
            self.load_shader(shader_name)
        return self.shaders_dict[shader_name]

    def _read_file_as_str(self, file_path):
        # ????????????????????????
        if not os.path.isfile(file_path):
            raise TypeError(file_path + " does not exist")

        all_the_text = open(file_path).read()
        return all_the_text

    def read_file_as_str(self, file_path):
        shaderSource = self._read_file_as_str(file_path)
        if "#include" in shaderSource:
            shader_source_split = shaderSource.split("#include ")
            final_shaderSource = shader_source_split[0]
            for i in range(len(shader_source_split)):
                if i > 0:
                    if shader_source_split[i - 1][-1] == '/':
                        final_shaderSource = final_shaderSource + shader_source_split[i]
                        continue
                    split_idx = (shader_source_split[1].find('\"', 1))
                    path_file = shader_source_split[1][1:split_idx]
                    left_str = shader_source_split[1][split_idx + 1:]
                    included_shaderSource = self.read_file_as_str(os.path.join(self.shaders_root_path, path_file))
                    final_shaderSource = final_shaderSource + included_shaderSource + left_str
            return final_shaderSource
        else:
            return shaderSource

    def bind_uniform_for_shader(self, shader_name, uniform_name, uniform_data):
        shaderProgram = self.get_shaderProgram(shader_name)
        glUseProgram(shaderProgram)
        self.uniform_dict[uniform_name] = uniform_data
        self.bind_shader_uniform.bind_uniform(shaderProgram, uniform_name, uniform_data)

    def activate_shader_for_geometry_pass(self):
        shader_name = "GeometryPass"
        shaderProgram = self.get_shaderProgram(shader_name)
        glUseProgram(shaderProgram)

        self.uniform_dict['ViewProjection'] = self.core_component.camera.get_modelviewprojection_mat()
        self.bind_shader_uniform.bind_uniform(shaderProgram, 'ViewProjection', self.uniform_dict['ViewProjection'])

        self.uniform_dict['model'] = np.eye(4, dtype=np.float32)
        self.bind_shader_uniform.bind_uniform(shaderProgram, 'model', self.uniform_dict['model'].T)

        self.core_component.opengl_pipe.bind_buffer.draw_VAO(push_pass="geometrypass")

    def activate_shader_for_shadow_pass(self):
        shader_name = "shadowpass"
        shaderProgram = self.get_shaderProgram(shader_name)
        glUseProgram(shaderProgram)

        _, center, eyeup = self.core_component.camera.get_look_at_params()
        eye = self.lightPos
        shadow_mat_v = compute_lookat_mat(eye, center, eyeup, my_dtype=np.float32)
        self.uniform_dict['ModelViewProjectionMatrix'] = shadow_mat_v.dot(self.core_component.camera.get_ortho_projection_mat())

        self.bind_shader_uniform.bind_uniform(shaderProgram,
                                              'ModelViewProjectionMatrix',
                                              self.uniform_dict['ModelViewProjectionMatrix'])

    def activate_deferred_shader(self, shader_name="LightingSimple"):
        shaderProgram = self.get_shaderProgram(shader_name)
        uniform_list = self.shaders_uniform_dict[shader_name]
        glUseProgram(shaderProgram)

        self.uniform_dict['viewPos'] = self.core_component.camera.eye

        for i in range(self.deferred_lightPos.shape[0]):
            self.uniform_dict['lights[{}].Position'.format(i)] = self.deferred_lightPos[i, :]
            self.uniform_dict['lights[{}].Color'.format(i)] = self.deferred_lightColor[i, :]

            self.uniform_dict['lights[{}].Linear'.format(i)] = 0.7
            self.uniform_dict['lights[{}].Quadratic'.format(i)] = 1.8

            max_Color_coeff = self.deferred_lightColor[i, :].max()
            radius = (-0.7 + np.sqrt(0.49 - 4 * 1.8 * (1.0 - (256.0 / 5.0) * max_Color_coeff))) / (2.0 * 1.8)
            self.uniform_dict['lights[{}].Radius'.format(i)] = radius

        for i, uniform_name in enumerate(self.uniform_dict.keys()):
            self.bind_shader_uniform.bind_uniform(shaderProgram, uniform_name, self.uniform_dict[uniform_name])

    def activate_forward_shader(self, shader_name, useShadow=False):
        shaderProgram = self.get_shaderProgram(shader_name)
        uniform_list = self.shaders_uniform_dict[shader_name]
        glUseProgram(shaderProgram)

        self.uniform_dict['ModelViewProjectionMatrix'] = self.core_component.camera.get_modelviewprojection_mat()  # update uniforms
        self.uniform_dict['ModelViewMatrix'] = self.core_component.camera.get_modelview_mat()
        self.uniform_dict['near_plane'] = self.core_component.camera.near
        self.uniform_dict['far_plane'] = self.core_component.camera.far

        self.uniform_dict['viewPos'] = self.core_component.camera.eye
        self.uniform_dict['lightPos'] = self.lightPos
        self.uniform_dict['lightColor'] = self.lightColor
        self.uniform_dict['useShadow'] = useShadow

        _, center, eyeup = self.core_component.camera.get_look_at_params()
        eye = self.uniform_dict['lightPos']
        shadow_mat_p = self.core_component.camera.get_ortho_projection_mat()
        shadow_mat_v = compute_lookat_mat(eye, center, eyeup, my_dtype=np.float32)

        if 'skybox' in uniform_list:
            self.core_component.scene_component.hlp_activate_cube_map(shaderProgram)
        if 'ShadowMatrix' in uniform_list and useShadow:
            self.uniform_dict['ShadowMatrix'] = shadow_mat_v.dot(shadow_mat_p)
        if 'shadowMap' in uniform_list and useShadow:
            glActiveTexture(GL_TEXTURE0 + self.depthTex)
            glBindTexture(GL_TEXTURE_2D, self.depthTex)
            SM_loc = glGetUniformLocation(shaderProgram, 'ShadowMap')
            glUniform1i(SM_loc, self.depthTex)
        if 'ShallowWaterColor' in uniform_list:
            self.uniform_dict['ShallowWaterColor'] = np.array([0.1, 0.7, 0.3], dtype=np.float32)
        if 'FarWaterColor' in uniform_list:
            self.uniform_dict['FarWaterColor'] = np.array([0.01, 0.07, 0.03], dtype=np.float32)
        if 'INV_PROJECTION' in uniform_list:
            self.uniform_dict['INV_PROJECTION'] = np.linalg.inv(self.core_component.camera.get_projection_mat())
        if 'INV_VIEW_ORIGIN' in uniform_list:
            self.uniform_dict['INV_VIEW_ORIGIN'] = np.linalg.inv(self.core_component.camera.get_view_mat())

        for i, uniform_name in enumerate(self.uniform_dict.keys()):
            self.bind_shader_uniform.bind_uniform(shaderProgram, uniform_name, self.uniform_dict[uniform_name])

        material_uniform_dict = self.core_component.resource_component.get_material_resource(shader_name).uniform_dict
        for i, uniform_name in enumerate(material_uniform_dict.keys()):
            self.bind_shader_uniform.bind_uniform(shaderProgram, uniform_name, material_uniform_dict[uniform_name])
