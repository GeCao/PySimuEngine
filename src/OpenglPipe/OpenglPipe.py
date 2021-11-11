import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from ..utils.Constants import *
from ..utils.utils import *
from .Shader import Shader
from .BindBuffer import BindBuffer


class OpenglPipe:
    def __init__(self, core_component):
        self.core_component = core_component
        self.camera = self.core_component.camera
        self.view = [-math.tan(self.camera.fov / 2.0) * self.camera.near,
                     math.tan(self.camera.fov / 2.0) * self.camera.near,
                     -math.tan(self.camera.fov / 2.0) * self.camera.near,
                     math.tan(self.camera.fov / 2.0) * self.camera.near,
                     self.camera.near, self.camera.far]
        self.width, self.height = self.camera.width, self.camera.height
        self.aspect = self.camera.aspect

        self.shader = Shader(core_component)
        self.bind_buffer = BindBuffer(core_component)

        self.need_update_VBO = True

        # Deferred Rendering related params
        self.use_deferred_shading = False
        self.gbuffer = None
        self.gPosition = None
        self.gNormal = None
        self.gAlbedoSpec = None
        self.renderDepth = None

        self.initialized = False

    def initialization(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
        glViewport(0, 0, self.width, self.height)

        self.bind_buffer.initialization()
        self.shader.initialization()

        self.initialized = True

    def clear_buffer(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def set_matrix(self):
        glViewport(0, 0, self.width, self.height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.camera.fov, self.camera.aspect, self.camera.near, self.camera.far)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            self.camera.eye[0], self.camera.eye[1], self.camera.eye[2],
            self.camera.look_at[0], self.camera.look_at[1], self.camera.look_at[2],
            self.camera.eye_up[0], self.camera.eye_up[1], self.camera.eye_up[2])

    def gbuffer_set(self, target_width=512, target_height=512):
        c_void_p_value = 0
        self.gbuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.gbuffer)

        # Pos buffer
        self.gPosition = glGenFramebuffers(1)
        glBindTexture(GL_TEXTURE_2D, self.gPosition)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, target_width, target_height, 0, GL_RGBA, GL_FLOAT,
                     c_void_p(c_void_p_value))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.gPosition, 0)
        self.core_component.log_component.InfoLog(
            "A message from PySimuEngine.OpenglPipe: A depth map {} has been established".format(self.gPosition))

        # Normal buffer
        self.gNormal = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.gNormal)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, target_width, target_height, 0, GL_RGBA, GL_FLOAT,
                     c_void_p(c_void_p_value))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, self.gNormal, 0)
        self.core_component.log_component.InfoLog(
            "A message from PySimuEngine.OpenglPipe: A depth map {} has been established".format(self.gNormal))

        # Albedo + Specualr buffer
        self.gAlbedoSpec = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.gAlbedoSpec)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, target_width, target_height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     c_void_p(c_void_p_value))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, self.gAlbedoSpec, 0)
        self.core_component.log_component.InfoLog(
            "A message from PySimuEngine.OpenglPipe: A depth map {} has been established".format(self.gAlbedoSpec))

        self.shader.bind_shader_uniform.texture_counter += 3

        attachments = (GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2)
        glDrawBuffers(3, attachments)

        # then also add render buffer object as depth buffer and check for completeness.
        self.renderDepth = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.renderDepth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, target_width, target_height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.renderDepth)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            self.core_component.log_component.ErrorLog("Did not created Framebuffer completely")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def gbuffer_pass(self):
        self.clear_buffer()

        glBindFramebuffer(GL_FRAMEBUFFER, self.gbuffer)
        self.clear_buffer()

        self.shader.activate_shader_for_geometry_pass()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind_all_render_targets(self, shader_name=None):
        glActiveTexture(GL_TEXTURE0 + self.gPosition)
        glBindTexture(GL_TEXTURE_2D, self.gPosition)
        glActiveTexture(GL_TEXTURE1 + self.gNormal)
        glBindTexture(GL_TEXTURE_2D, self.gNormal)
        glActiveTexture(GL_TEXTURE2 + self.gAlbedoSpec)
        glBindTexture(GL_TEXTURE_2D, self.gAlbedoSpec)

        if shader_name is None:
            shader_name = "LightingSimple"
        shaderProgram = self.shader.get_shaderProgram(shader_name)
        self.shader.bind_shader_uniform.bind_uniform(shaderProgram, "gPosition", TexID=self.gPosition, push_texture_change=True)
        self.shader.bind_shader_uniform.bind_uniform(shaderProgram, "gNormal", TexID=self.gNormal, push_texture_change=True)
        self.shader.bind_shader_uniform.bind_uniform(shaderProgram, "gAlbedoSpec", TexID=self.gAlbedoSpec, push_texture_change=True)

    def deferred_lighting_pass(self, target_width=512, target_height=512):
        self.clear_buffer()

        # also send light relevant uniforms
        self.shader.activate_deferred_shader()
        self.bind_all_render_targets()
        self.bind_buffer.draw_VAO(push_pass="LightingSimple")

        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.gbuffer)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, target_width, target_height, 0, 0, target_width, target_height, GL_DEPTH_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Forward Shading
        self.core_component.scene_component.render()

    def shadow_pass(self):
        self.clear_buffer()
        shadowMapWidth, shadowMapHeight = self.shader.get_shadowMap_size()
        old_eye = self.camera.eye
        self.core_component.camera.set_eye_position(new_eye=self.shader.lightPos)
        self.shader.activate_shader_for_shadow_pass()

        glViewport(0, 0, shadowMapWidth, shadowMapHeight)
        glBindFramebuffer(GL_FRAMEBUFFER, self.shader.shadowFBO)
        glClear(GL_DEPTH_BUFFER_BIT)
        glCullFace(GL_FRONT)
        self.bind_buffer.draw_VAO(push_pass="shadowpass")
        # glCullFace(GL_BACK)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self.core_component.camera.set_eye_position(new_eye=old_eye)

    def render_pass(self, useShadow=True):
        glViewport(0, 0, self.width, self.height)
        self.clear_buffer()

        self.set_matrix()

        for shader_name_in_using in self.core_component.resource_component.get_material_list():
            self.shader.activate_forward_shader(shader_name=shader_name_in_using,
                                        useShadow=useShadow)  # so called activate shader, actually means uniform binding
        glCullFace(GL_BACK)

        self.bind_buffer.draw_VAO()

        self.core_component.scene_component.render()

    def update(self):
        self.VBO_Resources()

        if self.use_deferred_shading:
            if self.gbuffer is None:
                self.gbuffer_set()
            self.gbuffer_pass()
            self.deferred_lighting_pass()
        else:
            self.shadow_pass()
            self.render_pass()

        # ---------------------------------------------------------------
        glFlush()  # 清空缓冲区，将指令送往硬件立即执行

    def VBO_Resources(self):
        if self.need_update_VBO:
            mesh_dict = self.core_component.resource_component.mesh_dict
            animation_dict = self.core_component.resource_component.animation_dict
            material_dict = self.core_component.resource_component.material_dict
            for resource_name in mesh_dict:
                point_info = self.core_component.resource_component.get_mesh_resource(resource_name).Pointer_info
                VAO, VBO = self.bind_buffer.bind_VAO(material_dict[resource_name],
                                                     data=mesh_dict[resource_name].data,
                                                     indices=mesh_dict[resource_name].faces,
                                                     Pointer_info=point_info)
                mesh_dict[resource_name].need_update_to_pipe = False
                if resource_name in animation_dict:
                    animation_dict[resource_name].VAO = VAO
                    animation_dict[resource_name].VBO = VBO
            self.need_update_VBO = False

        animation_dict = self.core_component.resource_component.animation_dict
        for resource_name in animation_dict:
            if animation_dict[resource_name].mesh.need_update_to_pipe:
                self.bind_buffer.sub_change_buffer(VAO=animation_dict[resource_name].VAO,
                                                   VBO=animation_dict[resource_name].VBO,
                                                   format=GL_ARRAY_BUFFER,
                                                   data=animation_dict[resource_name].mesh.data)
