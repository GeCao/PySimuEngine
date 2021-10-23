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
        self.view = [-math.tan(self.camera.fov / 2.0) * self.camera.near, math.tan(self.camera.fov / 2.0) * self.camera.near,
                     -math.tan(self.camera.fov / 2.0) * self.camera.near, math.tan(self.camera.fov / 2.0) * self.camera.near,
                     self.camera.near, self.camera.far]
        self.width, self.height = self.camera.width, self.camera.height
        self.aspect = self.camera.aspect

        self.shader = Shader(core_component)
        self.bind_buffer = BindBuffer(core_component)

        self.need_update_VBO = True

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
        self.bind_buffer.draw_VAO(push_for_shadow_pass=True)
        # glCullFace(GL_BACK)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self.core_component.camera.set_eye_position(new_eye=old_eye)

    def render_pass(self):
        glViewport(0, 0, self.width, self.height)
        self.clear_buffer()

        for shader_name_in_using in self.core_component.resource_component.get_material_list():
            self.shader.activate_shader(shader_name=shader_name_in_using, useShadow=True) # so called activate shader, actually means uniform binding
        glCullFace(GL_BACK)

        self.bind_buffer.draw_VAO()

    def update(self):
        self.VBO_Resources()

        self.shadow_pass()

        self.render_pass()

        # ---------------------------------------------------------------
        glFlush()  # 清空缓冲区，将指令送往硬件立即执行

    def VBO_Resources(self):
        if self.need_update_VBO:
            mesh_dict = self.core_component.resource_component.mesh_dict
            material_dict = self.core_component.resource_component.material_dict
            for resource_name in mesh_dict:
                point_info = self.core_component.resource_component.get_mesh_resource(resource_name).Pointer_info
                self.bind_buffer.bind_VAO(material_dict[resource_name],
                                          data=mesh_dict[resource_name].data,
                                          indices=mesh_dict[resource_name].faces,
                                          Pointer_info=point_info)
            self.need_update_VBO = False

