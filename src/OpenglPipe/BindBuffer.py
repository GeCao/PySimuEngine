import sys

from OpenGL.GL import *
from OpenGL.GLUT import *

class BindBuffer:
    def __init__(self, core_component):
        self.core_component = core_component
        self.VAO_dict = {}
        self.VAO_indices_size = {}

        self.initialized = False

    def initialization(self):
        self.initialized = True

    def bind_VAO(self, VAO_name, data=None, indices=None, Pointer_info=None):
        VAO = glGenVertexArrays(1)
        glBindVertexArray(VAO)

        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, sys.getsizeof(data), data, GL_STATIC_DRAW)

        index_p = 0
        size_p_info = {'vertices': [3, GL_FLOAT, 12], 'normals': [3, GL_FLOAT, 12], 'texcoords':[2, GL_FLOAT, 8]}
        stride = 0
        for key_str in size_p_info.keys():
            if key_str in Pointer_info:
                stride += size_p_info[key_str][2]
        c_void_p_value = 0
        for p_info in Pointer_info:
            glVertexAttribPointer(index_p, size_p_info[p_info][0], size_p_info[p_info][1], GL_FALSE, stride, c_void_p(c_void_p_value))
            glEnableVertexAttribArray(index_p)
            index_p += 1
            c_void_p_value += size_p_info[p_info][2]

        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sys.getsizeof(indices), indices, GL_STATIC_DRAW)

        if VAO_name in self.VAO_dict:
            self.VAO_dict[VAO_name].append(VAO)
            self.VAO_indices_size[VAO_name].append(len(indices))
        else:
            self.VAO_dict[VAO_name] = [VAO]
            self.VAO_indices_size[VAO_name] = [len(indices)]
        glBindVertexArray(0)  # 复位VAO

    def draw_VAO(self, push_for_shadow_pass=False):
        for VAO_name in self.VAO_dict.keys():
            shaderProgram = self.core_component.opengl_pipe.shader.get_shaderProgram(VAO_name)
            if push_for_shadow_pass:
                shaderProgram = self.core_component.opengl_pipe.shader.get_shaderProgram("shadowpass")
            glUseProgram(shaderProgram)
            for i, VAO in enumerate(self.VAO_dict[VAO_name]):
                len_indices = int(self.VAO_indices_size[VAO_name][i])

                glBindVertexArray(VAO)
                # glDrawArrays(GL_TRIANGLES, 0, 3)
                glDrawElements(GL_TRIANGLES, len_indices, GL_UNSIGNED_INT, None)
                glBindVertexArray(0)  # 复位VAO