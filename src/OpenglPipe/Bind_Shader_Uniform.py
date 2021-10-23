from OpenGL.GL import *
from OpenGL.GLU import *
from enum import Enum
import numpy as np


class DataType(Enum):
    NP_INT = np.dtype('int')
    NP_FLOAT = np.dtype('float32')
    NP_DOUBLE = np.dtype('float64')

class DataSize(Enum):
    SCALAR = (1, )

    VEC2 = (2, )
    VEC3 = (3, )
    VEC4 = (4, )

    MAT2 = (2, 2)
    MAT3 = (3, 3)
    MAT4 = (4, 4)

class BindShaderUniform:
    def __init__(self, core_component):
        self.core_component = core_component
        self.initialized = False
        self.texture_counter = 2
        self.texture_name_ID_map = {}

    def initialization(self):
        self.initialized = True

    """
    Always use this while assure that the shaderProgram has been used already!
    """
    def bind_uniform(self, shaderProgram, uniform_name, uniform_data, num=1, push_texture_change=False):
        location = glGetUniformLocation(shaderProgram, uniform_name)
        int_0 = int(0)
        float_0 = float(0)
        bool_true = True
        if type(uniform_data) == type(np.array([0])):
            data_shape = uniform_data.shape
            data_type = uniform_data.dtype
            if len(data_shape) == 3:
                if uniform_name in self.texture_name_ID_map and not push_texture_change:
                    return
                TexID = glGenTextures(1)
                print("TexID of ", uniform_name, ": = ", TexID)
                glBindTexture(GL_TEXTURE_2D, TexID)

                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB if uniform_data.shape[2] == 3 else GL_RGBA,
                             uniform_data.shape[0], uniform_data.shape[1], 0,
                             GL_RGB if uniform_data.shape[2] == 3 else GL_RGBA,
                             GL_FLOAT, uniform_data.data)
                glGenerateMipmap(GL_TEXTURE_2D)

                glActiveTexture(GL_TEXTURE0 + TexID)
                glBindTexture(GL_TEXTURE_2D, TexID)
                Tex_loc = glGetUniformLocation(shaderProgram, uniform_name)
                glUniform1i(Tex_loc, TexID)
                self.texture_name_ID_map[uniform_name] = TexID
                self.texture_counter += 1
            elif DataSize(data_shape) == DataSize.SCALAR:
                if(DataType(data_type) == DataType.NP_INT):
                    glUniform1i(location, uniform_data[0])
                elif(DataType(data_type) == DataType.NP_FLOAT):
                    glUniform1f(location, uniform_data[0])
                elif(DataType(data_type) == DataType.NP_DOUBLE):
                    glUniform1d(location, uniform_data[0])
            elif DataSize(data_shape) == DataSize.VEC2:
                if(DataType(data_type) == DataType.NP_INT):
                    glUniform2iv(location, num, uniform_data)
                elif(DataType(data_type) == DataType.NP_FLOAT):
                    glUniform2fv(location, num, uniform_data)
                elif(DataType(data_type) == DataType.NP_DOUBLE):
                    glUniform2dv(location, num, uniform_data)
            elif DataSize(data_shape) == DataSize.VEC3:
                if(DataType(data_type) == DataType.NP_INT):
                    glUniform3iv(location, num, uniform_data)
                elif(DataType(data_type) == DataType.NP_FLOAT):
                    glUniform3fv(location, num, uniform_data)
                elif(DataType(data_type) == DataType.NP_DOUBLE):
                    glUniform3dv(location, num, uniform_data)
            elif DataSize(data_shape) == DataSize.VEC4:
                if(DataType(data_type) == DataType.NP_INT):
                    glUniform4iv(location, num, uniform_data)
                elif(DataType(data_type) == DataType.NP_FLOAT):
                    glUniform4fv(location, num, uniform_data)
                elif(DataType(data_type) == DataType.NP_DOUBLE):
                    glUniform4dv(location, num, uniform_data)
            elif DataSize(data_shape) == DataSize.MAT2:
                if(DataType(data_type) == DataType.NP_INT):
                    pass
                elif(DataType(data_type) == DataType.NP_FLOAT):
                    glUniformMatrix2fv(location, num, GL_FALSE, uniform_data)
                elif(DataType(data_type) == DataType.NP_DOUBLE):
                    glUniformMatrix2dv(location, num, GL_FALSE, uniform_data)
            elif DataSize(data_shape) == DataSize.MAT3:
                if(DataType(data_type) == DataType.NP_INT):
                    pass
                elif(DataType(data_type) == DataType.NP_FLOAT):
                    glUniformMatrix3fv(location, num, GL_FALSE, uniform_data)
                elif(DataType(data_type) == DataType.NP_DOUBLE):
                    glUniformMatrix3dv(location, num, GL_FALSE, uniform_data)
            elif DataSize(data_shape) == DataSize.MAT4:
                if(DataType(data_type) == DataType.NP_INT):
                    pass
                elif(DataType(data_type) == DataType.NP_FLOAT):
                    glUniformMatrix4fv(location, num, GL_FALSE, list(uniform_data.flatten()))
                elif(DataType(data_type) == DataType.NP_DOUBLE):
                    glUniformMatrix4dv(location, num, GL_FALSE, uniform_data)

        elif type(uniform_data) == bool:
            glUniform1i(location, uniform_data)

        # else:
            # print("The shader uniform: {", uniform_name, "} has not been binded for some reason")