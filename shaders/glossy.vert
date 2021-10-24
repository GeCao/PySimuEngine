#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 Normal;
out vec3 Position;

uniform mat4 ModelViewProjectionMatrix;
uniform mat4 ModelViewMatrix;

void main()
{
    Normal = aNormal;
    Position = aPos;
    gl_Position = ModelViewProjectionMatrix * vec4(Position, 1.0);
}