#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texcoord;

uniform mat4 ModelViewProjectionMatrix;

void main() {
    gl_Position = ModelViewProjectionMatrix * vec4(aPos, 1.0);
}
