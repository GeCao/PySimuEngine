#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texcoord;

uniform mat4 ModelViewProjectionMatrix;
uniform mat4 ModelViewMatrix;

uniform mat4 ShadowMatrix;
uniform bool useShadow;

out vec3 Normal;
out vec3 FragPos;
out vec2 Texcoord;
out vec4 ShadowCoord;

void main()
{
    gl_Position = ModelViewProjectionMatrix * vec4(aPos, 1.0);

    if (useShadow) {
        FragPos = (vec4(aPos,1.0)).xyz;
        Normal = normalize(normal);
        ShadowCoord = ShadowMatrix * vec4(aPos,1.0);
        Texcoord = texcoord;
    }
}