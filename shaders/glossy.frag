#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 Position;
in vec4 ShadowCoord;

uniform vec3 viewPos;
uniform vec3 lightPos;
uniform samplerCube skybox;
uniform bool useShadow;

#include "PCF.glsl"

void main()
{
    vec3 I = normalize(Position - viewPos);
    vec3 R = reflect(I, normalize(Normal));

    float shadow = 1.0;
    if (useShadow) {
        vec3 lightDir = normalize(lightPos - Position);
        float bias = max(0.005 * (1.0 - dot(Normal, lightDir)), 0.005);
        shadow = ShadowCalculation(ShadowCoord, bias);
    }

    FragColor = vec4(texture(skybox, R).rgb * (1 - shadow), 1.0);
}