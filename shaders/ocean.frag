#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 Position;
in vec4 ShadowCoord;

uniform vec3 viewPos;
uniform vec3 lightPos;
//uniform samplerCube skybox;
uniform bool useShadow;

uniform vec3 ShallowWaterColor;
uniform vec3 FarWaterColor;

#include "PCF.glsl"

void main()
{
    float transition_depth = 0.3;
    float transitionCoeff = 1.0;
    float mydepth = GetMyDepth(ShadowCoord) / transition_depth;
    float depthCoeff = pow(clamp(mydepth, 0.0, 1.0), transitionCoeff);

    float shadow = 1.0;
    if (useShadow) {
        vec3 lightDir = normalize(lightPos - Position);
        float bias = max(0.005 * (1.0 - dot(Normal, lightDir)), 0.005);
        shadow = ShadowCalculation(ShadowCoord, bias);
    }

    FragColor = vec4((ShallowWaterColor + depthCoeff * (FarWaterColor - ShallowWaterColor)).rgb * (1 - shadow), 1.0);
}