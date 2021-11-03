#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;
in vec2 Texcoord;
in vec4 ShadowCoord;

uniform vec3 viewPos;
uniform vec3 lightPos;
uniform vec3 objectColor;
uniform vec3 lightColor;

uniform sampler2D sampler0;

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

uniform Material material;

uniform bool useShadow;

uniform float near_plane;
uniform float far_plane;

#include "PCF.glsl"

float LinearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0; // Back to NDC
    return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));
}

vec3 PhongModel() {
    vec3 n = Normal;
    if (!gl_FrontFacing) { n = -n;}
    vec3 s = normalize(lightPos - FragPos);
    vec3 v = normalize(-FragPos);
    vec3 r = reflect(-s, n);
    float s_dot_n = max(dot(s, n), 0.0);

    vec3 diffuseColor = vec3(texture2D(sampler0, Texcoord.st));
    vec3 diffuse = diffuseColor * lightColor * s_dot_n;

    vec3 specular = vec3(0.0);
    if (s_dot_n > 0.0) {
        specular = material.specular * lightColor * pow(max(dot(r, v), 0.0), material.shininess);
    }
    return diffuse;
    //return diffuse + specular;
}

void shadeWithShadow() {
    //vec3 FragobjectColor = vec3(texture2D(sampler0, Texcoord.st));
    vec3 ambient = material.ambient * lightColor;
    vec3 diffAndSpec = PhongModel();

    float shadow = 1.0;
    if (useShadow) {
        vec3 lightDir = normalize(lightPos - FragPos);
        float bias = max(0.005 * (1.0 - dot(Normal, lightDir)), 0.005);
        shadow = ShadowCalculation(ShadowCoord, bias);
    }

    FragColor = vec4((diffAndSpec * (1 - shadow) + ambient * (1 - shadow)), 1.0);
}

void main()
{
    if (useShadow){
        shadeWithShadow();
    }
}