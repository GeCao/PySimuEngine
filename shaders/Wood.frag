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

uniform sampler2D ShadowMap;
uniform bool useShadow;

uniform float near_plane;
uniform float far_plane;

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

float ShadowCalculation(vec4 fragPosLightSpace, float bias)
{
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(ShadowMap, projCoords.xy).r;
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // check whether current frag pos is in shadow
    vec2 texelSize = 1.0 / textureSize(ShadowMap, 0);
    float shadow = 0.0;
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = texture(ShadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;

    if (projCoords.z > 1.0) { shadow = 0.0; }

    return shadow;
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