#version 330

// Input vertex attributes
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

// Input uniform values
uniform mat4 mvp;
uniform mat4 matModel;
uniform mat4 matNormal;

#define     MAX_LIGHTS              4
#define     LIGHT_DIRECTIONAL       0
#define     LIGHT_POINT             1

struct MaterialProperty {
    vec3 color;
    int useSampler;
    sampler2D sampler;
};

struct Light {
    int enabled;
    int type;
    vec3 position;
    vec3 target;
    vec4 color;
};

// Input lighting values
uniform Light lights[MAX_LIGHTS];

// Output vertex attributes (to fragment shader)
out vec3 fragPosition;
out vec2 fragTexCoord;
out vec4 fragColor;
out vec3 fragNormal;
out vec3 lightPosition;

// NOTE: Add here your custom variables
uniform mat4 light_mvp;
uniform int model_id;
uniform vec3 lightPos;

#define MAX_MODELS   25

struct MatArr {
    mat4 mat;
};

uniform MatArr light_mvps[MAX_MODELS];

void main()
{
    mat4 light_mvp_from_arr = light_mvps[model_id].mat;
    // Send vertex attributes to fragment shader
    fragPosition = vertexPosition;
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = normalize(vec3(matNormal*vec4(vertexNormal, 1.0)));

    // Calculate final vertex position
    gl_Position = light_mvp_from_arr*vec4(vertexPosition, 1.0);
    lightPosition = lightPos;
}
