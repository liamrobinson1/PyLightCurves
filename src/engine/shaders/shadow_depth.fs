#version 330

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec3 fragNormal;
in vec3 lightPosition;

out vec4 fragColor;

uniform sampler2D texture0;

// Input lighting values

void main()
{
    // NOTE: Implement here your fragment shader code
    float x1 = fragPosition.x; //world coordinates of the fragment of interest
    float y1 = fragPosition.y;
    float z1 = fragPosition.z;

    float A = lightPosition.x; //for the equation of the light's plane as A(x-x0) + B(y-y0) + C(z-z0) = 0
    float B = lightPosition.y;
    float C = lightPosition.z;

    float x0 = lightPosition.x; //coordinates of a point we know lies in the plane
    float y0 = lightPosition.y;
    float z0 = lightPosition.z;

    float d = -dot(vec3(x1-x0, y1-y0, z1-z0), vec3(A, B, C))/20.0;
    fragColor = vec4(d, d, d, 1.0);
}
