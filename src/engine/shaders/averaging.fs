#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragPosition;

// Input uniform values
uniform sampler2D texture0;
uniform vec4 colDiffuse;

// Output fragment color
out vec4 finalColor;

// NOTE: Add here your custom variables
uniform int grid_width;

void main()
{
    ivec2 texSize = textureSize(texture0, 0);
    vec2 size = vec2(float(texSize.x), float(texSize.y));
    float texColumn = floor(fragTexCoord.x * texSize.x);
    float pixWidthPerColumn = size.x / grid_width;

    // Texel color fetching from texture sampler
    vec3 acculumatedColor = vec3(0.0, 0.0, 0.0);

    float columnPercent = texColumn / float(grid_width);

    for(float i = pixWidthPerColumn * texColumn; i < pixWidthPerColumn * (texColumn + 1); i++) {
        vec3 texCol = texture(texture0, vec2(i / size.x, fragTexCoord.y)).rgb;
        acculumatedColor = acculumatedColor + texCol;
    }
    
    acculumatedColor = acculumatedColor / pixWidthPerColumn;
    finalColor =  vec4(acculumatedColor, 1.0);
}