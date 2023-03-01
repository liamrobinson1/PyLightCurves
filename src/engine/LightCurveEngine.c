// Compile on Intel mac:
// clang -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL lib/libraylib.a LightCurveEngine.c -o LightCurveEngine
// Compile on M1 mac: (Liam's use only, assumes homebrew raylib)
// clang -framework CoreVideo -framework IOKit -framework Cocoa -framework GLUT -framework OpenGL /opt/homebrew/Cellar/raylib/4.0.0/lib/libraylib.a LightCurveEngine.c -o LightCurveEngine
/*******************************************************************************************
*
*   Light Curve Engine
*   Author: Liam Robinson
*   NOTE: This code is meant to interact with a light_curve.lcc (light curve command) file
*   written by MATLAB. All additional functionality should be implemented through the MATLAB
*   interface for future flexibility.
*
********************************************************************************************/

#include "include/raylib.h"
#include "include/rlgl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

//User-defined
#include "include/lightcurvelib.c"

#define RLIGHTS_IMPLEMENTATION
#include "include/rlights.h"

#define MAX_INSTANCES          25
#define MAX_DATA_POINTS        10000
#define MAX_FNAME_LENGTH       100
#define MAX_DIR_LENGTH         200

int main(int argc, char **argv)
{
    //--------------------------------------------------------------------------------------
    // Initialization
    //--------------------------------------------------------------------------------------
    int width_pix = 10*60;
    char command_filename[] = "light_curve0.lcc"; 
    int inst_count; 
    Vector3 sun_vectors[MAX_DATA_POINTS];
    Vector3 viewer_vectors[MAX_DATA_POINTS]; 
    int data_points;
    char results_file[MAX_FNAME_LENGTH] = "light_curve0.lcr";
    char model_name[MAX_FNAME_LENGTH] = "cube.obj";
    int frame_rate = 1000;
    int save_imgs = 0;
    char out_dir[MAX_DIR_LENGTH] = "out";
    bool show_window = false;
    int use_brdf;
    float cd = 0.5;
    float cs = 0.5;
    float n = 10;
    int provided_optical_coeffs = 0;
    int use_mtl_properties = 0;

    int opt;
    while((opt = getopt(argc, argv, "m:d:i:f:c:r:b:x:sD:S:N:vM")) != -1) 
    { 
        switch(opt) 
        { 
            case 'm': 
                strncpy(model_name, optarg, sizeof(model_name)); break;
            case 'd':
                width_pix = atoi(optarg);
                if(width_pix % 60) {
                  printf("Rejected argument (-d %d) \nFor shading accuracy, square pixel dimensions must be multiple of 60\n", width_pix);
                  exit(EXIT_FAILURE);
                }
                break;
            case 'i': 
                inst_count = atoi(optarg); 
                if(inst_count > MAX_INSTANCES || inst_count <= 0) {
                  printf("Rejected argument (-i %d) \nInstance count must be between 1 and %d\n", inst_count, MAX_INSTANCES);
                  exit(EXIT_FAILURE);
                }
                break; 
            case 'f': 
                frame_rate = atoi(optarg);
                if(frame_rate <= 0) {
                  printf("Rejected argument (-f %d) \nFrame rate must be positive \n", frame_rate);
                  exit(EXIT_FAILURE);
                }
                break;           
            case 'c':
                data_points = atoi(optarg);
                if(data_points > MAX_DATA_POINTS || data_points <= 0) {
                  printf("Rejected argument (-c %d) \nData count must be between 1 and %d\n", data_points, MAX_DATA_POINTS);
                  exit(EXIT_FAILURE);
                }
                break;
            case 'r':
                strncpy(results_file, optarg, sizeof(results_file)); break;
            case 'b':
                use_brdf = atoi(optarg); break;
            case 's':
                save_imgs = 1; break;
            case 'x':
                strncpy(out_dir, optarg, sizeof(out_dir)); break;
            case 'D':
                cd = atof(optarg); 
                provided_optical_coeffs++;
                break;
            case 'S':
                cs = atof(optarg); 
                provided_optical_coeffs++;
                break;
            case 'N':
                n = atof(optarg); 
                provided_optical_coeffs++;
                break;
            case 'v':
                show_window = true; break;
            case 'M':
                use_mtl_properties = 1;
            case ':': 
                printf("option needs a value\n"); break;
            case '?': 
                printf("unknown option: %c\n", optopt); break;
        }
    } 

    if(cd + cs > 1.0 || cd < 0 || cs < 0 || n < 0) {
      printf("Rejected BRDF Cd, Cs, or n. Make sure Cd + Cs <= 1, n > 0\n");
      exit(EXIT_FAILURE);
    }

    if(provided_optical_coeffs != 3 && provided_optical_coeffs > 0) {
      printf("All optical coefficients (cd, cs, n) must be provided if any are\n");
      exit(EXIT_FAILURE);
    }

    printf("Model name: %s\n", model_name);
    printf("Pixel width: %d\n", width_pix); 
    printf("Instance count: %d\n", inst_count);
    printf("Frame rate: %d\n", frame_rate);
    printf("Data points: %d\n", data_points);
    printf("Results file: %s\n", results_file);
    printf("BRDF index: %d\n", use_brdf);
    printf("BRDF cd: %.2f\n", cd);
    printf("BRDF cs: %.2f\n", cs);
    printf("BRDF n: %.2f\n", n);
    printf("Saving images: %s\n", (bool) save_imgs ? "true" : "false");
    printf(".mtl file properties %s be used\n", use_mtl_properties == 1 ? "WILL" : "will NOT");

    ReadLightCurveCommandFile(command_filename, sun_vectors, viewer_vectors, data_points);
    ClearLightCurveResults(results_file);

    Model model;
    Camera viewer_camera;
    Camera light_camera;
    float mesh_scale_factor;
    int grid_width;

    bool rendering = true;
    if(!show_window) {
      SetConfigFlags(FLAG_WINDOW_HIDDEN);
    }
    InitWindow(width_pix, width_pix, "Light Curve Engine");
    InitializeObjects(inst_count, &grid_width, model_name, &model, &viewer_camera, 
                      &light_camera, &mesh_scale_factor);

    // Loading depth shader
    Shader depthShader = LoadShader("shaders/shadow_depth.vs", "shaders/shadow_depth.fs");
    Shader lightingShader = LoadShader("shaders/lighting.vs", "shaders/lighting.fs");
    Shader min_shader = LoadShader("shaders/averaging.vs", "shaders/averaging.fs");

    int depth_light_mvp_locs[MAX_INSTANCES];
    int lighting_light_mvp_locs[MAX_INSTANCES];
    GetLCShaderLocations(&depthShader, &lightingShader, &min_shader, depth_light_mvp_locs, 
                         lighting_light_mvp_locs, inst_count);

    Light sun = CreateLight(LIGHT_DIRECTIONAL, Vector3Zero(), Vector3Zero(), WHITE, 
                            lightingShader);                    

    RenderTexture2D depthTex = LoadRenderTexture(width_pix, width_pix);
    RenderTexture2D renderedTex = LoadRenderTexture(width_pix, width_pix);
    RenderTexture2D averagedLCTex = LoadRenderTexture(ceil(sqrt(inst_count)), width_pix); 
    // Creates a RenderTexture2D minified (height x inst_count) for the light curve texture

    SetTargetFPS(frame_rate);

    float light_curve_results[MAX_DATA_POINTS];
    int frame_number = 0;
    bool rendered_first_frame = false;

    for(int m = 0; m < model.meshCount; m++) {
      Mesh mesh = model.meshes[m];
      rlUpdateVertexBuffer(mesh.vboId[0], mesh.vertices, mesh.vertexCount*3*sizeof(float), 0);
      // Update vertex position
      rlUpdateVertexBuffer(mesh.vboId[2], mesh.normals, mesh.vertexCount*3*sizeof(float), 0);
      // Update vertex normals
    }


    while (!WindowShouldClose() && rendering) // Detect window close button or ESC key
    {
      //----------------------------------------------------------------------------------
      // Update
      //----------------------------------------------------------------------------------
      BeginTextureMode(depthTex); // Enable drawing to texture
          ClearBackground(BLACK); // Clear texture background
      EndTextureMode();

      BeginTextureMode(renderedTex); // Enable drawing to texture
          ClearBackground(BLACK); // Clear texture background
      EndTextureMode();

      for(int inst = 0; inst < inst_count; inst++) {
        int render_index = inst + (frame_number * inst_count) % data_points; 
        // Selects the correct entry of the command file for this inst

        sun.position = sun_vectors[render_index];
        sun.position.x = 2.0f * sun.position.x;
        sun.position.y = 2.0f * sun.position.y;
        sun.position.z = 2.0f * sun.position.z;
        viewer_camera.position = viewer_vectors[render_index];
        viewer_camera.position.x = 2.0f * viewer_camera.position.x;
        viewer_camera.position.y = 2.0f * viewer_camera.position.y;
        viewer_camera.position.z = 2.0f * viewer_camera.position.z;
        light_camera.position = sun.position;
        UpdateLightValues(lightingShader, sun);
        UpdateLightValues(depthShader, sun);

        Vector3 mesh_offsets[MAX_INSTANCES] =             {0};
        Vector3 viewer_camera_transforms[MAX_INSTANCES] = {0};
        Vector3 light_camera_transforms[MAX_INSTANCES] =  {0};
        Matrix mvp_lights[MAX_INSTANCES] =                {0};
        Matrix mvp_viewer[MAX_INSTANCES] =                {0};
        Matrix mvp_light_biases[MAX_INSTANCES] =          {0};

        GenerateTranslations(mesh_offsets, viewer_camera, inst_count);
        viewer_camera_transforms[inst] = TransformOffsetToCameraPlane(viewer_camera, 
                                                                      mesh_offsets[inst]);
        light_camera_transforms[inst] = TransformOffsetToCameraPlane(light_camera, 
                                                                     mesh_offsets[inst]);

        // Calculates the model-view-projection matrix for the light_camera
        mvp_lights[inst] = CalculateMVPFromCamera(light_camera, mesh_offsets[inst]); 
        mvp_viewer[inst] = CalculateMVPFromCamera(viewer_camera, mesh_offsets[inst]);
        // Takes [-1, 1] -> [0, 1] for texture sampling
        mvp_light_biases[inst] = CalculateMVPBFromMVP(mvp_lights[inst]);

        SetShaderValue(lightingShader, lightingShader.locs[LIGHTING_SHADER_LOC_VIEW_POS], 
          &viewer_camera.position, SHADER_UNIFORM_VEC3); 
        SetShaderValue(lightingShader, lightingShader.locs[LIGHTING_SHADER_LOC_LIGHT_POS], 
          &sun.position, SHADER_UNIFORM_VEC3); 
        SetShaderValue(lightingShader, lightingShader.locs[LIGHTING_SHADER_LOC_BRDF_INDEX], 
          &use_brdf, SHADER_UNIFORM_INT); 

        //----------------------------------------------------------------------------------
        // Write to depth texture
        //----------------------------------------------------------------------------------
        BeginTextureMode(depthTex);
        // Enable drawing to texture
            BeginMode3D(light_camera);
            // Begin 3d mode drawing
                for(int m = 0; m < model.materialCount; m++) {
                  model.materials[m].shader = depthShader;
                }
                // Assign depth texture shader to model
                SetShaderValue(depthShader, depthShader.locs[LIGHTING_SHADER_LOC_MODEL_INSTANCE],
                   &inst, SHADER_UNIFORM_INT); 
                //Sends the light position vector to the lighting shader
                SetShaderValueMatrix(depthShader, depth_light_mvp_locs[inst], mvp_lights[inst]);
                SetShaderValue(depthShader, depthShader.locs[LIGHTING_SHADER_LOC_LIGHT_POS], 
                  &sun.position, SHADER_UNIFORM_VEC3);
                //Sends the light position vector to the depth shader
                DrawModel(model, Vector3Zero(), 0.0f, WHITE);
            EndMode3D();
        EndTextureMode();
        //----------------------------------------------------------------------------------
        // Write to the rendered texture
        //----------------------------------------------------------------------------------
        BeginTextureMode(renderedTex);
          for(int m = 0; m < model.materialCount; m++) {
            model.materials[m].shader = lightingShader;
          }
          // Sets the model's shader to the lighting shader (was the depth shader)

          for(int m = 0; m < model.meshCount; m++) {
          DrawTextureRec(depthTex.texture, (Rectangle){0, 0, 0, 0}, (Vector2){0, 0}, WHITE);
          SetShaderValueTexture(lightingShader, lightingShader.locs[LIGHTING_SHADER_LOC_DEPTH_TEX], depthTex.texture);
          // Sends depth texture to the main lighting shader    

            BeginMode3D(viewer_camera);
              SetShaderValueTexture(lightingShader, 
                lightingShader.locs[LIGHTING_SHADER_LOC_DEPTH_TEX], depthTex.texture);
              SetShaderValueMatrix(lightingShader, 
                lightingShader.locs[LIGHTING_SHADER_LOC_VIEW_MVP], mvp_viewer[inst]);
              SetShaderValueMatrix(lightingShader, 
                lightingShader.locs[LIGHTING_SHADER_LOC_LIGHT_MVP], mvp_light_biases[inst]);
              SetShaderValue(lightingShader, 
                lightingShader.locs[LIGHTING_SHADER_LOC_GRID_WIDTH], &grid_width, SHADER_UNIFORM_INT);
              SetShaderValue(lightingShader, 
                lightingShader.locs[LIGHTING_SHADER_LOC_BRDF_CD], &cd, SHADER_UNIFORM_FLOAT);
              SetShaderValue(lightingShader, 
                lightingShader.locs[LIGHTING_SHADER_LOC_BRDF_CS], &cs, SHADER_UNIFORM_FLOAT);
              SetShaderValue(lightingShader, 
                lightingShader.locs[LIGHTING_SHADER_LOC_BRDF_N], &n, SHADER_UNIFORM_FLOAT);
              SetShaderValue(lightingShader, 
                lightingShader.locs[LIGHTING_SHADER_LOC_USE_MTL_FLAG], &use_mtl_properties, SHADER_UNIFORM_INT);

              Matrix matTrans = MatrixTranslate(viewer_camera_transforms[inst].x, 
                                    viewer_camera_transforms[inst].y, 
                                    viewer_camera_transforms[inst].z);
              DrawMesh(model.meshes[m], model.materials[model.meshMaterial[m]], matTrans);
          EndMode3D();
          }
        EndTextureMode();
      }

      BeginTextureMode(averagedLCTex);
        ClearBackground(BLACK);
        BeginShaderMode(min_shader);
          SetShaderValue(min_shader, min_shader.locs[LIGHTING_SHADER_LOC_GRID_WIDTH], 
            &grid_width, SHADER_UNIFORM_INT); 
          // Sends the light position vector to the lighting shader
          DrawTextureRec(renderedTex.texture, (Rectangle){0, 0, (float) width_pix, 
                                                          (float) -width_pix},
                         (Vector2){0, 0}, WHITE);
        EndShaderMode();
      EndTextureMode();

      float clipping_area = CalculateCameraArea(viewer_camera);
      float lc_function[MAX_INSTANCES];
      CalculateLightCurveValues(lc_function, averagedLCTex, renderedTex,
                                clipping_area, inst_count, mesh_scale_factor);
      
      // Storing light curve function values
      for(int i = 0; i < inst_count; i++) {
        int data_point_index = (frame_number * inst_count) % data_points + i;
        light_curve_results[data_point_index] = lc_function[i];

        if(data_point_index + 1 == data_points) {
          WriteLightCurveResults(results_file, light_curve_results, data_points);
          rendering = false;
        }
      }

      BeginDrawing();
        ClearBackground(BLACK);
        DrawTextureRec(renderedTex.texture, (Rectangle){0, 0, depthTex.texture.width, 
                                                        (float) -depthTex.texture.height}, 
                                                        (Vector2){0, 0}, WHITE);
      EndDrawing();

      frame_number++;

      if(save_imgs == 1) {
        char img_fname[300];
        sprintf(img_fname, "%s/frame%d.png", out_dir, frame_number);
        SaveScreen(img_fname, renderedTex.texture);
        // char mask_fname[300];
        // sprintf(mask_fname, "%s/frame%d_mask.png", out_dir, frame_number);
        // SaveModelMask(mask_fname, renderedTex.texture);
        // char camera_depth_fname[300];
        // sprintf(camera_depth_fname, "%s/frame%d_cam_depth.png", out_dir, frame_number);
        // SaveCameraDepthMask(camera_depth_fname, depthTex.texture);
      }
    }

    CloseWindow();
    return 0;
}