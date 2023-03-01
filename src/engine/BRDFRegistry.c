#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>
#include<string.h>

const char* BRDFNames[] = {
    "diffuse",
    "phong",
    "blinn-phong",
    "ashikhmin-shirley",
    "cook-torrance",
    "glossy",
    "oren-nayar"
};
const int num_brdfs = sizeof(BRDFNames) / sizeof(BRDFNames[0]);

int main(int argc, char **argv) {
    if(argc == 2) {
        char* req_brdf_name = argv[1];
        if(strcmp(req_brdf_name, "all") != 0) {
            for(int i = 0; i < num_brdfs; i++) {
                if(strcmp(req_brdf_name, BRDFNames[i]) == 0) {
                    return i;
                }
            }
            printf("BRDF name %s not found in registry\n", req_brdf_name);
        }
        else {
            printf("-=-=- All available BRDFs -=-=-\n");
            for(int i = 0; i < num_brdfs; i++) {
                printf("%d: %s\n", i, BRDFNames[i]);
            }
            return -2;
        }
    }
    else {
        printf("The BRDF registry must be called with one integral command line argument\n");
    }
    return -1;
}