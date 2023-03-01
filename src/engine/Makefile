all: report_init make_registry make_engine

ifeq ($(OS),Windows_NT) 
    OS := Windows
	COMPILER := gcc
	ARCH := arm
else
    OS := $(shell sh -c 'uname 2>/dev/null || echo Unknown')
	ARCH := $(shell uname -p)
	COMPILER := clang
endif

ifeq (${ARCH},arm)
	USE_LIBRAYLIB := engine_lib/libraylib_m1.a
else
	USE_LIBRAYLIB := engine_lib/libraylib.a
endif

report_init:
	@echo "Found OS: ${OS}"
	@echo "Found architecture: ${ARCH}"
	@echo "Using compiler: ${COMPILER}"

make_registry:
	@echo "\n-=-=-=- Compiling BRDF registry -=-=-=-\n"
	${COMPILER} BRDFRegistry.c -o BRDFRegistry

make_engine:
	@echo "\n-=-=-=- Compiling Light Curve Engine -=-=-=-\n"
	${COMPILER} -framework CoreVideo -framework IOKit -framework Cocoa -framework OpenGL ${USE_LIBRAYLIB} LightCurveEngine.c -o LightCurveEngine