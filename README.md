# PyLightCurves
 Light curve inversion in python with OpenGL simulator backend

# Installation

## Mac (Intel or Apple CPU)
- Prerequisites: 
    - `clang` C compiler from XCode Command Line Tools. If installed, running `clang -v` in Terminal should return version information. If that errors, install directly via `xcode-select --install` or through the web at https://developer.apple.com/download/all/
    - `python3` should be installed by default on all Macs
    - `OpenGL` installed by default
- Clone repository to local folder
- Open terminal at repository top-level folder
- Run `source init_venv` to create and activate a python virtual environment, install dependencies, compile C executables for OpenGL light curve simulation, and run unit tests
- If all tests pass, the repository is fully initialized and all functions work as expected!
- Run `shape_invert_script.py` to perform a sample convex shape inversion

## Windows
- TBD, working on this on Friday with Alex

# Configuration
- The `MODELDIR` path set within `init_venv` can be set to a new directory containing `.obj` and `.mtl` files