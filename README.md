# APM Assignment

# Building

## Using the provided makefile

### On Ruche

Setup FreeImage with the instructions and using the default compiler. Then:

```shell
module load cuda/11.4.0/gcc-9.2.0
module load gcc/11.2.0/gcc-4.8.5
```

The makefile is unchanged, so I guess it should work as expected. Just type `make` to produce an executable.

## Using CMake

If you want to build with `clang++`, do `export CUDACXX=clang++`. If FreeImage is in an unusual location, set `FreeImage_HINT` to the root of the installation location.. To target a specific architecture
pass `CMAKE_CUDA_ARCHITECTURES=XX` to cmake.ls

Finally, build with:

```shell
mkdir build && cd build
cmake .. && cmake --build . --target main --parallel
```

# Execution

The first argument of the executable is the image path. The second is optional. If `-benchmark` is passed as the second argument, then all the filters will be benchmarked on the specified image, but no output will be
saved.

