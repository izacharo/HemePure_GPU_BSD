#!/bin/bash
## Compilation/build script for HEMELB
## Run from found location

## MODULE loads
export CC=$(which gcc)
export CXX=$(which g++)
export MPI_C_COMPILER=$(which mpicc)
export MPI_CXX_COMPILER=$(which mpicxx)

export SOURCE_DIR=/Dir_To_HemePure_GPU

## HEMELB build
# 1) Dependencies
BuildDep(){
cd $SOURCE_DIR/dep
rm -rf build
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} ..
make -j  && echo "Done HemeLB Dependencies"
cd ../..
}

#===============================================================================
# 2) Source code
# 2.1. Velocity (LADDIOLET) - Pressure (NashZerothOrderPressure) BCs
BuildSource_VP(){
cd $SOURCE_DIR/src
rm -rf build_VP
mkdir build_VP
cd build_VP
cmake  \
  -DCMAKE_CXX_FLAGS="-std=c++11 -g -Wno-narrowing" \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DHEMELB_USE_VELOCITY_WEIGHTS_FILE=ON \
  -DHEMELB_INLET_BOUNDARY=LADDIOLET \
  -DHEMELB_WALL_INLET_BOUNDARY=LADDIOLETSBB \
  -DHEMELB_OUTLET_BOUNDARY=NASHZEROTHORDERPRESSUREIOLET \
  -DHEMELB_WALL_OUTLET_BOUNDARY=NASHZEROTHORDERPRESSURESBB \
  -DHEMELB_LOG_LEVEL="Info" \
  -DHEMELB_USE_MPI_PARALLEL_IO=ON \
  -DCMAKE_CUDA_FLAGS="-ccbin g++ -gencode arch=compute_80,code=sm_80 -lineinfo --ptxas-options=-v --disable-warnings" \
        $SOURCE_DIR/src
make -j && echo "Done HemeLB Source"
cd ../..
}

#===============================================================================
# 2.2. Pressure - Pressure BCs
BuildSource_PP(){
cd $SOURCE_DIR/src
rm -rf build_PP
mkdir build_PP
cd build_PP
cmake  \
  -DCMAKE_CXX_FLAGS="-std=c++11 -g -Wno-narrowing" \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DHEMELB_USE_VELOCITY_WEIGHTS_FILE=OFF \
  -DHEMELB_INLET_BOUNDARY=NASHZEROTHORDERPRESSUREIOLET \
  -DHEMELB_WALL_INLET_BOUNDARY=NASHZEROTHORDERPRESSURESBB \
  -DHEMELB_OUTLET_BOUNDARY=NASHZEROTHORDERPRESSUREIOLET \
  -DHEMELB_WALL_OUTLET_BOUNDARY=NASHZEROTHORDERPRESSURESBB \
  -DHEMELB_LOG_LEVEL="Info" \
  -DHEMELB_USE_MPI_PARALLEL_IO=ON \
  -DCMAKE_CUDA_FLAGS="-ccbin g++ -gencode arch=compute_80,code=sm_80 -lineinfo --ptxas-options=-v --disable-warnings" \
        $SOURCE_DIR/src
make -j && echo "Done HemeLB Source"
cd ../..
}
#===============================================================================

#
BuildDep
BuildSource_PP
BuildSource_VP
echo "Done build all"
