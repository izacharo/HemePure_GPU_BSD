#!/bin/bash
## Compilation/build script for HEMELB
## Run from found location

export SOURCE_DIR=/grand/CompBioAffin/izacharo/hemeLB_GPU_BSD_5June2024/HemePure_GPU_BSD

# Load the modules - Here the ones for Polaris
module use /soft/modulefiles
module load spack-pe-base cmake
module load PrgEnv-gnu
module load nvhpc-mixed/23.9
module load craype-accel-nvidia80
module load cudatoolkit-standalone/12.4.0
module load craype-x86-milan
module load python
#
export CPATH=/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/include:$CPATH

export CC=$(which gcc)
export CXX=$(which g++)
export MPI_C_COMPILER=$(which mpicc)
export MPI_CXX_COMPILER=$(which mpicxx)
#


## HEMELB build
# 1) Dependencies
BuildDep(){
cd $SOURCE_DIR/dep
rm -rf build
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} ..
make -j32  && echo "Done HemeLB Dependencies"
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
  -DCMAKE_CXX_FLAGS="-std=c++17 -g -Wno-narrowing" \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DHEMELB_USE_VELOCITY_WEIGHTS_FILE=ON \
  -DHEMELB_INLET_BOUNDARY=LADDIOLET \
  -DHEMELB_WALL_INLET_BOUNDARY=LADDIOLETSBB \
  -DHEMELB_OUTLET_BOUNDARY=NASHZEROTHORDERPRESSUREIOLET \
  -DHEMELB_WALL_OUTLET_BOUNDARY=NASHZEROTHORDERPRESSURESBB \
  -DHEMELB_LOG_LEVEL="Info" \
  -DHEMELB_USE_MPI_PARALLEL_IO=ON \
  -DHEMELB_CUDA_AWARE_MPI=ON \
  -DCMAKE_CUDA_FLAGS="-ccbin g++ -gencode arch=compute_80,code=sm_80 -lineinfo --ptxas-options=-v --disable-warnings -maxrregcount 200" \
        $SOURCE_DIR/src
make -j32 && echo "Done HemeLB Source"
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
  -DCMAKE_CXX_FLAGS="-std=c++17 -g -Wno-narrowing" \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DHEMELB_USE_VELOCITY_WEIGHTS_FILE=OFF \
  -DHEMELB_INLET_BOUNDARY=NASHZEROTHORDERPRESSUREIOLET \
  -DHEMELB_WALL_INLET_BOUNDARY=NASHZEROTHORDERPRESSURESBB \
  -DHEMELB_OUTLET_BOUNDARY=NASHZEROTHORDERPRESSUREIOLET \
  -DHEMELB_WALL_OUTLET_BOUNDARY=NASHZEROTHORDERPRESSURESBB \
  -DHEMELB_LOG_LEVEL="Info" \
  -DHEMELB_USE_MPI_PARALLEL_IO=ON \
  -DHEMELB_CUDA_AWARE_MPI=ON \
  -DCMAKE_CUDA_FLAGS="-ccbin g++ -gencode arch=compute_80,code=sm_80 -lineinfo --ptxas-options=-v --disable-warnings -maxrregcount 200" \
        $SOURCE_DIR/src
make -j32 && echo "Done HemeLB Source"
cd ../..
}
#===============================================================================

#
BuildDep
BuildSource_PP
BuildSource_VP
echo "Done build all"
