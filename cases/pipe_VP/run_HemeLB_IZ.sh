#!/bin/bash

PROFILE_NSight_Sys=false
PROFILE_NSight_Com=false
PROFILE_NVPROF=false
CUDA_MEMCHECK=false
gdb_debug=false

# GPU version
##PATH_EXE=~/Documents/UCL_project/GPU_hackathon_UK_2022/hemeLB-GPU/HemePure-GPU/src/build_VP          
PATH_EXE=~/Documents/UCL_project/code_development/GPU_code/src_v2_2/build_VP
##PATH_EXE=/home/ioannis/Documents/UCL_project/code_development/GPU_code/src_v2_28a/build_VelPres
##PATH_EXE=~/Documents/UCL_project/code_development/GPU_code/src_v1_28a/build_VelInFile_PresOut_test
EXEC_FILE=hemepure_gpu
##EXEC_FILE=hemepure_gpu_F1e3

#CPU version for comparison
#PATH_EXE=~/Documents/UCL_project/code_development/HemePure_CPU/HemePure_test_validation/HemePure/src/buildVP
#EXEC_FILE=hemepure


INPUT_FILE=input.xml
nCPUs=4


rm -rf results

if [ "$PROFILE_NSight_Sys" = true ]
then
    echo 'Running and Profiling for NSight SYstems!'
    # Profiling using Nsight Systems
    #nsys profile --trace=cuda,mpi,nvtx --stats=true  mpirun -np $nCPUs $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results
    /opt/nvidia/nsight-systems/2021.3.2/target-linux-x64/nsys profile --force-overwrite=true --trace=cuda,mpi,nvtx --stats=true -o report_Test_NsightSys_n4_noPinned_reserve_vect_Direct_AsynchMcpy_movePreRec mpirun -np $nCPUs $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results
elif [ "$PROFILE_NSight_Com" = true ]
then
    echo 'Running and Profiling for NSight Compute!'
    # Profiling using Nsight Compute
    /opt/nvidia/nsight-compute/2020.3.1/target/linux-desktop-glibc_2_11_3-x64/ncu -f -o test_profile_ncu_hack --set detailed --target-processes all --replay-mode kernel --kernel-regex-base function --launch-skip-before-match 0 --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clo\
ck-control base --apply-rules yes --import-source yes --check-exit-code yes  mpirun -np 4 $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results

    #    /opt/nvidia/nsight-compute/2020.3.1/target/linux-desktop-glibc_2_11_3-x64/ncu -f -o test_profile_ncu --target-processes all --replay-mode kernel --kernel-regex-base function --launch-skip-before-match 0 --section LaunchStats --section Occupancy --section SpeedOfLight --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --apply-rules yes --import-source no --check-exit-code yes  mpirun -np 4 $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results

#    /opt/nvidia/nsight-compute/2022.1.0/ncu --set detailed -f -o testing_ncu --target-processes all --import-source yes  mpirun -np 4 $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results

    # /opt/nvidia/nsight-compute/2022.1.0/ncu --set detailed -f -o test_profile --target-processes all --import-source yes --check-exit-code yes  mpirun -np 4 $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results

 # /opt/nvidia/nsight-compute/2019.5.0/host/linux-desktop-glibc_2_11_3-x64/nv-nsight-cu --export "/home/ioannis/Documents/UCL_project/code_development/GPU_code/cases/bifurcation_hires_v1_30a/test_Prof_NsightComp_v2019_n2" --force-overwrite --target-processes all --section LaunchStats --section Occupancy --section SpeedOfLight --section SchedulerStats --section MemoryWorkloadAnalysis --section WarpStateStats --section SpeedOfLight_RooflineChart --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --apply-rules yes --import-source yes --check-exit-code yes  mpirun -np 4 $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results

#    /opt/nvidia/nsight-compute/2022.1.0/ncu --export "/home/ioannis/Documents/UCL_project/code_development/GPU_code/cases/bifurcation_hires_v1_30a/test_Prof_NsightComp_v2022" --force-overwrite --target-processes all --kernel-name-base function --launch-skip-before-match 0 --section LaunchStats --section Occupancy --section SpeedOfLight --section SchedulerStats --section MemoryWorkloadAnalysis --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --apply-rules yes --import-source no --check-exit-code yes  mpirun -np 4 $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results

    #ncu --export "/home/ioannis/Documents/UCL_project/code_development/GPU_code/cases/bifurcation_hires_v1_30a/test_Prof_NsightComp_v2022a" --force-overwrite --target-processes all mpirun -np 4 $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results
    
elif [ "$PROFILE_NVPROF" = true ]
then
    echo 'Running and Profiling with nvprof!'
     # Profiling using nvprof
    mpirun -np 4 nvprof -s -o nvprof_GeForce_NoCuAwMPI.%p.%q{OMPI_COMM_WORLD_RANK} --metrics gld_throughput  $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results
else
    if [ "$CUDA_MEMCHECK" = true ]
    then
	echo 'Performing CUDA memcheck!'
	mpirun -np $nCPUs cuda-memcheck --leak-check full --log-file "error_cuda_memcheck.txt"  $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results
    else
	#mpirun -np $nCPUs xterm -hold -e gdb -ex run --args $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results
	mpirun -np $nCPUs  $PATH_EXE/$EXEC_FILE -in $INPUT_FILE -out results
    fi    
fi
