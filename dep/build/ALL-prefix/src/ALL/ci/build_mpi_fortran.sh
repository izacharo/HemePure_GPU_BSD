#!/bin/bash
source $(cd "$(dirname "$0")"; pwd -P)/ci_funcs.sh

# create badge
create_badge "${BADGE_FILENAME}" build-mpi unknown "--color=#808080"
[[ ${BADGE_ONLY} ]] && pushbadge_exit "${BADGE_FILENAME}" 0

# load MPI environment
load_MPI

# build
mkdir -p build && cd build

if [[ $? == 0 ]]; then

    CC=mpicc CXX=mpicxx FC=mpif90 ${CMAKE} .. -DCM_ALL_FORTRAN=ON
    make VERBOSE=1 

    if [[ $? == 0 ]]; then
        create_badge "${BADGE_FILENAME}" build-mpi passed --color=green
        pushbadge_exit "${BADGE_FILENAME}" 0
    fi
fi

create_badge "${BADGE_FILENAME}" build-mpi failed --color=red
pushbadge_exit "${BADGE_FILENAME}" 1
