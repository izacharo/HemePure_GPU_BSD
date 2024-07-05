!Copyright 2018 Rene Halver, Forschungszentrum Juelich GmbH, Germany
!Copyright 2018 Godehard Sutmann, Forschungszentrum Juelich GmbH, Germany
!
!Redistribution and use in source and binary forms, with or without modification, are 
!permitted provided that the following conditions are met:
!
!1. Redistributions of source code must retain the above copyright notice, this list 
!   of conditions and the following disclaimer.
!
!2. Redistributions in binary form must reproduce the above copyright notice, this 
!   list of conditions and the following disclaimer in the documentation and/or 
!   other materials provided with the distribution.
!
!3. Neither the name of the copyright holder nor the names of its contributors 
!   may be used to endorse or promote products derived from this software without 
!   specific prior written permission.
!
!THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
!EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
!OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
!SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
!INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
!TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
!BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
!CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
!ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
!SUCH DAMAGE.

! module for ALL access from Fortran
MODULE ALL_module
    USE ISO_C_BINDING
    IMPLICIT NONE
    PUBLIC
        TYPE ALL_t
            PRIVATE
            TYPE(c_ptr) :: object = C_NULL_PTR
        END TYPE
    ! definitions of enum type for different methods
        INTEGER(c_short), parameter ::  ALL_STAGGERED = 0
        INTEGER(c_short), parameter ::  ALL_TENSOR = 1
        INTEGER(c_short), parameter ::  ALL_UNSTRUCTURED = 2
        INTEGER(c_short), parameter ::  ALL_CELLS = 3
        INTEGER(c_short), parameter ::  ALL_VORONOI = 4
    ! interface functions / subroutines to C++
    INTERFACE
        FUNCTION ALL_init_int(dim, gamma) RESULT(this) BIND(C,NAME="ALL_init_f")
            USE ISO_C_BINDING
            INTEGER(c_int),VALUE    :: dim
            REAL(c_double),VALUE    :: gamma
            TYPE(c_ptr)             :: this
        END FUNCTION
        SUBROUTINE ALL_set_work_int(obj, work) BIND(C,NAME="ALL_set_work_f")
            USE ISO_C_BINDING
            REAL(c_double),VALUE    ::  work
            TYPE(c_ptr), VALUE      ::  obj
        END SUBROUTINE
        SUBROUTINE ALL_set_vertices_int(obj, n, dim, vertices) BIND(C,NAME="ALL_set_vertices_f")
            USE ISO_C_BINDING
            TYPE(c_ptr), VALUE                  ::  obj
            INTEGER(c_int), VALUE               ::  n
            INTEGER(c_int), VALUE               ::  dim
            REAL(c_double), DIMENSION(n*dim)    ::  vertices
        END SUBROUTINE
        SUBROUTINE ALL_set_communicator_int(obj, comm) BIND(C,NAME="ALL_set_communicator_f")
            USE ISO_C_BINDING
            TYPE(c_ptr), VALUE              ::  obj
            INTEGER(c_int), VALUE           ::  comm
        END SUBROUTINE
        SUBROUTINE ALL_setup_int(obj, method) BIND(C,NAME="ALL_setup_f")
            USE ISO_C_BINDING
            TYPE(c_ptr), VALUE              ::  obj
            INTEGER(c_short), VALUE         ::  method
        END SUBROUTINE
        SUBROUTINE ALL_balance_int(obj, method) BIND(C,NAME="ALL_balance_f")
            USE ISO_C_BINDING
            TYPE(c_ptr), VALUE              ::  obj
            INTEGER(c_short), VALUE         ::  method
        END SUBROUTINE
        SUBROUTINE ALL_get_new_number_of_vertices_int(obj,n) &
            BIND(C,NAME="ALL_get_new_number_of_vertices_f")
            USE ISO_C_BINDING
            TYPE(c_ptr), VALUE              ::  obj
            INTEGER(c_int)                  ::  n
        END SUBROUTINE
        SUBROUTINE ALL_get_vertices_int(obj,n,vertices) &
            BIND(C,NAME="ALL_get_vertices_f")
            USE ISO_C_BINDING
            TYPE(c_ptr), VALUE              ::  obj
            INTEGER(c_int)                  ::  n
            REAL(c_double)                  ::  vertices(*) 
        END SUBROUTINE
        SUBROUTINE ALL_get_new_vertices_int(obj,n,vertices) &
            BIND(C,NAME="ALL_get_new_vertices_f")
            USE ISO_C_BINDING
            TYPE(c_ptr), VALUE              ::  obj
            INTEGER(c_int)                  ::  n
            REAL(c_double)                  ::  vertices(*) 
        END SUBROUTINE
    END INTERFACE
    ! module subroutines to be called from Fortran code
    CONTAINS
        SUBROUTINE ALL_init(obj, dim, gamma)
            TYPE(ALL_t)     :: obj
            INTEGER         :: dim
            REAL(8)         :: gamma
            obj%object = ALL_init_int(INT(dim,c_int),REAL(gamma,c_double))
        END SUBROUTINE
        SUBROUTINE ALL_set_work(obj, work)
            TYPE(ALL_t)     ::  obj
            REAL(8)         ::  work
            CALL ALL_set_work_int(obj%object, work)
        END SUBROUTINE
        SUBROUTINE ALL_set_vertices(obj, n, dim, vertices)
            TYPE(ALL_t)     ::  obj
            INTEGER         ::  n
            INTEGER         ::  dim
            REAL(8)         ::  vertices(n*dim)
            CALL ALL_set_vertices_int(obj%object, n, dim, vertices)
        END SUBROUTINE
        SUBROUTINE ALL_set_communicator(obj, comm)
            TYPE(ALL_t)     ::  obj
            INTEGER         ::  comm
            CALL ALL_set_communicator_int(obj%object, comm)
        END SUBROUTINE
        SUBROUTINE ALL_setup(obj, method)
            USE ISO_C_BINDING
            TYPE(ALL_t)         ::  obj
            INTEGER(c_short)    ::  method
            CALL ALL_setup_int(obj%object,method)
        END SUBROUTINE
        SUBROUTINE ALL_balance(obj, method)
            USE ISO_C_BINDING
            TYPE(ALL_t)         ::  obj
            INTEGER(c_short)    ::  method
            CALL ALL_balance_int(obj%object,method)
        END SUBROUTINE
        SUBROUTINE ALL_get_new_number_of_vertices(obj,n)
            TYPE(ALL_t)         ::  obj
            INTEGER             ::  n
            CALL ALL_get_new_number_of_vertices_int(obj%object,n)
        END SUBROUTINE
        SUBROUTINE ALL_get_vertices(obj,n,vertices)
            TYPE(ALL_t)         ::  obj
            INTEGER             ::  n
            REAL(8)             ::  vertices(*)
            CALL ALL_get_vertices_int(obj%object,n,vertices)
        END SUBROUTINE
        SUBROUTINE ALL_get_new_vertices(obj,n,vertices)
            TYPE(ALL_t)         ::  obj
            INTEGER             ::  n
            REAL(8)             ::  vertices(*)
            CALL ALL_get_new_vertices_int(obj%object,n,vertices)
        END SUBROUTINE
END MODULE ALL_module

