A Load Balancing Library (ALL)

The library aims to provide an easy way to include dynamic domain-based load balancing
into particle based simulation codes. The library is developed in the Simulation Laboratory Molecular Systems of the Juelich Supercomputing Centre at 
Forschungszentrum Juelich. 

It includes or is going to include several load-balancing schemes:

a) Tensor-Product:      the work on all processes is reduced over the cartesian 
            		    planes in the systems. This work is then equalized by 
		                adjusting the borders of the cartesian planes.
                 
b)  Staggered-grid:     a 3-step hierarchical approach is applied, where:
                        (i) work over the cartesian planes is reduced, before the borders of these 
                        planes are adjusted; (ii) in each of the cartesian planes the work is 
                        reduced for each cartesian column. These columns are then adjusted to each
                        other to homogenize the work in each column; (iii) the work between 
                        neighboring domains in each column is adjusted. Each adjustment is done
                        locally with the neighboring planes, columns or domains by adjusting the 
                        adjacent boundaries.
                    
c)  Topological Mesh:   In contrast to the previous methods this method adjusts 
                        domains not by moving boundaries but vertices, i.e. corner points, of
                        domains. For each vertex a force, based on the differences
                        in work of the neighboring domains, is computed and the
                        vertex is shifted in a way to equalize the work between these
                        neighboring domains.
                    
d)  Voronoi Mesh:       Similar to the topological mesh method, this method computes a
                        force, based on work differences. In contrast to the topological mesh
                        method, the force acts on a Voronoi point rather than a vertex, i.e. a 
                        point defining a Voronoi cell, which describes the domain. Consequently,
                        the number of neighbors is not a conserved quantity, i.e. the topology
                        may change over time. 

Installation & Requirements:

    Base requirements:
        C++11
        MPI
        CMake 3.1+

    Optional requirements:
        Fortran 2003
        VTK 7.1+ (for VTK-based output of the domains)

    Installation:

        1.) Either clone the library from 
                https://gitlab.version.fz-juelich.de/SLMS/loadbalancing
            or download it from the same location into a directory on
            your system ($ALL_ROOT_DIR).

        2.) Create a build-directory $ALL_BUILD_DIR in $ALL_ROOT_DIR, change into $ALL_BUILD_DIR and call
                cmake $ALL_ROOT_DIR <options>
            which sets up the installation. There are some optional features,
            you can set up with the following options:
                -DCMAKE_INSTALL_PREFIX=<$ALL_INSTALL_DIR> [default: depends on system]
                    sets the directory $ALL_INSTALL_DIR, into which ?make install? copies the compiled library
                    and examples
                -DCM_ALL_VTK_OUTPUT=ON/OFF [default: OFF]
                    enables/disables the VTK based output of the domains (requires
                    VTK 7.1+)
                -DCM_ALL_FORTRAN=ON/OFF [default: OFF]
                    compiles the Fortran interface and example

        3.) Execute make to compile and install the library to the previously set
            directory:
                make
                make install

    After "make install" the compiled library and the compiled examples are located in the directory $ALL_INSTALL_DIR.

Usage:
    
    ALL uses C++ template programming to deal with different data types that describe domain
    boundaries and domain based work. In order to capsulate the data, ALL uses a class in which
    required data and the computed results of a load-balancing routine are saved and can be
    accessed from. To include the library to an existing code, you need to do the following steps:

    1.) Create an object of the load-balancing class:
       
            ALL<T,W> ()
            ALL<T,W> ( int dimension, T gamma )  
            ALL<T,W> ( int dimension, std::vector<ALL_Point<T>> vertices, T gamma )

        As mentioned before the library uses template programming, where T is the data type used
        to describe boundaries between domains and vertices (usually float or double) and W is the
        data type used to describe the work-load of a process (usually float or double).
        The first version of the constructor defines a base object of the load-balancing class that
        contains no data, using a three-dimensional system. The second constructor sets up the
        system dimension (currently only three-dimensional systems are supported) and the relaxation
        parameter gamma, which controls the convergence of the load-balancing methods. In the third described version 
        of the ALL<T,W> constructor a set of vertices describing the local domain is already passed, using the
        ALL_Point Class, described below.

            ALL_Point<T>( const int dimension )
            ALL_Point<T>( const int dimension, const T* values )
            ALL_Point<T>( const std::vector<T>& values )

            void ALL_Point<T>::set_coordinates( const T* values )
            void ALL_Point<T>::set_coordinates( const std::vector<T> values )

            void ALL_Point<T>::set_coordinate(const int idx, const T& values )
            T ALL_Point<T>::x(int index)

        ALL_Point is a class describing a point in space, where the dimension of the space is given
        by the input parameter. It can be initialized by either using an array of datatype T or a
        std::vector<T>. In the latter case the passing of a dimension is not required and the dimension
        of the point is derived from the length of the std::vector. For the initialization with an
        array the user has to check that the passed array is of sufficient length, i.e. of length
        dimension (or longer).
        To update initialized ALL_Point objects, either all coordinates can be updated with the use
        of the set_coordinates methods, using either an array of data type T or a std::vector<T> as
        source for the new values. Like before in the case of the array, the user has to check for
        correctness of the array before passing it to the object. Single coordinates can be modified
        with the use of the set_coordinate method, while coordinate values can be accessed with the
        ALL_Point<T>::x method.

    2.) Setup basic parameters of the system:
        
        There are three input parameters that are required for the library:

            a) vertices describing the local domain
            b) work load for the local domain
            c) cartesian MPI communicator on which the program is executed
               (requirement of cartesian communicator is under review)

        These parameters can be set with the following methods:

            void ALL<T,W>::set_vertices(std::vector<ALL_Point<T>>& vertices)
            void ALL<T,W>::set_vertices(const int n, const int dimension, const T*)

            void ALL<T,W>::set_communicator(MPI_Comm comm)

            void ALL<T,W>::set_work(const W work)
       
        The vertices can be set by either using a std::vector of ALL_Point data, from which the number
        of vertices and the dimension of each point can be derived, or by passing an array of data type
        T, which requires that the number of vertices and the dimension of the vertices are also passed.
        For the MPI communicator the MPI communicator used by the calling program needs to be passed and
        for the work a single value of data type W needs to be passed.

    3.) Setting up the chosen load balancing routine

        To set up the required internal data structures a call of the following method is required:

            void ALL<T,W>::setup( short method )

            with ?method? being:
            
            ALL_LB_t::TENSOR
            ALL_LB_t::STAGGERED

        With the keyword method the load balancing strategy is chosen, given by the list above. Starting 
        point for all methods, described below is a given domain structure (e.g. the one which is initially 
        set up by the program). In the case of TENSOR and STAGGERED, the domains need to be orthogonal. 
        For these two methods, the procedure to adjust the work load between domains is a multi-step 
        approach where in each step, sets of domains are combined into super-domains, the work of which 
        is mutually adjusted. 

        Short overview about the methods:

            ALL_LB_t::TENSOR
            
                In order to equalize the load of individual domains, the assumption is made that this
                can be achieved by equalizing the work in each cartesian direction, i.e. the work
                of all domains having the same coordinate in a cartesian direction is collected and
                the width of all these domains in this direction is adjusted by comparing this
                collective work with the collective work of the neighboring domains. This is done
                independently for each cartesian direction in the system.

                Required number of vertices:
                    two, one describing the lower left front point and one describing the 
                    upper right back point of the domain

                Advantages:
                    - topology of the system is maintained (orthogonal domains, neighbor relations)
                    - no update for the neighbor relations need to be made in the calling code
                    - if a code was able to deal with orthogonal domains, only small changes are 
                      expected to include this strategy
                
                Disadvantages:
                    - due to the comparison of collective work loads in cartesian layers and 
                      restrictions resulting from this construction, the final result might 
                      lead to a sub-optimal domain distribution

            ALL_LB_t::STAGGERED:

                The staggered grid approach is a hierarchical one. In a first step the work of all
                domains sharing the same cartesian coordinate with respect to the highest dimension
                (z in three dimensions) is collected. Then, like in the TENSOR strategy the layer width in
                this dimension is adjusted based on comparison of the collected work with the collected
                work of the neighboring domains in the same dimension. As a second step each of these planes
                is divided into a set of columns, where all domains share the same cartesian coordinate
                in the next lower dimension (y in three dimensions). For each of these columns the
                before described procedure is repeated, i.e. work collected and the width of the
                columns adjusted accordingly. Finally, in the last step the work of individual domains
                is compared to direct neighbors and the width in the last dimension (x in three
                dimensions) adjusted. This leads to a staggered grid, that is much better suited to
                describe inhomogenous work distributions than the TENSOR strategy.

                Required number of vertices:
                    two, one describing the lower left front point and one describing the 
                    upper right back point of the domain

                Advantages:
                    - very good equalization results for the work load
                    - maintains orthogonal domains

                Disadvantages:
                    - changes topology of the domains and requires adjustment of neighbor relations
                    - communication pattern in the calling code might require adjustment to deal
                      with changing neighbors

    4.) Computing new boundaries / vertices

            void ALL<T,W>::balance(short method)
        
        The balance method starts the computation of the new vertices, according to the chosen method.
        The chosen method is required to be the same that was used in the call to the setup method. If
        required new neighbor relations are computed as well.

     5.) Accessing the results

            std::vector<ALL_Point<T>>& ALL<T,W>::get_result_vertices()
            
            int ALL<T,W> ALL<T,W>::get_n_neighbors(short method)
            void ALL<T,W> ALL<T,W>::get_neighbors(short method, int** neighbors)

            void ALL<T,W> ALL<T,W>::get_neighbors(short method, std::vector<int>& neighbors)


        In order to access the resulting vertices and neighbors, the above three methods can be used. If the
        array version of get_neighbors is used, the address to the neighbor array is returned in
        neighbors, therefore a int** is passed.

Examples:

    In the distribution two example codes are included, one written in C++, one written in Fortran. The main
    goal of these examples is to show how the library can be included into a particle code.

    ALL_test: 

        MPI C++ code that generates a particle distribution over the domains. At program start the domains form
        an orthogonal decomposition of the cubic 3d system. Each domain has the same volume as each other domain.
        Depending on the cartesian coordinates of the domain in this decomposition, a number of points is created
        on each domain. The points are then distributed uniformly over the domain. For the number of points a
        polynomial formula is used to create a non-uniform point disribution. As an estimation of the work load
        in this program the number of points within a domain was chosen. There is no particle movement included
        in the current version of the example code, therefore particle communication only takes place due to the
        shift of domain borders.

        The program creates three output files in its basic version:

            minmax.dat:
                contains four columns:
                    <iteration count> <W_min/W_avg> <W_max_W_avg> <(W_max-W_min)/(W_max+W_min)>
                explanation:
                    In order to try to give a comparable metric for the success of the load balancing procedure
                    the relatative difference of the minimum and maximum work loads in the system in relation
                    to the average work load of the system are given. The last column gives an indicator for
                    the inequality of work in the system, in a perfectly balanced system, this value should be
                    equal to zero.
            
            stddev.txt:
                contains two columns:
                    <iteration count> <std_dev>
                explanation:
                    The standard deviation of work load over all domains.

            domain_data.dat:
                contains two larger sets of data:
                    <rank> <x_min> <x_max> <y_min> <y_max> <z_min> <z_max> <W>
                explanation:
                    There are two blocks of rows, looking like this, the first block is the starting configuration
                    of the domains, which should be a uniform grid of orthogonal domains. The second block is the
                    configuration after 200 load balancing steps. In each line the MPI rank of the domain and the
                    extension in each cartesian direction is given, as well as the work load (the number of points).

        If the library is compiled with VTK support, ALL_test also creates a VTK based output of the domains in each
        iteration step. In order to work, a directory named "vtk_outline" needs to be created in the directory where
        the executable is located. The resulting output can be visualized with tools like ParaView or VisIt.

   ALL_test_f: 

        The Fortran example provides a more basic version of the test program ALL_test, as its main goal is to show
        the functuality of the Fortran interface. The code creates a basic uniform orthogonal domain decomposition
        and creates an inhomogenous particle distribution over these. Only one load balancing step is executed and
        the program prints out the domain distribution of the start configuration and of the final configuration.
