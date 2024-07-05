/*
Copyright 2018 Rene Halver, Forschungszentrum Juelich GmbH, Germany
Copyright 2018 Godehard Sutmann, Forschungszentrum Juelich GmbH, Germany

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this 
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this 
   list of conditions and the following disclaimer in the documentation and/or 
   other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may 
   be used to endorse or promote products derived from this software without 
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/

#include "../include/ALL.hpp"

extern "C"
{
    // wrapper to create a ALL<double,double> object (should be the
    // most commonly used one)
    ALL<double,double>* ALL_init_f(const int dim, double gamma)
    {
        return new ALL<double,double>(dim,gamma);
    }

    // wrapper to set the work (scalar only for the moment)
    void ALL_set_work_f(ALL<double,double>* all_obj, double work)
    {
        all_obj->set_work(work);
    }

    // wrapper to set the vertices (using an array of double values and dimension)
    void ALL_set_vertices_f(ALL<double,double>* all_obj, const int n, const int dim, const double* vertices)
    {
        all_obj->set_vertices(n,dim,vertices);
    }

    // wrapper to set the communicator
    void ALL_set_communicator_f(ALL<double,double>* all_obj, MPI_Comm comm)
    {
        all_obj->set_communicator(comm);
    }

    // wrapper to setup routine
    void ALL_setup_f(ALL<double,double>* all_obj, ALL_LB_t method)
    {
        all_obj->setup(method);
    }

    // wrapper to call balance routine
    void ALL_balance_f(ALL<double,double>* all_obj, ALL_LB_t method)
    {
        all_obj->balance(method);
    }

    // wrapper to get number of new vertices
    void ALL_get_new_number_of_vertices_f(ALL<double,double>* all_obj, int* n_vertices)
    {
        *n_vertices = (all_obj->get_result_vertices()).size(); 
    }

    // wrapper to return new vertices
    void ALL_get_vertices_f(ALL<double,double>* all_obj, int* n_vertices, double* vertices)
    {
        std::vector<ALL_Point<double>> tmp_vertices = all_obj->get_vertices();
        int dimension = all_obj->get_dimension();
        *n_vertices = tmp_vertices.size();
        for (int i = 0; i < *n_vertices; ++i)
        {
            for (int j = 0; j < dimension; ++j)
            {
                vertices[i * dimension + j] = tmp_vertices.at(i).x(j);
            }
        }
    }

    // wrapper to return new vertices
    void ALL_get_new_vertices_f(ALL<double,double>* all_obj, int* n_vertices, double* new_vertices)
    {
        std::vector<ALL_Point<double>> tmp_vertices = all_obj->get_result_vertices();
        int dimension = all_obj->get_dimension();
        *n_vertices = tmp_vertices.size();
        for (int i = 0; i < *n_vertices; ++i)
        {
            for (int j = 0; j < dimension; ++j)
            {
                new_vertices[i * dimension + j] = tmp_vertices.at(i).x(j);
            }
        }
    }
}
