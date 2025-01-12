#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <Sacado.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include "equation.h"

using namespace dealii;


template <int dim>
const double EulerEquations<dim>::gas_gamma = 1.4;

template <int dim>
EulerEquations<dim>::Postprocessor::
Postprocessor (const bool do_schlieren_plot)
		:
		do_schlieren_plot (do_schlieren_plot)
{}


// This is the only function worth commenting
// on. When generating graphical output, the
// DataOut and related classes will call this
// function on each cell, with values,
// gradients, hessians, and normal vectors
// (in case we're working on faces) at each
// quadrature point. Note that the data at
// each quadrature point is itself
// vector-valued, namely the conserved
// variables. What we're going to do here is
// to compute the quantities we're interested
// in at each quadrature point. Note that for
// this we can ignore the hessians ("dduh")
// and normal vectors; to avoid compiler
// warnings about unused variables, we
// comment out their names.
template <int dim>
void
EulerEquations<dim>::Postprocessor::evaluate_vector_field
(const DataPostprocessorInputs::Vector<dim> &input_data,
 std::vector<Vector<double>>                &computed_quantities) const
{
   // At the beginning of the function, let us
   // make sure that all variables have the
   // correct sizes, so that we can access
   // individual vector elements without
   // having to wonder whether we might read
   // or write invalid elements; we also check
   // that the <code>duh</code> vector only
   // contains data if we really need it (the
   // system knows about this because we say
   // so in the
   // <code>get_needed_update_flags()</code>
   // function below). For the inner vectors,
   // we check that at least the first element
   // of the outer vector has the correct
   // inner size:
   const std::vector<Vector<double>>& uh = input_data.solution_values;
   const std::vector<std::vector<Tensor<1,dim>>>& duh = input_data.solution_gradients;
   const unsigned int n_quadrature_points = uh.size();
   
   if (do_schlieren_plot == true)
      Assert (duh.size() == n_quadrature_points,
              ExcInternalError());
   
   Assert (computed_quantities.size() == n_quadrature_points,
           ExcInternalError());
   
   Assert (uh[0].size() == n_components,
           ExcInternalError());
   
   if (do_schlieren_plot == true)
      Assert (computed_quantities[0].size() == dim+2, ExcInternalError())
      else
         Assert (computed_quantities[0].size() == dim+1, ExcInternalError());
   
   // Then loop over all quadrature points and
   // do our work there. The code should be
   // pretty self-explanatory. The order of
   // output variables is first
   // <code>dim</code> velocities, then the
   // pressure, and if so desired the
   // schlieren plot. Note that we try to be
   // generic about the order of variables in
   // the input vector, using the
   // <code>first_momentum_component</code>
   // and <code>density_component</code>
   // information:
   for (unsigned int q=0; q<n_quadrature_points; ++q)
   {
      const double density = uh[q](density_component);
      
      for (unsigned int d=0; d<dim; ++d)
         computed_quantities[q](d) = uh[q](d) / density;
      
      computed_quantities[q](dim) = compute_pressure<double> (uh[q]);

      // Perturbation pressure useful for hydrostatic tests
      //computed_quantities[q](dim) = compute_pressure<double> (uh[q]) - std::exp(-evaluation_points[q][0]);
      //computed_quantities[q](dim) = compute_pressure<double> (uh[q]) - std::pow(1.0 - evaluation_points[q][0]/6.0,6);
      
      if (do_schlieren_plot == true)
         computed_quantities[q](dim+1) = duh[q][density_component] *
                                         duh[q][density_component];
   }
}


template <int dim>
std::vector<std::string>
EulerEquations<dim>::Postprocessor::get_names () const
{
  std::vector<std::string> names;
  names.push_back ("XVelocity");
  names.push_back ("YVelocity");
  if(dim==3)
     names.push_back ("ZVelocity");
  names.push_back ("Pressure");

  if (do_schlieren_plot == true)
    names.push_back ("schlieren_plot");

  return names;
}


template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
EulerEquations<dim>::Postprocessor::get_data_component_interpretation () const
{
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation (dim,
		    DataComponentInterpretation::component_is_part_of_vector);

  interpretation.push_back (DataComponentInterpretation::
			    component_is_scalar);

  if (do_schlieren_plot == true)
    interpretation.push_back (DataComponentInterpretation::
			      component_is_scalar);

  return interpretation;
}



template <int dim>
UpdateFlags
EulerEquations<dim>::Postprocessor::get_needed_update_flags () const
{
  if (do_schlieren_plot == true)
    return update_values | update_gradients | update_quadrature_points;
  else
    return update_values | update_quadrature_points;
}



template <int dim>
unsigned int
EulerEquations<dim>::Postprocessor::n_output_variables () const
{
  if (do_schlieren_plot == true)
    return dim+2;
  else
    return dim+1;
}

// To handle linking errors
template struct EulerEquations<2>;
