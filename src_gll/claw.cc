#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>
//#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition_block.h>

#include <Sacado.hpp>


#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include "claw.h"
#include "ic.h"

// Coefficients for SSP-RK scheme
double ark[3], bcrk[3];
unsigned int n_rk;

using namespace dealii;

//------------------------------------------------------------------------------
// @sect4{ConservationLaw::ConservationLaw}
//
// There is nothing much to say about
// the constructor. Essentially, it
// reads the input file and fills the
// parameter object with the parsed
// values:
//------------------------------------------------------------------------------
template <int dim>
ConservationLaw<dim>::ConservationLaw (const char *input_filename,
                                       const unsigned int degree,
                                       const FE_DGQArbitraryNodes<dim> &fe_scalar)
   :
   fe (fe_scalar, EulerEquations<dim>::n_components),
   dof_handler (triangulation),
   fe_cell (FE_DGQ<dim>(0)),
   dh_cell (triangulation),
   verbose_cout (std::cout, false)
{
   read_parameters (input_filename);
   
   // For MOOD method compute degree reduction matrices
   if(parameters.solver == Parameters::Solver::mood)
      compute_reduction_matrices ();
}

//------------------------------------------------------------------------------
// Constructor for Pk basis
//------------------------------------------------------------------------------
template <int dim>
ConservationLaw<dim>::ConservationLaw (const char *input_filename,
                                       const unsigned int degree,
                                       const FE_DGP<dim> &fe_scalar)
:
fe (fe_scalar, EulerEquations<dim>::n_components),
dof_handler (triangulation),
fe_cell (FE_DGQ<dim>(0)),
dh_cell (triangulation),
verbose_cout (std::cout, false)
{
   read_parameters (input_filename);
   
   // create map from dof index to total degree of basis function
   index_to_degree.resize(fe.base_element(0).dofs_per_cell);
   unsigned int c = 0;
   if(dim==2)
   {
      for(unsigned int j=0; j<=degree; ++j)
         for(unsigned int i=0; i<=degree-j; ++i)
         {
            index_to_degree[c++] = i+j;
         }
   }
   else
   {
      AssertThrow(false, ExcMessage("Not implemented for dim=3"));
   }
}

//------------------------------------------------------------------------------
// Read parameters from file and set some other parameters.
//------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::read_parameters (const char *input_filename)
{
   AssertThrow(dim == 2, ExcMessage("Only works for 2-D"));
   
   ParameterHandler prm;
   Parameters::AllParameters<dim>::declare_parameters (prm);
   
   prm.parse_input (input_filename);
   parameters.parse_parameters (prm);
   
   verbose_cout.set_condition (parameters.output == Parameters::Solver::verbose);
   
   // Save all parameters in xml format
   //std::ofstream xml_file ("input.xml");
   //prm.print_parameters (xml_file,  ParameterHandler::XML);
   
   // Set coefficients for SSPRK
   if(fe.degree == 0)
   {
      ark[0] = 0.0;
      bcrk[0] = 0.0;
      n_rk = 1;
   }
   else if(fe.degree==1)
   {
      ark[0] = 0.0;
      ark[1] = 1.0/2.0;
      bcrk[0] = 0.0; bcrk[1] = 1.0;
      n_rk = 2;
   }
   else
   {
      ark[0] = 0.0;
      ark[1] = 3.0/4.0;
      ark[2] = 1.0/3.0;
      bcrk[0] = 0.0; bcrk[1] = 1.0; bcrk[2] = 0.5;
      n_rk = 3;
   }
}

//------------------------------------------------------------------------------
// Return mapping type based on selected type
//------------------------------------------------------------------------------
template <int dim>
const Mapping<dim,dim>& ConservationLaw<dim>::mapping() const
{
   if(parameters.mapping_type == Parameters::AllParameters<dim>::q1)
   {
      static MappingQ1<dim> m;
      return m;
   }
   else if(parameters.mapping_type == Parameters::AllParameters<dim>::q2)
   {
      static MappingQ<dim,dim> m(2);
      return m;
   }
   else if(parameters.mapping_type == Parameters::AllParameters<dim>::cartesian)
   {
      static MappingCartesian<dim> m;
      return m;
   }
   else
   {
      AssertThrow (false, ExcMessage("Requested mapping type is unknown"));
      static MappingQ1<dim> m;
      return m;
   }

}

//------------------------------------------------------------------------------
// Compute dx, dy, dz for cartesian meshes.
// At present it only checks that dx == dy
//------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::compute_cartesian_mesh_size ()
{
   //const double geom_tol = 1.0e-12;
   const double geom_tol = 1.0e-8;
   
   typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   
   for (; cell!=endc; ++cell)
   {
      double xmin = 1.0e20, xmax = -1.0e20;
      double ymin = 1.0e20, ymax = -1.0e20;
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
         const Point<dim>& p = cell->face(f)->center();
         xmin = std::min(xmin, p[0]);
         xmax = std::max(xmax, p[0]);
         ymin = std::min(ymin, p[1]);
         ymax = std::max(ymax, p[1]);
      }
      double dx = xmax - xmin;
      double dy = ymax - ymin;
      AssertThrow(std::fabs(dx-dy) < geom_tol, ExcMessage("Cell is not square"));
   }
}

//------------------------------------------------------------------------------
// Our nodal basis uses Gauss points and we use Gauss quadrature with degree+1
// nodes. Then the mass matrix is diagonal. The mass matrix is integrated exactly
// for Cartesian cells but not exact for general cells.
//------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::compute_inv_mass_matrix ()
{
   QGaussLobatto<dim> quadrature(fe.degree+1);
   unsigned int n_q_points = quadrature.size();
   FEValues<dim> fe_values(mapping(), fe, quadrature, update_values | update_JxW_values);
   
   std::vector<unsigned int> dof_indices(fe.dofs_per_cell);
   
   typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   
   inv_mass_matrix_diag.resize(triangulation.n_active_cells(),
                          Vector<double>(fe.dofs_per_cell));
   for (; cell!=endc; ++cell)
   {
      unsigned int c = cell_number(cell);
      cell->get_dof_indices (dof_indices);
      fe_values.reinit(cell);
      for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
      {
         inv_mass_matrix_diag[c][i] = 0.0;
         for(unsigned int q=0; q<n_q_points; ++q)
            inv_mass_matrix_diag[c][i] += fe_values.shape_value(i,q) *
                                     fe_values.shape_value(i,q) *
                                     fe_values.JxW(q);
         inv_mass_matrix_diag[c][i] = 1.0 / inv_mass_matrix_diag[c][i];
      }
   }
}

//------------------------------------------------------------------------------
// @sect4{ConservationLaw::setup_system}
//
// The following (easy) function is called
// each time the mesh is changed. All it
// does is to resize the Trilinos matrix
// according to a sparsity pattern that we
// generate as in all the previous tutorial
// programs.
//------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::setup_system ()
{
   //DoFRenumbering::Cuthill_McKee (dof_handler);
   
   dof_handler.clear();
   dof_handler.distribute_dofs (fe);
   
   // Size all of the fields.
   old_solution.reinit (dof_handler.n_dofs());
   current_solution.reinit (dof_handler.n_dofs());
   predictor.reinit (dof_handler.n_dofs());
   right_hand_side.reinit (dof_handler.n_dofs());
   
   cell_average.resize (triangulation.n_active_cells(),
                        Vector<double>(EulerEquations<dim>::n_components));
   
   // Used for cell data like time step
   dh_cell.clear();
   dh_cell.distribute_dofs (fe_cell);
   mu_shock.reinit (dh_cell.n_dofs());
   shock_indicator.reinit (dh_cell.n_dofs());
   jump_indicator.reinit (dh_cell.n_dofs());

   // set cell index
   unsigned int index=0;
   
   for (typename Triangulation<dim>::active_cell_iterator cell=triangulation.begin_active();
        cell!=triangulation.end(); ++cell, ++index)
      cell->set_user_index(index);
   
   if(parameters.implicit == false)
   {
      std::cout << "Creating mass matrix ...\n";
      if(parameters.mass_matrix_type == Parameters::AllParameters<dim>::diagonal)
         compute_inv_mass_matrix ();
      else
      {
         Table<2,DoFTools::Coupling> coupling(EulerEquations<dim>::n_components,
                                              EulerEquations<dim>::n_components);
         for(unsigned int i=0; i<EulerEquations<dim>::n_components; ++i)
         {
            for(unsigned int j=0; j<EulerEquations<dim>::n_components; ++j)
               coupling[i][j] = DoFTools::none;
            coupling[i][i] = DoFTools::always;
         }
         DynamicSparsityPattern c_sparsity(dof_handler.n_dofs());
         DoFTools::make_sparsity_pattern (dof_handler, coupling, c_sparsity);
         sparsity_pattern.copy_from(c_sparsity);
         system_matrix.reinit (sparsity_pattern);
         MatrixCreator::create_mass_matrix (mapping(),
                                            dof_handler,
                                            QGauss<dim>(fe.degree+1),
                                            system_matrix);
         // Compute the inverse of the blocks
         unsigned int n_block_size = fe.dofs_per_cell / EulerEquations<dim>::n_components;
         std::cout << "   Block size = " << n_block_size << std::endl;
         inv_mass_matrix_full.initialize (system_matrix, n_block_size);
         // Once inv_mass_matrix is created, we can delete system_matrix
         system_matrix.clear();
         
         // Visualize sparsity pattern
         //std::ofstream out ("sparsity_pattern.1");
         //sparsity_pattern.print_gnuplot (out);
         //abort ();
      }
   }
   else
   {
      DynamicSparsityPattern c_sparsity(dof_handler.n_dofs());
      DoFTools::make_flux_sparsity_pattern (dof_handler, c_sparsity);
      sparsity_pattern.copy_from(c_sparsity);

      system_matrix.reinit (sparsity_pattern);
   }
   
   // Allocate memory for MOOD variables
   if(parameters.solver == Parameters::Solver::mood)
   {
      min_mood_var.reinit(triangulation.n_active_cells());
      max_mood_var.reinit(triangulation.n_active_cells());
      work1.reinit (dof_handler.n_dofs());
   }
   
   cell_degree.resize(triangulation.n_active_cells());
   re_update.resize(triangulation.n_active_cells());
   for(unsigned int i=0; i<triangulation.n_active_cells(); ++i)
   {
      cell_degree[i] = fe.degree;
      re_update[i] = true;
   }
   
   if(parameters.mapping_type == Parameters::AllParameters<dim>::cartesian)
      compute_cartesian_mesh_size ();

   if(parameters.mapping_type != Parameters::AllParameters<dim>::cartesian)
      return;
   
   // For each cell, find neighbourig cell
   // This is needed for limiter
   // CHECK: Should the size be n_active_cells() ?
   cell_data.resize(triangulation.n_active_cells());
   
   const double EPS = 1.0e-10;
   typename DoFHandler<dim>::active_cell_iterator
      cell = dh_cell.begin_active(),
      endc = dh_cell.end();
   for (; cell!=endc; ++cell)
   {
      unsigned int c = cell_number(cell);
      cell_data[c].lcell = endc;
      cell_data[c].rcell = endc;
      cell_data[c].bcell = endc;
      cell_data[c].tcell = endc;
      double dx = cell->diameter() / std::sqrt(1.0*dim);

      for (unsigned int face_no=0; face_no<GeometryInfo<dim>::faces_per_cell; ++face_no)
         if (! cell->at_boundary(face_no))
         {
            const typename DoFHandler<dim>::cell_iterator
               neighbor = cell->neighbor(face_no);
            Assert(neighbor->level() == cell->level() || neighbor->level() == cell->level()-1,
                   ExcInternalError());
            Tensor<1,dim> dr = neighbor->center() - cell->center();
            if(dr[0] < -0.5*dx)
               cell_data[c].lcell = neighbor;
            else if(dr[0] > 0.5*dx)
               cell_data[c].rcell = neighbor;
            else if(dr[1] < -0.5*dx)
               cell_data[c].bcell = neighbor;
            else if(dr[1] > 0.5*dx)
               cell_data[c].tcell = neighbor;
            else
            {
               std::cout << "dx, dy = " << dr[0] << "  " << dr[1] << std::endl;
               AssertThrow(false, ExcMessage("Did not find all neighbours"));
            }
         }
         else
         {
            unsigned int boundary_id = cell->face(face_no)->boundary_id();
            typename EulerEquations<dim>::BoundaryKind boundary_kind =
               parameters.boundary_conditions[boundary_id].kind;
            const Point<dim>& face_center = cell->face(face_no)->center();
            Tensor<1,dim> dr = cell->center() - face_center;
            if(dr[0] > 0.49*dx)
               cell_data[c].lbc = boundary_kind;
            else if(dr[0] < 0.49*dx)
               cell_data[c].rbc = boundary_kind;
            else if(dr[1] > 0.49*dx)
               cell_data[c].bbc = boundary_kind;
            else if(dr[1] < 0.49*dx)
               cell_data[c].tbc = boundary_kind;
            else
            {
               AssertThrow(false, ExcMessage("Error in getting boundary condition"));
            }
         }
   }

   // Visualize sparsity pattern
   //std::ofstream out ("sparsity_pattern.1");
   //sparsity_pattern.print_gnuplot (out);
   //abort ();
}

//------------------------------------------------------------------------------
// Create mesh worker for implicit integration
//------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::setup_mesh_worker (IntegratorImplicit<dim>& integrator)
{   
   const unsigned int n_gauss_points = fe.degree + 1;
   integrator.info_box.initialize_gauss_quadrature(n_gauss_points,
                                                   n_gauss_points,
                                                   n_gauss_points);
   
   integrator.info_box.initialize_update_flags ();
   integrator.info_box.add_update_flags_all (update_values | 
                                             update_quadrature_points |
                                             update_JxW_values);
   integrator.info_box.add_update_flags_cell     (update_gradients);
   integrator.info_box.add_update_flags_boundary (update_normal_vectors | update_gradients);
   integrator.info_box.add_update_flags_face     (update_normal_vectors | update_gradients);
   
   integrator.info_box.initialize (fe, mapping());
   
   integrator.assembler.initialize (system_matrix, right_hand_side);
}

//------------------------------------------------------------------------------
// Create mesh worker for explicit integration
// This computes only the right hand side
//------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::setup_mesh_worker (IntegratorExplicit<dim>& integrator)
{
   integrator.info_box.cell_quadrature = QGaussLobatto<dim>(fe.degree+1);
   integrator.info_box.face_quadrature = QGaussLobatto<dim-1>(fe.degree+1);
   integrator.info_box.boundary_quadrature = QGaussLobatto<dim-1>(fe.degree+1);
   
   integrator.info_box.initialize_update_flags ();
   integrator.info_box.add_update_flags_all (update_JxW_values);
   integrator.info_box.add_update_flags_cell     (update_gradients |
                                                  update_quadrature_points);
   integrator.info_box.add_update_flags_boundary (update_values |
                                                  update_normal_vectors |
                                                  update_quadrature_points); // TODO:ADIFF
   integrator.info_box.add_update_flags_face     (update_values |
                                                  update_normal_vectors); // TODO:ADIFF
   
   integrator.info_box.initialize (fe, mapping());
   
   AnyData rhs;
   Vector<double>* data = &right_hand_side;
   rhs.add< Vector<double>* > (data, "RHS");
   integrator.assembler.initialize (rhs);
}

//------------------------------------------------------------------------------
// Compute local time step for each cell
// We compute speed at quadrature points and take maximum over these values.
// This speed is used to compute time step in each cell.
//------------------------------------------------------------------------------
template <int dim>
void
ConservationLaw<dim>::compute_time_step ()
{
   // No need to compute time step for stationary flows
   if(parameters.is_stationary == true)
      return;

   // Update size of dt array since adaptation might have been performed
   dt.reinit (triangulation.n_cells());

   // If time step given in input file, then use it. This is only for global time stepping
   if(parameters.time_step_type == "global" && parameters.cfl <= 0.0)
   {
      dt = parameters.time_step;
      return;
   }

   if(parameters.mapping_type == Parameters::AllParameters<dim>::cartesian)
      compute_time_step_cartesian();
   else
      compute_time_step_q();


   // For global time step, use the minimum value
   if(parameters.time_step_type == "global")
   {
      if(global_dt > 0 && parameters.time_step > 0)
         global_dt = std::min(global_dt, parameters.time_step);
      if(elapsed_time + global_dt > parameters.final_time)
         global_dt = parameters.final_time - elapsed_time;
      dt = global_dt;
   }

}

//------------------------------------------------------------------------------
// Compute local time step for each cell
// This function is specified to cartesian cells.
//------------------------------------------------------------------------------
template <int dim>
void
ConservationLaw<dim>::compute_time_step_cartesian ()
{
   
   typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   
   global_dt = 1.0e20;
   
   for (; cell!=endc; ++cell)
   {
      const unsigned int c = cell_number (cell);
      const double h = cell->diameter() / std::sqrt(1.0*dim);
      const double sonic = EulerEquations<dim>::sound_speed(cell_average[c]);
      const double density = cell_average[c][EulerEquations<dim>::density_component];
      
      double max_eigenvalue = 0.0;
      for (unsigned int d=0; d<dim; ++d)
         max_eigenvalue += (sonic + std::fabs(cell_average[c][d]/density))/h;
      
      dt(c) = parameters.cfl / (max_eigenvalue * (2.0*fe.degree + 1.0));
      
      global_dt = std::min(global_dt, dt(c));
   }
   
}

//------------------------------------------------------------------------------
// Compute local time step for each cell
//------------------------------------------------------------------------------
template <int dim>
void
ConservationLaw<dim>::compute_time_step_q ()
{
   typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   
   global_dt = 1.0e20;
   
   for (; cell!=endc; ++cell)
   {
      const unsigned int c = cell_number (cell);
      const double h = cell->diameter() / std::sqrt(1.0*dim);
      const double sonic = EulerEquations<dim>::sound_speed(cell_average[c]);
      const double density = cell_average[c][EulerEquations<dim>::density_component];
      
      double q2 = 0.0;
      for (unsigned int d=0; d<dim; ++d)
         q2 += std::pow(cell_average[c][d]/density, 2.0);
      
      double max_eigenvalue = std::sqrt(q2) + sonic;
      dt(c) = parameters.cfl * h / (max_eigenvalue * (2.0*fe.degree + 1.0) * dim);
      
      global_dt = std::min(global_dt, dt(c));
   }
   
}

//------------------------------------------------------------------------------
// Compute cell average solution
//------------------------------------------------------------------------------
template <int dim>
void
ConservationLaw<dim>::compute_cell_average ()
{
   QGaussLobatto<dim>   quadrature_formula(fe.degree+1);
   FEValues<dim> fe_values (mapping(), fe,
                            quadrature_formula,
                            update_JxW_values);
   std::vector<unsigned int> local_dofs(fe.dofs_per_cell);
   
   typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   
   for (; cell!=endc; ++cell)
   {
      unsigned int cell_no = cell_number(cell);
      fe_values.reinit (cell);
         
      cell_average[cell_no] = 0.0;
         
      cell->get_dof_indices(local_dofs);
      for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
      {
         unsigned int c = fe.system_to_component_index(i).first;
         unsigned int q = fe.system_to_component_index(i).second;
         cell_average[cell_no][c] += current_solution(local_dofs[i]) * fe_values.JxW(q);
      }

      cell_average[cell_no] /= cell->measure();
   }
   
}

//------------------------------------------------------------------------------
// Computes total angular momentum in the computational domain
//------------------------------------------------------------------------------
template <int dim>
void
ConservationLaw<dim>::compute_angular_momentum ()
{
   AssertThrow(dim==2, ExcNotImplemented());
   
   QGauss<dim>   quadrature_formula(fe.degree+1);
   const unsigned int n_q_points = quadrature_formula.size();
   
   FEValues<dim> fe_values (mapping(), fe,
                            quadrature_formula,
                            update_values | update_quadrature_points | update_JxW_values);
   const FEValuesExtractors::Vector momentum (0);
   std::vector< Tensor<1,dim> > momentum_values(n_q_points);
   
   double angular_momentum = 0;
   typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
   
   for (; cell!=endc; ++cell)
   {
      fe_values.reinit(cell);
      fe_values[momentum].get_function_values(current_solution, momentum_values);
      const std::vector<Point<dim> >& qp = fe_values.get_quadrature_points();
      for(unsigned int q=0; q<n_q_points; ++q)
      {
         double cross = qp[q][0] * momentum_values[q][1] - qp[q][1] * momentum_values[q][0];
         angular_momentum += cross * fe_values.JxW(q);
      }
   }
   
   printf("Total angular momentum: %18.8e %24.14e\n", elapsed_time, angular_momentum);
}

//------------------------------------------------------------------------------
// @sect4{ConservationLaw::solve}
//
// Here, we actually solve the linear system,
// using either of Trilinos' Aztec or Amesos
// linear solvers. The result of the
// computation will be written into the
// argument vector passed to this
// function. The result is a pair of number
// of iterations and the final linear
// residual.
//------------------------------------------------------------------------------
template <int dim>
std::pair<unsigned int, double>
ConservationLaw<dim>::solve (Vector<double> &newton_update, 
                             double          current_residual)
{
   newton_update = 0;

   switch (parameters.solver)
   {
      case Parameters::Solver::umfpack:
      {
         SparseDirectUMFPACK  solver;
         solver.initialize(system_matrix);
         solver.vmult (newton_update, right_hand_side);
         return std::pair<unsigned int, double> (1, 0);
      }

      case Parameters::Solver::gmres:
      {
         SolverControl   solver_control (parameters.max_iterations, 
                                         parameters.linear_residual * current_residual);
         SolverGMRES<>::AdditionalData  gmres_data (30, true, true);
         SolverGMRES<>   solver (solver_control, gmres_data);

         PreconditionBlockSSOR<SparseMatrix<double>, float> preconditioner;
         preconditioner.initialize(system_matrix, fe.dofs_per_cell);

         // If gmres does not converge, print message and continue
         try
         {
            solver.solve (system_matrix,
                          newton_update,
                          right_hand_side,
                          preconditioner);
         }
         catch(...)
         {
            std::cout << "   *** No convergence in gmres ... continuing ***\n";
         }

         return std::pair<unsigned int, double> (solver_control.last_step(),
                                                 solver_control.last_value());
      }

      // We have equation M*du/dt = rhs, where M = mass matrix
      case Parameters::Solver::rk3:
      case Parameters::Solver::mood:
      {
         if(parameters.mass_matrix_type == Parameters::AllParameters<dim>::diagonal)
         {
            // Multiply newton_update by time step dt
            std::vector<unsigned int> dof_indices(fe.dofs_per_cell);
            typename DoFHandler<dim>::active_cell_iterator
               cell = dof_handler.begin_active(),
               endc = dof_handler.end();
            for (; cell!=endc; ++cell)
            {
               const unsigned int cell_no = cell_number (cell);
               
               cell->get_dof_indices (dof_indices);
               for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
                  newton_update(dof_indices[i]) = dt(cell_no) *
                                                  right_hand_side(dof_indices[i]) *
                                                  inv_mass_matrix_diag[cell_no][i];
            }
         }
         else
         {
            inv_mass_matrix_full.vmult (newton_update, right_hand_side);
            
            // Multiply newton_update by time step dt
            std::vector<unsigned int> dof_indices(fe.dofs_per_cell);
            typename DoFHandler<dim>::active_cell_iterator
               cell = dof_handler.begin_active(),
               endc = dof_handler.end();
            for (; cell!=endc; ++cell)
            {
               const unsigned int cell_no = cell_number (cell);
               
               cell->get_dof_indices (dof_indices);
               for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
                  newton_update(dof_indices[i]) *= dt(cell_no);
            }
         }
         return std::pair<unsigned int, double> (0,0);
      }

      default:
         Assert (false, ExcNotImplemented());
   }
   
   return std::pair<unsigned int, double> (0,0);
}

//------------------------------------------------------------------------------
// Perform RK update
//------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::iterate_explicit (IntegratorExplicit<dim>& integrator,
                                             Vector<double>& newton_update,
                                             double& res_norm0, double& res_norm)
{
   
   // Loop for newton iterations or RK stages
   for(unsigned int rk=0; rk<n_rk; ++rk)
   {
      // set time in boundary condition
      // NOTE: We need to check if this is time accurate.
      double bc_time = elapsed_time + bcrk[rk] * global_dt;
      for (unsigned int boundary_id=0; boundary_id<Parameters::AllParameters<dim>::max_n_boundaries;
           ++boundary_id)
      {
         parameters.boundary_conditions[boundary_id].values.set_time(bc_time);
      }
      
      assemble_system (integrator);
      
      res_norm = right_hand_side.l2_norm();
      if(rk == 0) res_norm0 = res_norm;
      
      std::pair<unsigned int, double> convergence
      = solve (newton_update, res_norm);
      
      // Forward euler step in case of explicit scheme
      // In case of implicit scheme, this is the update
      current_solution += newton_update;
      
      // current_solution = ark*old_solution + (1-ark)*current_solution
      current_solution.sadd (1.0-ark[rk], ark[rk], old_solution);
      
      compute_cell_average ();
      compute_shock_indicator ();
      apply_limiter ();
      
      if(parameters.pos_lim) apply_positivity_limiter ();
      
      std::printf("   %-16.3e %04d        %-5.2e\n",
                  res_norm, convergence.first, convergence.second);
      
   }
}

//------------------------------------------------------------------------------
// Perform SSPRK step using MOOD method
//------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::iterate_mood (IntegratorExplicit<dim>& integrator,
                                         Vector<double>& newton_update,
                                         double& res_norm0, double& res_norm)
{
   std::pair<unsigned int, double> convergence;
   
   // Loop for RK stages
   for(unsigned int rk=0; rk<n_rk; ++rk)
   {
      std::cout << "RK stage " << rk+1 << std::endl;
      std::cout << "\t Iter       n_reduce     n_re_update     n_reset\n";
      
      // set time in boundary condition
      // NOTE: We need to check if this is time accurate.
      for (unsigned int boundary_id=0; boundary_id<Parameters::AllParameters<dim>::max_n_boundaries;
           ++boundary_id)
      {
         parameters.boundary_conditions[boundary_id].values.set_time(elapsed_time);
      }
      
      compute_min_max_mood_var();
      
      // iterate forward euler until DMP is satisfied
      bool terminate = false;
      unsigned int mood_iter = 0;
      predictor = current_solution;
      shock_indicator = 0;
      
      while(!terminate)
      {         
         assemble_system (integrator);
         
         res_norm = right_hand_side.l2_norm();
         if(rk == 0) res_norm0 = res_norm;
         
         convergence = solve (newton_update, res_norm);
         
         if(mood_iter == 0)
         {
            // In first iteration all cells must be updated
            current_solution += newton_update;
            work1 = current_solution;
         }
         else
         {
            // Update cells with re_update == true
            std::vector<unsigned int> dof_indices(fe.dofs_per_cell);
            typename DoFHandler<dim>::active_cell_iterator
               cell = dof_handler.begin_active(),
               endc = dof_handler.end();
            for(; cell != endc; ++cell)
            {
               unsigned int c = cell_number(cell);
               if(re_update[c] == true)
               {
                  cell->get_dof_indices( dof_indices );
                  for(unsigned int i=0; i<fe.dofs_per_cell; ++i)
                  {
                     current_solution(dof_indices[i]) += newton_update(dof_indices[i]);
                     unsigned int base_i = fe.system_to_component_index(i).second;
                     if(index_to_degree[base_i] > cell_degree[c])
                        current_solution(dof_indices[i]) = 0.0;
                     work1(dof_indices[i]) = current_solution(dof_indices[i]);
                  }
               }
            }
         }

         compute_cell_average ();
         unsigned int n_reduce=0, n_re_update=0, n_reset=0;
         terminate = apply_mood (n_reduce, n_re_update, n_reset);
         
         ++mood_iter;
         std::printf("%12d %12d %12d %12d\n", mood_iter, n_reduce, n_re_update, n_reset);
      }
      
      // Forward euler has been accepted.
      // Now update rk stage.
      current_solution = work1;
      
      // current_solution = ark*old_solution + (1-ark)*current_solution
      current_solution.sadd (1.0-ark[rk], ark[rk], old_solution);
      
      // Reset degree and update flags
      for(unsigned int i=0; i<triangulation.n_active_cells(); ++i)
      {
         cell_degree[i] = fe.degree;
         re_update[i] = true;
      }
      compute_cell_average ();
      apply_limiter();
      if(parameters.pos_lim) apply_positivity_limiter ();
   }
}

//------------------------------------------------------------------------------
// Perform one step of implicit scheme
//------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::iterate_implicit (IntegratorImplicit<dim>& integrator,
                                             Vector<double>& newton_update,
                                             double& res_norm0, double& res_norm)
{
   // set time in boundary condition
   // NOTE: We need to check if this is time accurate.
   for (unsigned int boundary_id=0; boundary_id<Parameters::AllParameters<dim>::max_n_boundaries;
        ++boundary_id)
   {
      parameters.boundary_conditions[boundary_id].values.set_time(elapsed_time);
   }
   
   unsigned int nonlin_iter = 0;

   // Loop for newton iterations or RK stages
   while(true)
   {
      
      assemble_system (integrator);
      
      res_norm = right_hand_side.l2_norm();
      if(nonlin_iter == 0) res_norm0 = res_norm;
      
      std::pair<unsigned int, double> convergence
      = solve (newton_update, res_norm);
      
      // Forward euler step in case of explicit scheme
      // In case of implicit scheme, this is the update
      current_solution += newton_update;
      
      compute_cell_average ();
      compute_shock_indicator ();
      apply_limiter ();
      
      if(parameters.pos_lim) apply_positivity_limiter ();
      
      std::printf("   %-16.3e %04d        %-5.2e\n",
                  res_norm, convergence.first, convergence.second);
      
      ++nonlin_iter;
      
      // Check that newton iterations converged
      if(parameters.solver == Parameters::Solver::gmres &&
         nonlin_iter == parameters.max_nonlin_iter &&
         std::fabs(res_norm) > 1.0e-10)
         AssertThrow (nonlin_iter <= parameters.max_nonlin_iter,
                      ExcMessage ("No convergence in nonlinear solver"));
      
      // check stopping criterion
      if(parameters.solver == Parameters::Solver::gmres &&
         (nonlin_iter == parameters.max_nonlin_iter ||
          std::fabs(res_norm) <= 1.0e-10))
         break;
      else if(parameters.solver == Parameters::Solver::umfpack)
         break;
   }
}

//------------------------------------------------------------------------------
// @sect4{ConservationLaw::run}

// This function contains the top-level logic
// of this program: initialization, the time
// loop, and the inner Newton iteration.
//
// At the beginning, we read the mesh file
// specified by the parameter file, setup the
// DoFHandler and various vectors, and then
// interpolate the given initial conditions
// on this mesh. We then perform a number of
// mesh refinements, based on the initial
// conditions, to obtain a mesh that is
// already well adapted to the starting
// solution. At the end of this process, we
// output the initial solution.
//------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::run ()
{
   {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(triangulation);
      
      std::ifstream input_file(parameters.mesh_filename.c_str());
      Assert (input_file, ExcFileNotOpen(parameters.mesh_filename.c_str()));
      
      if(parameters.mesh_type == "ucd")
         grid_in.read_ucd(input_file);
      else if(parameters.mesh_type == "gmsh")
         grid_in.read_msh(input_file);
   }
   
//   {
//      // Use this refine around corner in forward step
//      unsigned int nrefine = 2;
//      for(unsigned int i=0; i<nrefine; ++i)
//         refine_forward_step ();
//   }
   
   /*
   static const HyperBallBoundary<dim> boundary_description;
   triangulation.set_boundary (1, boundary_description);
   */
   
   setup_system();
   set_initial_condition ();
   
   // Refine the initial mesh
   if (parameters.do_refine == true)
      for (unsigned int i=0; i<parameters.shock_levels; ++i)
      {
         Vector<double> refinement_indicators (triangulation.n_active_cells());
         
         compute_refinement_indicators(refinement_indicators);
         refine_grid(refinement_indicators);
         
         set_initial_condition ();
      }
   
   // Cell average of initial condition
   compute_cell_average ();

   // Limit the initial condition
   compute_shock_indicator ();
   apply_limiter();
   old_solution = current_solution;
   predictor = current_solution;
   
   // Reset time/iteration counters
   elapsed_time = 0;
   time_iter = 0;
   
   // Save initial condition to file
   output_results ();
   
   // We then enter into the main time
   // stepping loop.

   // Setup variables to decide if we need to save solution 
   double next_output_time = elapsed_time + parameters.output_time_step;
   int next_output_iter = time_iter + parameters.output_iter_step;

   // Variable to control grid refinement
   double next_refine_time = elapsed_time + parameters.refine_time_step;
   int    next_refine_iter = time_iter + parameters.refine_iter_step;

   Vector<double> newton_update (dof_handler.n_dofs());
   std::vector<double> residual_history;
   
   while (elapsed_time < parameters.final_time)
   {
      // compute time step in each cell using cfl condition
      compute_time_step ();
      
      std::cout << std::endl << "It=" << time_iter+1
                << ", T=" << elapsed_time + global_dt
                << ", dt=" << global_dt
                << ", cfl=" << parameters.cfl << std::endl
		          << "   Number of active cells:       "
		          << triangulation.n_active_cells()
		          << std::endl
		          << "   Number of degrees of freedom: "
		          << dof_handler.n_dofs()
		          << std::endl
		          << std::endl;
      
      unsigned int nonlin_iter = 0;
      double res_norm0 = 1.0;
      double res_norm  = 1.0;
      
      if(parameters.solver == Parameters::Solver::rk3)
      {
         IntegratorExplicit<dim> integrator_explicit (dof_handler);
         setup_mesh_worker (integrator_explicit);
         iterate_explicit(integrator_explicit, newton_update, res_norm0, res_norm);
      }
      else if(parameters.solver == Parameters::Solver::mood)
      {
         IntegratorExplicit<dim> integrator_explicit (dof_handler);
         setup_mesh_worker (integrator_explicit);
         iterate_mood(integrator_explicit, newton_update, res_norm0, res_norm);
      }
      else
      {
         std::cout << "   NonLin Res     Lin Iter       Lin Res" << std::endl
                   << "   _____________________________________" << std::endl;
         // With global time stepping, we can use predictor as initial
         // guess for the implicit scheme.
         current_solution = predictor;
         IntegratorImplicit<dim> integrator_implicit (dof_handler);
         setup_mesh_worker (integrator_implicit);
         iterate_implicit(integrator_implicit, newton_update, res_norm0, res_norm);
      }
      
      // Update counters
      elapsed_time += global_dt;
      ++time_iter;
      
      if(time_iter % parameters.ang_mom_step == 0)
         compute_angular_momentum();

      // Increase cfl
      if(parameters.solver == Parameters::Solver::gmres && 
         parameters.time_step_type == "local" &&
         time_iter >= 2)
      {
         //parameters.cfl *= 1.2;
         double factor = residual_history.back() / res_norm;
         factor = std::min( factor, 2.0 );
         factor = std::max( factor, 0.5 );
         parameters.cfl *= factor;
      }

      residual_history.push_back (res_norm);
      
      // Save solution for visualization
      if (elapsed_time >= next_output_time || time_iter == next_output_iter 
            || std::fabs(elapsed_time-parameters.final_time) < 1.0e-13)
      {
         output_results ();
         next_output_time = elapsed_time + parameters.output_time_step;
         next_output_iter = time_iter + parameters.output_iter_step;
      }
      
      // Compute predictor only for global time stepping
      // For local time stepping, this is meaningless.
      // If time step is changing, then also this is not correct.
      if(parameters.implicit || parameters.time_step_type == "global")
      {
         predictor = current_solution;
         predictor.sadd (2.0, -1.0, old_solution);
      }
      
      old_solution = current_solution;
      
      if (parameters.do_refine == true && 
          (elapsed_time >= next_refine_time || time_iter == next_refine_iter))
      {
         Vector<double> refinement_indicators (triangulation.n_active_cells());
         compute_refinement_indicators(refinement_indicators);
         
         refine_grid(refinement_indicators);
         
         newton_update.reinit (dof_handler.n_dofs());

         next_refine_time = elapsed_time + parameters.refine_time_step;
         next_refine_iter = time_iter + parameters.refine_iter_step;

         // We may need to reduce the cfl after refinement, only for steady state
         // problems when using gmres
         //parameters.cfl = 1.2;
      }
   }
}

//------------------------------------------------------------------------------
// Compute error norm at final time
//------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::compute_errors ()
{
   std::vector<double> L2_error(EulerEquations<2>::n_components);

   std::cout << "Computing errors at time " << elapsed_time << std::endl;
   
   if(parameters.ic_function == "isohydro")
   {
      VectorTools::interpolate(mapping(), dof_handler,
                               IsothermalHydrostatic<dim>(), old_solution);
   }
   else if(parameters.ic_function == "rt")
   {
      set_initial_condition_Rayleigh_Taylor ();
   }
   else if(parameters.ic_function == "rrt")
   {
      VectorTools::interpolate(mapping(), dof_handler,
                               RadialRayleighTaylor<dim>(), old_solution);
   }

   else if(parameters.ic_function == "rtb")
   {
      VectorTools::interpolate(mapping(), dof_handler,
                               RisingThermalBubble<dim>(parameters.gravity), old_solution);
   }

   else if(parameters.ic_function == "polyhydro")
   {
      VectorTools::interpolate(mapping(), dof_handler,
                               PolytropicHydrostatic<dim>(1.2), old_solution);
   }
   else
   {
      VectorTools::interpolate(mapping(), dof_handler,
                               parameters.initial_conditions, old_solution);
   }

   current_solution -= old_solution;
   output_results();
   
   Vector<double> difference_per_cell (triangulation.n_active_cells());
   
   for(unsigned c = 0; c < EulerEquations<dim>::n_components; ++c)
   {
      const ComponentSelectFunction<dim> component_mask (c, EulerEquations<dim>::n_components);
      VectorTools::integrate_difference (mapping(),
                                         dof_handler,
                                         current_solution,
                                         ZeroFunction<dim>(EulerEquations<dim>::n_components),
                                         /*UnsteadyGravity<dim>(elapsed_time),*/
                                         difference_per_cell,
                                         QGauss<dim>(fe.degree+2),
                                         VectorTools::L2_norm,
                                         &component_mask);
      L2_error[c] = difference_per_cell.l2_norm();
   }

   double area = M_PI; //4.0;
   std::cout << "Area used for error normalization = " << area << std::endl;
   for(unsigned int c=0; c<EulerEquations<2>::n_components; ++c)
      std::cout << L2_error[c]/std::sqrt(area) << "  ";
   std::cout << std::endl;
}

template class ConservationLaw<2>;
