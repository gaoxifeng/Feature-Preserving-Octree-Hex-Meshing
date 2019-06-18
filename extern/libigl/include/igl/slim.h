// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2016 Michael Rabinovich
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef SLIM_H
#define SLIM_H

#include "igl_inline.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

// This option makes the iterations faster (all except the first) by caching the 
// sparsity pattern of the matrix involved in the assembly. It should be on if you plan to do many iterations, off if you have to change the matrix structure at every iteration.
#define SLIM_CACHED 

#ifdef SLIM_CACHED
#include <igl/AtA_cached.h>
#endif

namespace igl
{

// Compute a SLIM map as derived in "Scalable Locally Injective Maps" [Rabinovich et al. 2016].
struct SLIMData
{
  double weight_opt = 1;
  // Input
  Eigen::MatrixXd V; // #V by 3 list of mesh vertex positions
  Eigen::MatrixXi F; // #F by 3/3 list of mesh faces (triangles/tets)
  std::vector<Eigen::MatrixXd> RF;
  enum SLIM_ENERGY
  {
    ARAP,
    LOG_ARAP,
    SYMMETRIC_DIRICHLET,
    CONFORMAL,
    EXP_CONFORMAL,
    EXP_SYMMETRIC_DIRICHLET
  };
  SLIM_ENERGY slim_energy;

  // Optional Input
  // soft constraints
  Eigen::VectorXi b;
  Eigen::MatrixXd bc;
  double soft_const_p = 0;
  //corner constraints
  Eigen::VectorXi ids_C;
  Eigen::MatrixXd C;
  double lamda_C = 0;
  //tagent plane constraints
  Eigen::VectorXi ids_T;
  Eigen::MatrixXd normal_T;
  Eigen::VectorXd dis_T;
  double lamda_T = 0;
  //feature line constraints
  uint32_t num_a;
  Eigen::VectorXi ids_L;
  Eigen::MatrixXd Axa_L;
  Eigen::MatrixXd origin_L;

  Eigen::VectorXd a_L;
  double lamda_L = 0;
//region
  Eigen::VectorXi regionb;
  Eigen::MatrixXd regionbc;
  double lamda_region = 0;
//equality constraint
  std::vector<std::vector<uint32_t>> Vgroups;
  double lamda_glue = 0;

  bool Projection = false;
  bool Global = false;

  double exp_factor; // used for exponential energies, ignored otherwise
  bool mesh_improvement_3d; // only supported for 3d

  // Output
  Eigen::MatrixXd V_o; // #V by dim list of mesh vertex positions (dim = 2 for parametrization, 3 otherwise)
  double energy; // objective value

  // INTERNAL
  Eigen::VectorXd M;
  double mesh_area;
  double avg_edge_length;
  int v_num;
  int f_num;
  double proximal_p;

  Eigen::VectorXd WGL_M;
  Eigen::VectorXd rhs;
  Eigen::MatrixXd Ri,Ji;
  Eigen::VectorXd W_11; Eigen::VectorXd W_12; Eigen::VectorXd W_13;
  Eigen::VectorXd W_21; Eigen::VectorXd W_22; Eigen::VectorXd W_23;
  Eigen::VectorXd W_31; Eigen::VectorXd W_32; Eigen::VectorXd W_33;
  Eigen::SparseMatrix<double> Dx,Dy,Dz;
  int f_n,v_n;
  bool first_solve;
  bool has_pre_calc = false;
  int dim;

#ifdef SLIM_CACHED
  Eigen::SparseMatrix<double> A;
  Eigen::VectorXi A_data;
  Eigen::SparseMatrix<double> AtA;
  igl::AtA_cached_data AtA_data;
#endif
};

// Compute necessary information to start using SLIM
// Inputs:
//		V           #V by 3 list of mesh vertex positions
//		F           #F by 3/3 list of mesh faces (triangles/tets)
//    b           list of boundary indices into V
//    bc          #b by dim list of boundary conditions
//    soft_p      Soft penalty factor (can be zero)
//    slim_energy Energy to minimize
IGL_INLINE void slim_precompute(Eigen::MatrixXd& V,
                                Eigen::MatrixXi& F,
                                Eigen::MatrixXd& V_init,
                                SLIMData& data,
                                SLIMData::SLIM_ENERGY slim_energy,
                                Eigen::VectorXi& b,
                                Eigen::MatrixXd& bc, 
	Eigen::VectorXi &ids_C_, Eigen::MatrixXd &C_,
	Eigen::VectorXi &ids_L_, Eigen::MatrixXd &Axa_L_, Eigen::MatrixXd &Origin_L_,
	Eigen::VectorXi &ids_T_, Eigen::MatrixXd &normal_T_, Eigen::VectorXd &dis_T_, 
	Eigen::VectorXi &regionb, Eigen::MatrixXd &regionbc,
	bool surface_projection, bool global_opt,
	std::vector<Eigen::MatrixXd> RF
	);

// Run iter_num iterations of SLIM
// Outputs:
//    V_o (in SLIMData): #V by dim list of mesh vertex positions
IGL_INLINE Eigen::MatrixXd slim_solve(SLIMData& data, int iter_num);

} // END NAMESPACE

#ifndef IGL_STATIC_LIBRARY
#  include "slim.cpp"
#endif

#endif // SLIM_H
