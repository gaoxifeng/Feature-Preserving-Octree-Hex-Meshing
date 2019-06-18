
#pragma once

#include "igl/igl_inline.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "global_types.h"
#include "global_functions.h"
// Compute a SLIM map as derived in "Scalable Locally Injective Maps" [Rabinovich et al. 2016].
struct SLIMData
{
  // Input
  Eigen::MatrixXd V; // #V by 3 list of mesh vertex positions
  Eigen::MatrixXi F; // #F by 3/3 list of mesh faces (triangles/tets)
  SLIM_ENERGY slim_energy;

  double weight_opt = 1;
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

  Tetralize_Set ts;
  double stop_threshold = 1.e-5;

  bool Projection = false;
  bool Global = false;

  double exp_factor; // used for exponential energies, ignored otherwise
  bool mesh_improvement_3d; // only supported for 3d

  double engery_quality = 0;
  double engery_soft = 0;
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
};

// Compute necessary information to start using SLIM
// Inputs:
//		V           #V by 3 list of mesh vertex positions
//		F           #F by 3/3 list of mesh faces (triangles/tets)
//    b           list of boundary indices into V
//    bc          #b by dim list of boundary conditions
//    soft_p      Soft penalty factor (can be zero)
//    slim_energy Energy to minimize
void slim_precompute(Eigen::MatrixXd& V,
                                Eigen::MatrixXi& F,
                                Eigen::MatrixXd& V_init,
                                SLIMData& data,
                                SLIM_ENERGY slim_energy,
                                Eigen::VectorXi& b,
                                Eigen::MatrixXd& bc, 
	Eigen::VectorXi &ids_C_, Eigen::MatrixXd &C_,
	Eigen::VectorXi &ids_L_, Eigen::MatrixXd &Axa_L_, Eigen::MatrixXd &Origin_L_,
	Eigen::VectorXi &ids_T_, Eigen::MatrixXd &normal_T_, Eigen::VectorXd &dis_T_, 
	Eigen::VectorXi &regionb, Eigen::MatrixXd &regionbc,
	bool surface_projection, bool global_opt
	);

void slim_precompute(Eigen::MatrixXd& V,
	Eigen::MatrixXi& F,
	Eigen::MatrixXd& V_init,
	SLIMData& data,
	SLIM_ENERGY slim_energy,
	Tetralize_Set ts
);
// Run iter_num iterations of SLIM
// Outputs:
//    V_o (in SLIMData): #V by dim list of mesh vertex positions
Eigen::MatrixXd slim_solve(SLIMData& data, int iter_num, int type, bool verbose=false);
Eigen::MatrixXd slim_solve2(SLIMData& data, int iter_num, int type, bool verbose=false);

//#ifndef IGL_STATIC_LIBRARY
//#  include "slim.cpp"
//#endif
