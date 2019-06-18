// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2016 Michael Rabinovich
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "slim.h"

#include "boundary_loop.h"
#include "cotmatrix.h"
#include "edge_lengths.h"
#include "grad.h"
#include "local_basis.h"
#include "repdiag.h"
#include "vector_area_matrix.h"
#include "arap.h"
#include "cat.h"
#include "doublearea.h"
#include "grad.h"
#include "local_basis.h"
#include "per_face_normals.h"
#include "slice_into.h"
#include "volume.h"
#include "polar_svd.h"
#include "flip_avoiding_line_search.h"

#include <iostream>
#include <map>
#include <set>
#include <vector>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
//#include <../../../tbb/tbb.h>
#include <../../../timer.h>

#include <igl/sparse_cached.h>
#include <igl/AtA_cached.h>
//#define CHOLMOD_
#ifdef CHOLMOD
#include <Eigen/CholmodSupport>
#endif
#include <unsupported/Eigen/SparseExtra>

namespace igl
{
  namespace slim
  {
    // Definitions of internal functions
    IGL_INLINE void compute_surface_gradient_matrix(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                                                    const Eigen::MatrixXd &F1, const Eigen::MatrixXd &F2,
                                                    Eigen::SparseMatrix<double> &D1, Eigen::SparseMatrix<double> &D2);
    //IGL_INLINE void buildA(igl::SLIMData& s, Eigen::SparseMatrix<double> &A);
	IGL_INLINE void buildA(igl::SLIMData& s, std::vector<Eigen::Triplet<double> > & IJV);
    IGL_INLINE void buildRhs(igl::SLIMData& s, const Eigen::SparseMatrix<double> &At);
    IGL_INLINE void add_soft_constraints(igl::SLIMData& s, Eigen::SparseMatrix<double> &L);
    IGL_INLINE double compute_energy(igl::SLIMData& s, Eigen::MatrixXd &V_new);
    IGL_INLINE double compute_soft_const_energy(igl::SLIMData& s,
                                                const Eigen::MatrixXd &V,
                                                const Eigen::MatrixXi &F,
                                                Eigen::MatrixXd &V_o);
    IGL_INLINE double compute_energy_with_jacobians(igl::SLIMData& s,
                                                    const Eigen::MatrixXd &V,
                                                    const Eigen::MatrixXi &F, const Eigen::MatrixXd &Ji,
                                                    Eigen::MatrixXd &uv, Eigen::VectorXd &areas);
    IGL_INLINE void solve_weighted_arap(igl::SLIMData& s,
                                        const Eigen::MatrixXd &V,
                                        const Eigen::MatrixXi &F,
                                        Eigen::MatrixXd &uv,
                                        Eigen::VectorXi &soft_b_p,
                                        Eigen::MatrixXd &soft_bc_p);
    IGL_INLINE void update_weights_and_closest_rotations( igl::SLIMData& s,
                                                          const Eigen::MatrixXd &V,
                                                          const Eigen::MatrixXi &F,
                                                          Eigen::MatrixXd &uv);
    IGL_INLINE void compute_jacobians(igl::SLIMData& s, const Eigen::MatrixXd &uv);
    IGL_INLINE void build_linear_system(igl::SLIMData& s, Eigen::SparseMatrix<double> &L);
    IGL_INLINE void pre_calc(igl::SLIMData& s);

    // Implementation
    IGL_INLINE void compute_surface_gradient_matrix(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                                                    const Eigen::MatrixXd &F1, const Eigen::MatrixXd &F2,
                                         Eigen::SparseMatrix<double> &D1, Eigen::SparseMatrix<double> &D2)
    {

      Eigen::SparseMatrix<double> G;
      igl::grad(V, F, G);
      Eigen::SparseMatrix<double> Dx = G.block(0, 0, F.rows(), V.rows());
      Eigen::SparseMatrix<double> Dy = G.block(F.rows(), 0, F.rows(), V.rows());
      Eigen::SparseMatrix<double> Dz = G.block(2 * F.rows(), 0, F.rows(), V.rows());

      D1 = F1.col(0).asDiagonal() * Dx + F1.col(1).asDiagonal() * Dy + F1.col(2).asDiagonal() * Dz;
      D2 = F2.col(0).asDiagonal() * Dx + F2.col(1).asDiagonal() * Dy + F2.col(2).asDiagonal() * Dz;
    }

    IGL_INLINE void compute_jacobians(igl::SLIMData& s, const Eigen::MatrixXd &uv)
    {
      if (s.F.cols() == 3)
      {
        // Ji=[D1*u,D2*u,D1*v,D2*v];
        s.Ji.col(0) = s.Dx * uv.col(0);
        s.Ji.col(1) = s.Dy * uv.col(0);
        s.Ji.col(2) = s.Dx * uv.col(1);
        s.Ji.col(3) = s.Dy * uv.col(1);
      }
      else /*tet mesh*/{
        // Ji=[D1*u,D2*u,D3*u, D1*v,D2*v, D3*v, D1*w,D2*w,D3*w];
        s.Ji.col(0) = s.Dx * uv.col(0);
        s.Ji.col(1) = s.Dy * uv.col(0);
        s.Ji.col(2) = s.Dz * uv.col(0);
        s.Ji.col(3) = s.Dx * uv.col(1);
        s.Ji.col(4) = s.Dy * uv.col(1);
        s.Ji.col(5) = s.Dz * uv.col(1);
        s.Ji.col(6) = s.Dx * uv.col(2);
        s.Ji.col(7) = s.Dy * uv.col(2);
        s.Ji.col(8) = s.Dz * uv.col(2);
      }
    }

    IGL_INLINE void update_weights_and_closest_rotations(igl::SLIMData& s,
                                              const Eigen::MatrixXd &V,
                                              const Eigen::MatrixXi &F,
                                              Eigen::MatrixXd &uv)
    {
      compute_jacobians(s, uv);

      const double eps = 1e-8;
      double exp_f = s.exp_factor;

      if (s.dim == 2)
      {
        for (int i = 0; i < s.Ji.rows(); ++i)
        {
          typedef Eigen::Matrix<double, 2, 2> Mat2;
          typedef Eigen::Matrix<double, 2, 1> Vec2;
          Mat2 ji, ri, ti, ui, vi;
          Vec2 sing;
          Vec2 closest_sing_vec;
          Mat2 mat_W;
          Vec2 m_sing_new;
          double s1, s2;

          ji(0, 0) = s.Ji(i, 0);
          ji(0, 1) = s.Ji(i, 1);
          ji(1, 0) = s.Ji(i, 2);
          ji(1, 1) = s.Ji(i, 3);

          igl::polar_svd(ji, ri, ti, ui, sing, vi);

          s1 = sing(0);
          s2 = sing(1);

          // Update Weights according to energy
          switch (s.slim_energy)
          {
            case igl::SLIMData::ARAP:
            {
              m_sing_new << 1, 1;
              break;
            }
            case igl::SLIMData::SYMMETRIC_DIRICHLET:
            {
              double s1_g = 2 * (s1 - pow(s1, -3));
              double s2_g = 2 * (s2 - pow(s2, -3));
              m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));
              break;
            }
            case igl::SLIMData::LOG_ARAP:
            {
              double s1_g = 2 * (log(s1) / s1);
              double s2_g = 2 * (log(s2) / s2);
              m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));
              break;
            }
            case igl::SLIMData::CONFORMAL:
            {
              double s1_g = 1 / (2 * s2) - s2 / (2 * pow(s1, 2));
              double s2_g = 1 / (2 * s1) - s1 / (2 * pow(s2, 2));

              double geo_avg = sqrt(s1 * s2);
              double s1_min = geo_avg;
              double s2_min = geo_avg;

              m_sing_new << sqrt(s1_g / (2 * (s1 - s1_min))), sqrt(s2_g / (2 * (s2 - s2_min)));

              // change local step
              closest_sing_vec << s1_min, s2_min;
              ri = ui * closest_sing_vec.asDiagonal() * vi.transpose();
              break;
            }
            case igl::SLIMData::EXP_CONFORMAL:
            {
              double s1_g = 2 * (s1 - pow(s1, -3));
              double s2_g = 2 * (s2 - pow(s2, -3));

              double geo_avg = sqrt(s1 * s2);
              double s1_min = geo_avg;
              double s2_min = geo_avg;

              double in_exp = exp_f * ((pow(s1, 2) + pow(s2, 2)) / (2 * s1 * s2));
              double exp_thing = exp(in_exp);

              s1_g *= exp_thing * exp_f;
              s2_g *= exp_thing * exp_f;

              m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));
              break;
            }
            case igl::SLIMData::EXP_SYMMETRIC_DIRICHLET:
            {
              double s1_g = 2 * (s1 - pow(s1, -3));
              double s2_g = 2 * (s2 - pow(s2, -3));

              double in_exp = exp_f * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2));
              double exp_thing = exp(in_exp);

              s1_g *= exp_thing * exp_f;
              s2_g *= exp_thing * exp_f;

              m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));
              break;
            }
          }

          if (std::abs(s1 - 1) < eps) m_sing_new(0) = 1;
          if (std::abs(s2 - 1) < eps) m_sing_new(1) = 1;
          mat_W = ui * m_sing_new.asDiagonal() * ui.transpose();

          s.W_11(i) = mat_W(0, 0);
          s.W_12(i) = mat_W(0, 1);
          s.W_21(i) = mat_W(1, 0);
          s.W_22(i) = mat_W(1, 1);

          // 2) Update local step (doesn't have to be a rotation, for instance in case of conformal energy)
          s.Ri(i, 0) = ri(0, 0);
          s.Ri(i, 1) = ri(1, 0);
          s.Ri(i, 2) = ri(0, 1);
          s.Ri(i, 3) = ri(1, 1);
        }
      }
      else
      {
        typedef Eigen::Matrix<double, 3, 1> Vec3;
        typedef Eigen::Matrix<double, 3, 3> Mat3;
        Mat3 ji;
        Vec3 m_sing_new;
        Vec3 closest_sing_vec;
        const double sqrt_2 = sqrt(2);
        for (int i = 0; i < s.Ji.rows(); ++i)
        {
          ji(0, 0) = s.Ji(i, 0);
          ji(0, 1) = s.Ji(i, 1);
          ji(0, 2) = s.Ji(i, 2);
          ji(1, 0) = s.Ji(i, 3);
          ji(1, 1) = s.Ji(i, 4);
          ji(1, 2) = s.Ji(i, 5);
          ji(2, 0) = s.Ji(i, 6);
          ji(2, 1) = s.Ji(i, 7);
          ji(2, 2) = s.Ji(i, 8);

          Mat3 ri, ti, ui, vi;
          Vec3 sing;
          igl::polar_svd(ji, ri, ti, ui, sing, vi);

          double s1 = sing(0);
          double s2 = sing(1);
          double s3 = sing(2);

          // 1) Update Weights
          switch (s.slim_energy)
          {
            case igl::SLIMData::ARAP:
            {
              m_sing_new << 1, 1, 1;
              break;
            }
            case igl::SLIMData::LOG_ARAP:
            {
              double s1_g = 2 * (log(s1) / s1);
              double s2_g = 2 * (log(s2) / s2);
              double s3_g = 2 * (log(s3) / s3);
              m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));
              break;
            }
            case igl::SLIMData::SYMMETRIC_DIRICHLET:
            {
              double s1_g = 2 * (s1 - pow(s1, -3));
              double s2_g = 2 * (s2 - pow(s2, -3));
              double s3_g = 2 * (s3 - pow(s3, -3));
              m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));
              break;
            }
            case igl::SLIMData::EXP_SYMMETRIC_DIRICHLET:
            {
              double s1_g = 2 * (s1 - pow(s1, -3));
              double s2_g = 2 * (s2 - pow(s2, -3));
              double s3_g = 2 * (s3 - pow(s3, -3));
              m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));

              double in_exp = exp_f * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2) + pow(s3, 2) + pow(s3, -2));
              double exp_thing = exp(in_exp);

              s1_g *= exp_thing * exp_f;
              s2_g *= exp_thing * exp_f;
              s3_g *= exp_thing * exp_f;

              m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));

              break;
            }
            case igl::SLIMData::CONFORMAL:
            {
              double common_div = 9 * (pow(s1 * s2 * s3, 5. / 3.));

              double s1_g = (-2 * s2 * s3 * (pow(s2, 2) + pow(s3, 2) - 2 * pow(s1, 2))) / common_div;
              double s2_g = (-2 * s1 * s3 * (pow(s1, 2) + pow(s3, 2) - 2 * pow(s2, 2))) / common_div;
              double s3_g = (-2 * s1 * s2 * (pow(s1, 2) + pow(s2, 2) - 2 * pow(s3, 2))) / common_div;

              double closest_s = sqrt(pow(s1, 2) + pow(s3, 2)) / sqrt_2;
              double s1_min = closest_s;
              double s2_min = closest_s;
              double s3_min = closest_s;

              m_sing_new << sqrt(s1_g / (2 * (s1 - s1_min))), sqrt(s2_g / (2 * (s2 - s2_min))), sqrt(
                  s3_g / (2 * (s3 - s3_min)));

              // change local step
              closest_sing_vec << s1_min, s2_min, s3_min;
              ri = ui * closest_sing_vec.asDiagonal() * vi.transpose();
              break;
            }
            case igl::SLIMData::EXP_CONFORMAL:
            {
              // E_conf = (s1^2 + s2^2 + s3^2)/(3*(s1*s2*s3)^(2/3) )
              // dE_conf/ds1 = (-2*(s2*s3)*(s2^2+s3^2 -2*s1^2) ) / (9*(s1*s2*s3)^(5/3))
              // Argmin E_conf(s1): s1 = sqrt(s1^2+s2^2)/sqrt(2)
              double common_div = 9 * (pow(s1 * s2 * s3, 5. / 3.));

              double s1_g = (-2 * s2 * s3 * (pow(s2, 2) + pow(s3, 2) - 2 * pow(s1, 2))) / common_div;
              double s2_g = (-2 * s1 * s3 * (pow(s1, 2) + pow(s3, 2) - 2 * pow(s2, 2))) / common_div;
              double s3_g = (-2 * s1 * s2 * (pow(s1, 2) + pow(s2, 2) - 2 * pow(s3, 2))) / common_div;

              double in_exp = exp_f * ((pow(s1, 2) + pow(s2, 2) + pow(s3, 2)) / (3 * pow((s1 * s2 * s3), 2. / 3)));;
              double exp_thing = exp(in_exp);

              double closest_s = sqrt(pow(s1, 2) + pow(s3, 2)) / sqrt_2;
              double s1_min = closest_s;
              double s2_min = closest_s;
              double s3_min = closest_s;

              s1_g *= exp_thing * exp_f;
              s2_g *= exp_thing * exp_f;
              s3_g *= exp_thing * exp_f;

              m_sing_new << sqrt(s1_g / (2 * (s1 - s1_min))), sqrt(s2_g / (2 * (s2 - s2_min))), sqrt(
                  s3_g / (2 * (s3 - s3_min)));

              // change local step
              closest_sing_vec << s1_min, s2_min, s3_min;
              ri = ui * closest_sing_vec.asDiagonal() * vi.transpose();
            }
          }
          if (std::abs(s1 - 1) < eps) m_sing_new(0) = 1;
          if (std::abs(s2 - 1) < eps) m_sing_new(1) = 1;
          if (std::abs(s3 - 1) < eps) m_sing_new(2) = 1;
          Mat3 mat_W;
          mat_W = ui * m_sing_new.asDiagonal() * ui.transpose();

          s.W_11(i) = mat_W(0, 0);
          s.W_12(i) = mat_W(0, 1);
          s.W_13(i) = mat_W(0, 2);
          s.W_21(i) = mat_W(1, 0);
          s.W_22(i) = mat_W(1, 1);
          s.W_23(i) = mat_W(1, 2);
          s.W_31(i) = mat_W(2, 0);
          s.W_32(i) = mat_W(2, 1);
          s.W_33(i) = mat_W(2, 2);

          // 2) Update closest rotations (not rotations in case of conformal energy)
          s.Ri(i, 0) = ri(0, 0);
          s.Ri(i, 1) = ri(1, 0);
          s.Ri(i, 2) = ri(2, 0);
          s.Ri(i, 3) = ri(0, 1);
          s.Ri(i, 4) = ri(1, 1);
          s.Ri(i, 5) = ri(2, 1);
          s.Ri(i, 6) = ri(0, 2);
          s.Ri(i, 7) = ri(1, 2);
          s.Ri(i, 8) = ri(2, 2);
        } // for loop end

      } // if dim end

    }

    IGL_INLINE void solve_weighted_arap(igl::SLIMData& s,
                                        const Eigen::MatrixXd &V,
                                        const Eigen::MatrixXi &F,
                                        Eigen::MatrixXd &uv,
                                        Eigen::VectorXi &soft_b_p,
                                        Eigen::MatrixXd &soft_bc_p)
    {
      using namespace Eigen;

	  Timer<> time0;
	  //time0.beginStage("build L");
      Eigen::SparseMatrix<double> L;
      build_linear_system(s,L);
	  //std::cout << "L innerSize(): " << L.innerSize() << std::endl;
	  //std::cout << "L outerSize(): " << L.outerSize() << std::endl;
	  //std::cout << "L nonzeros: " << L.nonZeros() << std::endl;
	  //std::cout<<"L size: "<< (uint32_t)L.data().size()<<std::endl;
	  //time0.endStage("end build L");
	  //std::cout << "Number of threads used by Eigen: " << Eigen::nbThreads() << std::endl;
	  //time0.beginStage("solve L");
	  // solve
	  Eigen::VectorXd Uc;

	  //std::cout << "SSSS " << (Eigen::MatrixXd(L) - Eigen::MatrixXd(L).transpose()).array().abs().maxCoeff() << std::endl;
	  //Eigen::saveMarket(L, "C:/xgao/meshing/Robust-Pure-Hex-Meshing/datasets/L.mat");

	  // std::cout << "before: " << L.norm() << std::endl;
	  // std::cout << "before: " << s.rhs.norm() << std::endl;

	  bool cholmod_definition = false;
		#ifdef CHOLMOD
		cholmod_definition=true;
		#endif

	  //if (true) {
	  if (!s.Vgroups.size() || cholmod_definition==false) {
		  if (s.dim == 2)
		  {
			  SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
			  Uc = solver.compute(L).solve(s.rhs);
		  }
		  else
		  { // seems like CG performs much worse for 2D and way better for 3D
			  uint32_t Num_variables = uv.rows() * s.dim + s.ids_L.rows();
			 if (s.Projection) 
				  Num_variables = uv.rows() * s.dim;

			  Eigen::VectorXd guess(Num_variables);
			  for (int i = 0; i < s.v_n; i++) for (int j = 0; j < s.dim; j++) guess(uv.rows() * j + i) = uv(i, j); // flatten vector
			  if (!s.Projection)
				  for (int i = 0; i < s.ids_L.rows(); i++) guess(uv.rows() * s.dim + i) = 0;//feature curve additional variable
			  ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Upper> solver;
			  //BiCGSTAB<SparseMatrix<double> >solver;
			  solver.setTolerance(1e-8);
			  solver.setMaxIterations(20);
			  Uc = solver.compute(L).solveWithGuess(s.rhs, guess);
			  //std::cout<<"error: "<<(L* Uc - s.rhs).norm() << std::endl;
		  }
	  }
	  else if(s.Vgroups.size() && s.lamda_glue>0){
	#ifdef CHOLMOD
		  CholmodSimplicialLDLT<Eigen::SparseMatrix<double> > solver;
		  Uc = solver.compute(L).solve(s.rhs);
	#endif
	  }
//#ifndef CHOLMOD
//#error should do a proper define of CHOLMOD, hacked with Teseo!
//      if (s.dim == 2)
//      {
//        SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
//        Uc = solver.compute(L).solve(s.rhs);
//      }
//      else
//      { // seems like CG performs much worse for 2D and way better for 3D
//		  uint32_t Num_variables = uv.rows() * s.dim + s.ids_L.rows();
//		  if (s.Projection) Num_variables = uv.rows() * s.dim;
//
//        Eigen::VectorXd guess(Num_variables);
//        for (int i = 0; i < s.v_n; i++) for (int j = 0; j < s.dim; j++) guess(uv.rows() * j + i) = uv(i, j); // flatten vector
//		if(!s.Projection)
//			for (int i = 0; i < s.ids_L.rows(); i++) guess(uv.rows() * s.dim + i) = 0;//feature curve additional variable
//        ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Upper> solver;
//        solver.setTolerance(1e-8);
//        Uc = solver.compute(L).solveWithGuess(s.rhs, guess);
//      }
//#else
//	  CholmodSimplicialLDLT<Eigen::SparseMatrix<double> > solver;
//	  Uc = solver.compute(L).solve(s.rhs);
//#endif

      for (int i = 0; i < s.dim; i++)
        uv.col(i) = Uc.block(i * s.v_n, 0, s.v_n, 1);
	  if (!s.Projection) {
		  for (int i = 0; i < s.ids_L.rows(); i++)
			  s.a_L(i) = Uc[s.dim * s.v_n + i];
	  }

	 //time0.endStage("end solve L");
    }


    IGL_INLINE void pre_calc(igl::SLIMData& s)
    {
      if (!s.has_pre_calc)
      {
        s.v_n = s.v_num;
        s.f_n = s.f_num;

        if (s.F.cols() == 3)
        {
          s.dim = 2;
          Eigen::MatrixXd F1, F2, F3;
          igl::local_basis(s.V, s.F, F1, F2, F3);
          //compute_surface_gradient_matrix(s.V, s.F, F1, F2, s.Dx, s.Dy);
		  /////////////////////////
		  Eigen::SparseMatrix<double> G;
		  igl::grad_ref(s.V, s.F,s.RF, G, true);
		  //igl::grad(s.V, s.F, G, false);
		  Eigen::SparseMatrix<double> Dx = G.block(0, 0, s.F.rows(), s.V.rows());
		  Eigen::SparseMatrix<double> Dy = G.block(s.F.rows(), 0, s.F.rows(), s.V.rows());
		  Eigen::SparseMatrix<double> Dz = G.block(2 * s.F.rows(), 0, s.F.rows(), s.V.rows());

		  s.Dx = F1.col(0).asDiagonal() * Dx + F1.col(1).asDiagonal() * Dy + F1.col(2).asDiagonal() * Dz;
		  s.Dy = F2.col(0).asDiagonal() * Dx + F2.col(1).asDiagonal() * Dy + F2.col(2).asDiagonal() * Dz;

		  /////////////////////////
          s.W_11.resize(s.f_n);
          s.W_12.resize(s.f_n);
          s.W_21.resize(s.f_n);
          s.W_22.resize(s.f_n);
        }
        else
        {
          s.dim = 3;
          Eigen::SparseMatrix<double> G;
          igl::grad_ref(s.V, s.F,s.RF, G,
                    s.mesh_improvement_3d /*use normal gradient, or one from a "regular" tet*/);
          s.Dx = G.block(0, 0, s.F.rows(), s.V.rows());
          s.Dy = G.block(s.F.rows(), 0, s.F.rows(), s.V.rows());
          s.Dz = G.block(2 * s.F.rows(), 0, s.F.rows(), s.V.rows());


          s.W_11.resize(s.f_n);
          s.W_12.resize(s.f_n);
          s.W_13.resize(s.f_n);
          s.W_21.resize(s.f_n);
          s.W_22.resize(s.f_n);
          s.W_23.resize(s.f_n);
          s.W_31.resize(s.f_n);
          s.W_32.resize(s.f_n);
          s.W_33.resize(s.f_n);
        }

        s.Dx.makeCompressed();
        s.Dy.makeCompressed();
        s.Dz.makeCompressed();
        s.Ri.resize(s.f_n, s.dim * s.dim);
        s.Ji.resize(s.f_n, s.dim * s.dim);
        s.rhs.resize(s.dim * s.v_num);

        // flattened weight matrix
        s.WGL_M.resize(s.dim * s.dim * s.f_n);
        for (int i = 0; i < s.dim * s.dim; i++)
          for (int j = 0; j < s.f_n; j++)
            s.WGL_M(i * s.f_n + j) = s.M(j);

        s.first_solve = true;
        s.has_pre_calc = true;
      }
    }

  //  IGL_INLINE void build_linear_system(igl::SLIMData& s, Eigen::SparseMatrix<double> &L)
  //  {
		////Timer<> time0;
		////time0.beginStage("buildA");

  //    // formula (35) in paper
  //    Eigen::SparseMatrix<double> A(s.dim * s.dim * s.f_n, s.dim * s.v_n);
  //    buildA(s,A);
	 // //time0.endStage("end buildA");

	 // //time0.beginStage("At.makeCompresse");
	 // Eigen::SparseMatrix<double> At = A.transpose();
  //    At.makeCompressed();
	 //// time0.endStage("end At.makeCompresse");
	 // //time0.beginStage("L");
  //    Eigen::SparseMatrix<double> id_m(At.rows(), At.rows());
  //    id_m.setIdentity();

  //    // add proximal penalty
  //    L = At * s.WGL_M.asDiagonal() * A + s.proximal_p * id_m; //add also a proximal term
  //    L.makeCompressed();
	 // //time0.endStage("end L");

	 // //time0.beginStage("buildRhs");
	 // buildRhs(s, At);
	 // //time0.endStage("end buildRhs");

	 //// time0.beginStage("add_soft_constraints");
  //    add_soft_constraints(s,L);
  //    L.makeCompressed();
	 // //time0.endStage("end add_soft_constraints");

  //  }
	IGL_INLINE void build_linear_system(igl::SLIMData& s, Eigen::SparseMatrix<double> &L)
	{
		std::vector<Eigen::Triplet<double> > IJV;

#ifdef SLIM_CACHED
		buildA(s, IJV);
		if (s.A.rows() == 0)
		{
			s.A = Eigen::SparseMatrix<double>(s.dim * s.dim * s.f_n, s.dim * s.v_n);
			igl::sparse_cached_precompute(IJV, s.A, s.A_data);
		}
		else {
			igl::sparse_cached(IJV, s.A, s.A_data);
		}
#else
		Eigen::SparseMatrix<double> A(s.dim * s.dim * s.f_n, s.dim * s.v_n);
		buildA(s, IJV);
		A.setFromTriplets(IJV.begin(), IJV.end());
		A.makeCompressed();
#endif

#ifdef SLIM_CACHED
#else
		Eigen::SparseMatrix<double> At = A.transpose();
		At.makeCompressed();
#endif

#ifdef SLIM_CACHED
		Eigen::SparseMatrix<double> id_m(s.A.cols(), s.A.cols());
#else
		Eigen::SparseMatrix<double> id_m(A.cols(), A.cols());
#endif

		id_m.setIdentity();

		// add proximal penalty
#ifdef SLIM_CACHED
		s.AtA_data.W = s.WGL_M;
		if (s.AtA.rows() == 0)
			igl::AtA_cached_precompute(s.A, s.AtA, s.AtA_data);
		else
			igl::AtA_cached(s.A, s.AtA, s.AtA_data);

		L = s.AtA + s.proximal_p * id_m; //add also a proximal 
		L.makeCompressed();

#else
		L = At * s.WGL_M.asDiagonal() * A + s.proximal_p * id_m; //add also a proximal term
		L.makeCompressed();
#endif

#ifdef SLIM_CACHED
		buildRhs(s, s.A);
#else
		buildRhs(s, A);
#endif

		Eigen::SparseMatrix<double> OldL = L;
		add_soft_constraints(s, L);
		L.makeCompressed();
	}

    IGL_INLINE void add_soft_constraints(igl::SLIMData& s, Eigen::SparseMatrix<double> &L)
    {
      int v_n = s.v_num;
	  //if (s.Projection) 
	  {
		  for (int d = 0; d < s.dim; d++)
		  {
			  for (int i = 0; i < s.b.rows(); i++)
			  {
				  int v_idx = s.b(i);
				  if (s.Projection) {
					  s.rhs(d * v_n + v_idx) += s.lamda_C * s.bc(i, d); // rhs
					  L.coeffRef(d * v_n + v_idx, d * v_n + v_idx) += s.lamda_C; // diagonal of matrix
				  }
				  else {
					  s.rhs(d * v_n + v_idx) += s.soft_const_p * s.bc(i, d); // rhs
					  L.coeffRef(d * v_n + v_idx, d * v_n + v_idx) += s.soft_const_p; // diagonal of matrix
				  }
			  }
		  }
	  }
	  {
		  //add equality
		  std::vector<Eigen::Triplet<double> > equality;
		  int32_t cn = 0;
		  for (uint32_t i = 0; i < s.Vgroups.size(); i++) cn += s.Vgroups[i].size();

		  Eigen::VectorXd b(cn * 3), A_Tb; cn = 0;
		  for (uint32_t i = 0; i < s.Vgroups.size(); i++) {
			  double coefficient = -1.0 / s.Vgroups[i].size();
			  for (auto vid: s.Vgroups[i]) {
				  for (int d = 0; d < s.dim; d++) {
					  equality.push_back(Eigen::Triplet<double>(s.dim * cn + d, d*s.v_num + vid, 1));
					  for (auto vid2 : s.Vgroups[i]) {
						  equality.push_back(Eigen::Triplet<double>(s.dim * cn + d, d*s.v_num + vid2, coefficient));
						  b[s.dim * cn + d] = 0;
					  }
				  }
				  cn++;
			  }
		  }
		  //for (uint32_t i = 0; i < s.Vgroups.size(); i++) cn += s.Vgroups[i].size();

		  //Eigen::VectorXd b(cn * s.dim), A_Tb; cn = 0;
		  //for (uint32_t i = 0; i < s.Vgroups.size(); i++) {
			 // for (uint32_t j = 0; j < s.Vgroups[i].size();j++) {
				//  int v0 = s.Vgroups[i][j], v1 = s.Vgroups[i][(j+1)% s.Vgroups[i].size()];
				//  for (int d = 0; d < s.dim; d++) {
				//	  equality.push_back(Eigen::Triplet<double>(s.dim * cn + d, d*s.v_num + v0, 1));
				//	  equality.push_back(Eigen::Triplet<double>(s.dim * cn + d, d*s.v_num + v1, -1));
				//	  b[s.dim * cn + d] = 0;
				//  }
				//  cn++;
			 // }
		  //}
		  int row_num = s.dim * cn, col_num = s.dim * s.v_num;
		  Eigen::SparseMatrix<double> A_(row_num, col_num), A_T(col_num, row_num), A_TA_(col_num, col_num);
		  A_.setFromTriplets(equality.begin(), equality.end());
		  A_T = A_.transpose(); A_TA_ = A_T * A_;
		  A_Tb = A_T * b;

		  L = L + s.lamda_glue * A_TA_; s.rhs = s.rhs + s.lamda_glue * A_Tb;
	  }
	  {//localize region
		  for (int d = 0; d < s.dim; d++)
		  {
			  for (int i = 0; i < s.regionb.rows(); i++)
			  {
				  int v_idx = s.regionb(i);
				  s.rhs(d * v_n + v_idx) += s.lamda_region * s.regionbc(i, d); // rhs
				  L.coeffRef(d * v_n + v_idx, d * v_n + v_idx) += s.lamda_region; // diagonal of matrix
			  }
		  }
	  }
	  if (!s.Projection) {
		  //add corner
		  for (int d = 0; d < s.dim; d++)
		  {
			  for (int i = 0; i < s.ids_C.rows(); i++)
			  {
				  int v_idx = s.ids_C(i);
				  s.rhs(d * v_n + v_idx) += s.lamda_C * s.C(i, d); // rhs
				  L.coeffRef(d * v_n + v_idx, d * v_n + v_idx) += s.lamda_C; // diagonal of matrix
			  }
		  }
		  //std::cout << "added corners  with coefficient "<< s.lamda_C << std::endl;

		  //add tagents
		  std::vector<Eigen::Triplet<double> > tagents;
		  Eigen::VectorXd b(s.ids_T.rows()), A_Tb;
		  for (int i = 0; i < s.ids_T.rows(); i++) {
			  int vid = s.ids_T[i];
			  for (int j = 0; j < s.dim; j++)
				  tagents.push_back(Eigen::Triplet<double>(i, j*s.v_num + vid, s.normal_T(i, j)));
			  b[i] = s.dis_T[i];
		  }
		  int row_num = s.ids_T.rows(), col_num = s.dim * s.v_num;
		  Eigen::SparseMatrix<double> A_(row_num, col_num), A_T(col_num, row_num), A_TA_(col_num, col_num);
		  A_.setFromTriplets(tagents.begin(), tagents.end());
		  A_T = A_.transpose(); A_TA_ = A_T * A_;
		  A_Tb = A_T * b;
	
		  L = L + s.lamda_T * A_TA_; s.rhs = s.rhs + s.lamda_T * A_Tb;
		 // std::cout << "added tagent with coefficient "<< s.lamda_T << std::endl;

		  ////add curve
		  std::vector<Eigen::Triplet<double> > curves;
		  Eigen::VectorXd bl_(s.dim * s.ids_L.rows() + s.ids_L.rows()), Al_Tbl_;
		  for (int i = 0; i < s.ids_L.rows(); i++) {
			  int vid = s.ids_L[i];
			  for (int d = 0; d < s.dim; d++) {
				  curves.push_back(Eigen::Triplet<double>(s.dim * i + d, d * s.v_n + vid, 1.0));
				  curves.push_back(Eigen::Triplet<double>(s.dim * i + d, s.dim * s.v_n + i, -s.Axa_L(i, d)));
				  //curves.push_back(Eigen::Triplet<double>(s.dim * i + d, s.dim * s.v_n + i, 0));
				  bl_[s.dim * i + d] = s.origin_L(i, d);
			  }
		  }
		  //for (int i = 0; i < s.ids_L.rows(); i++) {
			 // curves.push_back(Eigen::Triplet<double>(s.dim * s.ids_L.rows() + i, s.dim * s.v_n + i, 1.0));
			 // bl_[s.dim * s.ids_L.rows() + i] = 0.0;
		  //}
  		  row_num = 3 * s.ids_L.rows() + s.ids_L.rows(), col_num = s.dim * s.v_num + s.ids_L.rows();

		  Eigen::SparseMatrix<double> Al_(row_num, col_num), Al_T(col_num, row_num), Al_TAl_(col_num, col_num);
		  Al_.setFromTriplets(curves.begin(), curves.end());
		  Al_T = Al_.transpose(); Al_TAl_ = Al_T * Al_;
		  Al_Tbl_ = Al_T * bl_;

		  Eigen::SparseMatrix<double> L_(col_num, col_num); 
		  curves.clear();
		  for (int k = 0; k<L.outerSize(); ++k)
			  for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it)
				  curves.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
		  L_.setFromTriplets(curves.begin(), curves.end());
		  //L.conservativeResize(col_num, col_num);
		  L = L_ + s.lamda_L * Al_TAl_;

		  s.rhs.conservativeResizeLike(Eigen::VectorXd::Zero(col_num)); s.rhs = s.rhs + s.lamda_L * Al_Tbl_;
		  //std::cout << "added curve with coefficient " << s.lamda_L << std::endl;

		  ////add feature lines
		  //for (int d = 0; d < s.dim; d++)
		  //{
			 // for (int i = 0; i < s.ids_L.rows(); i++)
			 // {
				//  int v_idx = s.ids_L(i);
				//  s.rhs(d * v_n + v_idx) += s.lamda_L * s.origin_L(i, d); // rhs
				//  L.coeffRef(d * v_n + v_idx, d * v_n + v_idx) += s.lamda_L; // diagonal of matrix
			 // }
		  //}
	  }
    }

    IGL_INLINE double compute_energy(igl::SLIMData& s, Eigen::MatrixXd &V_new)
    {
      compute_jacobians(s,V_new);
	  double e = compute_energy_with_jacobians(s, s.V, s.F, s.Ji, V_new, s.M);
	  //std::cout << "jacobian energy e: " << e << std::endl;
	  double esoft= compute_soft_const_energy(s, s.V, s.F, V_new);
		//std::cout << "soft constraints energy e: " << esoft << std::endl;
		e += esoft;
		//std::cout << "energy e: " << e<< std::endl;
		return e;
	}


    IGL_INLINE double compute_soft_const_energy(igl::SLIMData& s,
                                                const Eigen::MatrixXd &V,
                                                const Eigen::MatrixXi &F,
                                                Eigen::MatrixXd &V_o)
    {
      double e = 0;
	  {
		  for (int i = 0; i < s.b.rows(); i++)
		  {
			  if(s.Projection) e += s.lamda_T * (s.bc.row(i) - V_o.row(s.b(i))).squaredNorm();
			  else  e += s.soft_const_p * (s.bc.row(i) - V_o.row(s.b(i))).squaredNorm();
		  }
	  }
	  {
		  for (uint32_t i = 0; i < s.Vgroups.size(); i++) {
			  double coefficient = 1.0 / s.Vgroups[i].size();
			  Eigen::RowVectorXd cv(s.dim);
			  cv.setZero();
			  for (uint32_t j = 0; j < s.Vgroups[i].size(); j++) {
				  int v0 = s.Vgroups[i][j], v1 =s.Vgroups[i][(j+1)% s.Vgroups[i].size()];
				  e += s.lamda_glue * (V_o.row(v0) - V_o.row(v1)).squaredNorm();
			  }
		  }
		  //std::cout << "after glue soft constraints energy e: " << e << std::endl;
	  }
	  {//localize region
		  for (int i = 0; i < s.regionb.rows(); i++)
		  {
			  e += s.lamda_region * (s.regionbc.row(i) - V_o.row(s.regionb(i))).squaredNorm();
		  }
		  //std::cout << "after local region soft constraints energy e: " << e << std::endl;
	  }
	  if (!s.Projection) {
		  //add corner
		  for (int i = 0; i < s.ids_C.rows(); i++)
		  {
			  e += s.lamda_C * (s.C.row(i) - V_o.row(s.ids_C(i))).squaredNorm();
		  }
		  //add tagents
		  for (int i = 0; i < s.ids_T.rows(); i++)
		  {
			  double abs_dis = s.normal_T.row(i).dot(V_o.row(s.ids_T[i])) - s.dis_T[i];
			  e += s.lamda_T * abs_dis * abs_dis;
		  }
		  //add curves
		  for (int i = 0; i < s.ids_L.rows(); i++)
		  {
			  double abs_dis = (V_o.row(s.ids_L[i]) - s.a_L[i] * s.Axa_L.row(i) - s.origin_L.row(i)).squaredNorm();
			  e += s.lamda_L * abs_dis;
			  //e += s.a_L[i] * s.a_L[i];
		  }
		  //std::cout << "after feature soft constraints energy e: " << e << std::endl;
	  }
	  return e;
    }

    IGL_INLINE double compute_energy_with_jacobians(igl::SLIMData& s,
                                                    const Eigen::MatrixXd &V,
                                                    const Eigen::MatrixXi &F, const Eigen::MatrixXd &Ji,
                                                    Eigen::MatrixXd &uv, Eigen::VectorXd &areas)
    {

      double energy = 0;
      if (s.dim == 2)
      {
        Eigen::Matrix<double, 2, 2> ji;
        for (int i = 0; i < s.f_n; i++)
        {
          ji(0, 0) = Ji(i, 0);
          ji(0, 1) = Ji(i, 1);
          ji(1, 0) = Ji(i, 2);
          ji(1, 1) = Ji(i, 3);

          typedef Eigen::Matrix<double, 2, 2> Mat2;
          typedef Eigen::Matrix<double, 2, 1> Vec2;
          Mat2 ri, ti, ui, vi;
          Vec2 sing;
          igl::polar_svd(ji, ri, ti, ui, sing, vi);
          double s1 = sing(0);
          double s2 = sing(1);

          switch (s.slim_energy)
          {
            case igl::SLIMData::ARAP:
            {
              energy += areas(i) * (pow(s1 - 1, 2) + pow(s2 - 1, 2));
              break;
            }
            case igl::SLIMData::SYMMETRIC_DIRICHLET:
            {
              energy += areas(i) * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2));
              break;
            }
            case igl::SLIMData::EXP_SYMMETRIC_DIRICHLET:
            {
              energy += areas(i) * exp(s.exp_factor * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2)));
              break;
            }
            case igl::SLIMData::LOG_ARAP:
            {
              energy += areas(i) * (pow(log(s1), 2) + pow(log(s2), 2));
              break;
            }
            case igl::SLIMData::CONFORMAL:
            {
              energy += areas(i) * ((pow(s1, 2) + pow(s2, 2)) / (2 * s1 * s2));
              break;
            }
            case igl::SLIMData::EXP_CONFORMAL:
            {
              energy += areas(i) * exp(s.exp_factor * ((pow(s1, 2) + pow(s2, 2)) / (2 * s1 * s2)));
              break;
            }

          }

        }
      }
      else
      {
        Eigen::Matrix<double, 3, 3> ji;
        for (int i = 0; i < s.f_n; i++)
        {
          ji(0, 0) = Ji(i, 0);
          ji(0, 1) = Ji(i, 1);
          ji(0, 2) = Ji(i, 2);
          ji(1, 0) = Ji(i, 3);
          ji(1, 1) = Ji(i, 4);
          ji(1, 2) = Ji(i, 5);
          ji(2, 0) = Ji(i, 6);
          ji(2, 1) = Ji(i, 7);
          ji(2, 2) = Ji(i, 8);

          typedef Eigen::Matrix<double, 3, 3> Mat3;
          typedef Eigen::Matrix<double, 3, 1> Vec3;
          Mat3 ri, ti, ui, vi;
          Vec3 sing;
          igl::polar_svd(ji, ri, ti, ui, sing, vi);
          double s1 = sing(0);
          double s2 = sing(1);
          double s3 = sing(2);

		  //std::cout << s1 << " " << s2 << " " << s3 << std::endl;
          switch (s.slim_energy)
          {
            case igl::SLIMData::ARAP:
            {
              energy += areas(i) * (pow(s1 - 1, 2) + pow(s2 - 1, 2) + pow(s3 - 1, 2));
              break;
            }
            case igl::SLIMData::SYMMETRIC_DIRICHLET:
            {
              energy += areas(i) * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2) + pow(s3, 2) + pow(s3, -2));
              break;
            }
            case igl::SLIMData::EXP_SYMMETRIC_DIRICHLET:
            {
              energy += areas(i) * exp(s.exp_factor *
                                       (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2) + pow(s3, 2) + pow(s3, -2)));
              break;
            }
            case igl::SLIMData::LOG_ARAP:
            {
              energy += areas(i) * (pow(log(s1), 2) + pow(log(std::abs(s2)), 2) + pow(log(std::abs(s3)), 2));
              break;
            }
            case igl::SLIMData::CONFORMAL:
            {
              energy += areas(i) * ((pow(s1, 2) + pow(s2, 2) + pow(s3, 2)) / (3 * pow(s1 * s2 * s3, 2. / 3.)));
              break;
            }
            case igl::SLIMData::EXP_CONFORMAL:
            {
              energy += areas(i) * exp((pow(s1, 2) + pow(s2, 2) + pow(s3, 2)) / (3 * pow(s1 * s2 * s3, 2. / 3.)));
              break;
            }
          }
		  //std::cout << "energy i " <<i<<" "<< energy << std::endl;
        }
      }

      return energy;
    }
IGL_INLINE void buildA(igl::SLIMData& s, std::vector<Eigen::Triplet<double> > & IJV)
{
	// formula (35) in paper
	if (s.dim == 2)
	{
		IJV.reserve(4 * (s.Dx.outerSize() + s.Dy.outerSize()));

		/*A = [W11*Dx, W12*Dx;
		W11*Dy, W12*Dy;
		W21*Dx, W22*Dx;
		W21*Dy, W22*Dy];*/
		for (int k = 0; k < s.Dx.outerSize(); ++k)
		{
			for (Eigen::SparseMatrix<double>::InnerIterator it(s.Dx, k); it; ++it)
			{
				int dx_r = it.row();
				int dx_c = it.col();
				double val = it.value();

				IJV.push_back(Eigen::Triplet<double>(dx_r, dx_c, val * s.W_11(dx_r)));
				IJV.push_back(Eigen::Triplet<double>(dx_r, s.v_n + dx_c, val * s.W_12(dx_r)));

				IJV.push_back(Eigen::Triplet<double>(2 * s.f_n + dx_r, dx_c, val * s.W_21(dx_r)));
				IJV.push_back(Eigen::Triplet<double>(2 * s.f_n + dx_r, s.v_n + dx_c, val * s.W_22(dx_r)));
			}
		}

		for (int k = 0; k < s.Dy.outerSize(); ++k)
		{
			for (Eigen::SparseMatrix<double>::InnerIterator it(s.Dy, k); it; ++it)
			{
				int dy_r = it.row();
				int dy_c = it.col();
				double val = it.value();

				IJV.push_back(Eigen::Triplet<double>(s.f_n + dy_r, dy_c, val * s.W_11(dy_r)));
				IJV.push_back(Eigen::Triplet<double>(s.f_n + dy_r, s.v_n + dy_c, val * s.W_12(dy_r)));

				IJV.push_back(Eigen::Triplet<double>(3 * s.f_n + dy_r, dy_c, val * s.W_21(dy_r)));
				IJV.push_back(Eigen::Triplet<double>(3 * s.f_n + dy_r, s.v_n + dy_c, val * s.W_22(dy_r)));
			}
		}
	}
	else
	{

		/*A = [W11*Dx, W12*Dx, W13*Dx;
		W11*Dy, W12*Dy, W13*Dy;
		W11*Dz, W12*Dz, W13*Dz;
		W21*Dx, W22*Dx, W23*Dx;
		W21*Dy, W22*Dy, W23*Dy;
		W21*Dz, W22*Dz, W23*Dz;
		W31*Dx, W32*Dx, W33*Dx;
		W31*Dy, W32*Dy, W33*Dy;
		W31*Dz, W32*Dz, W33*Dz;];*/
		IJV.reserve(9 * (s.Dx.outerSize() + s.Dy.outerSize() + s.Dz.outerSize()));
		for (int k = 0; k < s.Dx.outerSize(); k++)
		{
			for (Eigen::SparseMatrix<double>::InnerIterator it(s.Dx, k); it; ++it)
			{
				int dx_r = it.row();
				int dx_c = it.col();
				double val = it.value();

				IJV.push_back(Eigen::Triplet<double>(dx_r, dx_c, val * s.W_11(dx_r)));
				IJV.push_back(Eigen::Triplet<double>(dx_r, s.v_n + dx_c, val * s.W_12(dx_r)));
				IJV.push_back(Eigen::Triplet<double>(dx_r, 2 * s.v_n + dx_c, val * s.W_13(dx_r)));

				IJV.push_back(Eigen::Triplet<double>(3 * s.f_n + dx_r, dx_c, val * s.W_21(dx_r)));
				IJV.push_back(Eigen::Triplet<double>(3 * s.f_n + dx_r, s.v_n + dx_c, val * s.W_22(dx_r)));
				IJV.push_back(Eigen::Triplet<double>(3 * s.f_n + dx_r, 2 * s.v_n + dx_c, val * s.W_23(dx_r)));

				IJV.push_back(Eigen::Triplet<double>(6 * s.f_n + dx_r, dx_c, val * s.W_31(dx_r)));
				IJV.push_back(Eigen::Triplet<double>(6 * s.f_n + dx_r, s.v_n + dx_c, val * s.W_32(dx_r)));
				IJV.push_back(Eigen::Triplet<double>(6 * s.f_n + dx_r, 2 * s.v_n + dx_c, val * s.W_33(dx_r)));
			}
		}

		for (int k = 0; k < s.Dy.outerSize(); k++)
		{
			for (Eigen::SparseMatrix<double>::InnerIterator it(s.Dy, k); it; ++it)
			{
				int dy_r = it.row();
				int dy_c = it.col();
				double val = it.value();

				IJV.push_back(Eigen::Triplet<double>(s.f_n + dy_r, dy_c, val * s.W_11(dy_r)));
				IJV.push_back(Eigen::Triplet<double>(s.f_n + dy_r, s.v_n + dy_c, val * s.W_12(dy_r)));
				IJV.push_back(Eigen::Triplet<double>(s.f_n + dy_r, 2 * s.v_n + dy_c, val * s.W_13(dy_r)));

				IJV.push_back(Eigen::Triplet<double>(4 * s.f_n + dy_r, dy_c, val * s.W_21(dy_r)));
				IJV.push_back(Eigen::Triplet<double>(4 * s.f_n + dy_r, s.v_n + dy_c, val * s.W_22(dy_r)));
				IJV.push_back(Eigen::Triplet<double>(4 * s.f_n + dy_r, 2 * s.v_n + dy_c, val * s.W_23(dy_r)));

				IJV.push_back(Eigen::Triplet<double>(7 * s.f_n + dy_r, dy_c, val * s.W_31(dy_r)));
				IJV.push_back(Eigen::Triplet<double>(7 * s.f_n + dy_r, s.v_n + dy_c, val * s.W_32(dy_r)));
				IJV.push_back(Eigen::Triplet<double>(7 * s.f_n + dy_r, 2 * s.v_n + dy_c, val * s.W_33(dy_r)));
			}
		}

		for (int k = 0; k < s.Dz.outerSize(); k++)
		{
			for (Eigen::SparseMatrix<double>::InnerIterator it(s.Dz, k); it; ++it)
			{
				int dz_r = it.row();
				int dz_c = it.col();
				double val = it.value();

				IJV.push_back(Eigen::Triplet<double>(2 * s.f_n + dz_r, dz_c, val * s.W_11(dz_r)));
				IJV.push_back(Eigen::Triplet<double>(2 * s.f_n + dz_r, s.v_n + dz_c, val * s.W_12(dz_r)));
				IJV.push_back(Eigen::Triplet<double>(2 * s.f_n + dz_r, 2 * s.v_n + dz_c, val * s.W_13(dz_r)));

				IJV.push_back(Eigen::Triplet<double>(5 * s.f_n + dz_r, dz_c, val * s.W_21(dz_r)));
				IJV.push_back(Eigen::Triplet<double>(5 * s.f_n + dz_r, s.v_n + dz_c, val * s.W_22(dz_r)));
				IJV.push_back(Eigen::Triplet<double>(5 * s.f_n + dz_r, 2 * s.v_n + dz_c, val * s.W_23(dz_r)));

				IJV.push_back(Eigen::Triplet<double>(8 * s.f_n + dz_r, dz_c, val * s.W_31(dz_r)));
				IJV.push_back(Eigen::Triplet<double>(8 * s.f_n + dz_r, s.v_n + dz_c, val * s.W_32(dz_r)));
				IJV.push_back(Eigen::Triplet<double>(8 * s.f_n + dz_r, 2 * s.v_n + dz_c, val * s.W_33(dz_r)));
			}
		}
	}
}
    IGL_INLINE void buildRhs(igl::SLIMData& s, const Eigen::SparseMatrix<double> &A)
    {
		Eigen::VectorXd f_rhs(s.dim * s.dim * s.f_n);
		f_rhs.setZero();
		if (s.dim == 2)
		{
			/*b = [W11*R11 + W12*R21; (formula (36))
			W11*R12 + W12*R22;
			W21*R11 + W22*R21;
			W21*R12 + W22*R22];*/
			for (int i = 0; i < s.f_n; i++)
			{
				f_rhs(i + 0 * s.f_n) = s.W_11(i) * s.Ri(i, 0) + s.W_12(i) * s.Ri(i, 1);
				f_rhs(i + 1 * s.f_n) = s.W_11(i) * s.Ri(i, 2) + s.W_12(i) * s.Ri(i, 3);
				f_rhs(i + 2 * s.f_n) = s.W_21(i) * s.Ri(i, 0) + s.W_22(i) * s.Ri(i, 1);
				f_rhs(i + 3 * s.f_n) = s.W_21(i) * s.Ri(i, 2) + s.W_22(i) * s.Ri(i, 3);
			}
		}
		else
		{
			/*b = [W11*R11 + W12*R21 + W13*R31;
			W11*R12 + W12*R22 + W13*R32;
			W11*R13 + W12*R23 + W13*R33;
			W21*R11 + W22*R21 + W23*R31;
			W21*R12 + W22*R22 + W23*R32;
			W21*R13 + W22*R23 + W23*R33;
			W31*R11 + W32*R21 + W33*R31;
			W31*R12 + W32*R22 + W33*R32;
			W31*R13 + W32*R23 + W33*R33;];*/
			for (int i = 0; i < s.f_n; i++)
			{
				f_rhs(i + 0 * s.f_n) = s.W_11(i) * s.Ri(i, 0) + s.W_12(i) * s.Ri(i, 1) + s.W_13(i) * s.Ri(i, 2);
				f_rhs(i + 1 * s.f_n) = s.W_11(i) * s.Ri(i, 3) + s.W_12(i) * s.Ri(i, 4) + s.W_13(i) * s.Ri(i, 5);
				f_rhs(i + 2 * s.f_n) = s.W_11(i) * s.Ri(i, 6) + s.W_12(i) * s.Ri(i, 7) + s.W_13(i) * s.Ri(i, 8);
				f_rhs(i + 3 * s.f_n) = s.W_21(i) * s.Ri(i, 0) + s.W_22(i) * s.Ri(i, 1) + s.W_23(i) * s.Ri(i, 2);
				f_rhs(i + 4 * s.f_n) = s.W_21(i) * s.Ri(i, 3) + s.W_22(i) * s.Ri(i, 4) + s.W_23(i) * s.Ri(i, 5);
				f_rhs(i + 5 * s.f_n) = s.W_21(i) * s.Ri(i, 6) + s.W_22(i) * s.Ri(i, 7) + s.W_23(i) * s.Ri(i, 8);
				f_rhs(i + 6 * s.f_n) = s.W_31(i) * s.Ri(i, 0) + s.W_32(i) * s.Ri(i, 1) + s.W_33(i) * s.Ri(i, 2);
				f_rhs(i + 7 * s.f_n) = s.W_31(i) * s.Ri(i, 3) + s.W_32(i) * s.Ri(i, 4) + s.W_33(i) * s.Ri(i, 5);
				f_rhs(i + 8 * s.f_n) = s.W_31(i) * s.Ri(i, 6) + s.W_32(i) * s.Ri(i, 7) + s.W_33(i) * s.Ri(i, 8);
			}
		}
		Eigen::VectorXd uv_flat(s.dim *s.v_n);
		for (int i = 0; i < s.dim; i++)
			for (int j = 0; j < s.v_n; j++)
				uv_flat(s.v_n * i + j) = s.V_o(j, i);

		s.rhs = (f_rhs.transpose() * s.WGL_M.asDiagonal() * A).transpose() + s.proximal_p * uv_flat;
    }

  }
}

/// Slim Implementation

IGL_INLINE void igl::slim_precompute(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &V_init, SLIMData &data,
                                     SLIMData::SLIM_ENERGY slim_energy, Eigen::VectorXi &b, Eigen::MatrixXd &bc,
	Eigen::VectorXi &ids_C_, Eigen::MatrixXd &C_,
	Eigen::VectorXi &ids_L_, Eigen::MatrixXd &Axa_L_, Eigen::MatrixXd &Origin_L_,
	Eigen::VectorXi &ids_T_, Eigen::MatrixXd &normal_T_, Eigen::VectorXd &dis_T_, 
	Eigen::VectorXi &regionb, Eigen::MatrixXd &regionbc,
	bool surface_projection, bool global_opt,
	std::vector<Eigen::MatrixXd> RF
)
{

  data.V = V;
  data.F = F;
  data.RF = RF;
  data.V_o = V_init;

  data.v_num = V.rows();
  data.f_num = F.rows();

  data.slim_energy = slim_energy;

  data.b = b;
  data.bc = bc;
  
 //feature constraints
  data.ids_C = ids_C_;
  data.C = C_;
  data.ids_T = ids_T_;
  data.normal_T = normal_T_;
  data.dis_T = dis_T_;
  data.ids_L = ids_L_;
  data.Axa_L = Axa_L_;
  data.origin_L = Origin_L_;
  data.a_L.resize(data.ids_L.rows()); data.a_L.setZero();
//region
  data.regionb = regionb;
  data.regionbc = regionbc;


  data.Projection = surface_projection;
  data.Global = global_opt;

 // data.proximal_p = 0.0001;

//igl::doublearea(V, F, data.M);
  // igl::volume(V, F, data.M);
  // for (int i = 0; i < data.M.size(); i++)
  // {
	//   if (data.M[i] < 1e-2)
	// 	  data.M[i] = 1e-2;
  // }
	data.proximal_p = 0.00001 * data.weight_opt;

  data.M.resize(F.rows());
	data.M.setConstant(data.weight_opt);

  data.M /= 2.;
  data.mesh_area = data.M.sum();
  if(global_opt)  data.mesh_improvement_3d = true;
  else data.mesh_improvement_3d = false; // whether to use a jacobian derived from a real mesh or an abstract regular mesh (used for mesh improvement)
  data.exp_factor = 1.0; // param used only for exponential energies (e.g exponential symmetric dirichlet)

  assert (F.cols() == 3 || F.cols() == 4);

  igl::slim::pre_calc(data);
  data.energy = igl::slim::compute_energy(data,data.V_o) / data.mesh_area;
}


IGL_INLINE Eigen::MatrixXd igl::slim_solve(SLIMData &data, int iter_num)
{
  for (int i = 0; i < iter_num; i++)
  {
    Eigen::MatrixXd dest_res;
    dest_res = data.V_o;

    // Solve Weighted Proxy
	Timer<> time0, time1, time2;
	//time0.beginStage("update weight");
	//if(i==0)
	    igl::slim::update_weights_and_closest_rotations(data,data.V, data.F, dest_res);
	//time0.endStage("end update weight");
	//time0.beginStage("solve weight");
    igl::slim::solve_weighted_arap(data,data.V, data.F, dest_res, data.b, data.bc);
	//time0.endStage("end solve weight");

    double old_energy = data.energy;
	//time2.beginStage(" flip_avoiding");
    std::function<double(Eigen::MatrixXd &)> compute_energy = [&](
        Eigen::MatrixXd &aaa) { return igl::slim::compute_energy(data,aaa); };

    data.energy = igl::flip_avoiding_line_search(data.F, data.V_o, dest_res, compute_energy,
                                                 data.energy * data.mesh_area) / data.mesh_area;
	std::cout << "energy: " << data.energy << std::endl;
	//time2.endStage("end flip_avoiding");
  }
  return data.V_o;
}

#ifdef IGL_STATIC_LIBRARY
#endif
#ifdef WIN32
template void igl::slim_precompute(class Eigen::Matrix<double, -1, -1, 0, -1, -1> &, class Eigen::Matrix<int, -1, -1, 0, -1, -1> &, class Eigen::Matrix<double, -1, -1, 0, -1, -1> &, struct igl::SLIMData &, enum igl::SLIMData::SLIM_ENERGY, class Eigen::Matrix<int, -1, 1, 0, -1, 1> &, class Eigen::Matrix<double, -1, -1, 0, -1, -1> &, class Eigen::Matrix<int, -1, 1, 0, -1, 1> &, class Eigen::Matrix<double, -1, -1, 0, -1, -1> &, class Eigen::Matrix<int, -1, 1, 0, -1, 1> &, class Eigen::Matrix<double, -1, -1, 0, -1, -1> &, class Eigen::Matrix<double, -1, -1, 0, -1, -1> &, class Eigen::Matrix<int, -1, 1, 0, -1, 1> &, class Eigen::Matrix<double, -1, -1, 0, -1, -1> &, class Eigen::Matrix<double, -1, 1, 0, -1, 1> &, class Eigen::Matrix<int, -1, 1, 0, -1, 1> &, class Eigen::Matrix<double, -1, -1, 0, -1, -1> &, bool, bool, class std::vector<class Eigen::Matrix<double, -1, -1, 0, -1, -1>, class std::allocator<class Eigen::Matrix<double, -1, -1, 0, -1, -1> > >);
#endif
