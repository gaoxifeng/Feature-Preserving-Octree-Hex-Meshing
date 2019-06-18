
#include "slim_m.h"

#include "igl/boundary_loop.h"
#include "igl/cotmatrix.h"
#include "igl/edge_lengths.h"
#include "igl/local_basis.h"
#include "igl/repdiag.h"
#include "igl/vector_area_matrix.h"
#include "igl/arap.h"
#include "igl/cat.h"
#include "igl/doublearea.h"
#include "igl/grad.h"
#include "igl/local_basis.h"
#include "igl/per_face_normals.h"
#include "igl/slice_into.h"
#include "igl/volume.h"
#include "igl/polar_svd.h"
#include "igl/flip_avoiding_line_search.h"

#include"igl/slice.h"
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <algorithm>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

//#ifdef CHOLMOD_
#ifdef CHOLMOD_
  #include <Eigen/CholmodSupport>
#endif
#include <unsupported/Eigen/SparseExtra>

    // Definitions of internal functions
     void compute_surface_gradient_matrix(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
                                                    const Eigen::MatrixXd &F1, const Eigen::MatrixXd &F2,
                                                    Eigen::SparseMatrix<double> &D1, Eigen::SparseMatrix<double> &D2);
     void buildA(SLIMData& s, Eigen::SparseMatrix<double> &A);
     void buildRhs(SLIMData& s, const Eigen::SparseMatrix<double> &At);
     void add_soft_constraints(SLIMData& s, Eigen::SparseMatrix<double> &L);
     double compute_energy_total(SLIMData& s, Eigen::MatrixXd &V_new);
     double compute_soft_const_energy(SLIMData& s,
                                                const Eigen::MatrixXd &V,
                                                const Eigen::MatrixXi &F,
                                                Eigen::MatrixXd &V_o);
     double compute_energy_with_jacobians(SLIMData& s,
                                                    const Eigen::MatrixXd &V,
                                                    const Eigen::MatrixXi &F, const Eigen::MatrixXd &Ji,
                                                    Eigen::MatrixXd &uv, Eigen::VectorXd &areas);
     void solve_weighted_arap(SLIMData& s,
                                        const Eigen::MatrixXd &V,
                                        const Eigen::MatrixXi &F,
                                        Eigen::MatrixXd &uv,
                                        Eigen::VectorXi &soft_b_p,
                                        Eigen::MatrixXd &soft_bc_p);
     void update_weights_and_closest_rotations(SLIMData& s,
                                                          const Eigen::MatrixXd &V,
                                                          const Eigen::MatrixXi &F,
                                                          Eigen::MatrixXd &uv);
     void compute_jacobians(SLIMData& s, const Eigen::MatrixXd &uv);
     void build_linear_system(SLIMData& s, Eigen::SparseMatrix<double> &L);
     void pre_calc(SLIMData& s);

    // Implementation
     void compute_surface_gradient_matrix(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,
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

     void compute_jacobians(SLIMData& s, const Eigen::MatrixXd &uv)
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

     void update_weights_and_closest_rotations(SLIMData& s,
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
            case SLIM_ENERGY::ARAP:
            {
              m_sing_new << 1, 1;
              break;
            }
            case SLIM_ENERGY::SYMMETRIC_DIRICHLET:
            {
              double s1_g = 2 * (s1 - pow(s1, -3));
              double s2_g = 2 * (s2 - pow(s2, -3));
              m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));
              break;
            }
            case SLIM_ENERGY::LOG_ARAP:
            {
              double s1_g = 2 * (log(s1) / s1);
              double s2_g = 2 * (log(s2) / s2);
              m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1)));
              break;
            }
            case SLIM_ENERGY::CONFORMAL:
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
            case SLIM_ENERGY::EXP_CONFORMAL:
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
            case SLIM_ENERGY::EXP_SYMMETRIC_DIRICHLET:
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
            case SLIM_ENERGY::ARAP:
            {
              m_sing_new << 1, 1, 1;
              break;
            }
            case SLIM_ENERGY::LOG_ARAP:
            {
              double s1_g = 2 * (log(s1) / s1);
              double s2_g = 2 * (log(s2) / s2);
              double s3_g = 2 * (log(s3) / s3);
              m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));
              break;
            }
            case SLIM_ENERGY::SYMMETRIC_DIRICHLET:
            {
              double s1_g = 2 * (s1 - pow(s1, -3));
              double s2_g = 2 * (s2 - pow(s2, -3));
              double s3_g = 2 * (s3 - pow(s3, -3));
              m_sing_new << sqrt(s1_g / (2 * (s1 - 1))), sqrt(s2_g / (2 * (s2 - 1))), sqrt(s3_g / (2 * (s3 - 1)));
              break;
            }
            case SLIM_ENERGY::EXP_SYMMETRIC_DIRICHLET:
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
            case SLIM_ENERGY::CONFORMAL:
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
            case SLIM_ENERGY::EXP_CONFORMAL:
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

     void solve_weighted_arap(SLIMData& s,
                                        const Eigen::MatrixXd &V,
                                        const Eigen::MatrixXi &F,
                                        Eigen::MatrixXd &uv,
                                        Eigen::VectorXi &soft_b_p,
                                        Eigen::MatrixXd &soft_bc_p)
    {
      using namespace Eigen;

      Eigen::SparseMatrix<double> L;
      build_linear_system(s,L);

      bool cholmod_definition = false;
      #ifdef CHOLMOD_
      cholmod_definition=true;
      #endif

	  // solve
	  Eigen::VectorXd Uc;

	  if (s.ts.known_value_post) {
		  int N = s.dim * s.ts.post_index + s.ts.fc.ids_L.rows();
		  int NC = s.dim * (s.V.rows() - s.ts.post_index);
		  Eigen::SparseMatrix<double> L_ii(N, N), L_ib(N, NC);

		  VectorXi R(N), C(N);
		  for (int i = 0; i < s.dim; i++)
		  {
			  for (int j = 0; j < s.ts.post_index; j++)
			  {
				  R[i*s.ts.post_index + j] = i*s.V.rows() + j;
			  }
		  }
		  for (int j = 0; j < s.ts.fc.ids_L.rows(); j++)
		  {
			  R[s.dim * s.ts.post_index + j] = s.dim * s.V.rows() + j;
		  }
		  C = R;
		  igl::slice(L, R, C, L_ii);

		  C.resize(NC);
		  for (int i = 0; i<s.dim; i++)
			  for (int j = 0; j < s.v_n - s.ts.post_index; j++)
			  {
				  C[i*(s.v_n - s.ts.post_index) + j] = i*s.V.rows() + s.ts.post_index + j;
			  }
		  igl::slice(L, R, C, L_ib);

		  Eigen::VectorXd b_i(N), Aibi, X_b(NC);
		  for(int m = 0; m < s.dim; m++)
			  X_b.segment(m * (s.v_n - s.ts.post_index), 1 * (s.v_n - s.ts.post_index)) = s.ts.post_Variables.col(m);

		  Aibi = L_ib*X_b;
		  for (int m = 0; m < s.dim; m++) 
			  b_i.segment(m * s.ts.post_index, 1 * s.ts.post_index) = s.rhs.segment(m * s.v_n, s.ts.post_index) - Aibi.segment(m * s.ts.post_index, 1 * s.ts.post_index);
		  b_i.segment(s.dim * s.ts.post_index, s.ts.fc.ids_L.rows()) = s.rhs.segment(s.dim * s.v_n, s.ts.fc.ids_L.rows()) - Aibi.segment(s.dim * s.ts.post_index, s.ts.fc.ids_L.rows());

		  L = L_ii;
		  s.rhs = b_i;

		  if (s.dim == 2)
		  {
			  SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
			  Uc = solver.compute(L).solve(s.rhs);
		  }
		  else if(s.dim == 3 && cholmod_definition)
      {
      	#ifdef CHOLMOD_
		    CholmodSimplicialLDLT<Eigen::SparseMatrix<double> > solver;
		    Uc = solver.compute(L).solve(s.rhs);
	      #endif
      }
      else{
			  uint32_t Num_variables = s.ts.post_index * s.dim + s.ts.fc.ids_L.rows();
			  
			  Eigen::VectorXd guess(Num_variables);
			  for (int i = 0; i < s.ts.post_index; i++) for (int j = 0; j < s.dim; j++) guess(s.ts.post_index * j + i) = uv(i, j); // flatten vector
			  //if (!s.ts.projection)
			  for (int i = 0; i < s.ts.fc.ids_L.rows(); i++) guess(s.ts.post_index * s.dim + i) = 0;//feature curve additional variable
			  //Eigen::VectorXd guess(s.dim * s.ts.post_index);
			  //for (int i = 0; i < s.ts.post_index; i++) for (int j = 0; j < s.dim; j++) guess(s.ts.post_index * j + i) = uv(i, j); // flatten vector
			  ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Upper> solver;
			  solver.setTolerance(1e-8);
			  solver.setMaxIterations(20);
			  Uc = solver.compute(L).solveWithGuess(s.rhs, guess);
		  }

		  for (int i = 0; i < s.dim; i++)
			  uv.col(i).segment(0, s.ts.post_index) = Uc.segment(i * s.ts.post_index, 1 * s.ts.post_index);
		  //if (!s.ts.projection && !s.ts.glue) {
		  for (int i = 0; i < s.ts.fc.ids_L.rows(); i++)
			  s.a_L(i) = Uc[s.dim * s.ts.post_index + i];
		  //}
		  return;
	  }

      if (s.dim == 2)
      {
        SimplicialLDLT<Eigen::SparseMatrix<double> > solver;
        Uc = solver.compute(L).solve(s.rhs);
      }
      else
      { // seems like CG performs much worse for 2D and way better for 3D
		  uint32_t Num_variables = uv.rows() * s.dim + s.ts.fc.ids_L.rows();
		  if (s.ts.projection) Num_variables = uv.rows() * s.dim;

        Eigen::VectorXd guess(Num_variables);
        for (int i = 0; i < s.v_n; i++) for (int j = 0; j < s.dim; j++) guess(uv.rows() * j + i) = uv(i, j); // flatten vector
		if(!s.ts.projection)
			for (int i = 0; i < s.ts.fc.ids_L.rows(); i++) guess(uv.rows() * s.dim + i) = 0;//feature curve additional variable
        ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Upper> solver;
        solver.setTolerance(1e-8);
		solver.setMaxIterations(20);
        Uc = solver.compute(L).solveWithGuess(s.rhs, guess);
      }

      for (int i = 0; i < s.dim; i++)
        uv.col(i) = Uc.block(i * s.v_n, 0, s.v_n, 1);
      if (!s.ts.projection) {
        for (int i = 0; i < s.ts.fc.ids_L.rows(); i++)
          s.a_L(i) = Uc[s.dim * s.v_n + i];
      }
    }


     void pre_calc(SLIMData& s)
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
          compute_surface_gradient_matrix(s.V, s.F, F1, F2, s.Dx, s.Dy);

          s.W_11.resize(s.f_n);
          s.W_12.resize(s.f_n);
          s.W_21.resize(s.f_n);
          s.W_22.resize(s.f_n);
        }
        else
        {
          s.dim = 3;
          Eigen::SparseMatrix<double> G;
          //igl::grad(s.V, s.F, G,
          //          s.mesh_improvement_3d /*use normal gradient, or one from a "regular" tet*/);
		  igl::grad_ref(s.V, s.F, s.ts.RT, G, true);
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

     void build_linear_system(SLIMData& s, Eigen::SparseMatrix<double> &L)
    {
      // formula (35) in paper
      Eigen::SparseMatrix<double> A(s.dim * s.dim * s.f_n, s.dim * s.v_n);
      buildA(s,A);

      Eigen::SparseMatrix<double> At = A.transpose();
      At.makeCompressed();

      Eigen::SparseMatrix<double> id_m(At.rows(), At.rows());
      id_m.setIdentity();

      // add proximal penalty
      L = At * s.WGL_M.asDiagonal() * A + s.proximal_p * id_m; //add also a proximal term
      L.makeCompressed();

      buildRhs(s, At);
      Eigen::SparseMatrix<double> OldL = L;

	  //cout << L << endl;
	  //cout << s.rhs << endl;
	
      add_soft_constraints(s,L);
      L.makeCompressed();

	  //cout << L << endl;
	  //cout << s.rhs << endl;
    }

     void add_soft_constraints(SLIMData& s, Eigen::SparseMatrix<double> &L)
    {
      int v_n = s.v_num;
	  if (s.ts.projection) {
		  for (int d = 0; d < s.dim; d++)
		  {
			  for (int i = 0; i < s.b.rows(); i++)
			  {
				  int v_idx = s.b(i);
   				s.rhs(d * v_n + v_idx) += s.ts.lamda_projection * s.bc(i, d); // rhs
				L.coeffRef(d * v_n + v_idx, d * v_n + v_idx) += s.ts.lamda_projection; // diagonal of matrix
			  }
		  }
	  }
	  if (s.ts.glue) {
		  //add equality
		  std::vector<Eigen::Triplet<double> > equality;
		  uint32_t cn = s.ts.V_ranges[s.ts.V_ranges.size() - 1] + 1;
		  Eigen::VectorXd b(cn * s.dim), A_Tb; cn = 0;
		  for (uint32_t i = 0; i < s.ts.V_ranges.size(); i++) {
			  int start = 0, end = s.ts.V_ranges[i];
			  if (i != 0) start = s.ts.V_ranges[i - 1] + 1;

			  double coefficient = -1.0 / (end - start);
			  for (uint32_t j = start; j < end + 1; j++) {
				  for (int d = 0; d < s.dim; d++) {
					  equality.push_back(Eigen::Triplet<double>(s.dim * cn + d, d*s.v_num + j, 1));
					  if (end == start) {
						  equality.push_back(Eigen::Triplet<double>(s.dim * cn + d, d*s.v_num + j, -1));
						  b[s.dim * cn + d] = 0;
						  continue;
					  }
					  for (uint32_t k = start; k < end + 1; k++) {
						  if (k == j) continue;
						  equality.push_back(Eigen::Triplet<double>(s.dim * cn + d, d*s.v_num + k, coefficient));
						  b[s.dim * cn + d] = 0;
					  }
				  }
				  cn++;
			  }
		  }
		  int row_num = s.dim * cn, col_num = s.dim * s.v_num;
		  Eigen::SparseMatrix<double> A_(row_num, col_num), A_T(col_num, row_num), A_TA_(col_num, col_num);
		  A_.setFromTriplets(equality.begin(), equality.end());
		  A_T = A_.transpose(); A_TA_ = A_T * A_;
		  A_Tb = A_T * b;

		  L = L + s.ts.lamda_glue * A_TA_; s.rhs = s.rhs + s.ts.lamda_glue * A_Tb;
		  //std::cout << "added equality with coefficient "<< s.ts.lamda_glue << std::endl;
	  }
	  if (!s.Projection && !s.ts.glue) {
		  //add corner
		  for (int d = 0; d < s.dim; d++)
		  {
			  for (int i = 0; i < s.ts.fc.ids_C.rows(); i++)
			  {
				  int v_idx = s.ts.fc.ids_C(i);
				  s.rhs(d * v_n + v_idx) += s.ts.fc.lamda_C * s.ts.fc.C(i, d); // rhs
				  L.coeffRef(d * v_n + v_idx, d * v_n + v_idx) += s.ts.fc.lamda_C; // diagonal of matrix
			  }
		  }
		  //std::cout << "added corners  with coefficient "<< s.lamda_C << std::endl;

		  //add tagents
		  std::vector<Eigen::Triplet<double> > tagents;
		  Eigen::VectorXd b(s.ts.fc.ids_T.rows()), A_Tb;
		  for (int i = 0; i < s.ts.fc.ids_T.rows(); i++) {
			  int vid = s.ts.fc.ids_T[i];
			  for (int j = 0; j < s.dim; j++)
				  tagents.push_back(Eigen::Triplet<double>(i, j*s.v_num + vid, s.ts.fc.normal_T(i, j)));
			  b[i] = s.ts.fc.dis_T[i];
		  }
		  int row_num = s.ts.fc.ids_T.rows(), col_num = s.dim * s.v_num;
		  Eigen::SparseMatrix<double> A_(row_num, col_num), A_T(col_num, row_num), A_TA_(col_num, col_num);
		  A_.setFromTriplets(tagents.begin(), tagents.end());
		  A_T = A_.transpose(); A_TA_ = A_T * A_;
		  A_Tb = A_T * b;

		  L = L + s.ts.fc.lamda_T * A_TA_; s.rhs = s.rhs + s.ts.fc.lamda_T * A_Tb;
		  // std::cout << "added tagent with coefficient "<< s.lamda_T << std::endl;

		  ////add curve
		  std::vector<Eigen::Triplet<double> > curves;
		  Eigen::VectorXd bl_(s.dim * s.ts.fc.ids_L.rows() + s.ts.fc.ids_L.rows()), Al_Tbl_;
		  for (int i = 0; i < s.ts.fc.ids_L.rows(); i++) {
			  int vid = s.ts.fc.ids_L[i];
			  for (int d = 0; d < s.dim; d++) {
				  curves.push_back(Eigen::Triplet<double>(s.dim * i + d, d * s.v_n + vid, 1.0));
				  curves.push_back(Eigen::Triplet<double>(s.dim * i + d, s.dim * s.v_n + i, -s.ts.fc.Axa_L(i, d)));
				  //curves.push_back(Eigen::Triplet<double>(s.dim * i + d, s.dim * s.v_n + i, 0));
				  bl_[s.dim * i + d] = s.ts.fc.origin_L(i, d);
			  }
		  }
		  //for (int i = 0; i < s.ids_L.rows(); i++) {
		  // curves.push_back(Eigen::Triplet<double>(s.dim * s.ids_L.rows() + i, s.dim * s.v_n + i, 1.0));
		  // bl_[s.dim * s.ids_L.rows() + i] = 0.0;
		  //}
		  row_num = 3 * s.ts.fc.ids_L.rows() + s.ts.fc.ids_L.rows(), col_num = s.dim * s.v_num + s.ts.fc.ids_L.rows();

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
		  L = L_ + s.ts.fc.lamda_L * Al_TAl_;

		  s.rhs.conservativeResizeLike(Eigen::VectorXd::Zero(col_num)); s.rhs = s.rhs + s.ts.fc.lamda_L * Al_Tbl_;
		  //std::cout << "added curve with coefficient " << s.lamda_L << std::endl;
	  }

    }

     double compute_energy_total(SLIMData& s, Eigen::MatrixXd &V_new)
    {
      compute_jacobians(s,V_new);
      //return compute_energy_with_jacobians(s, s.V, s.F, s.Ji, V_new, s.M) +
      //       compute_soft_const_energy(s, s.V, s.F, V_new);

	  double e = compute_energy_with_jacobians(s, s.V, s.F, s.Ji, V_new, s.M);

    s.engery_quality = e;
	  //std::cout << "jacobian energy e: " << e << std::endl;
	  double esoft = compute_soft_const_energy(s, s.V, s.F, V_new);

    s.engery_soft = esoft;
	  //std::cout << "soft constraints energy e: " << esoft << std::endl;
	  e += esoft;
	  //std::cout << "energy e: " << e << std::endl;
	  return e;
    }

     double compute_soft_const_energy(SLIMData& s,
                                                const Eigen::MatrixXd &V,
                                                const Eigen::MatrixXi &F,
                                                Eigen::MatrixXd &V_o)
    {
      double e = 0;
	  if (s.ts.projection) {
		  for (int i = 0; i < s.b.rows(); i++)
		  {
				e += s.ts.lamda_projection * (s.bc.row(i) - V_o.row(s.b(i))).squaredNorm();
		  }
		 //cout << "soft energy: " << e << endl;
	  }
	  if(s.ts.glue) {
		  //add equality
		  for (uint32_t i = 0; i < s.ts.V_ranges.size(); i++) {
			  int start = 0, end = s.ts.V_ranges[i];
			  if (i != 0) start = s.ts.V_ranges[i - 1] + 1;
			  if (end == start) continue;
			  double coefficient = 1.0 / (end - start);
			  for (uint32_t j = start; j < end + 1; j++) {
				  RowVectorXd cv(s.dim);
				  cv.setZero();
				  for (uint32_t k = start; k < end + 1; k++) {
					  if (j == k) continue; 
					  cv += V_o.row(k);
				  }
				  cv *= coefficient;
				  e += s.ts.lamda_glue * (V_o.row(j) - cv).squaredNorm();
				  //cout << "coefficient e " <<coefficient<<" "<< e<< endl;
			  }
		  }
	  }
	  if (!s.Projection &&!s.ts.glue) {
		  //add corner
		  for (int i = 0; i < s.ts.fc.ids_C.rows(); i++)
		  {
			  e += s.ts.fc.lamda_C * (s.ts.fc.C.row(i) - V_o.row(s.ts.fc.ids_C(i))).squaredNorm();
		  }
		  //add tagents
		  for (int i = 0; i < s.ts.fc.ids_T.rows(); i++)
		  {
			  double abs_dis = s.ts.fc.normal_T.row(i).dot(V_o.row(s.ts.fc.ids_T[i])) - s.ts.fc.dis_T[i];
			  e += s.ts.fc.lamda_T * abs_dis * abs_dis;
		  }
		  //add curves
		  for (int i = 0; i < s.ts.fc.ids_L.rows(); i++)
		  {
			  double abs_dis = (V_o.row(s.ts.fc.ids_L[i]) - s.a_L[i] * s.ts.fc.Axa_L.row(i) - s.ts.fc.origin_L.row(i)).squaredNorm();
			  e += s.ts.fc.lamda_L * abs_dis;
			  //e += s.a_L[i] * s.a_L[i];
		  }
		  //std::cout << "after feature soft constraints energy e: " << e << std::endl;
	  }
	  return e;
    }

     double compute_energy_with_jacobians(SLIMData& s,
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
            case SLIM_ENERGY::ARAP:
            {
              energy += areas(i) * (pow(s1 - 1, 2) + pow(s2 - 1, 2));
              break;
            }
            case SLIM_ENERGY::SYMMETRIC_DIRICHLET:
            {
              energy += areas(i) * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2));
              break;
            }
            case SLIM_ENERGY::EXP_SYMMETRIC_DIRICHLET:
            {
              energy += areas(i) * exp(s.exp_factor * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2)));
              break;
            }
            case SLIM_ENERGY::LOG_ARAP:
            {
              energy += areas(i) * (pow(log(s1), 2) + pow(log(s2), 2));
              break;
            }
            case SLIM_ENERGY::CONFORMAL:
            {
              energy += areas(i) * ((pow(s1, 2) + pow(s2, 2)) / (2 * s1 * s2));
              break;
            }
            case SLIM_ENERGY::EXP_CONFORMAL:
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

          switch (s.slim_energy)
          {
            case SLIM_ENERGY::ARAP:
            {
              energy += areas(i) * (pow(s1 - 1, 2) + pow(s2 - 1, 2) + pow(s3 - 1, 2));
              break;
            }
            case SLIM_ENERGY::SYMMETRIC_DIRICHLET:
            {
              energy += areas(i) * (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2) + pow(s3, 2) + pow(s3, -2));
              break;
            }
            case SLIM_ENERGY::EXP_SYMMETRIC_DIRICHLET:
            {
              energy += areas(i) * exp(s.exp_factor *
                                       (pow(s1, 2) + pow(s1, -2) + pow(s2, 2) + pow(s2, -2) + pow(s3, 2) + pow(s3, -2)));
              break;
            }
            case SLIM_ENERGY::LOG_ARAP:
            {
              energy += areas(i) * (pow(log(s1), 2) + pow(log(std::abs(s2)), 2) + pow(log(std::abs(s3)), 2));
              break;
            }
            case SLIM_ENERGY::CONFORMAL:
            {
              energy += areas(i) * ((pow(s1, 2) + pow(s2, 2) + pow(s3, 2)) / (3 * pow(s1 * s2 * s3, 2. / 3.)));
              break;
            }
            case SLIM_ENERGY::EXP_CONFORMAL:
            {
              energy += areas(i) * exp((pow(s1, 2) + pow(s2, 2) + pow(s3, 2)) / (3 * pow(s1 * s2 * s3, 2. / 3.)));
              break;
            }
          }
        }
      }

      return energy;
    }

     void buildA(SLIMData& s, Eigen::SparseMatrix<double> &A)
    {
      // formula (35) in paper
      std::vector<Eigen::Triplet<double> > IJV;
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
      A.setFromTriplets(IJV.begin(), IJV.end());
    }

     void buildRhs(SLIMData& s, const Eigen::SparseMatrix<double> &At)
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

      s.rhs = (At * s.WGL_M.asDiagonal() * f_rhs + s.proximal_p * uv_flat);
    }

/// Slim Implementation
 void slim_precompute(Eigen::MatrixXd& V,
	Eigen::MatrixXi& F,
	Eigen::MatrixXd& V_init,
	SLIMData& data,
	SLIM_ENERGY slim_energy,
	Tetralize_Set ts
) {
	data.V = V;
	data.F = F;
	data.V_o = V_init;

	data.v_num = V.rows();
	data.f_num = F.rows();

	data.slim_energy = slim_energy;

	data.ts = ts;
	data.b = ts.b; 
	data.bc = ts.bc;

	//feature constraints
	data.a_L.resize(ts.fc.ids_L.rows()); data.a_L.setZero();

	data.proximal_p = 0.00001 * data.weight_opt;

  data.M.resize(F.rows());
	data.M.setConstant(data.weight_opt);
	data.mesh_area = data.M.sum();
	if (ts.global)  data.mesh_improvement_3d = true;
	else data.mesh_improvement_3d = false; // whether to use a jacobian derived from a real mesh or an abstract regular mesh (used for mesh improvement)
	data.exp_factor = 1.0; // param used only for exponential energies (e.g exponential symmetric dirichlet)

	assert(F.cols() == 3 || F.cols() == 4);

	pre_calc(data);
	data.energy = compute_energy_total(data, data.V_o) / data.mesh_area;

}
 void slim_precompute(Eigen::MatrixXd &V, Eigen::MatrixXi &F, Eigen::MatrixXd &V_init, SLIMData &data,
                                     SLIM_ENERGY slim_energy, Eigen::VectorXi &b, Eigen::MatrixXd &bc,
	Eigen::VectorXi &ids_C_, Eigen::MatrixXd &C_,
	Eigen::VectorXi &ids_L_, Eigen::MatrixXd &Axa_L_, Eigen::MatrixXd &Origin_L_,
	Eigen::VectorXi &ids_T_, Eigen::MatrixXd &normal_T_, Eigen::VectorXd &dis_T_, 
	Eigen::VectorXi &regionb, Eigen::MatrixXd &regionbc,
	bool surface_projection, bool global_opt
)
{

  data.V = V;
  data.F = F;
  data.V_o = V_init;

  data.v_num = V.rows();
  data.f_num = F.rows();

  data.slim_energy = slim_energy;

  data.b = b;
  data.bc = bc;
 //data.soft_const_p = soft_p;
  
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

  data.proximal_p = 0.0001;

  igl::doublearea(V, F, data.M);
  data.M /= 2.;
  data.mesh_area = data.M.sum();
  if(global_opt)  data.mesh_improvement_3d = true;
  else data.mesh_improvement_3d = false; // whether to use a jacobian derived from a real mesh or an abstract regular mesh (used for mesh improvement)
  data.exp_factor = 1.0; // param used only for exponential energies (e.g exponential symmetric dirichlet)

  assert (F.cols() == 3 || F.cols() == 4);

  pre_calc(data);
  data.energy = compute_energy_total(data,data.V_o) / data.mesh_area;
}

 Eigen::MatrixXd slim_solve2(SLIMData &data, int iter_num, int type, bool verbose)
{
  if (iter_num == -1) {
	  double ratio = 1; 
	  int iter = 0; double energy_pre = -1, energy_cur = -1;
	  int max_iter = 50;
    double MESHRATIO = 1;
    double LAMDA_GLUE_MIN_BOUND = 1e+4;
    double LAMDA_GLUE_MAX_BOUND = 1e+16;
	  double hex_minJ = -1;
    data.ts.lamda_glue = LAMDA_GLUE_MIN_BOUND;
    
	  Mesh_Quality mq;
    
	  std::vector<bool> H_inout_tag(data.ts.m.Hs.size(), true);
	  std::vector<uint32_t> Hids;
	  for (uint32_t i = 0; i < data.ts.m.Hs.size(); i++)Hids.push_back(i);

	  do {
			  
		  Eigen::MatrixXd mV = data.ts.m.V.transpose();
		  compute_referenceMesh(mV, data.ts.m.Hs, H_inout_tag, Hids, data.ts.RT, false);

		  Eigen::MatrixXd dest_res;
		  data.ts.V = dest_res = data.V_o;

		  // Solve Weighted Proxy
		  update_weights_and_closest_rotations(data, data.V, data.F, dest_res);
		  solve_weighted_arap(data, data.V, data.F, dest_res, data.b, data.bc);

		  double old_energy = data.energy;

		  std::function<double(Eigen::MatrixXd &)> compute_energy = [&](
			  Eigen::MatrixXd &aaa) { return compute_energy_total(data, aaa); };

		  data.energy = igl::flip_avoiding_line_search(data.F, data.V_o, dest_res, compute_energy,
			  data.energy * data.mesh_area) / data.mesh_area;

		  if(verbose){
			  cout << "energy at iter: " << iter << " :" << data.energy << endl;
        cout << "energys: " << data.engery_quality << " :" << data.engery_soft << " "<<data.ts.lamda_glue<< endl;
      }
		  iter++;

		  energy_pre = energy_cur;
		  energy_cur = data.energy;
		  if (iter > 1) ratio = (energy_pre - energy_cur) / energy_cur;

		  if (type == 0)//projection
		  {
			  data.ts.m.V= data.V_o.transpose();
			  for (int k = 0; k < data.b.size();k++)
				  data.ts.m.V.col(data.b[k]) = data.bc.row(k);
		  }
		  else if (type == 1)//gluing
		  {
			  data.ts.m.V.resize(3, data.ts.O_Vranges.size());
			   MatrixXd tempV = data.V_o;

			   for (uint32_t i = 0; i < data.ts.V.rows(); i++) tempV.row(i) = data.V_o.row(data.ts.mappedV[i]);
			   for (uint32_t i = 0; i < data.ts.O_Vranges.size(); i++) 
			   {
					int start = 0, end = data.ts.O_Vranges[i];
					if (i != 0) start = data.ts.O_Vranges[i - 1] + 1;
					Vector3d v; v.setZero();
					for (uint32_t j = start; j <= end; j++) v += tempV.row(j);
					v /= (end - start + 1);
					data.ts.m.V.col(i) = v;
			   }
         data.ts.lamda_glue = std::min(LAMDA_GLUE_MIN_BOUND * std::max(data.engery_quality/data.engery_soft * data.ts.lamda_glue, 1.0), LAMDA_GLUE_MAX_BOUND);
		  }
		  scaled_jacobian(data.ts.m, mq);
		  if (mq.min_Jacobian > 0) break;
		  else if (verbose)
			  std::cout << "min J " << mq.min_Jacobian << "; ave J " << mq.ave_Jacobian << endl;
	  } while (iter < max_iter);

	  if (verbose)
		cout << "iterations taken " << iter << endl;
  }
  else {
	  for (int i = 0; i < iter_num; i++)
	  {
		  Eigen::MatrixXd dest_res;
		  dest_res = data.V_o;

		  // Solve Weighted Proxy
		  update_weights_and_closest_rotations(data, data.V, data.F, dest_res);
		  solve_weighted_arap(data, data.V, data.F, dest_res, data.b, data.bc);

		  double old_energy = data.energy;

		  std::function<double(Eigen::MatrixXd &)> compute_energy = [&](
			  Eigen::MatrixXd &aaa) { return compute_energy_total(data, aaa); };

		  data.energy = igl::flip_avoiding_line_search(data.F, data.V_o, dest_res, compute_energy,
			  data.energy * data.mesh_area) / data.mesh_area;
		  cout << "energy at iter: " << i << " :" << data.energy << endl;
	  }
  }

  return data.V_o;
}

 Eigen::MatrixXd slim_solve(SLIMData &data, int iter_num, int type, bool verbose)
{
  if (iter_num == -1) {
	  double ratio = 1; 
	  int iter = 0; double energy_pre = -1, energy_cur = -1;
	  int max_iter = 50;
    //double MESHRATIO = global_boundary_ratio(data.ts.m, false);
    double MESHRATIO = 1;
    double LAMDA_GLUE_BOUND = MESHRATIO * 1e+8;
	  double hex_minJ = -1;
	  Mesh_Quality mq;

	  std::vector<bool> H_inout_tag(data.ts.m.Hs.size(), true);
	  std::vector<uint32_t> Hids;
	  for (uint32_t i = 0; i < data.ts.m.Hs.size(); i++)Hids.push_back(i);

	  do {

		  if (iter > 3 && data.ts.lamda_glue < LAMDA_GLUE_BOUND)
			  data.ts.lamda_glue *=2;

		  Eigen::MatrixXd mV = data.ts.m.V.transpose();
		  compute_referenceMesh(mV, data.ts.m.Hs, H_inout_tag, Hids, data.ts.RT, false);

		  Eigen::MatrixXd dest_res;
		  data.ts.V = dest_res = data.V_o;

		  // Solve Weighted Proxy
		  update_weights_and_closest_rotations(data, data.V, data.F, dest_res);
		  solve_weighted_arap(data, data.V, data.F, dest_res, data.b, data.bc);

		  double old_energy = data.energy;

		  std::function<double(Eigen::MatrixXd &)> compute_energy = [&](
			  Eigen::MatrixXd &aaa) { return compute_energy_total(data, aaa); };

		  data.energy = igl::flip_avoiding_line_search(data.F, data.V_o, dest_res, compute_energy,
			  data.energy * data.mesh_area) / data.mesh_area;

		  if(verbose)
			cout << "energy at iter: " << iter << " :" << data.energy << endl;
		  iter++;

		  energy_pre = energy_cur;
		  energy_cur = data.energy;
		  if (iter > 1) ratio = (energy_pre - energy_cur) / energy_cur;

		  if (type == 0)//projection
		  {
			  data.ts.m.V= data.V_o.transpose();
			  for (int k = 0; k < data.b.size();k++)
				  data.ts.m.V.col(data.b[k]) = data.bc.row(k);
		  }
		  else if (type == 1)//gluing
		  {
			  data.ts.m.V.resize(3, data.ts.O_Vranges.size());
			   MatrixXd tempV = data.V_o;

			   for (uint32_t i = 0; i < data.ts.V.rows(); i++) tempV.row(i) = data.V_o.row(data.ts.mappedV[i]);
			   for (uint32_t i = 0; i < data.ts.O_Vranges.size(); i++) 
			   {
					int start = 0, end = data.ts.O_Vranges[i];
					if (i != 0) start = data.ts.O_Vranges[i - 1] + 1;
					Vector3d v; v.setZero();
					for (uint32_t j = start; j <= end; j++) v += tempV.row(j);
					v /= (end - start + 1);
					data.ts.m.V.col(i) = v;
			   }
		  }
		  scaled_jacobian(data.ts.m, mq);
		  if (mq.min_Jacobian > 0) break;
		  else if (verbose)
			  std::cout << "min J " << mq.min_Jacobian << "; ave J " << mq.ave_Jacobian << endl;

	  } while (iter < max_iter);// || hex_minJ < 0);

	  if (verbose)
		cout << "iterations taken " << iter << endl;
  }
  else {
	  for (int i = 0; i < iter_num; i++)
	  {
		  Eigen::MatrixXd dest_res;
		  dest_res = data.V_o;

		  // Solve Weighted Proxy
		  update_weights_and_closest_rotations(data, data.V, data.F, dest_res);
		  solve_weighted_arap(data, data.V, data.F, dest_res, data.b, data.bc);

		  double old_energy = data.energy;

		  std::function<double(Eigen::MatrixXd &)> compute_energy = [&](
			  Eigen::MatrixXd &aaa) { return compute_energy_total(data, aaa); };

		  data.energy = igl::flip_avoiding_line_search(data.F, data.V_o, dest_res, compute_energy,
			  data.energy * data.mesh_area) / data.mesh_area;
		  cout << "energy at iter: " << i << " :" << data.energy << endl;
	  }
  }

  return data.V_o;
}
