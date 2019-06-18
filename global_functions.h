#pragma once
#include "io.h"
#include "global_types.h"
#include "Eigen/Dense"
#include <algorithm>
#include <iterator>
//#include <omp.h>
#include <iostream>
#include <map>
#include <set>
#include <queue>
#include <tbb/tbb.h>
#include "igl/hausdorff.h"
#include <string>
#include <limits> // for numeric_limits
#include <utility> // for pair
using namespace Eigen;
using namespace std;

//===================================mesh connectivities===================================
void build_connectivity(Mesh &hmi);
void topology_info(Mesh &mesh, Frame &frame, Mesh_Topology & mt);
bool manifoldness_closeness_check(Mesh &mesh);
bool disk_polygon(Mesh &mesh, Frame &frame, vector<vector<uint32_t>> &fes, vector<short> &E_flag, vector<short> &V_flag, const bool &Ismesh);
bool sphere_polyhedral(Mesh &mesh, Frame &frame, vector<vector<uint32_t>> &F_nvs, vector<vector<uint32_t>> &pfs, vector<bool> &F_flag, vector<short> &E_flag, vector<short> &V_flag, const bool &Ismesh);
bool comp_topology(Mesh_Topology & mt0, Mesh_Topology & mt1);
bool redundentV_check(Mesh &meshI, Mesh &meshO);
double average_edge_length(const Mesh &mesh);
void edge_length(const Mesh &mesh, Mesh_Quality &mq);
void compute_volume(Mesh &mesh, Mesh_Quality &mq);
void area_volume(const Mesh &mesh, double &volume);
double global_boundary_ratio(Mesh &mesh, bool medial_surface=true);
void re_indexing_connectivity(Mesh &hmi, MatrixXi &H);
void re_indexing_connectivity(Mesh &hmi, vector<bool> &H_flag, Mesh &Ho, vector<int32_t> &V_map, vector<int32_t> &V_map_reverse, vector<int32_t> &H_map, vector<int32_t> &H_map_reverse);
void re_indexing_connectivity_sur(Mesh &mi, vector<bool> &f_flag, Mesh &mo, vector<int32_t> &V_map, vector<int32_t> &V_map_reverse, vector<int32_t> &f_map, vector<int32_t> &f_map_reverse);

void decompose_components(Mesh &mesh, std::vector<Mesh> &components);
void combine_components(std::vector<Mesh> &components, Mesh &mesh);

void move_boundary_vertices_back(Mesh_Domain &md, std::vector<int> &hex2Octree_map);
void refine_surface_mesh(const Mesh &meshi, Mesh &mesho, int iter);
void triangulation(Mesh &meshi, Mesh &mesho);
void extract_surface_mesh(Mesh &meshi, Mesh &mesho);
void extract_surface_conforming_mesh(Mesh &meshi, Mesh &mesho, vector<int32_t> &V_map, vector<int32_t> &V_map_reverse, vector<int32_t> &F_map, vector<int32_t> &F_map_reverse);
void  orient_surface_mesh(Mesh &hmi);
void  orient_triangle_mesh(Mesh &hmi); 
void  orient_triangle_mesh_acw(Mesh &hmi);
void  orient_triangle_mesh(MatrixXd &Tri_V, MatrixXi &Tri_F);
Float	uctet(vector<Float> a, vector<Float> b, vector<Float> c, vector<Float> d);

void extract_boundary_from_surface_mesh(const Mesh &meshi, Mesh &mesho);
int face_from_vs(Mesh &m, vector<vector<uint32_t>> &fss);
void loops_from_es(const std::vector<Hybrid_E> &es, std::vector<std::vector<int>> &loops);
void direction_correction(std::vector<std::vector<int>> &loops, Mesh &m);
void face_soup_info(Mesh &hmi,vector<uint32_t> &fs_soup, vector<bool> &E_tag, vector<uint32_t> &bvs);
void face_soup_info(Mesh &hmi, vector<uint32_t> &fs_soup, vector<bool> &E_tag, vector<uint32_t> &bes, vector<uint32_t> &bvs);
void face_soup_roll_region(Mesh &hmi, uint32_t sfid, vector<bool> &E_tag, vector<uint32_t> &bes, vector<uint32_t> &fs, vector<uint32_t> &bvs);
void face_soup_expand_region(Mesh &hmi, vector<uint32_t> &fs_soup, vector<bool> &E_tag, vector<bool> &F_tag, vector<uint32_t> &bes);
typedef int vertex_t;
typedef double weight_t;
struct neighbor {
	vertex_t target;
	weight_t weight;
	neighbor(vertex_t arg_target, weight_t arg_weight)
		: target(arg_target), weight(arg_weight) { }
};
typedef std::vector<std::vector<neighbor> > adjacency_list_t;
typedef std::pair<weight_t, vertex_t> weight_vertex_pair_t;
const weight_t max_weight = std::numeric_limits<double>::infinity();
void DijkstraComputePaths(vertex_t source, const adjacency_list_t &adjacency_list, std::vector<weight_t> &min_distance, std::vector<vertex_t> &previous);
void DijkstraComputePaths(vector<vertex_t> &source, const adjacency_list_t &adjacency_list, std::vector<weight_t> &min_distance, std::vector<vertex_t> &previous);
std::vector<vertex_t> DijkstraGetShortestPathTo(vertex_t vertex, const std::vector<vertex_t> &previous);

bool curve_parameterization(const vector<Eigen::Vector3d> &l0, const vector<Eigen::Vector3d> &l1,
	vector<Eigen::Vector3d> &lo, bool circle, bool uniform = true);
//===================================mesh quality==========================================
void reorder_quad_mesh_propogation(Mesh &mi);

void reorder_hex_mesh(Mesh &hmi);
void reorder_hex_mesh_propogation(Mesh &hmi);
bool scaled_jacobian(Mesh &hmi, Mesh_Quality &mq);
Float a_jacobian(Vector3d &v0, Vector3d &v1, Vector3d &v2, Vector3d &v3);
Float a_jacobian_nonscaled(Vector3d &v0, Vector3d &v1, Vector3d &v2, Vector3d &v3);
double a_jacobian(VectorXd &v0, VectorXd &v1, VectorXd &v2, VectorXd &v3);
double a_jacobian(VectorXd &v0, VectorXd &v1, VectorXd &v2);
//===================================feature v tags==========================================
bool quad_mesh_feature(Mesh_Feature &mf);
bool triangle_mesh_feature(Mesh_Feature &mf);
void build_feature_graph(Mesh_Feature &mf, Feature_Graph &fg);
bool triangle_mesh_feature(Mesh_Feature &mf, Mesh &hmi);
bool initial_feature(Mesh_Feature &mf, Feature_Constraints &fc, Mesh &hmi);
bool project_surface_update_feature(Mesh_Feature &mf, Feature_Constraints &fc, MatrixXd &V, VectorXi &b, MatrixXd &bc, uint32_t Loop = 1);
bool phong_projection(vector<uint32_t> &tids, uint32_t Loop, uint32_t &tid, Vector3d &v, Vector3d &interpolP, Vector3d &interpolN, Vector3d &PreinterpolP, Vector3d &PreinterpolN);
void point_line_projection(const Vector3d &v1, const Vector3d &v2, const Vector3d &v, Vector3d &pv, double &t);
void projectPointOnQuad(const vector<Vector3d>& quad_vs, vector<Vector3d> & vs_normals, const Vector3d& p, Vector2d& uv, Vector3d& interpolP, Vector3d& interpolN);
void projectPointOnTriangle(const vector<Vector3d>& tri_vs, const vector<Vector3d> & vs_normals, const Vector3d& p, Vector2d& uv, Vector3d& interpolP, Vector3d& interpolN);
template <typename T>
T bilinear(const T& v1, const T& v2, const T& v3, const T& v4, const Vector2d& uv) {
	return (1 - uv.x()) * ((1 - uv.y()) * v1 + uv.y() * v4) + uv.x() * ((1 - uv.y()) * v2 + uv.y() * v3);
}
template <typename T>
T barycentric(const T& v1, const T& v2, const T& v3, const Vector2d& uv)
{
	return uv.x() * v1 + uv.y() * v2 + (1 - uv.x() - uv.y()) * v3;
}
template <typename T>
bool num_equal(const T& x, const T& y, const double &precision) {
	return std::abs(x - y) <= (std::max)(precision, precision * (std::max)(std::abs(x), std::abs(y)));
}

void nearest_point_loop(const MatrixXd &P, const MatrixXd &V, VectorXi &I);
void hausdorff_dis(Mesh &mesh0, Mesh & mesh1);
bool hausdorff_dis(Mesh &mesh0, Mesh & mesh1, double &hausdorff_ratio_threshould);
bool hausdorff_dis(Mesh &mesh0, Mesh & mesh1, vector<int> & outlierVs, double &hausdorff_dis_threshold);

Float rescale(Mesh &mesh, Float scaleI, bool inverse);
void translate_rescale(const Mesh &ref, Mesh &mesh);
void compute_referenceMesh(MatrixXd &V, vector<Hybrid_F> &F, vector<uint32_t> &Fs, vector<MatrixXd> &Vout, bool square=false);
void quad2square(MatrixXd &V, const vector<uint32_t> &vs, vector<MatrixXd> &vout, bool square = false);

void compute_referenceMesh(MatrixXd &V, vector<Hybrid> &H, vector<bool> &H_flag, vector<uint32_t> &Hs, vector<MatrixXd> &Vout, bool cube);
void hex2cuboid_(MatrixXd &V, const vector<uint32_t> &vs, vector<MatrixXd> &vout, bool &cube, bool &h_flag);

void compute_referenceMesh(MatrixXd &V, vector<Hybrid> &H, vector<uint32_t> &Hs, vector<MatrixXd> &Vout);
void compute_referenceMesh(MatrixXd &V, vector<Hybrid> &H, vector<uint32_t> &Hs, vector<MatrixXd> &Vout, bool cube, vector<double> &Vols);
void hex2cuboid(MatrixXd &V, const vector<uint32_t> &vs, vector<MatrixXd> &vout);
void hex2cuboid(MatrixXd &V, const vector<uint32_t> &vs, vector<MatrixXd> &vout, bool cube, double vol);
void hex2cuboid(MatrixXd &V, const vector<uint32_t> &vs, MatrixXd &vout, bool cube);
void hex2cuboid(double elen, MatrixXd &vout);
void hex2tet24(MatrixXd &V, const vector<uint32_t> &vs, double & volume);

//===================================mesh inside/outside judgement==========================================
void points_inside_mesh(MatrixXd &Ps, Mesh &tmi, VectorXd &signed_dis);
//===================================PCA Bounding Box==========================================
bool PCA_BBOX(MatrixXd &Ps, MatrixXd &T, VectorXd &S);
