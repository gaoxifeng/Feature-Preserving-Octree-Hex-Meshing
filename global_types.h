#pragma once
#include <cstdlib>
#include <vector>
#include <igl/signed_distance.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_geometry.h>
#include <geogram/mesh/mesh_io.h>

#include "Eigen/Dense"
using namespace Eigen;
using namespace std;

/*typedefs*/
#if defined(SINGLE_PRECISION)
typedef float Float;
#else
typedef double Float;
#endif

#define Interior_RegularE 4
#define Boundary_RegularE 2

#define Precision 1.e-7
#define Precision_Pro 1.0e-5
#define Jacobian_Bound 1.0e-4

#define PAI 3.1415926
enum Base_Set{
	SHEET=0,
	CHORD
};
typedef std::tuple<uint32_t, Base_Set, Float> Tuple_Candidate;//id, type, weight

enum Feature_V_Type {
	INTERIOR = -4,
	CORNER,
	LINE,
	REGULAR
};
enum Element_Type {
	Tetrahedral = 0,
	Slab,
	Pyramid,
	Prism,
	PyramidCombine,
	TetCombine,
	Hexahedral
};
enum Which_plane {
	N = -1,
	X,
	Y,
	Z
};

enum SLIM_ENERGY
{
	ARAP,
	LOG_ARAP,
	SYMMETRIC_DIRICHLET,
	CONFORMAL,
	EXP_CONFORMAL,
	EXP_SYMMETRIC_DIRICHLET
};
enum Deform_V_Type {
	Fixed = 0,
	Movable,
	Free = 2
};

const int tetra_table[4][4] =
{
	{ 0,2,1,3 },
	{ 1,0,2,3 },
	{ 2,1,0,3 },
	{ 3,0,1,2 },
};
const int tet_faces[4][3] = {
	{ 1, 0, 2 },
	{ 3, 2, 0 },
	{ 1, 2, 3 },
	{ 0, 1, 3 } 
};
const int tet_edges[6][2] = {
	{ 0, 1},
	{ 0, 2},
	{ 0, 3 },
	{ 1, 2},
	{ 1, 3 },
	{ 2, 3 }
};
//slab:{0, 1, 2, 3, 4}
const int slab_faces[4][4] = {
	{ 0, 1, 2, 3 },
	{ 4, 1, 2, 3 },
	{ 0, 1, 4, -1 },
	{ 0, 3, 4, -1 }
};
const int slab_edges[7][2] = {
	{ 0, 1 },
	{ 1, 2 },
	{ 2, 3 },
	{ 3, 0 },
	{ 0, 4 },
	{ 1, 4 },
	{ 3, 4 }
};
//pyramid:{0, 1, 2, 3, 4}
const int pyramid_faces[5][4] = {
	{ 0, 1, 2, 3 },
	{ 1, 0, 4, -1 },
	{ 2, 1, 4, -1 },
	{ 3, 2, 4, -1 },
	{ 0, 3, 4, -1 }
};
const int pyramid_edges[8][2] = {
	{ 0, 1 },
	{ 1, 2 },
	{ 2, 3 },
	{ 3, 0 },
	{ 0, 4 },
	{ 1, 4 },
	{ 2, 4 },
	{ 3, 4 }
};
//prism:{0, 1, 2, 3, 4, 5}
const int prism_faces[5][4] = {
	{ 0, 1, 2, -1 },
	{ 3, 4, 5, -1 },
	{ 2, 1, 4, 5 },
	{ 1, 0, 3, 4 },
	{ 0, 2, 5, 3 }
};
const int prism_edges[9][2] = {
	{ 0, 1 },
	{ 1, 2 },
	{ 2, 0 },
	{ 3, 4 },
	{ 4, 5 },
	{ 5, 3 },
	{ 0, 3 },
	{ 1, 4 },
	{ 2, 5 }
};
//pyramidcombine:{0, 1, 2, 3, 4,5,6}
const int pyramidcombine_faces[6][4] = {
	{ 0, 1, 2, 3 },
	{ 0, 4, 5, 6 },
	{ 1, 2, 5, 4 },
	{ 2, 3, 6, 5 },
	{ 0, 1, 4, -1 },
	{ 0, 3, 6, -1 }
};
const int hex_face_table[6][4] =
{
	{ 0,1,2,3 },
	{ 4,7,6,5 },
	{ 0,4,5,1 },
	{ 0,3,7,4 },
	{ 3,2,6,7 },
	{ 1,5,6,2 },
};
const int hex_tetra_table[8][4] =
{
	{ 0,3,4,1 },
	{ 1,0,5,2 },
	{ 2,1,6,3 },
	{ 3,2,7,0 },
	{ 4,7,5,0 },
	{ 5,4,6,1 },
	{ 6,5,7,2 },
	{ 7,6,4,3 },
};
const int hex_edge_tetra_table[32][4] =
{
	{ 0,3,4,1 },
	{ 1,0,5,2 },
	{ 2,1,6,3 },
	{ 3,2,7,0 },
	{ 4,7,5,0 },
	{ 5,4,6,1 },
	{ 6,5,7,2 },
	{ 7,6,4,3 },
	{ 0,1,2,6 },
	{ 1,2,3,7 },
	{ 2,3,0,4 },
	{ 3,0,1,5 },
	{ 4,7,6,2 },
	{ 7,6,5,1 },
	{ 6,5,4,0 },
	{ 5,4,7,3 },
	{ 1,0,4,7 },
	{ 3,2,6,5 },
	{ 0,3,7,6 },
	{ 2,1,5,4 },
	{ 0,1,2,4 },
	{ 1,2,3,5 },
	{ 2,3,0,6 },
	{ 3,0,1,7 },
	{ 4,7,6,0 },
	{ 7,6,5,3 },
	{ 6,5,4,2 },
	{ 5,4,7,1 },
	{ 4,0,3,5 },
	{ 2,6,7,1 },
	{ 3,7,4,2 },
	{ 1,5,6,0 },
};
const int quad_tri_table[4][3] =
{
	{ 0,1,3 },
	{ 1,2,0 },
	{ 2,3,1 },
	{ 3,0,2 },
};
const double hex_ref_shape[8][3] =
{
	{ 0, 0, 0 },
	{ 1, 0, 0 },
	{ 1, 1, 0 },
	{ 0, 1, 0 },
	{ 0, 0, 1 },
	{ 1, 0, 1 },
	{ 1, 1, 1 },
	{ 0, 1, 1 },
};
const double tri_ref_shape[3][2] =
{
	{ 0, 0 },
	{ 1, 0 },
	{ 0.5, 0.866025405 },//1.73205081 * 0.5
};

//-------------------------------------------------------------------
//---For Hybrid mesh-------------------------------------------------
struct Hybrid_V
{
	uint32_t id, svid, fvid;
	vector<Float> v;
	vector<uint32_t> neighbor_vs;
	vector<uint32_t> neighbor_es;
	vector<uint32_t> neighbor_fs;
	vector<uint32_t> neighbor_hs;
	Eigen::Vector3d normal;
	bool boundary;
	bool on_medial_surface = false;
};
struct Hybrid_E
{
	uint32_t id;
	vector<uint32_t> vs;
	vector<uint32_t> neighbor_fs;
	vector<uint32_t> neighbor_hs;
	
	bool boundary;

	bool hex_edge = false;
};
struct Hybrid_F
{
	uint32_t id;
	vector<uint32_t> vs;
	vector<uint32_t> es;
	vector<uint32_t> neighbor_hs;
	bool boundary;
	bool on_medial_surface = false;
};

struct Hybrid
{
	uint32_t id;
	vector<uint32_t> vs;
	vector<uint32_t> es;
	vector<uint32_t> fs;
	bool boundary;
};
//-------------------------------------------------------------------
//-------------------------------------------------------------------
struct Singular_V
{
	uint32_t id, hid;
	bool boundary;

	vector<uint32_t> neighbor_svs;//singular vs
	vector<uint32_t> neighbor_ses;//singular es
	bool fake;

	uint32_t which_singularity;
	uint32_t which_singularity_type;
};
struct Singular_E
{
	uint32_t id;
	std::vector<uint32_t> vs;//singular v
	vector<uint32_t> es_link;//hex e
	vector<uint32_t> vs_link;//hex v
	bool boundary;

	vector<uint32_t> neighbor_ses;//singular es
	bool circle;//0 for normal, 2 for circle, 
};
struct Frame_V
{
	uint32_t id, hid, svid;
	uint32_t what_type;//1 singular, 2, extra-nordinary, 3, regular
				  //0 non extra-ordinary node and at surface,1 extra-ordinary node and at surface;
				  //2 non extra-ordinary node and inside volume,3 extra-ordinary node and inside volume;
	vector<uint32_t>  neighbor_fvs;
	vector<uint32_t>  neighbor_fes;
	vector<uint32_t>  neighbor_ffs;
	vector<uint32_t>  neighbor_fhs;
	bool boundary;
};
struct Frame_E
{
	uint32_t id;
	bool singular = false;
	std::vector<uint32_t> vs;
	bool boundary;//0 interior, 1 boundary, 2 first and last slice boundary
	vector<uint32_t>  vs_link;//v of hex_v
	vector<uint32_t>  es_link;//e of hex_e
	vector<uint32_t>  neighbor_fes;
	vector<uint32_t>  neighbor_ffs;
	vector<uint32_t>  neighbor_fhs;
};
struct Frame_F
{
	uint32_t id;
	bool boundary;
	uint32_t F_location;//for rendering

	vector<uint32_t> vs;
	vector<uint32_t> es;
	vector<uint32_t>  fvs_net;
	vector<uint32_t>  ffs_net;
	vector<uint32_t>  neighbor_ffs;
	vector<uint32_t>  neighbor_fhs;
	uint32_t Color_ID;
};
struct Frame_H
{
	uint32_t id;

	std::vector<uint32_t> vs;
	std::vector<uint32_t> es;
	std::vector<uint32_t> fs;
	vector<vector<vector<uint32_t> >> vs_net;
	vector<uint32_t>  fs_net;
	vector<uint32_t>  hs_net;
	vector<uint32_t>  neighbor_fhs;//neighboring cube	
	uint32_t Color_ID;
};


enum Sheet_type {
	open = 0,
	closetype,
	tagent,
	intersect,
	mobius
};
struct Sheet
{
	uint32_t id;
	Sheet_type type;
	bool fake;
	std::vector<uint32_t> ns;
	std::vector<uint32_t> es;
	std::vector<uint32_t> fs;
	std::vector<uint32_t> cs;

	std::vector<uint32_t> middle_es, middle_es_b, left_es, right_es;
	std::vector<uint32_t> middle_fs, side_fs, left_fs, right_fs;

	vector<vector<uint32_t>> vs_pairs, vs_links, Vs_Group;
	VectorXi target_vs;
	MatrixXd target_coords;

	Float weight, weight_val_average, weight_val_max, weight_val_min, weight_len;
	bool valence_filter;
};
struct CHord
{
	uint32_t id;

	Sheet_type type;
	bool fake;
	bool side;//false 0, true 1 
	std::vector<uint32_t> ns;
	std::vector<uint32_t> es;
	std::vector<uint32_t> fs;
	std::vector<uint32_t> cs;

	std::vector<uint32_t> parallel_ns[4];
	std::vector<uint32_t> parallel_es[4], vertical_es[4];
	std::vector<uint32_t> parallel_fs, vertical_fs[4];;

	vector<uint32_t> tangent_vs;
	vector<uint32_t> tangent_es;
	vector<uint32_t> tangent_fs;
	vector<uint32_t> tangent_cs;


	vector<vector<uint32_t>> Vs_Group;
	VectorXi target_vs;
	MatrixXd target_coords;

	Float weight, weight_val_average, weight_val_max, weight_val_min, weight_len;
	bool valence_filter;

};
//-------------------------------------------------------------------
//-------------------------------------------------------------------
struct Feature_Corner {
	int id;
	vector<uint32_t> vs;//mesh v-id
	vector<uint32_t> ring_vs;//
	vector<int> ring_vs_tag;//-1, regular; natural int, index of feature line;
	vector<uint32_t> neighbor_cs;
	vector<uint32_t> neighbor_ls;
	vector<uint32_t> neighbor_rs;
	Vector3d original, projected;
};
struct Feature_Line {
	int id;
	vector<uint32_t> cs;//corner id
	vector<uint32_t> vs;
	vector<uint32_t> es;
	vector<uint32_t> neighbor_rs;
	vector<int> guiding_vs;
	vector<Vector3d> guiding_v;
	bool boundary;
	bool circle = false;
	bool broken = false;
};
struct Feature_Region {
	int id;
	vector<uint32_t> cs;//corner id
	vector<uint32_t> ls;//line id
	vector<uint32_t> vs, tris;
	int Color_ID;
};
//-------------------------------------------------------------------
//-------------------------------------------------------------------
struct Singularity
{
	vector<Singular_V> SVs;
	vector<Singular_E> SEs;
};
struct Frame
{
	vector<Frame_V> FVs;
	vector<Frame_E> FEs;
	vector<Frame_F> FFs;
	vector<Frame_H> FHs;
};
enum Mesh_type {
	line = -1,
	Tri,
	Qua,
	HSur,
	Tet,
	Hyb,
	Hex
};
struct Mesh_Topology
{
	bool euler_problem;
	bool manifoldness_problem;

	int genus;
	int surface_euler;
	int volume_euler;
	bool surface_manifoldness;
	bool volume_manifoldness;

	bool frame_euler_problem;
	bool frame_manifoldness_problem;

	int frame_genus;
	int frame_surface_euler;
	int frame_volume_euler;
	bool frame_surface_manifoldness;
	bool frame_volume_manifoldness;
};

struct Mesh_Quality
{
	std::string Name;
	double min_Jacobian;
	double ave_Jacobian;
	double deviation_Jacobian;
	VectorXd V_Js;
	VectorXd H_Js;
	VectorXd Num_Js;

	int32_t V_num, H_num;
	int32_t BV_num, BC_num;
	int32_t RemovedSheetChord_num;
	double RemovedCuboid_ratio;

	double Hausdorff_ratio;
	vector<double> max_hausdorff;
	vector<double> ave_hausdorff;
	double timings = -1;

	std::vector<double> ELens;
	std::vector<double> Vols;
	std::vector<double> H_mineLens;
	std::vector<double> H_maxeLens;
};
struct Mesh
{
	short type;//Mesh_type
	Which_plane plane = Which_plane::N;
	MatrixXd V;
	vector<Hybrid_V> Vs;
	vector<Hybrid_E> Es;
	vector<Hybrid_F> Fs;
	vector<Hybrid> Hs;
};
struct Mesh_Feature
{//ground-truth feature
	Mesh tri;
	vector<int> V_map, V_map_reverse;

	vector<Vector3d> Tcenters;
	double ave_length;
	double angle_threshold = 0;
	bool orphan_curve = true;
	bool orphan_curve_single = true;
	//read_from_file 
	bool read_from_file = false;
	vector<vector<uint32_t>> IN_v_pairs;
	vector<uint32_t> IN_corners;
	vector<bool> E_feature_flag;

	vector<uint32_t> corners;
	vector<vector<uint32_t>> corner_curves;


	vector<vector<uint32_t>> curve_vs;
	vector<vector<uint32_t>> curve_es;
	vector<bool> broken_curves;
	vector<bool> circles;//true -- circle, false -- not circle

	MatrixXd normal_V, normal_Tri;

	vector<int> v_types;//Regular --> V, edge -- curve id, corner -- V
};

struct Mapping_Info {
	std::vector<MatrixXd> Vs;
	std::vector<MatrixXi> Fs;
	std::vector<MatrixXd> mapped_Vs;
	std::vector<MatrixXi> mapped_Fs;

	std::vector<MatrixXd> Draw_Vs;
	std::vector<MatrixXi> Draw_Es;
	MatrixXd all_Vs;
	MatrixXd mapped_all_Vs;
	//gluing clusters
	vector<vector<RowVector3d>> Vgroups;


	MatrixXd negative_Vs;
	MatrixXd target_negative_Vs;

	MatrixXd LocalRegion_Vs;

	std::vector<MatrixXd> V_layers_coords;
};

struct Feature_Graph
{
	Mesh_Feature mf;
	vector<Feature_Corner> Cs;
	vector<Feature_Line> Ls;
	vector<Feature_Region> Rs;
	vector<int> F_tag;
};

struct Feature_Constraints
{
	vector<Feature_V_Type> V_types;//interior, corner, line, regular
	vector<int> V_ids;//Regular --> V/T, edge -- curve id, corner -- V
	vector<bool> RV_type;//true -- T, false -- V

	//corner constraints
	Eigen::VectorXi ids_C;
	Eigen::MatrixXd C;
	double lamda_C = 0;
	//tagent plane constraints
	Eigen::VectorXi ids_T;
	Eigen::MatrixXd normal_T;
	Eigen::VectorXd dis_T;
	Eigen::MatrixXd V_T;
	double lamda_T = 0;
	//feature line constraints
	uint32_t num_a;
	Eigen::VectorXi ids_L;
	Eigen::MatrixXd Axa_L;
	Eigen::MatrixXd origin_L;
	double lamda_L = 0;
	//
	vector<vector<uint32_t>> curve_vs;
	vector<int> curveIds;

	Mapping_Info MI;
};
struct Tetralize_Set {
	vector<uint32_t> V_map, Reverse_V_map;
	MatrixXd V;
	MatrixXi T;
	MatrixXd UV;
	vector<MatrixXd> RT;
	VectorXi b;
	MatrixXd bc;
	Feature_Constraints fc;
	double lamda_b = 0;

	//project back
	bool projection;
	Eigen::VectorXi s;
	Eigen::MatrixXd sc;
	//global optimization
	bool global;
	
	double lamda_region = 1.0e+7;
	Eigen::VectorXi regionb;
	Eigen::MatrixXd regionbc;

	vector<vector<uint32_t>> Vgroups;
	//V: multi-copies
	bool glue;
	double lamda_glue = 1.e1;
	double lamda_projection = 1.e5;

	vector<uint32_t> V_ranges, O_Vranges;
	//know variable values
	bool known_value_post = false;
	int post_index = -1;
	MatrixXd post_Variables;

	SLIM_ENERGY energy_type = SLIM_ENERGY::CONFORMAL;
	vector<int32_t> mappedV, mappedV_reverse;
	bool record_Sequence = false;

	Mesh m;
};
struct Collapse_Info {
	vector<vector<uint32_t>> V_Groups;	
	VectorXi target_vs;
	MatrixXd target_coords;
	vector<uint32_t> hs;

	vector<uint32_t> fs_before, before_region;
	vector<uint32_t> fs_after, after_region;
	vector<uint32_t> fs_subdivided, subd_region;

	vector<uint32_t> Hsregion;
};
struct arguments{
	string choice;
	double Hausdorff_ratio_t = 0.005;
	double edge_length_ratio = 15;
	double edge_length = 0;
	int Iteration_Base = 3;
	bool Hard_Feature = true;
	string input = "", output = "";
	bool octree = true;
	bool whole_domain = false;
	int num_cell = 30;
	int scaffold_layer = 3;
//
	bool pca_oobb = true;
	int scaffold_type = 1;//1 box scaffold free boundary; 2 layered scaffold free boundary; 3 box scaffold fixed boundary; 4 layered scaffold fixed boundary.
	double weight_opt = 1;
	double feature_weight = 0.05;
};
struct Treestr {
	Mesh mesh;
	std::vector<int> v_map;
	std::vector<int> v_map_reverse;
	igl::AABB<Eigen::MatrixXd, 3> tree;
	Eigen::MatrixXd TriV;
	Eigen::MatrixXi TriF;
	Eigen::MatrixXd TriFN, TriVN, TriEN;
	Eigen::MatrixXi TriE;
	Eigen::VectorXi TriEMAP;
};
struct Mesh_Domain{
	bool known_value_post = false;
	int post_index = -1;
	Mesh mesh_entire;
	Mesh mesh_subA;
	Mesh mesh_subB;//scaffold
	vector<bool> H_flag;
	vector<int32_t> V_map, V_map_reverse, F_map, F_map_reverse, H_map,H_map_reverse;
	//features
	Treestr quad_tree;
	Mesh_Feature mf_quad;
	Feature_Graph Qfg, Qfg_sur;
	vector<int32_t> TV_map, TV_map_reverse, TF_map, TF_map_reverse;
	vector<int32_t> QV_map, QV_map_reverse, QF_map, QF_map_reverse;
	//graph matching
	vector<vector<uint32_t>> graph_matches;
	//bijective mapping
	vector<Treestr> tri_Trees;


	void clear_Qfg(){
		Qfg.F_tag.clear(); 
		Qfg.mf = Mesh_Feature();
		Qfg.Cs.clear(); 
		Qfg.Ls.clear(); 
		Qfg.Rs.clear();
	}
	void clear_mf_quad(){
		mf_quad.corners.clear(); 
		mf_quad.corner_curves.clear();
		mf_quad.curve_vs.clear();
		mf_quad.curve_es.clear(); 		
		mf_quad.circles.clear();
	}

};	

extern arguments args;
extern Mesh_Feature mf;
extern GEO::Mesh M_i;	
extern Feature_Graph fg;
extern Feature_Graph Q_final_fg;
extern vector<vector<uint32_t>> GRAPH_MATCHES;

extern char path_out[300];
//parallel
extern int32_t GRAIN_SIZE;
//meshes

extern double diagonal_len;