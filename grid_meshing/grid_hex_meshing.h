#pragma once
#include "../global_functions.h"
#include "../io.h"
//#include "octree.h"
//#include "common.h"
#include "voxelization.h"
#include <algorithm>
//for signed distance
#include <igl/edge_lengths.h>
#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/signed_distance.h>
#include <Eigen/Sparse>
#include "igl/bounding_box_diagonal.h"
#include "../metro_hausdorff.h"
class grid_hex_meshing_bijective
{
	enum NV_type {
		IV = 0,//inner vertex
		EV,//edge vertex
		DEV,//edge vertex shared by two base cells
		NRV,//new regular edge vertex
		NV//not new vertex
	};
	enum class V_Pos { 
		Boundary_full = 0,
		Boundary_single,
		Interior_full,
		Not_involve
	};
	struct Local_V {
		int id = -1;
		int hvid = -1;
		int id_global = -1;
		V_Pos pos;
		std::vector<uint32_t> nfs;
		std::vector<uint32_t> nhs;
		bool manifold = true;
		bool boundary = false;
		bool top = true;
	};

public:
	grid_hex_meshing_bijective();
	bool initialization(string path);
	void pipeline(string path);
	bool pure_hex_meshing();

	void initialize_mesh();
	//mesh generation
		void voxel_meshing(GEO::Mesh &mi, Mesh &hmi);
		void dual_octree_meshing(GEO::Mesh &mi, Mesh &mo);
			bool octree_mesh(GEO::Mesh &mi, Mesh &mo, OctreeGrid &octree, Eigen::Vector3i &grid_size);
			void conforming_mesh(Mesh &mo, Mesh &hybrid, OctreeGrid &octree, Eigen::Vector3i &grid_size);
			void dual_conforming_mesh(Mesh &mo, Mesh &hybrid, Mesh &hybrid_standard, vector<Element_Type> &h_type);
			void connectivity_modification(Mesh &hybrid_standard, vector<Element_Type> &h_type, vector<NV_type> &nv_tag, Mesh &pure_hex_mesh);
	void clean_hex_mesh(Mesh &tmi, Mesh_Domain &md);
		void tagging_uneven_element(const Mesh &mi, vector<bool> &H_flag);
		void clean_non_manifold_ve(Mesh &mi, Mesh &hmi_local, vector<int> &V_map, vector<int> &V_map_reverse,
			vector<int> &H_map, vector<int> &H_map_reverse, VectorXd &signed_dis, vector<bool> &H_flag);
		void drop_small_pieces(Mesh_Domain &md);
		void scaffold(Mesh_Domain &md);
	//boundary projection & feature capture
	bool surface_mapping(Mesh &tmi, Mesh_Domain &md);
		void break_circles(Mesh_Feature &mf_temp, vector<bool> &Corner_tag, vector<vector<uint32_t>> &circle2curve_map);
		bool node_mapping(Mesh_Feature &mf_temp, Feature_Graph &Tfg, Mesh_Domain &md);
		bool curve_mapping(Feature_Graph &Tfg, Mesh_Domain &md);
		bool reunion_circles(Mesh_Feature &mf_temp, vector<bool> &Corner_tag, vector<vector<uint32_t>> &circle2curve_map, Mesh_Domain &md);
		bool patch_mapping(Mesh_Domain &md);
		void patch_trees(Mesh_Domain &md);
	//deformation
	bool deformation(Mesh_Domain &md);
	bool smoothing(Mesh_Domain &md);
	//global padding
	bool mesh_padding(Mesh_Domain &md);
		void padding_connectivity(Mesh_Domain &md);
		bool padding_local_untangle_geometry(Mesh_Domain &md);
	//surface deformation
		void projection_smooth(const Mesh &hmi, Feature_Constraints &fc);
	//local padding
	bool local_padding(Mesh_Domain &md);
		void local_padding_connectivity_robust(Mesh_Domain &md);
		void padding_arbitrary_hex_mesh_step1(Mesh_Domain &md, const vector<uint32_t> &region,vector<Local_V> &local_vs,
vector<bool> &H_tag, vector<bool> &F_tag, vector<int> &local_fs, vector<int> &local_hs);
		void padding_arbitrary_hex_mesh_step2(Mesh_Domain &md, const vector<uint32_t> &region,vector<Local_V> &local_vs,
vector<bool> &H_tag, vector<int> &local_fs);
		//bool local_padding_local_untangle_geometry(Mesh_Domain &md);
		
	void feature_qmesh_udpate(Mesh_Domain &md);
	//feature alignment
	bool feature_alignment(Mesh_Domain &md);

	void dirty_region_identification(Mesh_Domain &md, std::vector<int> &Dirty_Vs);
	void dirty_local_feature_update(Mesh_Domain &md, std::vector<int> &Dirty_Vs);
	void dirty_graph_projection(Mesh_Domain &md, Feature_Constraints &fc, std::vector<int> &Dirty_Vs);

	void tet_assembling(const Mesh &m, const vector<Hybrid> &Hs_copies, const vector<bool> &H_flag, const vector<bool> &Huntangle_flag, bool local);
	void local_region(const Mesh &m, const vector<bool> &HQ_flag, vector<bool> &H_flag, int &hn, VectorXi &b, MatrixXd &bc);
	bool stop_criterior_satisfied(Mesh_Domain &md, const int iter_after_untangle, const Mesh &tm0, Mesh &tm1, bool &max_dis_satisified, bool & ave_dis_satisfied);

	void geomesh2mesh(GEO::Mesh &gm, Mesh &m);
	void build_aabb_tree(Mesh &tmi, Treestr &a_tree, bool is_tri = false);

	bool hausdorff_ratio_check(Mesh &m0, Mesh &m1);
bool hausdorff_ratio_check(Mesh &m0, Mesh &m1, double & hausdorff_dis_threshold);
	~grid_hex_meshing_bijective();
public:
#define	untangling_Iter_DEFAULT 30;

	int num_voxels = 1048576;
	bool simplification_bool = false;
	double HR = 0.005;
	double weight_opt = 1;
	int STOP_EXTENT_MIN;
	int STOP_EXTENT_MAX;
	vector<int> tb_subdivided_cells;
	vector<int> hex2Octree_map;
	int local_refinement_ringN = 2;
	int region_Size_untangle = 2;
	double Ratio_grow = 0.1;
	int start_ave_hausdorff_count_Iter = 5;
	int improve_Quality_after_Untangle_Iter_MAX = 10;
	std::vector<double>ave_Hausdorff_dises;
	std::vector<double>ratio_max_Hausdorff;
	std::vector<double>ratio_ave_Hausdorff;
	double STOP_AVE_HAUSDORFF_THRESHOLD = 0.01;
	bool re_Octree_Meshing = false;

	double hausdorff_ratio_threshould = 0.01;
	double hausdorff_ratio = 0;
	OctreeGrid octree;
	h_io io;
	Mesh_Quality mq;
	Mesh_Domain MD;
	double voxel_size = 1;
	bool graded = false;
	bool paired = true;
	//cleaning
	vector<NV_type> nv_TAG;
	//build AABB tree for a triangle mesh
	Treestr tri_tree;
	//optimization
	Feature_Constraints fc;
	Tetralize_Set ts;
	double Tiny_angle_threshold = 30.0 / 180.0 * PAI;
	std::vector<int> Dirty_Vertices;
	std::vector<int> Dirty_Quads;
	//local padding
	vector<vector<uint32_t>> Q_regions;
	vector<vector<uint32_t>> V_layers, H_Tlayers, H_Nlayers;
	vector<int32_t> V2V_Map_Reverse_final;//
	//pocket
	vector<int> Pocket_vs;
	//target volume of reference shape
	double VOLUME_input;
	vector<double> Target_Vols;
	//for debugging
	int file_id = 0;
	double max_HR = 0; 

};

