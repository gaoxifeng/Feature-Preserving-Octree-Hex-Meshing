#include "grid_hex_meshing.h"
#include "igl/procrustes.h"
#include "../optimization.h"
#include <unordered_map>

grid_hex_meshing_bijective::grid_hex_meshing_bijective(){
	voxel_size = 5;
	graded = true;
	paired = true;
	HR = 0.005;
}
bool grid_hex_meshing_bijective::initialization(string path) {
	GEO::initialize();

	GEO::mesh_load(path + ".obj", M_i);
	geomesh2mesh(M_i, mf.tri);

	if (!mf.tri.Vs.size() || !mf.tri.Fs.size())
	{
		std::cout << "empty file" << std::endl;
		return false;
	}
	
	mf.angle_threshold = 0;
	mf.orphan_curve = true;
	mf.orphan_curve_single = true;

	if (io.read_feature_Graph_FGRAPH(mf, path + ".fgraph")) {
		mf.read_from_file = true;
	}
	else {
		std::cout << "no feature file, detect based on angle" << std::endl;
		mf.read_from_file = false;
	}


	triangle_mesh_feature(mf);
	build_feature_graph(mf, fg);

	num_voxels = 1048576;
	
	MD.mesh_entire.type = Mesh_type::Hex;
	MD.mesh_subA.type = Mesh_type::Hex;
	MD.mesh_subB.type = Mesh_type::Hex;

	if(args.Hausdorff_ratio_t != 0)
		HR = args.Hausdorff_ratio_t;

	STOP_EXTENT_MIN = STOP_EXTENT_MAX = 15;
	if(args.edge_length_ratio != 0)
		STOP_EXTENT_MIN = STOP_EXTENT_MAX = args.edge_length_ratio;
	return true;
}
void grid_hex_meshing_bijective::pipeline(string path) {
	if (HR <= 0) {
		cout << "please provide a reasonable hausdorff distance ratio > 0" << endl;
		return;
	}

	area_volume(mf.tri, VOLUME_input);
	build_aabb_tree(mf.tri, tri_tree);

	octree = OctreeGrid();
	tb_subdivided_cells.clear();
	hex2Octree_map.clear();

	hausdorff_ratio_threshould = HR;
	double hausdorff_dis_threshold = 0;
	int iterN = 0;
	re_Octree_Meshing = true;

	int griding_resolution = args.num_cell;
	int griding_resolution_step = 10;

	auto HAUSDORFF_REFINEMENT = [&](double threshold)->bool{
		if (!hausdorff_ratio_check(mf.tri, MD.mesh_subA, threshold)) {

			Mesh htri; htri.type = Mesh_type::Tri;
			vector<int32_t> v_map, v_map_reverse, f_map, f_map_reverse;
			extract_surface_conforming_mesh(MD.mesh_subA, htri, v_map, v_map_reverse, f_map, f_map_reverse);
			vector<int> ref_vs;
			hausdorff_dis(mf.tri, htri, ref_vs, threshold);
			for (auto &vid : ref_vs) {
				vid = MD.V_map_reverse[v_map_reverse[vid]];
				tb_subdivided_cells.push_back(vid);
			}

			STOP_EXTENT_MAX--;
			cout << "hausdorff distance too large, refine once!" << endl;
			return false;
		}
		return true;
	};

	while (true) {
		
		vector<uint32_t> tb_subdivided_cells_;
		for (uint32_t j = 0; j < local_refinement_ringN; j++) {
			vector<int> nvs_;
			for (auto nvid : tb_subdivided_cells)
				nvs_.insert(nvs_.end(), MD.mesh_entire.Vs[nvid].neighbor_vs.begin(), MD.mesh_entire.Vs[nvid].neighbor_vs.end());
			sort(nvs_.begin(), nvs_.end()); nvs_.erase(unique(nvs_.begin(), nvs_.end()), nvs_.end());
			tb_subdivided_cells = nvs_;
		}
		std::sort(tb_subdivided_cells.begin(), tb_subdivided_cells.end());
		tb_subdivided_cells.erase(std::unique(tb_subdivided_cells.begin(), tb_subdivided_cells.end()), tb_subdivided_cells.end());
		cout << "size of cells: " << tb_subdivided_cells.size() << endl;
		iterN++;

		MD = Mesh_Domain();
		
		cout << "min max: " << STOP_EXTENT_MIN << " " << STOP_EXTENT_MAX << endl;

		if (STOP_EXTENT_MIN < 1) break;
		if (STOP_EXTENT_MAX < 1)STOP_EXTENT_MAX = 1;

		if (STOP_EXTENT_MAX < STOP_EXTENT_MIN - 5) {
			cout << "too many refinements!" << endl;
			STOP_EXTENT_MIN--;
			STOP_EXTENT_MAX = STOP_EXTENT_MIN;
			cout << "re-start!" << endl;
			re_Octree_Meshing = true;

			octree = OctreeGrid();
			tb_subdivided_cells.clear();
			hex2Octree_map.clear();

			continue;
		}

		if(!args.octree){
			args.num_cell +=griding_resolution_step;
			tb_subdivided_cells.clear();
		}

		if (!pure_hex_meshing()) {
			STOP_EXTENT_MIN--;
			STOP_EXTENT_MAX = STOP_EXTENT_MIN;
			cout << "no hex created, re-start!" << endl;
			re_Octree_Meshing = true;

			octree = OctreeGrid();
			tb_subdivided_cells.clear();
			hex2Octree_map.clear();

			continue;
		}

		cout << "Feature Mapping" << endl;
		if (!surface_mapping(mf.tri, MD)) {
			STOP_EXTENT_MAX--;
			cout << "refine once!" << endl;
			continue;
		}

		cout << "deformation after feature mapping" << endl;
		if (args.octree && !deformation(MD) && !mf.curve_vs.size()) {
			HAUSDORFF_REFINEMENT(HR * 3); continue;
		}
		cout << "global Padding" << endl;
		if(!mesh_padding(MD)) {
			STOP_EXTENT_MAX--;
			cout << "flipped elements refine once!" << endl;
			continue;
		}
		cout << "deformation after global padding" << endl;
		smoothing(MD);
		cout << "local Padding" << endl;
		if (mf.curve_vs.size() || mf.corners.size()) {
			if (!local_padding(MD)) {
				STOP_EXTENT_MAX--;
				cout << "flipped elements refine once!" << endl;
				continue;
			}
			cout << "deformation after local padding" << endl;
			deformation(MD);
		}
		
		cout << "Feature Alignment" << endl;
		if (!feature_alignment(MD)) {
			STOP_EXTENT_MAX--;
			continue;
		}
		if(!HAUSDORFF_REFINEMENT(HR)) {
			continue; 
		}
		else 
			break;
	}

	mq.V_num = MD.mesh_subA.Vs.size();
	mq.H_num = MD.mesh_subA.Hs.size();
	for (int id = 0;id<MD.H_flag.size();id++)
		MD.H_flag[id] = !MD.H_flag[id];
	re_indexing_connectivity(MD.mesh_entire, MD.H_flag, MD.mesh_subB, MD.V_map, MD.V_map_reverse, MD.H_map, MD.H_map_reverse);
}

bool grid_hex_meshing_bijective::pure_hex_meshing() {
	
	MD.mesh_entire.type = Mesh_type::Hex;
	if(!args.octree){
		num_voxels= args.num_cell;
		voxel_meshing(M_i, MD.mesh_entire);
	}
	else
		dual_octree_meshing(M_i, MD.mesh_entire);
	if (!MD.mesh_entire.Hs.size()) { cout << "no hex-mesh generated, exit!"; return false;}

	clean_hex_mesh(mf.tri, MD);
	if (!MD.mesh_subA.Hs.size()) { cout << "no hex-mesh inside the object, exit!"; return false; }
	
	return true;
}
//mesh generation
void grid_hex_meshing_bijective::voxel_meshing(GEO::Mesh &mi, Mesh &hmi){
	// Initialize voxel grid and AABB tree
	GEO::vec3 min_corner, max_corner;
	GEO::get_bbox(mi, &min_corner[0], &max_corner[0]);
	GEO::vec3 extent = max_corner - min_corner;
	if (num_voxels > 0) {
		// Force number of voxels along longest axis
		double max_extent = std::max(extent[0], std::max(extent[1], extent[2]));
		voxel_size = max_extent / num_voxels;
	}
	const auto& len =voxel_size;
	Vector3i dim(std::ceil(extent[0]/len),std::ceil(extent[1]/len),std::ceil(extent[2]/len));
	Vector3f grid_length(extent[0]/dim[0],extent[1]/dim[1],extent[2]/dim[2]);

	hmi.Vs.resize(dim[0]*dim[1]*dim[2]);
	int vn =0;
	for (int i = 0; i <dim[0]; i++)
	{
		double x = min_corner[0] + grid_length[0] * i;
		for (int j = 0; j <dim[1]; j++)
		{
			double y = min_corner[1] + grid_length[1] * j;

			for (int k = 0; k <dim[2]; k++)
			{
				double z = min_corner[2] + grid_length[2] * k;

				vn = i * dim[1] * dim[2] + j * dim[2] + k;

				Hybrid_V v;
				v.id = vn;
				v.boundary = false;
				v.v.push_back(x);
				v.v.push_back(y);
				v.v.push_back(z);
				hmi.Vs[vn] = v;
			}
		}
	}
	hmi.V.resize(3, hmi.Vs.size());
	for(auto &v: hmi.Vs){
		hmi.V(0, v.id) = v.v[0];
		hmi.V(1, v.id) = v.v[1];
		hmi.V(2, v.id) = v.v[2];
	}

	Hybrid h;	
	h.vs.resize(8);
	std::vector<int> coord(3);
	for(int i = 0; i < hmi.Vs.size(); i++)
	{
		int a = i/(dim[1]*dim[2]);
		int b = (i%(dim[1]*dim[2]))/dim[2];
		int c = i%dim[2];

		h.id = hmi.Hs.size();
		int a_, b_, c_, id;
		h.vs[0] = i;
		bool not_found= false;
		for(int j = 1; j< 8; j++)
		{
			switch(j)
			{
				case 1: a_ = a + 1, b_ = b    , c_ = c    ; break;
				case 2: a_ = a + 1, b_ = b + 1, c_ = c    ; break;
				case 3: a_ = a    , b_ = b + 1, c_ = c    ; break;
				case 4: a_ = a    , b_ = b    , c_ = c + 1; break;
				case 5: a_ = a + 1, b_ = b    , c_ = c + 1; break;
				case 6: a_ = a + 1, b_ = b + 1, c_ = c + 1; break;
				case 7: a_ = a    , b_ = b + 1, c_ = c + 1; break;
			}
			if(a_ < 0 || b_ < 0 || c_ < 0) {not_found = true; break;}
			if(a_ >= dim[0] || b_ >= dim[1] || c_ >= dim[2]) {not_found = true; break;}
		
			h.vs[j] = a_ * dim[1] * dim[2] + b_ * dim[2] + c_;
		}
		if(!not_found)
			hmi.Hs.push_back(h);
	}

	build_connectivity(hmi);
}
void grid_hex_meshing_bijective::dual_octree_meshing(GEO::Mesh &mi, Mesh &mo) {
	
	Eigen::Vector3i grid_size;
	if(!octree_mesh(mi, mo, octree, grid_size))
		return;
	Mesh hybrid;
	conforming_mesh(mo, hybrid, octree, grid_size);

	Mesh hybrid_standard;
	vector<Element_Type> h_type(hybrid_standard.Hs.size());
	dual_conforming_mesh(mo, hybrid, hybrid_standard, h_type);

	if (!hybrid_standard.Hs.size()) 
	{ 
		cout << "no elements, exit!" << endl; 
		mo.Vs.clear(); mo.Es.clear(); mo.Fs.clear(); mo.Hs.clear();
		return; 
	}

	//hybrid-mesh ---> Pure hex-mesh
	//all the pyramid tips, and find all the quads that neighboring to pyramids,
	enum Vertex_Tag {
		Regular = 0,
		Pyramid_tip,
		Pyramid_bottom,
		Prism_tip
	};
	vector<Vertex_Tag> vertex_type(hybrid_standard.Vs.size(), Vertex_Tag::Regular);
	for (uint32_t i = 0; i < h_type.size(); i++)
		if (h_type[i] == Element_Type::Pyramid) {
			for (uint32_t j = 0; j < 4; j++) vertex_type[hybrid_standard.Hs[i].vs[j]] = Vertex_Tag::Pyramid_bottom;
			vertex_type[hybrid_standard.Hs[i].vs[4]] = Vertex_Tag::Pyramid_tip;
		}
	vector<bool> tip_quads(hybrid_standard.Fs.size(), false);
	for (auto &f : hybrid_standard.Fs) {
		bool notip = false;
		for (auto vid : f.vs)if (vertex_type[vid] != Vertex_Tag::Pyramid_tip)notip = true;
		if (!notip)tip_quads[f.id] = true;
	}
	//base-cell: four pyramids, four prisms, a hex
	struct Base_Cell {
		int hex_id;
		int quad_id;
		vector<int> pyramid_ids;
		vector<int> prism_ids;
		vector<vector<int>> base_bvs;//00, 01, 02; 10, 11, 12; 20, 21, 22; 30, 31, 32;
		vector<int> base_qvs;// 03, 13, 23, 33;
		vector<uint32_t> top_qvs;//0, 1, 2, 3;

		vector<vector<int>> split_mvs;//0e0, 1e0; 1e1, 2e1; 2e2, 3e2; 3e3, 0e3;
		vector<int> split_ivs;//0q,1q,2q,3q.
		Vector3d direction;//from base to top		
	};
	vector<Base_Cell> bcells(hybrid_standard.Fs.size());
	
	for (uint32_t i = 0; i < tip_quads.size(); i++) {
		if (!tip_quads[i])continue;

		Base_Cell &bc = bcells[i];
		bc.hex_id = -1;
		bc.quad_id = i;
		bc.top_qvs = hybrid_standard.Fs[i].vs;
		bc.base_bvs.resize(4);

		for (auto nhid : hybrid_standard.Fs[i].neighbor_hs) {
			vector<int> nBV;
			for (auto hvid : hybrid_standard.Hs[nhid].vs)if (vertex_type[hvid] == Vertex_Tag::Pyramid_bottom) nBV.push_back(hvid);
			if (nBV.size() == 4) {
				int nP = 0;
				for (auto fid : hybrid_standard.Hs[nhid].fs) for (auto nnhid : hybrid_standard.Fs[fid].neighbor_hs)
					if (nnhid != nhid && h_type[nnhid] == Element_Type::Prism)nP++;
				if (nP != 4)continue;

				bc.hex_id = nhid;
				for (auto vid : bc.top_qvs) for (auto vid1 : nBV) {
					vector<uint32_t> es0 = hybrid_standard.Vs[vid].neighbor_es, es1 = hybrid_standard.Vs[vid1].neighbor_es, sharedes;
					sort(es0.begin(), es0.end()); sort(es1.begin(), es1.end());
					set_intersection(es0.begin(), es0.end(), es1.begin(), es1.end(), back_inserter(sharedes));
					if (sharedes.size()) { bc.base_qvs.push_back(vid1); break; }
				}
			}
		}
		if (bc.hex_id == -1) {
			tip_quads[i] = false; continue;
		}
		//prism
		for (auto eid : hybrid_standard.Fs[i].es) {
			vector<uint32_t> fs_local = hybrid_standard.Es[eid].neighbor_fs, fs_hex = hybrid_standard.Hs[bc.hex_id].fs, fs_shared;
			sort(fs_local.begin(), fs_local.end()); sort(fs_hex.begin(), fs_hex.end());
			set_intersection(fs_local.begin(), fs_local.end(), fs_hex.begin(), fs_hex.end(), back_inserter(fs_shared));
			int side_fid = -1;
			for (auto fshared : fs_shared)if (fshared != i) side_fid = fshared;
			for (auto nhid : hybrid_standard.Fs[side_fid].neighbor_hs) if (nhid != bc.hex_id) bc.prism_ids.push_back(nhid);
		}
		//pyramid
		for (uint32_t j = 0; j < bc.prism_ids.size(); j++) {
			int prism = bc.prism_ids[j];
			vector<int> ps;
			for (auto pfid : hybrid_standard.Hs[prism].fs)for (auto nhid : hybrid_standard.Fs[pfid].neighbor_hs)if (h_type[nhid] == Element_Type::Pyramid) ps.push_back(nhid);
			for (auto p : ps)if (hybrid_standard.Hs[p].vs[4] == bc.top_qvs[j]) { bc.pyramid_ids.push_back(p); break; }
		}
		//Vertices
		vector<vector<int>> othervs(4);
		for (uint32_t j = 0; j < 4; j++) {
			int which_index = -1, pid = bc.pyramid_ids[j];
			for (uint32_t k = 0; k < 4; k++) if (bc.base_qvs[j] == hybrid_standard.Hs[pid].vs[k]) which_index = k;
			bc.base_bvs[j].push_back(hybrid_standard.Hs[pid].vs[(which_index + 2) % 4]);
			othervs[j].push_back(hybrid_standard.Hs[pid].vs[(which_index + 1) % 4]);
			othervs[j].push_back(hybrid_standard.Hs[pid].vs[(which_index + 3) % 4]);
		}
		for (uint32_t j = 0; j < 4; j++) {
			int pid = bc.prism_ids[j];
			for (auto vid : hybrid_standard.Hs[pid].vs) if (vid == othervs[j][0] || vid == othervs[j][1]) bc.base_bvs[j].push_back(vid);
			for (auto vid : hybrid_standard.Hs[pid].vs) if (vid == othervs[(j + 1) % 4][0] || vid == othervs[(j + 1) % 4][1]) bc.base_bvs[j].push_back(vid);
		}
		Vector3d bcenter, tcenter;
		bcenter.setZero(); tcenter.setZero();
		for (uint32_t j = 0; j < 4; j++) {
			bcenter += hybrid_standard.V.col(bc.base_qvs[j]);
			tcenter += hybrid_standard.V.col(bc.top_qvs[j]);
		}
		bc.direction = tcenter - bcenter;

		bc.split_ivs.resize(4);
		bc.split_mvs.resize(4);
		for (auto &am : bc.split_mvs)am.resize(2);
	}
	//classify into separate groups,
	vector<vector<uint32_t>> f_layers;
	vector<bool> tq_tag(tip_quads.size(), true);
	while (true) {

		int f_start = -1;

		for (uint32_t i = 0; i<tq_tag.size(); i++)
			if (tq_tag[i] && tip_quads[i]) { f_start = i; tq_tag[i] = false; break; }
		if (f_start == -1)break;

		vector<uint32_t> a_layer, layer_temp, layer_temp2;
		a_layer.push_back(f_start); layer_temp.push_back(f_start);
		while (layer_temp.size()) {
			for (auto fid : layer_temp)for (auto vid : hybrid_standard.Fs[fid].vs)
				for (auto nfid : hybrid_standard.Vs[vid].neighbor_fs) {
					if (tq_tag[nfid] && tip_quads[nfid]) {
						tq_tag[nfid] = false; layer_temp2.push_back(nfid);
					}
				}
			if (layer_temp2.size()) {
				a_layer.insert(a_layer.end(), layer_temp2.begin(), layer_temp2.end());
				layer_temp = layer_temp2;
				layer_temp2.clear();
			}
			else break;
		}
		f_layers.push_back(a_layer);
	}

	Mesh pure_hex_mesh; pure_hex_mesh.type = Mesh_type::Hex;
	nv_TAG.clear();
	connectivity_modification(hybrid_standard, h_type, nv_TAG, pure_hex_mesh);
	mo = pure_hex_mesh;
	build_connectivity(mo);
}
bool grid_hex_meshing_bijective::octree_mesh(GEO::Mesh &mi, Mesh &mo, OctreeGrid &octree,
	Eigen::Vector3i &grid_size)
{
	GEO::vec3 min_corner, max_corner;
	GEO::get_bbox(mi, &min_corner[0], &max_corner[0]);
	GEO::vec3 mesh_center = (min_corner + max_corner) / 2;
	GEO::vec3 extent = (max_corner - min_corner);
	if (num_voxels > 0) {
		double max_extent = std::max(extent[0], std::max(extent[1], extent[2]));
		voxel_size = max_extent / num_voxels;
	}

	if (args.scaffold_type == 2 || args.scaffold_type == 3)
	{
		int scaffold_extent = pow(2, STOP_EXTENT_MIN);
		min_corner -= args.scaffold_layer * 2 * scaffold_extent * voxel_size * GEO::vec3(1, 1, 1);
		max_corner += args.scaffold_layer * 2 * scaffold_extent * voxel_size* GEO::vec3(1, 1, 1);
		extent = (max_corner - min_corner);
		mesh_center = (min_corner + max_corner) / 2;
	}

	GEO::MeshFacetsAABB aabb_tree(mi);

	int	padding = 0;
	GEO::vec3 origin = min_corner - padding * voxel_size * GEO::vec3(1, 1, 1);
	grid_size = Eigen::Vector3i(
		next_pow2(std::ceil(extent[0] / voxel_size) + 2 * padding),
		next_pow2(std::ceil(extent[1] / voxel_size) + 2 * padding),
		next_pow2(std::ceil(extent[2] / voxel_size) + 2 * padding)
	);
	GEO::vec3 origin_max = origin + GEO::vec3(voxel_size * grid_size[0], voxel_size * grid_size[1], voxel_size * grid_size[2]);

	GEO::vec3 origin_center = (origin_max + origin) * 0.5;
	Eigen::Vector3d mesh_transform(mesh_center[0]-origin_center[0], mesh_center[1] - origin_center[1], mesh_center[2] - origin_center[2]);

	int stop_extent = pow(2, STOP_EXTENT_MAX);
	if (!octree.numNodes() || re_Octree_Meshing) {
		octree.OctreeGrid_initialize(grid_size);
		re_Octree_Meshing = false;
		stop_extent = pow(2, STOP_EXTENT_MIN);
	}

	auto should_subdivide = [&](int x, int y, int z, int extent) {
		if (extent <= stop_extent) {
			return false; 
		}
		GEO::Box box;
		box.xyz_min[0] = mesh_transform[0] + origin[0] + voxel_size * x;
		box.xyz_min[1] = mesh_transform[1] + origin[1] + voxel_size * y;
		box.xyz_min[2] = mesh_transform[2] + origin[2] + voxel_size * z;
		box.xyz_max[0] = box.xyz_min[0] + voxel_size * extent;
		box.xyz_max[1] = box.xyz_min[1] + voxel_size * extent;
		box.xyz_max[2] = box.xyz_min[2] + voxel_size * extent;
		bool has_triangles = false;
		auto action = [&has_triangles](int id) { has_triangles = true; };
		aabb_tree.compute_bbox_facet_bbox_intersections(box, action);
		return has_triangles;
	};
	if (tb_subdivided_cells.size()) {
		for(auto &tbv:tb_subdivided_cells) tbv = hex2Octree_map[tbv];
		octree.subdivide(should_subdivide, tb_subdivided_cells, graded, paired);
		tb_subdivided_cells.clear();
	}
	else
		octree.subdivide(should_subdivide, graded, paired);
	hex2Octree_map.clear();

	// Octree-mesh
	Eigen::Vector3d o(origin[0], origin[1], origin[2]);
	Eigen::Vector3d s(voxel_size, voxel_size, voxel_size);

	mo.Vs.clear(); mo.Vs.resize(octree.numNodes());
	for (int idx = 0; idx < octree.numNodes(); ++idx) {
		Eigen::Vector3d pos = mesh_transform + o + octree.nodePos(idx).cast<double>().cwiseProduct(s);
		Hybrid_V v; v.id = idx;
		v.v.push_back(pos[0]);
		v.v.push_back(pos[1]);
		v.v.push_back(pos[2]);
		mo.Vs[idx] = v;
	}
	mo.V.resize(3, mo.Vs.size());
	for (uint32_t i = 0; i < mo.Vs.size(); i++) {
		mo.V(0, i) = mo.Vs[i].v[0];
		mo.V(1, i) = mo.Vs[i].v[1];
		mo.V(2, i) = mo.Vs[i].v[2];
	}
	// Count num of leaf cells
	int numLeaves = 0;
	for (int c = 0; c < octree.numCells(); ++c) if (octree.cellIsLeaf(c)) { ++numLeaves; }
	mo.Hs.clear(); mo.Hs.resize(numLeaves);
	for (int q = 0, c = 0; q < octree.numCells(); ++q) {
		if (!octree.cellIsLeaf(q)) continue;
		
		hex2Octree_map.push_back(q);

		Hybrid h;
		h.id = c;
		h.vs.resize(8);
		for (GEO::index_t lv = 0; lv < 8; ++lv) {
			h.vs[lv] = octree.cellCornerId(q, lv);
		}
		mo.Hs[c++] = h;
	}
	if (!mo.Hs.size()) { cout << "No octants, exit!" << endl; return false;}
	build_connectivity(mo);

	return true;
}
void grid_hex_meshing_bijective::conforming_mesh(Mesh &mo, Mesh &hybrid, OctreeGrid &octree, Eigen::Vector3i &grid_size)
{
	//quad relationship
	vector<bool> v_boundary(mo.Vs.size(), false);
	for (uint32_t i = 0; i < octree.numNodes(); i++) {
		Vector3i pos = octree.nodePos(i);
		if ((pos[0] == 0 || pos[0] == grid_size[0]) || (pos[1] == 0 || pos[1] == grid_size[1])
			|| (pos[2] == 0 || pos[2] == grid_size[2]))
			v_boundary[i] = true;
	}
	vector<vector<int>> f_relations(mo.Fs.size());
	vector<vector<int>> f_corvs(mo.Fs.size());
	for (uint32_t i = 0; i < mo.Vs.size(); i++) {
		if (mo.Vs[i].boundary && !v_boundary[i]) {
			//has complete 6 neighbors
			Node &n = octree.m_Nodes[i];
			if (find(n.neighNodeId.begin(), n.neighNodeId.end(), -1) != n.neighNodeId.end()) {//T-node
				int pos0 = find(n.neighNodeId.begin(), n.neighNodeId.end(), -1) - n.neighNodeId.begin();
				int pos1 = pos0 + 1; if (pos0 % 2 == 1)pos1 = pos0 - 1;
				vector<int> vs;//4 vs
				for (uint32_t j = 0; j < n.neighNodeId.size(); j++)if (j == pos0 || j == pos1)continue; else vs.push_back(n.neighNodeId[j]);
				if (find(vs.begin(), vs.end(), -1) != vs.end()) continue;//notT-node  but at two sides
																		 //4 faces
				sort(vs.begin(), vs.end());
				vector<int> fs4, vs_total;
				for (auto fid : mo.Vs[i].neighbor_fs) {
					vector<uint32_t> vs0 = mo.Fs[fid].vs;
					sort(vs0.begin(), vs0.end());
					vector<uint32_t> vs_shared;
					set_intersection(vs0.begin(), vs0.end(), vs.begin(), vs.end(), back_inserter(vs_shared));
					if (vs_shared.size() == 2) {
						fs4.push_back(fid);
						vs_total.insert(vs_total.end(), vs0.begin(), vs0.end());
					}
				}
				sort(vs_total.begin(), vs_total.end()); vs_total.erase(std::unique(vs_total.begin(), vs_total.end()), vs_total.end());
				//4 corners
				vs.push_back(i);
				vector<int> vs_corners;
				for (auto vid : vs_total) if (find(vs.begin(), vs.end(), vid) != vs.end())continue; else vs_corners.push_back(vid);
				//big face
				sort(vs_corners.begin(), vs_corners.end());
				if (vs_corners.size() == 0) { cout << "bug check" << endl; system("PAUSE"); }
				int ff = -1;
				for (auto fid : mo.Vs[vs_corners[0]].neighbor_fs) {
					vector<uint32_t> vs0 = mo.Fs[fid].vs;
					sort(vs0.begin(), vs0.end());
					vector<uint32_t> vs_shared;
					set_intersection(vs0.begin(), vs0.end(), vs_corners.begin(), vs_corners.end(), back_inserter(vs_shared));
					if (vs_shared.size() == 4) ff = fid;
				}
				if (ff != -1) {
					f_relations[ff] = fs4;
					f_corvs[ff] = vs;
				}
			}
		}
	}
	//dual mesh
	//cout << "start computing conforming hybrid mesh.." << endl;
	vector<int> e_midv(mo.Es.size(), -1);
	//replace big faces
	for (uint32_t i = 0; i < f_relations.size(); i++) {
		if (f_relations[i].size() == 4) {
			int hid = mo.Fs[i].neighbor_hs[0];
			vector<uint32_t> &fs = mo.Hs[hid].fs;
			fs.erase(std::remove(fs.begin(), fs.end(), i), fs.end());
			fs.insert(fs.end(), f_relations[i].begin(), f_relations[i].end());
			for (auto eid : mo.Fs[i].es) {
				//mid v for each edge
				int v0 = mo.Es[eid].vs[0], v1 = mo.Es[eid].vs[1];
				vector<uint32_t> nvs0 = mo.Vs[v0].neighbor_vs, nvs1 = mo.Vs[v1].neighbor_vs;
				sort(nvs0.begin(), nvs0.end()); sort(nvs1.begin(), nvs1.end());
				vector<uint32_t> vs_shared;
				set_intersection(nvs0.begin(), nvs0.end(), nvs1.begin(), nvs1.end(), back_inserter(vs_shared));
				if (vs_shared.size() == 1 && find(f_corvs[i].begin(), f_corvs[i].end(), vs_shared[0]) != f_corvs[i].end())
					e_midv[eid] = vs_shared[0];
			}
		}
	}
	//replace face edges
	for (uint32_t i = 0; i < e_midv.size(); i++) {
		if (e_midv[i] != -1) {
			int v0 = mo.Es[i].vs[0], v1 = mo.Es[i].vs[1];
			for (auto fid : mo.Es[i].neighbor_fs) {
				vector<uint32_t> vs;
				int size = mo.Fs[fid].vs.size();
				for (uint32_t j = 0; j < size; j++) {
					int v_cur = mo.Fs[fid].vs[j], v_next = mo.Fs[fid].vs[(j + 1) % size];
					vs.push_back(v_cur);
					if ((v_cur == v0 && v_next == v1) || (v_cur == v1 && v_next == v0))
						vs.push_back(e_midv[i]);
				}
				mo.Fs[fid].vs = vs;
			}
		}
	}
	//rebuild connectivity
	//cout << "build conforming hybrid mesh.." << endl;
	hybrid.type = Mesh_type::Hyb;
	hybrid.V = mo.V;
	for (uint32_t i = 0; i < mo.Vs.size(); i++) {
		Hybrid_V v;
		v.id = i; v.v = mo.Vs[i].v;
		hybrid.Vs.push_back(v);
	}
	vector<int> f_map(mo.Fs.size(), -1);
	for (uint32_t i = 0; i < mo.Fs.size(); i++) {
		if (f_relations[i].size() == 4)continue;
		Hybrid_F f;
		f.id = hybrid.Fs.size();
		f.vs = mo.Fs[i].vs;
		hybrid.Fs.push_back(f);
		f_map[i] = f.id;
	}
	for (uint32_t i = 0; i < mo.Hs.size(); i++) {
		Hybrid h;
		h.id = hybrid.Hs.size();
		for (auto fid : mo.Hs[i].fs) {
			h.fs.push_back(f_map[fid]);
			h.vs.insert(h.vs.end(), hybrid.Fs[f_map[fid]].vs.begin(), hybrid.Fs[f_map[fid]].vs.end());
		}
		sort(h.vs.begin(), h.vs.end()); h.vs.erase(unique(h.vs.begin(), h.vs.end()), h.vs.end());
		hybrid.Hs.push_back(h);

		for (auto vid : h.vs)hybrid.Vs[vid].neighbor_hs.push_back(h.id);
	}
	build_connectivity(hybrid);
}
void grid_hex_meshing_bijective::dual_conforming_mesh(Mesh &mo, Mesh &hybrid, Mesh &hybrid_standard, vector<Element_Type> &h_type)
{
	hybrid_standard.type = Mesh_type::Hyb;
	hybrid_standard.V.resize(3, hybrid.Hs.size());
	hybrid_standard.Vs.resize(hybrid.Hs.size());
	for (auto h : mo.Hs) {
		Hybrid_V v;
		v.id = h.id; v.v.resize(3, 0);
		for (auto vid : h.vs)for (uint32_t i = 0; i < 3; i++)v.v[i] += mo.Vs[vid].v[i];
		for (uint32_t i = 0; i < 3; i++) {
			v.v[i] /= 8;
			hybrid_standard.V(i, v.id) = v.v[i];
		}
		hybrid_standard.Vs[h.id] = v;
	}
	vector<int> e_tag(hybrid.Es.size(), -1);
	vector<bool> h_tag(hybrid.Hs.size(), false);
	for (uint32_t i = 0; i < hybrid.Es.size(); i++) {
		if (hybrid.Es[i].boundary)continue;
		Hybrid_F f;
		f.id = hybrid_standard.Fs.size();
		vector<uint32_t> &hs = hybrid.Es[i].neighbor_hs;
		uint32_t sh = hs[0];
		for (int j = 0; j < hs.size(); j++) {
			f.vs.push_back(sh); h_tag[sh] = true;
			for (auto fid : hybrid.Hs[sh].fs) {
				if (hybrid.Fs[fid].neighbor_hs.size() == 1)continue;

				uint32_t hid = hybrid.Fs[fid].neighbor_hs[0];
				if (hid == sh)  hid = hybrid.Fs[fid].neighbor_hs[1];
				if (!h_tag[hid] && find(hs.begin(), hs.end(), hid) != hs.end()) {
					sh = hid; break;
				}
			}
		}
		for (auto hid : f.vs)h_tag[hid] = false;

		hybrid_standard.Fs.push_back(f);
		e_tag[i] = f.id;
	}
	for (uint32_t i = 0; i < hybrid.Vs.size(); i++) {
		if (hybrid.Vs[i].boundary) continue;
		Hybrid h;
		h.id = hybrid_standard.Hs.size();
		for (auto eid : hybrid.Vs[i].neighbor_es)h.fs.push_back(e_tag[eid]);

		for (auto fid : h.fs) {
			h.vs.insert(h.vs.end(), hybrid_standard.Fs[fid].vs.begin(), hybrid_standard.Fs[fid].vs.end());
		}
		sort(h.vs.begin(), h.vs.end()); h.vs.erase(unique(h.vs.begin(), h.vs.end()), h.vs.end());
		hybrid_standard.Hs.push_back(h);

		for (auto vid : h.vs)hybrid_standard.Vs[vid].neighbor_hs.push_back(h.id);
	}
	build_connectivity(hybrid_standard);
	
	//completing elements
	h_type.resize(hybrid_standard.Hs.size());
	int tetrahedralN = 0, slabN = 0, pyramidN = 0, prismN = 0, pyramidcombineN = 0, tetcombineN = 0, hexN = 0;
	for (auto &h : hybrid_standard.Hs) {

		int triN = 0, quadN = 0; vector<int> tris, quads;
		for (auto fid : h.fs) {
			if (hybrid_standard.Fs[fid].vs.size() == 3) { triN++; tris.push_back(fid); }
			else if (hybrid_standard.Fs[fid].vs.size() == 4) { quadN++; quads.push_back(fid); }
		}

		vector<uint32_t> hvs = h.vs; h.vs.clear();

		if (h.fs.size() == 4) {
			if (triN == 4) {
				tetrahedralN++;
				h_type[h.id] = Element_Type::Tetrahedral;
				std::cout << "tetrahedral appear bug" << endl;
				system("PAUSE");
			}
			else if (triN == 2 && quadN == 2) {
				slabN++;
				h_type[h.id] = Element_Type::Slab;
				vector<uint32_t> sharedvs, vs0 = hybrid_standard.Fs[tris[0]].vs, vs1 = hybrid_standard.Fs[tris[1]].vs,
					vs2 = hybrid_standard.Fs[quads[0]].vs, vs3 = hybrid_standard.Fs[quads[1]].vs;
				sort(vs0.begin(), vs0.end()); sort(vs1.begin(), vs1.end());
				set_intersection(vs0.begin(), vs0.end(), vs1.begin(), vs1.end(), back_inserter(sharedvs));
				if (!sharedvs.size()) { std::cout << "bug" << endl; }
				int v0 = sharedvs[0]; int id = -1;
				if (find(vs2.begin(), vs2.end(), v0) != vs2.end()) {
					id = find(vs2.begin(), vs2.end(), v0) - vs2.begin();
					for (uint32_t j = 0; j < vs2.size(); j++) h.vs.push_back(vs2[(id + j) % vs2.size()]);
				}
				else {
					id = find(vs3.begin(), vs3.end(), v0) - vs3.begin();
					for (uint32_t j = 0; j < vs3.size(); j++) h.vs.push_back(vs3[(id + j) % vs3.size()]);
				}
				h.vs.push_back(sharedvs[1]);
			}
		}
		else if (h.fs.size() == 5) {
			if (triN == 4 && quadN == 1) {
				pyramidN++;
				h_type[h.id] = Element_Type::Pyramid;
				h.vs = hybrid_standard.Fs[quads[0]].vs;
				for (auto vid : hybrid_standard.Fs[tris[0]].vs) if (find(h.vs.begin(), h.vs.end(), vid) != h.vs.end())continue;
				else {
					h.vs.push_back(vid); break;
				}
			}
			else if (triN == 2 && quadN == 3) {
				prismN++;
				h_type[h.id] = Element_Type::Prism;
				h.vs = hybrid_standard.Fs[tris[0]].vs;
				for (auto vid : hybrid_standard.Fs[tris[0]].vs) for (auto vid2 : hybrid_standard.Fs[tris[1]].vs) {
					vector<uint32_t> es0 = hybrid_standard.Vs[vid].neighbor_es, es1 = hybrid_standard.Vs[vid2].neighbor_es, sharedes;
					sort(es0.begin(), es0.end()); sort(es1.begin(), es1.end());
					set_intersection(es0.begin(), es0.end(), es1.begin(), es1.end(), back_inserter(sharedes));
					if (sharedes.size()) {
						h.vs.push_back(vid2); break;
					}
				}
			}
		}
		else if (h.fs.size() == 6) {
			if (triN == 2 && quadN == 4) {
				pyramidcombineN++;
				h_type[h.id] = Element_Type::PyramidCombine;

				vector<uint32_t> sharedvs, vs0 = hybrid_standard.Fs[tris[0]].vs, vs1 = hybrid_standard.Fs[tris[1]].vs;
				sort(vs0.begin(), vs0.end()); sort(vs1.begin(), vs1.end());
				set_intersection(vs0.begin(), vs0.end(), vs1.begin(), vs1.end(), back_inserter(sharedvs));
				if (!sharedvs.size()) { std::cout << "bug" << endl; }
				int v0 = sharedvs[0]; int id = -1;
				for (auto fid : quads) {
					vector<uint32_t> vs = hybrid_standard.Fs[fid].vs;
					if (find(vs.begin(), vs.end(), v0) != vs.end()) {
						id = find(vs.begin(), vs.end(), v0) - vs.begin();
						for (uint32_t j = 0; j < vs.size(); j++) h.vs.push_back(vs[(id + j) % vs.size()]);

						vector<uint32_t> vs3, vs3_ordered;
						for (auto vid : hvs)if (find(h.vs.begin(), h.vs.end(), vid) != h.vs.end())continue; else vs3.push_back(vid);

						for (uint32_t j = 1; j < h.vs.size(); j++) for (auto vid2 : vs3) {
							vector<uint32_t> es0 = hybrid_standard.Vs[h.vs[j]].neighbor_es, es1 = hybrid_standard.Vs[vid2].neighbor_es, sharedes;
							sort(es0.begin(), es0.end()); sort(es1.begin(), es1.end());
							set_intersection(es0.begin(), es0.end(), es1.begin(), es1.end(), back_inserter(sharedes));
							if (sharedes.size()) {
								vs3_ordered.push_back(vid2); break;
							}
						}
						h.vs.insert(h.vs.end(), vs3_ordered.begin(), vs3_ordered.end());
						break;
					}
				}
			}
			else if (quadN == 6) {
				hexN++;
				h_type[h.id] = Element_Type::Hexahedral;
				h.vs = hybrid_standard.Fs[quads[0]].vs;
				vector<uint32_t> vs4;
				for (auto vid : hvs)if (find(h.vs.begin(), h.vs.end(), vid) != h.vs.end())continue; else vs4.push_back(vid);
				for (auto vid : hybrid_standard.Fs[quads[0]].vs) for (auto vid2 : vs4) {
					vector<uint32_t> es0 = hybrid_standard.Vs[vid].neighbor_es, es1 = hybrid_standard.Vs[vid2].neighbor_es, sharedes;
					sort(es0.begin(), es0.end()); sort(es1.begin(), es1.end());
					set_intersection(es0.begin(), es0.end(), es1.begin(), es1.end(), back_inserter(sharedes));
					if (sharedes.size()) {
						h.vs.push_back(vid2); break;
					}
				}
			}
			else if (triN == 4 && quadN == 2) {
				tetcombineN++;
				h_type[h.id] = Element_Type::TetCombine;
				h.vs = hvs;
			}
		}
	}
	std::cout << "total, tetN, slabN, pyramidN, prismN, pyramidcombineN, tetcombineN, hexN: " << hybrid_standard.Hs.size() << " " << tetrahedralN << " " << slabN << " " << pyramidN << " " << prismN << " " << pyramidcombineN << " " << tetcombineN << " " << hexN << endl;
}
void grid_hex_meshing_bijective::connectivity_modification(Mesh &hybrid_standard, vector<Element_Type> &h_type, vector<NV_type> &nv_tag, Mesh &pure_hex_mesh) {
	
	enum Vertex_Tag {
		Regular = 0,
		Pyramid_tip,
		Pyramid_bottom,
		Prism_tip
	};
	vector<Vertex_Tag> vertex_type(hybrid_standard.Vs.size(), Vertex_Tag::Regular);
	for (uint32_t i = 0; i < h_type.size(); i++)
		if (h_type[i] == Element_Type::Pyramid) {
			for (uint32_t j = 0; j < 4; j++) vertex_type[hybrid_standard.Hs[i].vs[j]] = Vertex_Tag::Pyramid_bottom;
			vertex_type[hybrid_standard.Hs[i].vs[4]] = Vertex_Tag::Pyramid_tip;
		}
	vector<bool> tip_quads(hybrid_standard.Fs.size(), false);
	for (auto &f : hybrid_standard.Fs) {
		bool notip = false;
		for (auto vid : f.vs)if (vertex_type[vid] != Vertex_Tag::Pyramid_tip)notip = true;		
		if (!notip)tip_quads[f.id] = true;
	}
	//base-cell: four pyramids, four prisms, a hex
	struct Base_Cell {
		int hex_id;
		int quad_id;
		vector<int> pyramid_ids;
		vector<int> prism_ids;
		vector<vector<int>> base_bvs;//00, 01, 02; 10, 11, 12; 20, 21, 22; 30, 31, 32;
		vector<int> base_qvs;// 03, 13, 23, 33;
		vector<uint32_t> top_qvs;//0, 1, 2, 3;

		vector<vector<int>> split_mvs;//0e0, 1e0; 1e1, 2e1; 2e2, 3e2; 3e3, 0e3;
		vector<int> split_ivs;//0q,1q,2q,3q.
		Vector3d direction;//from base to top		
	};
	vector<Base_Cell> bcells(hybrid_standard.Fs.size());

	for (uint32_t i = 0; i < tip_quads.size(); i++) {
		if (!tip_quads[i])continue;

		Base_Cell &bc = bcells[i];
		bc.hex_id = -1;
		bc.quad_id = i;
		bc.top_qvs = hybrid_standard.Fs[i].vs;
		bc.base_bvs.resize(4);

		for (auto nhid : hybrid_standard.Fs[i].neighbor_hs) {
			vector<int> nBV;
			for (auto hvid : hybrid_standard.Hs[nhid].vs)if (vertex_type[hvid] == Vertex_Tag::Pyramid_bottom) nBV.push_back(hvid);
			if (nBV.size() == 4) {
				int nP = 0;
				for (auto fid : hybrid_standard.Hs[nhid].fs) for (auto nnhid : hybrid_standard.Fs[fid].neighbor_hs)
					if (nnhid != nhid && h_type[nnhid] == Element_Type::Prism)nP++;

				if (nP != 4)continue;


				bc.hex_id = nhid;
				for (auto vid : bc.top_qvs) for (auto vid1 : nBV) {
					vector<uint32_t> es0 = hybrid_standard.Vs[vid].neighbor_es, es1 = hybrid_standard.Vs[vid1].neighbor_es, sharedes;
					sort(es0.begin(), es0.end()); sort(es1.begin(), es1.end());
					set_intersection(es0.begin(), es0.end(), es1.begin(), es1.end(), back_inserter(sharedes));
					if (sharedes.size()) { bc.base_qvs.push_back(vid1); break; }
				}
			}
		}
		if (bc.hex_id == -1) {
			tip_quads[i] = false; continue;
		}
		//prism
		for (auto eid : hybrid_standard.Fs[i].es) {
			vector<uint32_t> fs_local = hybrid_standard.Es[eid].neighbor_fs, fs_hex = hybrid_standard.Hs[bc.hex_id].fs, fs_shared;
			sort(fs_local.begin(), fs_local.end()); sort(fs_hex.begin(), fs_hex.end());
			set_intersection(fs_local.begin(), fs_local.end(), fs_hex.begin(), fs_hex.end(), back_inserter(fs_shared));
			int side_fid = -1;
			for (auto fshared : fs_shared)if (fshared != i) side_fid = fshared;
			for (auto nhid : hybrid_standard.Fs[side_fid].neighbor_hs) if (nhid != bc.hex_id) bc.prism_ids.push_back(nhid);
		}
		//pyramid
		for (uint32_t j = 0; j < bc.prism_ids.size(); j++) {
			int prism = bc.prism_ids[j];
			vector<int> ps;
			for (auto pfid : hybrid_standard.Hs[prism].fs)for (auto nhid : hybrid_standard.Fs[pfid].neighbor_hs)if (h_type[nhid] == Element_Type::Pyramid) ps.push_back(nhid);
			for (auto p : ps)if (hybrid_standard.Hs[p].vs[4] == bc.top_qvs[j]) { bc.pyramid_ids.push_back(p); break; }
		}
		//Vertices
		vector<vector<int>> othervs(4);
		for (uint32_t j = 0; j < 4; j++) {
			int which_index = -1, pid = bc.pyramid_ids[j];
			for (uint32_t k = 0; k < 4; k++) if (bc.base_qvs[j] == hybrid_standard.Hs[pid].vs[k]) which_index = k;
			bc.base_bvs[j].push_back(hybrid_standard.Hs[pid].vs[(which_index + 2) % 4]);
			othervs[j].push_back(hybrid_standard.Hs[pid].vs[(which_index + 1) % 4]);
			othervs[j].push_back(hybrid_standard.Hs[pid].vs[(which_index + 3) % 4]);
		}
		for (uint32_t j = 0; j < 4; j++) {
			int pid = bc.prism_ids[j];
			for (auto vid : hybrid_standard.Hs[pid].vs) if (vid == othervs[j][0] || vid == othervs[j][1]) bc.base_bvs[j].push_back(vid);
			for (auto vid : hybrid_standard.Hs[pid].vs) if (vid == othervs[(j + 1) % 4][0] || vid == othervs[(j + 1) % 4][1]) bc.base_bvs[j].push_back(vid);
		}
		Vector3d bcenter, tcenter;
		bcenter.setZero(); tcenter.setZero();
		for (uint32_t j = 0; j < 4; j++) {
			bcenter += hybrid_standard.V.col(bc.base_qvs[j]);
			tcenter += hybrid_standard.V.col(bc.top_qvs[j]);
		}
		bc.direction = tcenter/4 - bcenter/4;

		bc.split_ivs.resize(4);
		bc.split_mvs.resize(4);
		for (auto &am : bc.split_mvs)am.resize(2);
	}
	//classify into separate groups,
	vector<vector<uint32_t>> f_layers;
	vector<bool> tq_tag(tip_quads.size(), true);
	while (true) {

		int f_start = -1;

		for (uint32_t i = 0; i<tq_tag.size(); i++)
			if (tq_tag[i] && tip_quads[i]) { f_start = i; tq_tag[i] = false; break; }
		if (f_start == -1)break;

		vector<uint32_t> a_layer, layer_temp, layer_temp2;
		a_layer.push_back(f_start); layer_temp.push_back(f_start);
		while (layer_temp.size()) {
			for (auto fid : layer_temp)for (auto vid : hybrid_standard.Fs[fid].vs)
				for (auto nfid : hybrid_standard.Vs[vid].neighbor_fs) {
					if (tq_tag[nfid] && tip_quads[nfid]) {
						tq_tag[nfid] = false; layer_temp2.push_back(nfid);
					}
				}
			if (layer_temp2.size()) {
				a_layer.insert(a_layer.end(), layer_temp2.begin(), layer_temp2.end());
				layer_temp = layer_temp2;
				layer_temp2.clear();
			}
			else break;
		}
		f_layers.push_back(a_layer);
	}
	for (uint32_t i = 0; i < f_layers.size(); i++) for (auto fid : f_layers[i])for (auto vid : hybrid_standard.Fs[fid].vs) {
		;
	}

	//Quad_Tag 
	enum Quad_Tag {
		Entire_Split = 0,
		One_Split_x,//corresponds to edge index 0
		One_Split_y,//corresponds to edge index 1
		No_Split,
		Initial_State
	};
	vector<Quad_Tag> q_tag(hybrid_standard.Fs.size(), Quad_Tag::Initial_State);
	fill(tq_tag.begin(), tq_tag.end(), false);
	int which_group = -1;
	for (const auto &a_group : f_layers) {
		which_group++;
		//quad-tag
		while (true) {
			int s_fid = -1;
			for (auto fid : a_group)if (!tq_tag[fid]) {
				vector<int> e_indices;
				for (uint32_t i = 0; i < 4; i++) {
					int eid = hybrid_standard.Fs[fid].es[i];
					bool found = false;
					for (auto efid : hybrid_standard.Es[eid].neighbor_fs)
						if (efid != fid && tip_quads[efid])found = true;
					if (!found) e_indices.push_back(i);
				}
				if (e_indices.size() == 2) { if ((e_indices[0] + 1) % 4 == e_indices[1] || (e_indices[1] + 1) % 4 == e_indices[0]) { s_fid = fid; break; } }
				else if (e_indices.size() > 2) { s_fid = fid; break; }
			}
			if (s_fid == -1) {
				for (auto fid : a_group)if (!tq_tag[fid]) {
					vector<int> e_indices;
					for (uint32_t i = 0; i < 4; i++) {
						int eid = hybrid_standard.Fs[fid].es[i];
						for (auto nfid : hybrid_standard.Es[eid].neighbor_fs)if (fid != nfid && tip_quads[nfid]) {
							vector<int> pyramids0 = bcells[fid].pyramid_ids, pyramids1 = bcells[nfid].pyramid_ids, sharedpyramids;
							sort(pyramids0.begin(), pyramids0.end()); sort(pyramids1.begin(), pyramids1.end());
							set_intersection(pyramids0.begin(), pyramids0.end(), pyramids1.begin(), pyramids1.end(), back_inserter(sharedpyramids));
							if (sharedpyramids.size() == 0) {
								e_indices.push_back(i); break;
							}
						}
					}
					if (e_indices.size() == 2) { if ((e_indices[0] + 1) % 4 == e_indices[1] || (e_indices[1] + 1) % 4 == e_indices[0]) { s_fid = fid; break; } }
					else if (e_indices.size() > 2) { s_fid = fid; break; }
				}
			}
			if (s_fid == -1) {
				for (const auto &fid : a_group) if (!tq_tag[fid]) {
					vector<int> e_indices_share, e_indices_notshare;
					for (uint32_t i = 0; i < 4; i++) {
						int eid = hybrid_standard.Fs[fid].es[i];
						for (auto nfid : hybrid_standard.Es[eid].neighbor_fs)if (fid != nfid && tip_quads[nfid]) {
							vector<int> pyramids0 = bcells[fid].pyramid_ids, pyramids1 = bcells[nfid].pyramid_ids, sharedpyramids;
							sort(pyramids0.begin(), pyramids0.end()); sort(pyramids1.begin(), pyramids1.end());
							set_intersection(pyramids0.begin(), pyramids0.end(), pyramids1.begin(), pyramids1.end(), back_inserter(sharedpyramids));
							if (sharedpyramids.size() == 0) {
								e_indices_notshare.push_back(i); break;
							}else {
								e_indices_share.push_back(i); break;
							}	
						}
					}
					if (e_indices_notshare.size() == 2 && e_indices_share.size()==1) { 
						if ((e_indices_notshare[0] + 2) % 4 == e_indices_notshare[1] || (e_indices_notshare[1] + 2) % 4 == e_indices_notshare[0]) {
						s_fid = fid; break; } }

				}
			}
			if (s_fid == -1) {//only 4 quads:open cylinder
				for (const auto &fid : a_group) if (!tq_tag[fid]) {
					vector<int> e_indices_share, e_indices_notshare;
					for (uint32_t i = 0; i < 4; i++) {
						int eid = hybrid_standard.Fs[fid].es[i];
						for (auto nfid : hybrid_standard.Es[eid].neighbor_fs)if (fid != nfid && tip_quads[nfid]) {
							vector<int> pyramids0 = bcells[fid].pyramid_ids, pyramids1 = bcells[nfid].pyramid_ids, sharedpyramids;
							sort(pyramids0.begin(), pyramids0.end()); sort(pyramids1.begin(), pyramids1.end());
							set_intersection(pyramids0.begin(), pyramids0.end(), pyramids1.begin(), pyramids1.end(), back_inserter(sharedpyramids));
							if (sharedpyramids.size() == 0) {
								e_indices_notshare.push_back(i); break;
							}
							else {
								e_indices_share.push_back(i); break;
							}
						}
					}
					if (e_indices_notshare.size() == 2) {
						if ((e_indices_notshare[0] + 2) % 4 == e_indices_notshare[1] || (e_indices_notshare[1] + 2) % 4 == e_indices_notshare[0]) {
							s_fid = fid; break;
						}
					}

				}
			}
			if (s_fid == -1) {//more than 4 quads:open cylinder
				for (const auto &fid : a_group) if (!tq_tag[fid]) {
					vector<int> e_indices_share, e_indices_notshare;
					for (uint32_t i = 0; i < 4; i++) {
						int eid = hybrid_standard.Fs[fid].es[i];
						for (auto nfid : hybrid_standard.Es[eid].neighbor_fs)if (fid != nfid && tip_quads[nfid]) {
							vector<int> pyramids0 = bcells[fid].pyramid_ids, pyramids1 = bcells[nfid].pyramid_ids, sharedpyramids;
							sort(pyramids0.begin(), pyramids0.end()); sort(pyramids1.begin(), pyramids1.end());
							set_intersection(pyramids0.begin(), pyramids0.end(), pyramids1.begin(), pyramids1.end(), back_inserter(sharedpyramids));
							if (sharedpyramids.size() == 0) {
								e_indices_notshare.push_back(i); break;
							}
							else {
								e_indices_share.push_back(i); break;
							}
						}
					}
					if (e_indices_notshare.size() == 1 && e_indices_share.size() == 2) {
						if ((e_indices_share[0] + 1) % 4 == e_indices_share[1] || (e_indices_share[1] + 1) % 4 == e_indices_share[0]) {
							s_fid = fid; break;
						}
					}

				}
			}
			if (s_fid == -1) {
				for (const auto &fid : a_group) if (!tq_tag[fid]) {
					cout << "didn't find starting quad at group: " << which_group<< endl;
				}
				break;
			}

			tq_tag[s_fid] = true;
			q_tag[s_fid] = Quad_Tag::Entire_Split;
			vector<int> pool, pool_; pool.push_back(s_fid);
			while (pool.size()) {
				for (auto fid : pool)for (uint32_t i = 0; i < 4; i++) {
					int eid = hybrid_standard.Fs[fid].es[i];
					for (auto nfid : hybrid_standard.Es[eid].neighbor_fs)if (!tq_tag[nfid] && tip_quads[nfid] && q_tag[nfid] == Quad_Tag::Initial_State) {

						int whichindex = -1;
						for (uint32_t j = 0; j < 4; j++) if (eid == hybrid_standard.Fs[nfid].es[j]) { whichindex = j; break; }
						int whichindex_o = -1;
						for (uint32_t j = 0; j < 4; j++) if (eid == hybrid_standard.Fs[fid].es[j]) { whichindex_o = j; break; }

						vector<int> pyramids0 = bcells[fid].pyramid_ids, pyramids1 = bcells[nfid].pyramid_ids, sharedpyramids;
						sort(pyramids0.begin(), pyramids0.end()); sort(pyramids1.begin(), pyramids1.end());
						set_intersection(pyramids0.begin(), pyramids0.end(), pyramids1.begin(), pyramids1.end(), back_inserter(sharedpyramids));

						if (q_tag[fid] == Quad_Tag::Entire_Split) {
							if (sharedpyramids.size() == 0) q_tag[nfid] = Quad_Tag::Entire_Split;
							else if (sharedpyramids.size() == 1) q_tag[nfid] = Quad_Tag::No_Split;
							else if (sharedpyramids.size() == 2) {
								if (whichindex == 0 || whichindex == 2)q_tag[nfid] = Quad_Tag::One_Split_x;
								else if (whichindex == 1 || whichindex == 3)q_tag[nfid] = Quad_Tag::One_Split_y;
							}
						}
						else if (q_tag[fid] == Quad_Tag::One_Split_x) {
							if (sharedpyramids.size() == 0) {
								if (whichindex == 0 || whichindex == 2)q_tag[nfid] = Quad_Tag::One_Split_y;
								else if (whichindex == 1 || whichindex == 3)q_tag[nfid] = Quad_Tag::One_Split_x;
							}
							else {
								if (whichindex_o == 0 || whichindex_o == 2)q_tag[nfid] = Quad_Tag::Entire_Split;
								else if (whichindex_o == 1 || whichindex_o == 3)q_tag[nfid] = Quad_Tag::No_Split;
							}
						}
						else if (q_tag[fid] == Quad_Tag::One_Split_y) {
							if (sharedpyramids.size() == 0) {
								if (whichindex == 0 || whichindex == 2)q_tag[nfid] = Quad_Tag::One_Split_y;
								else if (whichindex == 1 || whichindex == 3)q_tag[nfid] = Quad_Tag::One_Split_x;
							}
							else {
								if (whichindex_o == 0 || whichindex_o == 2)q_tag[nfid] = Quad_Tag::No_Split;
								else if (whichindex_o == 1 || whichindex_o == 3)q_tag[nfid] = Quad_Tag::Entire_Split;
							}
						}
						else if (q_tag[fid] == Quad_Tag::No_Split) {
							if (whichindex == 0 || whichindex == 2)q_tag[nfid] = Quad_Tag::One_Split_y;
							else if (whichindex == 1 || whichindex == 3)q_tag[nfid] = Quad_Tag::One_Split_x;
						}
						tq_tag[nfid] = true;
						pool_.push_back(nfid);
					}
				}
				pool = pool_; pool_.clear();
			}
		}
	}

	//tip split, hex built
	for (auto v : hybrid_standard.Vs) {
		Hybrid_V v_; v_.id = v.id;
		v_.v = v.v;
		pure_hex_mesh.Vs.push_back(v_);
		nv_tag.push_back(NV_type::NV);
	}
	//tip split
	double w_e1 = 0.75, w_e2 = 0.75, w_iq = 0.3, w_eb = 0.75;
	typedef std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> tuple_Tip_Split;//tip_id, splited new v_id, edge id, tip quad_id //, at which edge index of tip quad_id
	vector<vector<int>> v2tuple_map(vertex_type.size());
	vector<tuple_Tip_Split> tip_splits;
	vector<bool> edge_tag(hybrid_standard.Es.size(), false);
	for (auto agroup : f_layers) for (auto fid : agroup)if (q_tag[fid] == Quad_Tag::Entire_Split) for (auto eid : hybrid_standard.Fs[fid].es) { edge_tag[eid] = true; }

	for (uint32_t i = 0; i < vertex_type.size(); i++) {
		if (vertex_type[i] != Vertex_Tag::Pyramid_tip) continue;

		vector<int> es, fs;//tips
		for (auto neid : hybrid_standard.Vs[i].neighbor_es) if (edge_tag[neid]) es.push_back(neid);

		for (auto nfid : hybrid_standard.Vs[i].neighbor_fs)if (q_tag[nfid] == Quad_Tag::Entire_Split) {
			Hybrid_V v;
			v.v.resize(3, 0);
			v.id = pure_hex_mesh.Vs.size();
			int which_v = -1;
			Base_Cell &bc = bcells[nfid];
			for (uint32_t j = 0; j < 4; j++)if (bc.top_qvs[j] == i) { 
				which_v = bc.base_qvs[j]; 
				bc.split_ivs[j] = v.id;
			}
			for (uint32_t j = 0; j < 3; j++) v.v[j] = hybrid_standard.Vs[which_v].v[j] + w_iq * bc.direction[j];

			pure_hex_mesh.Vs.push_back(v);
			nv_tag.push_back(NV_type::IV);
			hex2Octree_map.push_back(hex2Octree_map[i]);

			v2tuple_map[i].push_back(tip_splits.size());
			tip_splits.push_back(tuple_Tip_Split(i, v.id, -1, nfid));
			fs.push_back(nfid);
		}
		for (auto eid : es) {
			vector<int> nfs;
			for (auto nfid : hybrid_standard.Es[eid].neighbor_fs)if (q_tag[nfid] == Quad_Tag::Entire_Split)nfs.push_back(nfid);
			Hybrid_V v;
			v.v.resize(3, 0);
			v.id = pure_hex_mesh.Vs.size();

			if (nfs.size() == 1) {
				int which_v = -1;
				Base_Cell &bc = bcells[nfs[0]];
				for (uint32_t j = 0; j < 4; j++)if (hybrid_standard.Fs[nfs[0]].es[j] == eid) {
					if (bc.top_qvs[j] == i) {
						which_v = bc.base_bvs[j][1];
						bc.split_mvs[j][0] = v.id;
					}
					else if (bc.top_qvs[(j + 1) % 4] == i) {
						which_v = bc.base_bvs[j][2];
						bc.split_mvs[j][1] = v.id;
					}
				}
				for (uint32_t j = 0; j < 3; j++) v.v[j] = hybrid_standard.Vs[which_v].v[j] + w_e1 * bc.direction[j];
				v2tuple_map[i].push_back(tip_splits.size());
				tip_splits.push_back(tuple_Tip_Split(i, v.id, eid, nfs[0]));
				nv_tag.push_back(NV_type::EV);
				hex2Octree_map.push_back(hex2Octree_map[i]);
			}
			else if (nfs.size() == 2) {
				int which_v = -1;
				for (auto fid : nfs) {
					Base_Cell &bc = bcells[fid];
					for (uint32_t j = 0; j < 4; j++)if (hybrid_standard.Fs[fid].es[j] == eid) {
						if (bc.top_qvs[j] == i) {
							which_v = bc.base_bvs[j][1];
							bc.split_mvs[j][0] = v.id;
						}
						else if (bc.top_qvs[(j + 1) % 4] == i) {
							which_v = bc.base_bvs[j][2];
							bc.split_mvs[j][1] = v.id;
						}
					}
					for (uint32_t j = 0; j < 3; j++) v.v[j] += hybrid_standard.Vs[which_v].v[j] + w_e2 * bc.direction[j];
				}
				for (uint32_t j = 0; j < 3; j++) v.v[j] /= 2;
				v2tuple_map[i].push_back(tip_splits.size());
				tip_splits.push_back(tuple_Tip_Split(i, v.id, eid, nfs[0]));
				v2tuple_map[i].push_back(tip_splits.size());
				tip_splits.push_back(tuple_Tip_Split(i, v.id, eid, nfs[1]));
				nv_tag.push_back(NV_type::DEV);
				hex2Octree_map.push_back(hex2Octree_map[i]);
			}
			else {
				cout << "bug!!!!!!!!!!" << endl; 
			}
			pure_hex_mesh.Vs.push_back(v);
		}
	}
	//assign splited vertices to base-cells
	for (auto a_tip : tip_splits) {
		int which_corner = -1, which_side = -1;
		Base_Cell &bc = bcells[get<3>(a_tip)];
		for (uint32_t i = 0; i < 4; i++) {
			if (get<0>(a_tip) == bc.top_qvs[i])which_corner = i;
			if (get<2>(a_tip) != -1 && get<2>(a_tip) == hybrid_standard.Fs[bc.quad_id].es[i])which_side = i;
		}
		if (get<2>(a_tip) == -1) {
			if (which_corner != -1) bc.split_ivs[which_corner] = get<1>(a_tip);
			else { cout << "BUG!!!!";}
		}
		else {
			if (which_side != -1) {
				if (which_side == which_corner) bc.split_mvs[which_side][0] = get<1>(a_tip);
				else if (which_side == (which_corner + 1) % 4) bc.split_mvs[which_side][1] = get<1>(a_tip);
			}
			else { cout << "BUG!!!!";}
		}
	}
	//hex built:iterate quads
	vector<bool> h_tag(hybrid_standard.Hs.size(), true);
	for (uint32_t i = 0; i < q_tag.size(); i++) {
		if (q_tag[i] == Quad_Tag::Initial_State || q_tag[i] == Quad_Tag::No_Split) continue;

		Base_Cell &bc = bcells[i];

		Hybrid h; h.vs.resize(8);
		if (q_tag[i] == Quad_Tag::Entire_Split) {
			h_tag[bc.hex_id] = false;
			for (auto p : bc.prism_ids)h_tag[p] = false;
			for (auto p : bc.pyramid_ids)h_tag[p] = false;
			//bottom 8
			for (uint32_t j = 0; j < 4; j++) {
				h.id = pure_hex_mesh.Hs.size();
				h.vs[0] = bc.base_bvs[j][0];
				h.vs[1] = bc.base_bvs[j][1];
				h.vs[2] = bc.base_qvs[j];
				h.vs[3] = bc.base_bvs[(j - 1 + 4) % 4][2];

				h.vs[4] = bc.top_qvs[j];
				h.vs[5] = bc.split_mvs[j][0];
				h.vs[6] = bc.split_ivs[j];
				h.vs[7] = bc.split_mvs[(j - 1 + 4) % 4][1];
				pure_hex_mesh.Hs.push_back(h);

				h.id = pure_hex_mesh.Hs.size();
				h.vs[0] = bc.base_bvs[j][1];
				h.vs[1] = bc.base_bvs[j][2];
				h.vs[2] = bc.base_qvs[(j + 1) % 4];
				h.vs[3] = bc.base_qvs[j];

				h.vs[4] = bc.split_mvs[j][0];
				h.vs[5] = bc.split_mvs[j][1];
				h.vs[6] = bc.split_ivs[(j + 1) % 4];
				h.vs[7] = bc.split_ivs[j];
				pure_hex_mesh.Hs.push_back(h);
			}
			//bottom middle 
			h.id = pure_hex_mesh.Hs.size();
			for (uint32_t j = 0; j < 4; j++) {
				h.vs[j] = bc.base_qvs[j];
				h.vs[j + 4] = bc.split_ivs[j];
			}
			pure_hex_mesh.Hs.push_back(h);
			//middle 3
			h.id = pure_hex_mesh.Hs.size();
			h.vs[0] = bc.top_qvs[0];
			h.vs[1] = bc.split_mvs[0][0];
			h.vs[2] = bc.split_ivs[0];
			h.vs[3] = bc.split_mvs[(0 - 1 + 4) % 4][1];

			h.vs[4] = bc.top_qvs[3];
			h.vs[5] = bc.split_mvs[2][1];
			h.vs[6] = bc.split_ivs[3];
			h.vs[7] = bc.split_mvs[3][0];
			pure_hex_mesh.Hs.push_back(h);

			h.id = pure_hex_mesh.Hs.size();
			h.vs[0] = bc.split_mvs[0][0];
			h.vs[1] = bc.split_mvs[0][1];
			h.vs[2] = bc.split_ivs[1];
			h.vs[3] = bc.split_ivs[0];

			h.vs[4] = bc.split_mvs[2][1];
			h.vs[5] = bc.split_mvs[2][0];
			h.vs[6] = bc.split_ivs[2];
			h.vs[7] = bc.split_ivs[3];
			pure_hex_mesh.Hs.push_back(h);

			h.id = pure_hex_mesh.Hs.size();
			h.vs[0] = bc.top_qvs[1];
			h.vs[1] = bc.split_mvs[1][0];
			h.vs[2] = bc.split_ivs[1];
			h.vs[3] = bc.split_mvs[0][1];

			h.vs[4] = bc.top_qvs[2];
			h.vs[5] = bc.split_mvs[1][1];
			h.vs[6] = bc.split_ivs[2];
			h.vs[7] = bc.split_mvs[2][0];
			pure_hex_mesh.Hs.push_back(h);
			//top 1
			h.id = pure_hex_mesh.Hs.size();
			h.vs[0] = bc.top_qvs[0];
			h.vs[1] = bc.top_qvs[1];
			h.vs[2] = bc.split_mvs[0][1];
			h.vs[3] = bc.split_mvs[0][0];

			h.vs[4] = bc.top_qvs[3];
			h.vs[5] = bc.top_qvs[2];
			h.vs[6] = bc.split_mvs[2][0];
			h.vs[7] = bc.split_mvs[2][1];
			pure_hex_mesh.Hs.push_back(h);
		}
		else {
			h_tag[bc.hex_id] = false;
			vector<vector<int>> vs0(3), vs2(3);
			Base_Cell bc0, bc2;
			int e0 = -1, e2 = -1, q0 = -1, q2 = -1;
			vector<int> ps;
			if (q_tag[i] == Quad_Tag::One_Split_x) {
				h_tag[bc.prism_ids[1]] = h_tag[bc.prism_ids[3]] = false;
				ps.push_back(bc.prism_ids[1]);
				ps.push_back(bc.prism_ids[3]);

				e0 = hybrid_standard.Fs[i].es[0]; e2 = hybrid_standard.Fs[i].es[2];
				vs0[0].push_back(hybrid_standard.Fs[i].vs[0]);
				vs0[0].push_back(hybrid_standard.Fs[i].vs[1]);
				vs2[0].push_back(hybrid_standard.Fs[i].vs[3]);
				vs2[0].push_back(hybrid_standard.Fs[i].vs[2]);
			}
			else if (q_tag[i] == Quad_Tag::One_Split_y) {
				h_tag[bc.prism_ids[0]] = h_tag[bc.prism_ids[2]] = false;
				ps.push_back(bc.prism_ids[0]);
				ps.push_back(bc.prism_ids[2]);


				e0 = hybrid_standard.Fs[i].es[1]; e2 = hybrid_standard.Fs[i].es[3];
				vs0[0].push_back(hybrid_standard.Fs[i].vs[1]);
				vs0[0].push_back(hybrid_standard.Fs[i].vs[2]);
				vs2[0].push_back(hybrid_standard.Fs[i].vs[0]);
				vs2[0].push_back(hybrid_standard.Fs[i].vs[3]);
			}
			for (auto nfid : hybrid_standard.Es[e0].neighbor_fs)
				if (nfid != i && q_tag[nfid] == Quad_Tag::Entire_Split) {
					q0 = nfid; break;
				}
			for (auto nfid : hybrid_standard.Es[e2].neighbor_fs)
				if (nfid != i && q_tag[nfid] == Quad_Tag::Entire_Split) {
					q2 = nfid; break;
				}
			if (q0 == -1 || q2 == -1)cout << "BUGGGGGGG" << endl;
			bc0 = bcells[q0]; bc2 = bcells[q2];
			
			vector<int> ids0, ids2;
			for (auto vid : vs0[0])if (find(bc0.top_qvs.begin(), bc0.top_qvs.end(), vid) != bc0.top_qvs.end()) {
				ids0.push_back(find(bc0.top_qvs.begin(), bc0.top_qvs.end(), vid) - bc0.top_qvs.begin());
			}
			for (auto vid : vs2[0])if (find(bc2.top_qvs.begin(), bc2.top_qvs.end(), vid) != bc2.top_qvs.end()) {
				ids2.push_back(find(bc2.top_qvs.begin(), bc2.top_qvs.end(), vid) - bc2.top_qvs.begin());
			}
			if ((ids0[0]+1)%4 == ids0[1]) {
				vs0[1] = bc0.split_mvs[ids0[0]];
				vs0[2].insert(vs0[2].end(), bc0.base_bvs[ids0[0]].begin(), bc0.base_bvs[ids0[0]].end());
				vs0[2].push_back(bc0.base_bvs[ids0[1]][0]);
			}
			else {
				vs0[1].push_back(bc0.split_mvs[ids0[1]][1]);
				vs0[1].push_back(bc0.split_mvs[ids0[1]][0]);
				vs0[2].push_back(bc0.base_bvs[ids0[0]][0]);
				vs0[2].push_back(bc0.base_bvs[ids0[1]][2]);
				vs0[2].push_back(bc0.base_bvs[ids0[1]][1]);
				vs0[2].push_back(bc0.base_bvs[ids0[1]][0]);
			}
			if ((ids2[0]+1)%4 == ids2[1]) {
				vs2[1] = bc2.split_mvs[ids2[0]];
				vs2[2].insert(vs2[2].end(), bc2.base_bvs[ids2[0]].begin(), bc2.base_bvs[ids2[0]].end());
				vs2[2].push_back(bc2.base_bvs[ids2[1]][0]);
			}
			else {
				vs2[1].push_back(bc2.split_mvs[ids2[1]][1]);
				vs2[1].push_back(bc2.split_mvs[ids2[1]][0]);
				vs2[2].push_back(bc2.base_bvs[ids2[0]][0]);
				vs2[2].push_back(bc2.base_bvs[ids2[1]][2]);
				vs2[2].push_back(bc2.base_bvs[ids2[1]][1]);
				vs2[2].push_back(bc2.base_bvs[ids2[1]][0]);
			}
			//top 1
			h.id = pure_hex_mesh.Hs.size();
			h.vs[0] = vs0[0][0];
			h.vs[1] = vs0[0][1];
			h.vs[2] = vs0[1][1];
			h.vs[3] = vs0[1][0];

			h.vs[4] = vs2[0][0];
			h.vs[5] = vs2[0][1];
			h.vs[6] = vs2[1][1];
			h.vs[7] = vs2[1][0];
			pure_hex_mesh.Hs.push_back(h);
			//bottom 3
			h.id = pure_hex_mesh.Hs.size();
			h.vs[0] = vs0[0][0];
			h.vs[1] = vs0[1][0];
			h.vs[2] = vs0[2][1];
			h.vs[3] = vs0[2][0];

			h.vs[4] = vs2[0][0];
			h.vs[5] = vs2[1][0];
			h.vs[6] = vs2[2][1];
			h.vs[7] = vs2[2][0];
			pure_hex_mesh.Hs.push_back(h);

			h.id = pure_hex_mesh.Hs.size();
			h.vs[0] = vs0[1][0];
			h.vs[1] = vs0[1][1];
			h.vs[2] = vs0[2][2];
			h.vs[3] = vs0[2][1];

			h.vs[4] = vs2[1][0];
			h.vs[5] = vs2[1][1];
			h.vs[6] = vs2[2][2];
			h.vs[7] = vs2[2][1];
			pure_hex_mesh.Hs.push_back(h);

			h.id = pure_hex_mesh.Hs.size();
			h.vs[0] = vs0[1][1];
			h.vs[1] = vs0[0][1];
			h.vs[2] = vs0[2][3];
			h.vs[3] = vs0[2][2];

			h.vs[4] = vs2[1][1];
			h.vs[5] = vs2[0][1];
			h.vs[6] = vs2[2][3];
			h.vs[7] = vs2[2][2];
			pure_hex_mesh.Hs.push_back(h);
		}
	}
	//hex built:iterate edges
	for (uint32_t i = 0; i < edge_tag.size(); i++) if (edge_tag[i]) {
		//cout << i << endl;
		
		vector<int> nfs_entire, nfs_all;
		for (auto nfid : hybrid_standard.Es[i].neighbor_fs)if (tip_quads[nfid]) {
			nfs_all.push_back(nfid);

			if (q_tag[nfid] == Quad_Tag::Entire_Split) nfs_entire.push_back(nfid);
		}

		Hybrid h; h.vs.resize(8);
		if (nfs_entire.size() == 2) {
			//cout << "nfs_entire 2" << endl;
			Base_Cell &bc = bcells[nfs_entire[0]];
			int which_index = -1;
			for (uint32_t j = 0; j < 4; j++) {
				int eid = hybrid_standard.Fs[nfs_entire[0]].es[j];
				if (eid == i) { which_index = j; break; }
			}
			int pid = bc.prism_ids[which_index];
			int hid = -1; vector<int> pids;
			for (auto fid : hybrid_standard.Hs[pid].fs) {
				if (find(hybrid_standard.Fs[fid].es.begin(), hybrid_standard.Fs[fid].es.end(), i) != hybrid_standard.Fs[fid].es.end()) {
					for (auto nhid : hybrid_standard.Fs[fid].neighbor_hs)
						if (h_type[nhid] == Element_Type::Hexahedral && bc.hex_id != nhid) hid = nhid;
				}
			}
			for (auto fid : hybrid_standard.Hs[hid].fs)for (auto nhid : hybrid_standard.Fs[fid].neighbor_hs)
				if (h_type[nhid] == Element_Type::PyramidCombine) pids.push_back(nhid);
			if (!pids.size()) {
				for (auto fid : hybrid_standard.Hs[hid].fs)for (auto nhid : hybrid_standard.Fs[fid].neighbor_hs)
					if (h_type[nhid] == Element_Type::TetCombine) pids.push_back(nhid);
				if (!pids.size()) {
					cout << "BUGGGGGGGGGGGGG" << endl;
				}
				h_tag[pids[0]] = h_tag[pids[1]] = h_tag[hid] = false;

				vector<int> vs0, vs1;
				vs0.push_back(hybrid_standard.Es[i].vs[0]);
				for (auto tid : v2tuple_map[vs0[0]])
					if (vs0[0] == std::get<0>(tip_splits[tid]) && i == std::get<2>(tip_splits[tid])) { vs0.push_back(get<1>(tip_splits[tid])); break; }
				vs1.push_back(hybrid_standard.Es[i].vs[1]);
				for (auto tid : v2tuple_map[vs1[0]])
					if (vs1[0] == std::get<0>(tip_splits[tid]) && i == std::get<2>(tip_splits[tid])) { vs1.push_back(get<1>(tip_splits[tid])); break; }

				if (find(hybrid_standard.Hs[pids[0]].vs.begin(), hybrid_standard.Hs[pids[0]].vs.end(), vs1[0]) != hybrid_standard.Hs[pids[0]].vs.end())
					swap(pids[0], pids[1]);

				vector<int> vs00, vs11; int hfid0 = -1, hfid1 = -1;
				for (auto vfid : hybrid_standard.Vs[vs0[0]].neighbor_fs) {
					if (find(hybrid_standard.Hs[hid].fs.begin(), hybrid_standard.Hs[hid].fs.end(), vfid) != hybrid_standard.Hs[hid].fs.end()) {
						vector<uint32_t> &fvs = hybrid_standard.Fs[vfid].vs;
						if (find(fvs.begin(), fvs.end(), vs1[0]) != fvs.end())continue;
						for (uint32_t j = 0; j < 4; j++)if (fvs[j] == vs0[0]) vs00.push_back(fvs[(j + 2) % 4]);
						hfid0 = vfid;
						if (vs00.size())break;
					}
				}
				for (auto vfid : hybrid_standard.Vs[vs1[0]].neighbor_fs) {
					if (find(hybrid_standard.Hs[hid].fs.begin(), hybrid_standard.Hs[hid].fs.end(), vfid) != hybrid_standard.Hs[hid].fs.end()) {
						vector<uint32_t> &fvs = hybrid_standard.Fs[vfid].vs;
						if (find(fvs.begin(), fvs.end(), vs0[0]) != fvs.end())continue;
						for (uint32_t j = 0; j < 4; j++)if (fvs[j] == vs1[0]) vs11.push_back(fvs[(j + 2) % 4]);
						hfid1 = vfid;
						if (vs11.size())break;
					}
				}
				int oeid = -1;
				vector<uint32_t> es00 = hybrid_standard.Vs[vs00[0]].neighbor_es, es11 = hybrid_standard.Vs[vs11[0]].neighbor_es, sharedes;
				sort(es00.begin(), es00.end()); sort(es11.begin(), es11.end());
				set_intersection(es00.begin(), es00.end(), es11.begin(), es11.end(), back_inserter(sharedes));
				if (!sharedes.size()) { cout << "bug" << endl; }
				oeid = sharedes[0];
				edge_tag[oeid] = false;
				for (auto tid : v2tuple_map[vs00[0]])
					if (vs00[0] == std::get<0>(tip_splits[tid]) && oeid == std::get<2>(tip_splits[tid])) { vs00.push_back(get<1>(tip_splits[tid])); break; }
				for (auto tid : v2tuple_map[vs11[0]])
					if (vs11[0] == std::get<0>(tip_splits[tid]) && oeid == std::get<2>(tip_splits[tid])) { vs11.push_back(get<1>(tip_splits[tid])); break; }
				
				if (vs00.size() != 2 || vs11.size() != 2)continue;

				//p0
				h.id = pure_hex_mesh.Hs.size();
				for (uint32_t j = 0; j < 4; j++) {
					int fvid = hybrid_standard.Fs[hfid0].vs[j];
					if (fvid == vs0[0]) {
						h.vs[j] = fvid;
						h.vs[j + 4] = vs0[1];
					}
					else if (fvid == vs00[0]) {
						h.vs[j] = fvid;
						h.vs[j + 4] = vs00[1];
					}
					else {
						for (auto vvid : hybrid_standard.Vs[fvid].neighbor_vs) {
							if (find(hybrid_standard.Hs[pids[0]].vs.begin(), hybrid_standard.Hs[pids[0]].vs.end(), vvid) != hybrid_standard.Hs[pids[0]].vs.end())
								if (vvid != vs0[0] && vvid != vs00[0]) {
									h.vs[j] = vvid; break;
								}
						}
						h.vs[j + 4] = fvid;
					}
				}
				pure_hex_mesh.Hs.push_back(h);
				//p1
				h.id = pure_hex_mesh.Hs.size();
				for (uint32_t j = 0; j < 4; j++) {
					int fvid = hybrid_standard.Fs[hfid1].vs[j];
					if (fvid == vs1[0]) {
						h.vs[j] = fvid;
						h.vs[j + 4] = vs1[1];
					}
					else if (fvid == vs11[0]) {
						h.vs[j] = fvid;
						h.vs[j + 4] = vs11[1];
					}
					else {
						for (auto vvid : hybrid_standard.Vs[fvid].neighbor_vs) {
							if (find(hybrid_standard.Hs[pids[1]].vs.begin(), hybrid_standard.Hs[pids[1]].vs.end(), vvid) != hybrid_standard.Hs[pids[1]].vs.end())
								if (vvid != vs1[0] && vvid != vs11[0]) {
									h.vs[j] = vvid; break;
								}
						}
						h.vs[j + 4] = fvid;
					}
				}
				pure_hex_mesh.Hs.push_back(h);
				//hex
				h.id = pure_hex_mesh.Hs.size(); 
				h.vs = hybrid_standard.Hs[hid].vs;
				for (auto &vid : h.vs)if (vid == vs0[0]) vid = vs0[1]; else if (vid == vs1[0])vid = vs1[1];
				else if (vid == vs00[0]) vid = vs00[1]; else if (vid == vs11[0])vid = vs11[1];
				pure_hex_mesh.Hs.push_back(h);

				continue;
			}
			h_tag[pids[0]] = h_tag[pids[1]] = h_tag[hid] = false;

			vector<int> vs0, vs1;
			vs0.push_back(hybrid_standard.Es[i].vs[0]);
			for (auto tid : v2tuple_map[vs0[0]])
				if (vs0[0] == std::get<0>(tip_splits[tid]) && i == std::get<2>(tip_splits[tid])) { vs0.push_back(get<1>(tip_splits[tid])); break; }
			vs1.push_back(hybrid_standard.Es[i].vs[1]);
			for (auto tid : v2tuple_map[vs1[0]])
				if (vs1[0] == std::get<0>(tip_splits[tid]) && i == std::get<2>(tip_splits[tid])) { vs1.push_back(get<1>(tip_splits[tid])); break; }
			if (find(hybrid_standard.Hs[pids[0]].vs.begin(), hybrid_standard.Hs[pids[0]].vs.end(), vs1[0]) != hybrid_standard.Hs[pids[0]].vs.end())
				swap(pids[0], pids[1]);
			//p0
			h.id = pure_hex_mesh.Hs.size();
			for (uint32_t j = 0; j < 4; j++) {
				h.vs[j] = hybrid_standard.Hs[pids[0]].vs[j];
				if (j == 0) h.vs[j + 4] = vs0[1];
				else h.vs[j + 4] = hybrid_standard.Hs[pids[0]].vs[j + 3];
			}
			pure_hex_mesh.Hs.push_back(h);
			//p1
			h.id = pure_hex_mesh.Hs.size();
			for (uint32_t j = 0; j < 4; j++) {
				h.vs[j] = hybrid_standard.Hs[pids[1]].vs[j];
				if (j == 0) h.vs[j + 4] = vs1[1];
				else h.vs[j + 4] = hybrid_standard.Hs[pids[1]].vs[j + 3];
			}
			pure_hex_mesh.Hs.push_back(h);
			//hex
			h.id = pure_hex_mesh.Hs.size();
			h.vs = hybrid_standard.Hs[hid].vs;
			for (auto &vid : h.vs)if (vid == vs0[0]) vid = vs0[1]; else if (vid == vs1[0])vid = vs1[1];
			pure_hex_mesh.Hs.push_back(h);


		}
		else if (nfs_entire.size() == 1 && nfs_all.size() == 1) {
			//cout << "nfs_entire 1" << endl;
			Base_Cell &bc0 = bcells[nfs_entire[0]];
			int which_index = -1;
			for (uint32_t j = 0; j < 4; j++) {
				int eid = hybrid_standard.Fs[nfs_entire[0]].es[j];
				if (eid == i) { which_index = j; break; }
			}
			int pid = bc0.prism_ids[which_index];
			int hid = -1; vector<int> pids, fids;
			for (auto fid : hybrid_standard.Hs[pid].fs) {
				if (find(hybrid_standard.Fs[fid].es.begin(), hybrid_standard.Fs[fid].es.end(), i) != hybrid_standard.Fs[fid].es.end()) {
					for (auto nhid : hybrid_standard.Fs[fid].neighbor_hs)
						if (h_type[nhid] == Element_Type::Hexahedral && bc0.hex_id != nhid) hid = nhid;
				}
			}
			if (hid == -1)continue;
			for (auto fid : hybrid_standard.Hs[hid].fs)for (auto nhid : hybrid_standard.Fs[fid].neighbor_hs)
				if (h_type[nhid] == Element_Type::Slab) {
					pids.push_back(nhid); fids.push_back(fid);
				}
			if (pids.size() != 2) {
				cout << "One side slab, check!" << endl;
				continue;
			}
			if (h_tag[pids[0]]) {
				h_tag[pids[0]] = h_tag[pids[1]] = h_tag[hid] = false;
			}
			else continue;

			vector<vector<int>> vs_layers(5);
			vs_layers[0].resize(2);
			vs_layers[1].resize(4);
			vs_layers[2].resize(4);
			vs_layers[3].resize(2);

			vs_layers[1][0] = hybrid_standard.Es[i].vs[0];
			vs_layers[1][1] = hybrid_standard.Es[i].vs[1];

			for (auto vfid : hybrid_standard.Vs[vs_layers[1][0]].neighbor_fs) {
				if (find(hybrid_standard.Hs[hid].fs.begin(), hybrid_standard.Hs[hid].fs.end(), vfid) != hybrid_standard.Hs[hid].fs.end()) {
					vector<uint32_t> &fvs = hybrid_standard.Fs[vfid].vs;
					if (find(fvs.begin(), fvs.end(), vs_layers[1][1]) != fvs.end())continue;
					for (uint32_t j = 0; j < 4; j++)if (fvs[j] == vs_layers[1][0]) vs_layers[1][2] = fvs[(j + 2) % 4];
				}
			}
			for (auto vfid : hybrid_standard.Vs[vs_layers[1][1]].neighbor_fs) {
				if (find(hybrid_standard.Hs[hid].fs.begin(), hybrid_standard.Hs[hid].fs.end(), vfid) != hybrid_standard.Hs[hid].fs.end()) {
					vector<uint32_t> &fvs = hybrid_standard.Fs[vfid].vs;
					if (find(fvs.begin(), fvs.end(), vs_layers[1][0]) != fvs.end())continue;
					for (uint32_t j = 0; j < 4; j++)if (fvs[j] == vs_layers[1][1]) vs_layers[1][3] = fvs[(j + 2) % 4];
				}
			}
			int eid1 = -1;
			vector<uint32_t> es00 = hybrid_standard.Vs[vs_layers[1][2]].neighbor_es, es11 = hybrid_standard.Vs[vs_layers[1][3]].neighbor_es, sharedes;
			sort(es00.begin(), es00.end()); sort(es11.begin(), es11.end());
			set_intersection(es00.begin(), es00.end(), es11.begin(), es11.end(), back_inserter(sharedes));
			if (!sharedes.size()) { cout << "bug" << endl; }
			eid1 = sharedes[0];
			edge_tag[eid1] = false;


			vector<int> nfs_entire_, nfs_all_;
			for (auto nfid : hybrid_standard.Es[eid1].neighbor_fs)if (tip_quads[nfid]) {
				nfs_all_.push_back(nfid);
				if (q_tag[nfid] == Quad_Tag::Entire_Split || q_tag[nfid] == Quad_Tag::One_Split_x
					|| q_tag[nfid] == Quad_Tag::One_Split_y || q_tag[nfid] == Quad_Tag::No_Split) nfs_entire_.push_back(nfid);
			}
			if (!(nfs_entire_.size() == 1 && nfs_all_.size() == 1)) { 
				cout << "BUG" << endl; 
				cout << nfs_entire_.size() << "" << nfs_all_.size() << endl;
				continue; 
			}

			Base_Cell &bc1 = bcells[nfs_entire_[0]];

			vector<int> ids0, ids1;
			for (uint32_t j = 0; j < 2; j++) {
				int vid = vs_layers[1][j];
				if (find(bc0.top_qvs.begin(), bc0.top_qvs.end(), vid) != bc0.top_qvs.end()) {
					ids0.push_back(find(bc0.top_qvs.begin(), bc0.top_qvs.end(), vid) - bc0.top_qvs.begin());
				}
			}
			for (uint32_t j = 0; j < 2; j++) {
				int vid = vs_layers[1][j + 2];
				if (find(bc1.top_qvs.begin(), bc1.top_qvs.end(), vid) != bc1.top_qvs.end()) {
					ids1.push_back(find(bc1.top_qvs.begin(), bc1.top_qvs.end(), vid) - bc1.top_qvs.begin());
				}
			}
			if ((ids0[0]+1)%4 == ids0[1]) {
				vs_layers[2][0] = bc0.split_mvs[ids0[0]][0];
				vs_layers[2][1] = bc0.split_mvs[ids0[0]][1];
				vs_layers[4].insert(vs_layers[4].end(), bc0.base_bvs[ids0[0]].begin(), bc0.base_bvs[ids0[0]].end());
				vs_layers[4].push_back(bc0.base_bvs[ids0[1]][0]);
			}
			else {
				vs_layers[2][0] = bc0.split_mvs[ids0[1]][1];
				vs_layers[2][1] = bc0.split_mvs[ids0[1]][0];
				vs_layers[4].push_back(bc0.base_bvs[ids0[0]][0]);
				vs_layers[4].push_back(bc0.base_bvs[ids0[1]][2]);
				vs_layers[4].push_back(bc0.base_bvs[ids0[1]][1]);
				vs_layers[4].push_back(bc0.base_bvs[ids0[1]][0]);
			}
			if ((ids1[0]+1)%4 == ids1[1]) {
				vs_layers[2][2] = bc1.split_mvs[ids1[0]][0];
				vs_layers[2][3] = bc1.split_mvs[ids1[0]][1];
			}
			else {
				vs_layers[2][2] = bc1.split_mvs[ids1[1]][1];
				vs_layers[2][3] = bc1.split_mvs[ids1[1]][0];
			}

			vector<uint32_t> &vs0 = hybrid_standard.Fs[fids[0]].vs, &vs1 = hybrid_standard.Fs[fids[1]].vs;
			if (find(vs0.begin(), vs0.end(), vs_layers[1][0]) != vs0.end()) {
				for (auto vid : vs0)if (vid != vs_layers[1][0] && vid != vs_layers[1][2] && vid != vs_layers[4][1])vs_layers[0][0] = vid;
				for (auto vid : vs1)if (vid != vs_layers[1][1] && vid != vs_layers[1][3] && vid != vs_layers[4][2])vs_layers[0][1] = vid;
			}
			else {
				for (auto vid : vs1)if (vid != vs_layers[1][0] && vid != vs_layers[1][2] && vid != vs_layers[4][1])vs_layers[0][0] = vid;
				for (auto vid : vs0)if (vid != vs_layers[1][1] && vid != vs_layers[1][3] && vid != vs_layers[4][2])vs_layers[0][1] = vid;
			}

			for (uint32_t j = 0; j < 2; j++) {//vs_layers[4][1],vs_layers[4][2]
				Hybrid_V v;
				v.id = pure_hex_mesh.Vs.size();
				//coords TO-DO
				v.v.resize(3, 0);
				vector<double> v0 = hybrid_standard.Vs[vs_layers[0][j]].v, v1 = hybrid_standard.Vs[vs_layers[4][j + 1]].v,
					v2 = hybrid_standard.Vs[vs_layers[4][j + 2]].v;
				Vector3d t2d_dir, edge_dir, edge_dirnd;
				for (uint32_t k = 0; k < 3; k++) {
					t2d_dir[k] = v0[k] - v1[k];
					edge_dir[k] = v2[k] - v1[k];
					if(j==1) edge_dir[k] = v1[k] - v2[k];
				}
				edge_dirnd = edge_dir.normalized();
				double l = t2d_dir[0] * edge_dirnd[0] + t2d_dir[1] * edge_dirnd[1] + t2d_dir[2] * edge_dirnd[2];
				for (uint32_t k = 0; k < 3; k++) v.v[k] = v1[k] + w_eb *(t2d_dir[k] - l* edge_dirnd[k]) + 0.3 * edge_dir[k];
				vs_layers[3][j] = v.id;
				pure_hex_mesh.Vs.push_back(v);
				nv_tag.push_back(NV_type::NRV);
				hex2Octree_map.push_back(hex2Octree_map[vs_layers[4][j + 1]]);
			}

			//bottom 3
			h.id = pure_hex_mesh.Hs.size();
			h.vs[0] = vs_layers[1][0];
			h.vs[1] = vs_layers[4][0];
			h.vs[2] = vs_layers[1][2];
			h.vs[3] = vs_layers[0][0];

			h.vs[4] = vs_layers[2][0];
			h.vs[5] = vs_layers[4][1];
			h.vs[6] = vs_layers[2][2];
			h.vs[7] = vs_layers[3][0];
			pure_hex_mesh.Hs.push_back(h);

			h.id = pure_hex_mesh.Hs.size();
			h.vs[0] = vs_layers[2][0];
			h.vs[1] = vs_layers[4][1];
			h.vs[2] = vs_layers[2][2];
			h.vs[3] = vs_layers[3][0];

			h.vs[4] = vs_layers[2][1];
			h.vs[5] = vs_layers[4][2];
			h.vs[6] = vs_layers[2][3];
			h.vs[7] = vs_layers[3][1];
			pure_hex_mesh.Hs.push_back(h);

			h.id = pure_hex_mesh.Hs.size();
			h.vs[0] = vs_layers[2][1];
			h.vs[1] = vs_layers[4][2];
			h.vs[2] = vs_layers[2][3];
			h.vs[3] = vs_layers[3][1];

			h.vs[4] = vs_layers[1][1];
			h.vs[5] = vs_layers[4][3];
			h.vs[6] = vs_layers[1][3];
			h.vs[7] = vs_layers[0][1];
			pure_hex_mesh.Hs.push_back(h);
			//top 2
			h.id = pure_hex_mesh.Hs.size();
			h.vs[0] = vs_layers[0][0];
			h.vs[1] = vs_layers[0][1];
			h.vs[2] = vs_layers[1][3];
			h.vs[3] = vs_layers[1][2];

			h.vs[4] = vs_layers[3][0];
			h.vs[5] = vs_layers[3][1];
			h.vs[6] = vs_layers[2][3];
			h.vs[7] = vs_layers[2][2];
			pure_hex_mesh.Hs.push_back(h);

			h.id = pure_hex_mesh.Hs.size();
			h.vs[0] = vs_layers[0][0];
			h.vs[1] = vs_layers[0][1];
			h.vs[2] = vs_layers[1][1];
			h.vs[3] = vs_layers[1][0];

			h.vs[4] = vs_layers[3][0];
			h.vs[5] = vs_layers[3][1];
			h.vs[6] = vs_layers[2][1];
			h.vs[7] = vs_layers[2][0];
			pure_hex_mesh.Hs.push_back(h);
		}
	}
	for (uint32_t i = 0; i < h_tag.size(); i++) {
		if (!h_tag[i]) continue;
		Hybrid h;
		h.id = pure_hex_mesh.Hs.size();
		h.vs = hybrid_standard.Hs[i].vs;
		if (h.vs.size() != 8) {
			cout <<endl<< i << " " << h_type[i] << endl;
			for(auto vid: h.vs) cout << " "<<vid;
			continue;
		}
		pure_hex_mesh.Hs.push_back(h);
	}
	pure_hex_mesh.V.resize(3, pure_hex_mesh.Vs.size());
	for (uint32_t i = 0; i < pure_hex_mesh.Vs.size(); i++) {
		pure_hex_mesh.V(0, i) = pure_hex_mesh.Vs[i].v[0];
		pure_hex_mesh.V(1, i) = pure_hex_mesh.Vs[i].v[1];
		pure_hex_mesh.V(2, i) = pure_hex_mesh.Vs[i].v[2];
	}
	cout << "finished pure hex-meshing: V H"<< pure_hex_mesh.Vs.size()<<" "<< pure_hex_mesh.Hs.size() << endl;
}

void grid_hex_meshing_bijective::clean_hex_mesh(Mesh &tmi, Mesh_Domain &md) {
	
	auto &mi = md.mesh_entire;
	reorder_hex_mesh(mi);

	MatrixXd Ps(mi.Hs.size(),3);
	for (uint32_t i = 0; i < mi.Hs.size(); i++) {
		Eigen::MatrixXd VV(3, 8);
		for (int j = 0; j < 8;j++) {
			auto &vid  = mi.Hs[i].vs[j];
			VV.col(j) = mi.V.col(vid);
		}
		Ps.row(i) = (VV.rowwise().maxCoeff()+ VV.rowwise().minCoeff())/2;
	}
	VectorXd signed_dis(mi.Hs.size());

	points_inside_mesh(Ps, tmi, signed_dis);

	md.H_flag.resize(mi.Hs.size());
	std::fill(md.H_flag.begin(), md.H_flag.end(), false);
	for (uint32_t j = 0; j < signed_dis.size(); j++) if (signed_dis[j] < 0) { md.H_flag[j] = true; }

	tagging_uneven_element(mi, md.H_flag);
	re_indexing_connectivity(mi, md.H_flag, md.mesh_subA, md.V_map, md.V_map_reverse, md.H_map, md.H_map_reverse);

	if (!md.mesh_subA.Hs.size()) 
	{ 
		cout << "no elements inside the object, exit";
		return; 
	}
	//remove non-manifold edge and vertex
	clean_non_manifold_ve(mi, md.mesh_subA, md.V_map, md.V_map_reverse, md.H_map, md.H_map_reverse, signed_dis, md.H_flag);

	drop_small_pieces(md);

	if (args.scaffold_type == 2 || args.scaffold_type == 3)
		scaffold(md);

	for(auto&f:md.mesh_entire.Fs)
	{
		if(
			(f.boundary && md.H_flag[f.neighbor_hs[0]])||
			!f.boundary && (md.H_flag[f.neighbor_hs[0]] !=md.H_flag[f.neighbor_hs[1]])
		)
		{
			f.on_medial_surface = true;
			for(auto &vid: f.vs)
				md.mesh_entire.Vs[vid].on_medial_surface = true;
		}
	}
}
void grid_hex_meshing_bijective::tagging_uneven_element(const Mesh &mi, vector<bool> &H_flag) {

	for (int i = 0; i < H_flag.size(); i++) {

		int bn = 0, in = 0;
		if (H_flag[i]) {
			for (const auto fid : mi.Hs[i].fs) {
				if (mi.Fs[fid].neighbor_hs.size() == 1)bn++;
				else if (!H_flag[mi.Fs[fid].neighbor_hs[0]] || !H_flag[mi.Fs[fid].neighbor_hs[1]])bn++;
				else if (H_flag[mi.Fs[fid].neighbor_hs[0]] && H_flag[mi.Fs[fid].neighbor_hs[1]])in++;
			}
			if (bn == 5 && in == 1)H_flag[i] = false;
		}
		else {
			for (const auto fid : mi.Hs[i].fs) {
				if (mi.Fs[fid].neighbor_hs.size() == 1)bn++;
				else if (H_flag[mi.Fs[fid].neighbor_hs[0]] || H_flag[mi.Fs[fid].neighbor_hs[1]])bn++;
				else if (!H_flag[mi.Fs[fid].neighbor_hs[0]] && !H_flag[mi.Fs[fid].neighbor_hs[1]])in++;
			}
			if (bn == 5 && in == 1)H_flag[i] = true;
		}
	}
}
void grid_hex_meshing_bijective::clean_non_manifold_ve(Mesh &mi, Mesh &hmi_local, vector<int> &V_map, vector<int> &V_map_reverse,
	vector<int> &H_map, vector<int> &H_map_reverse, VectorXd &signed_dis, vector<bool> &H_flag) {
	while (true) {
		//cout << "start cleaning non-manifold v" << endl;
		bool changed = false;

		vector<bool> v_tag(hmi_local.Vs.size(), false);
		for (auto v : hmi_local.Vs) {
			if (!v.boundary)continue;
			if (v_tag[v.id])continue; v_tag[v.id] = true;

			bool non_manifold = false;

			vector<uint32_t> bfs;
			vector<bool> f_tag_;
			for (auto fid : v.neighbor_fs)if (hmi_local.Fs[fid].boundary)bfs.push_back(fid);
			f_tag_.resize(bfs.size(), false);
			vector<uint32_t> f_pool; f_pool.push_back(bfs[0]);
			while (true) {
				vector<uint32_t> f_pool_;
				for (auto fid : f_pool) {
					if (f_tag_[find(bfs.begin(), bfs.end(), fid) - bfs.begin()]) continue;
					else f_tag_[find(bfs.begin(), bfs.end(), fid) - bfs.begin()] = true;
					for (auto eid : hmi_local.Fs[fid].es) {
						for (auto nfid : hmi_local.Es[eid].neighbor_fs)
							if (find(bfs.begin(), bfs.end(), nfid) != bfs.end()) {
								if (!f_tag_[find(bfs.begin(), bfs.end(), nfid) - bfs.begin()]) f_pool_.push_back(nfid);
							}
					}
				}
				f_pool.clear();
				if (f_pool_.size())f_pool = f_pool_;
				else  if (find(f_tag_.begin(), f_tag_.end(), false) != f_tag_.end()) {
					non_manifold = true;
					break;
				}
				else break;
				f_pool_.clear();
			}
			if (non_manifold) {
				for (uint32_t i = 0; i < f_tag_.size(); i++) {
					if (!f_tag_[i]) {
						int nhid = hmi_local.Fs[bfs[i]].neighbor_hs[0];
						H_flag[H_map_reverse[nhid]] = false;
						for (auto vid : hmi_local.Hs[nhid].vs) v_tag[vid] = true;
					}
				}
				changed = true;
			}
		}
		//cout << "start cleaning non-manifold e" << endl;
		if (!changed) {
			vector<bool> e_tag(hmi_local.Es.size(), false);
			for (auto e : hmi_local.Es) {
				if (!e.boundary)continue;
				if (e_tag[e.id])continue; e_tag[e.id] = true;

				vector<uint32_t> bfs;
				vector<bool> f_tag_;
				for (auto fid : e.neighbor_fs)if (hmi_local.Fs[fid].boundary)bfs.push_back(fid);
				if (bfs.size() != 2) {
					changed = true;

					int nhid = hmi_local.Fs[bfs[0]].neighbor_hs[0];
					H_flag[H_map_reverse[nhid]] = false;
					for (auto fid : hmi_local.Hs[nhid].fs) for (auto eid : hmi_local.Fs[fid].es)e_tag[eid] = true;
				}
			}
			//cout << "finished cleaning non-manifold e" << endl;
			if (!changed)
				break;
		}
		re_indexing_connectivity(mi, H_flag, hmi_local, V_map, V_map_reverse, H_map, H_map_reverse);
	}
}
void grid_hex_meshing_bijective::drop_small_pieces(Mesh_Domain &md) {
	
	auto &hmi_local = md.mesh_subA;

	vector<vector<uint32_t>> hgroups;
	vector<bool> h_flag_(hmi_local.Hs.size(), false);

	vector<uint32_t> a_group; a_group.push_back(0);
	while (true) {
		for (uint32_t i = 0; i < a_group.size(); i++) {
			int hid = a_group[i];
			if (h_flag_[hid]) continue; else h_flag_[hid] = true;

			for (auto fid : hmi_local.Hs[hid].fs) {
				for (auto nhid : hmi_local.Fs[fid].neighbor_hs)
					if (nhid != hid && !h_flag_[nhid]) a_group.push_back(nhid);
			}
		}
		hgroups.push_back(a_group);
		if (find(h_flag_.begin(), h_flag_.end(), false) != h_flag_.end()) {
			a_group.clear();
			int id = find(h_flag_.begin(), h_flag_.end(), false) - h_flag_.begin();
			a_group.push_back(id);
		}
		else break;
	}

	if (hgroups.size() > 1) {
		int size_max = hgroups[0].size();
		int which = 0;
		for (uint32_t i = 1; i < hgroups.size();i++) 
		if(hgroups[i].size()>size_max){
			size_max = hgroups[i].size();
			which = i;
		}

		fill(h_flag_.begin(), h_flag_.end(), false);

		fill(md.H_flag.begin(), md.H_flag.end(), false);
		for (const auto &hid : hgroups[which])md.H_flag[md.H_map_reverse[hid]] = true;

		re_indexing_connectivity(md.mesh_entire, md.H_flag, md.mesh_subA, md.V_map, md.V_map_reverse, md.H_map, md.H_map_reverse);
	}
}
void grid_hex_meshing_bijective::scaffold(Mesh_Domain &md)
{
	std::vector<bool> S_flag = md.H_flag;
	for (int i = 0; i < args.scaffold_layer; i++) {
		std::vector<bool> S_flag_ (md.H_flag.size(),false);
		for (uint32_t j = 0; j < S_flag.size(); j++) if(S_flag[j]){
			for (const auto &vid : md.mesh_entire.Hs[j].vs)
				for (const auto &nhid : md.mesh_entire.Vs[vid].neighbor_hs)
					S_flag_[nhid] = true;
		}
		S_flag = S_flag_;
	}

	Mesh mi;
	re_indexing_connectivity(md.mesh_entire, S_flag, mi, md.V_map, md.V_map_reverse, md.H_map, md.H_map_reverse);
	md.mesh_entire = mi;
	reorder_hex_mesh_propogation(md.mesh_entire);
	std::vector<bool> H_flag(md.mesh_entire.Hs.size(), false);
	for (int i = 0; i < md.H_map_reverse.size(); i++)
	{
		int rhid = md.H_map_reverse[i];
		if (md.H_flag[rhid])
			H_flag[i] = true;
	}

	md.H_flag = H_flag;
	move_boundary_vertices_back(md, hex2Octree_map);
	}

//boundary projection & feature capture
bool grid_hex_meshing_bijective::surface_mapping(Mesh &tmi, Mesh_Domain &md) 
{
	Mesh_Feature mf_temp;
	vector<bool> Corner_tag;
	vector<vector<uint32_t>> circle2curve_map;
	break_circles(mf_temp, Corner_tag, circle2curve_map);

	//build surface feture graph
	Feature_Graph Tfg;
	
	md.clear_Qfg();	
	md.clear_mf_quad();
	Mesh htri; 
	htri.type = Mesh_type::Tri;
	auto &hqua = md.quad_tree.mesh; 
	hqua.type = Mesh_type::Qua;

	extract_surface_conforming_mesh(md.mesh_subA, htri, md.TV_map, md.TV_map_reverse, md.TF_map, md.TF_map_reverse);
	extract_surface_conforming_mesh(md.mesh_subA, hqua, md.QV_map, md.QV_map_reverse, md.QF_map, md.QF_map_reverse);
	//aabb stree for surface of the hex-mesh
	build_aabb_tree(htri, md.quad_tree);

	cout << "node mapping" << endl;
	if(!node_mapping(mf_temp,Tfg, md))
		return false;
	cout << "curve mapping" << endl;
	curve_mapping(Tfg, md);
	if(!reunion_circles(mf_temp,Corner_tag, circle2curve_map ,md))
		return false;
	cout << "patch mapping" << endl;
	if(!patch_mapping(md))
		return false;
	patch_trees(md);
	
	fc.V_types.resize(md.mesh_entire.Vs.size()); 
	fill(fc.V_types.begin(), fc.V_types.end(), Feature_V_Type::INTERIOR);

	for (uint32_t i = 0; i < md.mf_quad.corners.size();i++) 
		fc.V_types[md.mf_quad.corners[i]] = Feature_V_Type::CORNER;
	for (uint32_t i = 0; i < md.mf_quad.curve_vs.size();i++)
		for (auto vid : md.mf_quad.curve_vs[i]) 
		{
			if (fc.V_types[vid] == Feature_V_Type::CORNER || fc.V_types[vid] == Feature_V_Type::LINE)
				continue;
			fc.V_types[vid] = Feature_V_Type::LINE;
		}
	for (auto v:md.mesh_entire.Vs)
		if (v.on_medial_surface && fc.V_types[v.id] == Feature_V_Type::INTERIOR)
			fc.V_types[v.id] = Feature_V_Type::REGULAR;
	
	vector<bool> V_tag(md.mesh_entire.Vs.size(), false);
	for (auto &R : md.Qfg.Rs) 
	{
		for (const auto & fid : R.tris)
			for (const auto &vid : md.Qfg.mf.tri.Fs[fid].vs)
			{
				auto &rvid = md.V_map_reverse[md.QV_map_reverse[vid]];
				if (!V_tag[rvid] && fc.V_types[rvid] == Feature_V_Type::REGULAR) {
					R.vs.push_back(rvid);
					V_tag[rvid] = true;
				}
			}
	}
 
 	Q_final_fg = md.Qfg;
	GRAPH_MATCHES = md.graph_matches;

	return true;
}
void grid_hex_meshing_bijective::break_circles(Mesh_Feature &mf_temp, vector<bool> &Corner_tag, vector<vector<uint32_t>> &circle2curve_map)
{
	mf_temp.corners = mf.corners;
	mf_temp.corner_curves = mf.corner_curves;
	mf_temp.curve_vs = mf.curve_vs;

	Corner_tag.resize(mf.corners.size(),true);
	circle2curve_map.resize(mf.circles.size());
	for (uint32_t i = 0; i < mf.circles.size(); i++) {
		if (mf.circles[i]) {
			int lenv= mf.curve_vs[i].size();
			int vid0=mf.curve_vs[i][0], vid1 = mf.curve_vs[i][lenv/2];
			mf_temp.corners.push_back(vid0);
			Corner_tag.push_back(false);
			mf_temp.corners.push_back(vid1);
			Corner_tag.push_back(false);
			vector<uint32_t> curve_vs;
			for (auto vid : mf.curve_vs[i])if (vid == vid1) {
				curve_vs.push_back(vid);
				mf_temp.curve_vs[i] = curve_vs;
				curve_vs.clear();
				curve_vs.push_back(vid);
			}else curve_vs.push_back(vid);
			curve_vs.push_back(vid0);
			mf_temp.curve_vs.push_back(curve_vs);
			vector<uint32_t> curve_ids;
			curve_ids.push_back(i);
			curve_ids.push_back(mf_temp.curve_vs.size() - 1);
			mf_temp.corner_curves.push_back(curve_ids);
			mf_temp.corner_curves.push_back(curve_ids);
			circle2curve_map[i] = curve_ids;
		}
		else circle2curve_map[i].push_back(i);
	}
}
bool grid_hex_meshing_bijective::node_mapping(Mesh_Feature &mf_temp, Feature_Graph &Tfg, Mesh_Domain &md)
{
	auto &quad_tree = md.quad_tree;	
	auto &TF_map_reverse = md.TF_map_reverse;
	auto &QV_map_reverse = md.QV_map_reverse;
	auto &QF_map = md.QF_map;
	auto &Qfg = md.Qfg;
	//find corresponding vs of corners on the triangle mesh
	MatrixXd Ps(mf_temp.corners.size(), 3); int num_corner = 0;
	for (auto vid : mf_temp.corners) { Ps.row(num_corner++) = mf.tri.V.col(vid).transpose(); }

	VectorXd signed_dis;
	VectorXi ids;
	MatrixXd VT, NT;
	signed_distance_pseudonormal(Ps, quad_tree.TriV, quad_tree.TriF, quad_tree.tree, quad_tree.TriFN, quad_tree.TriVN, quad_tree.TriEN, quad_tree.TriEMAP, signed_dis, ids, VT, NT);
	//build nodes of the graph for both triangle mesh and quad mesh
	vector<bool> TE_tag(mf.tri.Es.size(), false), QE_tag(quad_tree.mesh.Es.size(), false);
	vector<bool> V_flag(quad_tree.mesh.Vs.size(), true);

	for (uint32_t i = 0; i < mf_temp.corners.size(); i++) {
		Vector3d v = mf.tri.V.col(mf_temp.corners[i]);
		int qid = QF_map[TF_map_reverse[ids[i]]];
		int vid = -1, vid_min = -1; double min_dis = std::numeric_limits<double>::max();
		for (uint32_t j = 0; j < quad_tree.mesh.Fs[qid].vs.size(); j++) {
			int vid_ = quad_tree.mesh.Fs[qid].vs[j];
			if (!V_flag[vid_])continue;//already taken by other corners
			double dis = (v - quad_tree.mesh.V.col(vid_)).norm();
			if (j == 0 || min_dis > dis) {
				vid_min = vid_; min_dis = dis;
				if (quad_tree.mesh.Vs[vid_].neighbor_es.size() >= mf_temp.corner_curves[i].size())
					vid = vid_;
			}
		}
		Feature_Corner c;
		c.id = Qfg.Cs.size();
		c.original = v;
		c.projected = VT.row(i);

		if (vid == -1) {
			if (vid_min == -1) {
				vid_min = quad_tree.mesh.Fs[qid].vs[0];
			}
			for (auto nfid : quad_tree.mesh.Vs[vid_min].neighbor_fs) {
				for (uint32_t j = 0; j < quad_tree.mesh.Fs[nfid].vs.size(); j++) {
					int vid_ = quad_tree.mesh.Fs[nfid].vs[j];
					if (!V_flag[vid_])continue;
					
					double dis = (v - quad_tree.mesh.V.col(vid_)).norm();
					if (quad_tree.mesh.Vs[vid_].neighbor_es.size() >= mf_temp.corner_curves[i].size() && (j == 0 || min_dis > dis)) 
					{ 
						vid = vid_; min_dis = dis; 
					}
				}
			}
			if (vid == -1) {
				auto g_id = md.V_map_reverse[QV_map_reverse[vid_min]];
				vector<uint32_t> nvs = md.mesh_entire.Vs[g_id].neighbor_vs;
				tb_subdivided_cells.insert(tb_subdivided_cells.end(), nvs.begin(), nvs.end());
			}
			else c.vs.push_back(vid);
		}
		else c.vs.push_back(vid);

		for (auto vid : c.vs) {
			V_flag[vid] = false;
		}
		Qfg.Cs.push_back(c);
	}

	if (tb_subdivided_cells.size()) return false;
//Tfg
	for (uint32_t i = 0; i < mf_temp.corners.size(); i++) {
		Feature_Corner tc;
		tc.id = Tfg.Cs.size();
		tc.vs.push_back(mf_temp.corners[i]);
		//neighbor vs
		vector<uint32_t> nts;
		for (auto vid : tc.vs)nts.insert(nts.end(), mf.tri.Vs[vid].neighbor_fs.begin(), mf.tri.Vs[vid].neighbor_fs.end());
		std::sort(nts.begin(), nts.end()); nts.erase(unique(nts.begin(), nts.end()), nts.end());
		face_soup_info(mf.tri, nts, TE_tag, tc.ring_vs);

		Tfg.Cs.push_back(tc);
	}
	Tfg.Ls.resize(mf_temp.curve_vs.size());
	for (uint32_t i = 0; i < Tfg.Cs.size(); i++) {
		Feature_Corner &tc = Tfg.Cs[i];
		tc.neighbor_ls = mf_temp.corner_curves[i];
		for (auto cid : mf_temp.corner_curves[i]) Tfg.Ls[cid].cs.push_back(tc.id);
	}
	for (int i = 0; i < Tfg.Ls.size();i++) {
		Tfg.Ls[i].id = i;
		Tfg.Ls[i].vs = mf_temp.curve_vs[i];
	}
	//corner mismatch check
	for (uint32_t i = 0; i < mf_temp.corners.size(); i++) {
		Feature_Corner &c = Qfg.Cs[i];
		//neighbor vs
		vector<uint32_t> nqs;
		for (auto vid : c.vs)nqs.insert(nqs.end(), quad_tree.mesh.Vs[vid].neighbor_fs.begin(), quad_tree.mesh.Vs[vid].neighbor_fs.end());
		std::sort(nqs.begin(), nqs.end()); nqs.erase(unique(nqs.begin(), nqs.end()), nqs.end());
		vector<uint32_t> ring_vs;
		face_soup_info(quad_tree.mesh, nqs, QE_tag, ring_vs);

		vector<int> wrong_corners;
		for (auto vid : ring_vs) {
			bool is_neighbor = false;
			for (auto vid_ : c.vs) {
				if (find(quad_tree.mesh.Vs[vid_].neighbor_vs.begin(), quad_tree.mesh.Vs[vid_].neighbor_vs.end(), vid) != quad_tree.mesh.Vs[vid_].neighbor_vs.end()) {
					is_neighbor = true; break;
				}
			}
			if (!is_neighbor) continue;
			//conflicting corners, ring_vs with other corners, ring_vs of other corners
			bool wrong_connection = false;
			if (!V_flag[vid]) {
				for (const auto &c_ : Qfg.Cs) if (c_.vs[0] == vid) {
					auto es0 = Tfg.Cs[c.id].neighbor_ls, es1 = Tfg.Cs[c_.id].neighbor_ls;
					vector<uint32_t> sharedes;
					std::sort(es0.begin(), es0.end()); std::sort(es1.begin(), es1.end());
					set_intersection(es0.begin(), es0.end(), es1.begin(), es1.end(), back_inserter(sharedes));
					if (!sharedes.size()) {
						wrong_corners.push_back(c_.vs[0]);
						wrong_connection = true;// break;
					}
				}
				else {
					for (auto ovid : c_.ring_vs)if (vid == ovid) {
						auto es0 = Tfg.Cs[c.id].neighbor_ls, es1 = Tfg.Cs[c_.id].neighbor_ls;
						vector<uint32_t> sharedes;
						std::sort(es0.begin(), es0.end()); std::sort(es1.begin(), es1.end());
						set_intersection(es0.begin(), es0.end(), es1.begin(), es1.end(), back_inserter(sharedes));
						if (!sharedes.size()) {
							wrong_corners.push_back(c_.vs[0]);
							wrong_connection = true; //break;
						}
					}
				}
			}
			if (wrong_connection) continue;
			c.ring_vs.push_back(vid);
		}
		if (c.ring_vs.size() < mf_temp.corner_curves[i].size()) {
			auto g_id = md.V_map_reverse[QV_map_reverse[c.vs[0]]];
			tb_subdivided_cells.push_back(g_id);
			for (auto wc : wrong_corners)
			{
				auto g_id = md.V_map_reverse[QV_map_reverse[wc]];
				tb_subdivided_cells.push_back(g_id);
			} 
		}
		for (auto vid : c.ring_vs)V_flag[vid] = false;
	}
	if (tb_subdivided_cells.size()) return false;

	//corner direction match
	Qfg.Ls.resize(mf_temp.curve_vs.size());
	for (uint32_t i = 0; i < Tfg.Cs.size(); i++) {
		Feature_Corner &tc = Tfg.Cs[i];
		tc.ring_vs_tag.resize(tc.ring_vs.size(), -1);
		tc.neighbor_ls = mf_temp.corner_curves[i];

		vector<uint32_t> vs;
		for (auto cid : mf_temp.corner_curves[i]) {
			vector<uint32_t> &curve = mf_temp.curve_vs[cid];
			if (curve[0] == mf_temp.corners[i]) vs.push_back(curve[1]);
			else if (curve[curve.size() - 1] == mf_temp.corners[i]) vs.push_back(curve[curve.size() - 2]);
		}
		vector<Vector3d> Tdirs; vector<int> lids;
		for (uint32_t j = 0; j < tc.ring_vs.size(); j++) {
			if (find(vs.begin(), vs.end(), tc.ring_vs[j]) != vs.end()) {
				tc.ring_vs_tag[j] = mf_temp.corner_curves[i][find(vs.begin(), vs.end(), tc.ring_vs[j]) - vs.begin()];
				lids.push_back(tc.ring_vs_tag[j]);
				Tdirs.push_back((mf.tri.V.col(tc.ring_vs[j]) - mf.tri.V.col(tc.vs[0])).normalized());
			}
		}
		Feature_Corner &qc = Qfg.Cs[i];
		vector<Vector3d> Qdirs;
		for (uint32_t j = 0; j < qc.ring_vs.size(); j++) Qdirs.push_back((quad_tree.mesh.V.col(qc.ring_vs[j]) - quad_tree.mesh.V.col(qc.vs[0])).normalized());
		//compare Tdirs and Qdirs
		vector<bool> Qdirs_tag(Qdirs.size(), false);
		std::fill(Qdirs_tag.end() - Tdirs.size(), Qdirs_tag.end(), true);
		vector<int> best_set;
		double max_alignment = -std::numeric_limits<double>::infinity();
		do {
			vector<int> cur_set;
			for (uint32_t j = 0; j < Qdirs_tag.size(); j++) if (Qdirs_tag[j]) cur_set.push_back(j);

			int start_id = -1;
			for (uint32_t j = 0; j < cur_set.size(); j++) {
				double cost = 0;
				for (uint32_t k = 0; k<cur_set.size(); k++) cost += Tdirs[k].dot(Qdirs[cur_set[(k + j) % cur_set.size()]]);
				if (cost > max_alignment) {
					start_id = j;
					max_alignment = cost;
				}
			}
			if (start_id != -1) {
				best_set.clear();
				for (uint32_t k = 0; k<cur_set.size(); k++) best_set.push_back(cur_set[(k + start_id) % cur_set.size()]);
			}
		} while (std::next_permutation(Qdirs_tag.begin(), Qdirs_tag.end()));

		qc.ring_vs_tag.resize(qc.ring_vs.size(), -1);
		for (uint32_t j = 0; j < best_set.size(); j++)qc.ring_vs_tag[best_set[j]] = lids[j];
	}
	//corner deep mismatch check
	fill(V_flag.begin(), V_flag.end(), true);
	for (const auto &c : Qfg.Cs) {
		for (const auto &vid : c.vs) V_flag[vid] = false;
		for (const auto &vid : c.ring_vs) V_flag[vid] = false;
	}
	for (const auto &c : Qfg.Cs) {
		vector<int> wrong_corners;
		for (int i = 0; i < c.ring_vs_tag.size();i++) {
			if (c.ring_vs_tag[i] == -1)continue;
			int lid = c.ring_vs_tag[i];
			const auto vid = c.ring_vs[i];
			bool wrong_connection = false;
			if (!V_flag[vid]) {
				for (const auto &c_ : Qfg.Cs) if (c_.vs[0] == vid) {
					if (std::find(c_.ring_vs_tag.begin(), c_.ring_vs_tag.end(), lid) != c_.ring_vs_tag.end()) {
						auto id = std::find(c_.ring_vs_tag.begin(), c_.ring_vs_tag.end(), lid) - c_.ring_vs_tag.begin();
						if (c_.ring_vs[id] != c.vs[0]) {
							wrong_corners.push_back(c_.vs[0]);
							wrong_connection = true;// break;
						}
					}
					else {
						wrong_corners.push_back(c_.vs[0]);
						wrong_connection = true;// break;
					}
				}
				else {
					for (auto ovid : c_.ring_vs)if (vid == ovid) {
						if (std::find(c_.ring_vs_tag.begin(), c_.ring_vs_tag.end(), lid) != c_.ring_vs_tag.end()) {
							auto id = std::find(c_.ring_vs_tag.begin(), c_.ring_vs_tag.end(), lid) - c_.ring_vs_tag.begin();
							if (c_.ring_vs[id] != c.ring_vs[i]) {
								wrong_corners.push_back(c_.vs[0]);
								wrong_connection = true;// break;
							}
						}
						else {
							wrong_corners.push_back(c_.vs[0]);
							wrong_connection = true;// break;
						}
					}
				}
			}
		}
		if (wrong_corners.size()) {
			auto g_id = md.V_map_reverse[QV_map_reverse[c.vs[0]]];
			tb_subdivided_cells.push_back(g_id);
			for (auto wc : wrong_corners) 
			{
				auto g_id = md.V_map_reverse[QV_map_reverse[wc]];
				tb_subdivided_cells.push_back(g_id);
			}
		}
	}
	if (tb_subdivided_cells.size()) return false;
	return true;
}
bool grid_hex_meshing_bijective::curve_mapping(Feature_Graph &Tfg, Mesh_Domain &md)
{
	auto &quad_tree = md.quad_tree;	
	auto &TF_map_reverse = md.TF_map_reverse;
	auto &QF_map = md.QF_map;
	auto &Qfg = md.Qfg;
//setup before line trace
	vector<bool> V_flag(quad_tree.mesh.Vs.size(), true);
	for (auto &c : Qfg.Cs) {
		for (auto vid : c.vs) V_flag[vid] = false;
		for (uint32_t i = 0; i < c.ring_vs_tag.size(); i++) if (c.ring_vs_tag[i] != -1) V_flag[c.ring_vs[i]] = false;
	}

	adjacency_list_t adjacency_list(quad_tree.mesh.Vs.size()), adjacency_list_w; std::vector<weight_t> min_distance; std::vector<vertex_t> previous;
	for (auto &v : quad_tree.mesh.Vs) {
		for (auto &nv : v.neighbor_vs) adjacency_list[v.id].push_back(neighbor(nv, 1));
	}
	adjacency_list_w = adjacency_list;
	for (auto &v : quad_tree.mesh.Vs) {
		if (V_flag[v.id]) continue;
		for (auto &nv : v.neighbor_vs) adjacency_list[v.id].clear();
	}

	//trace lines
	for (uint32_t i = 0; i < Tfg.Ls.size(); i++) {
		Qfg.Ls[i].id = i;
		Qfg.Ls[i].cs = Tfg.Ls[i].cs;
		//start to trace a curve
		vector<uint32_t> &curve = Tfg.Ls[i].vs;
		int start_cid = Tfg.Ls[i].cs[0], start2_vid = -1, end2_vid = -1, end_cid = Tfg.Ls[i].cs[1];

		if (curve[0] == Tfg.Cs[end_cid].vs[0]) swap(start_cid, end_cid);
		int which_id = find(Qfg.Cs[start_cid].ring_vs_tag.begin(), Qfg.Cs[start_cid].ring_vs_tag.end(), i) - Qfg.Cs[start_cid].ring_vs_tag.begin();
		start2_vid = Qfg.Cs[start_cid].ring_vs[which_id];
		which_id = find(Qfg.Cs[end_cid].ring_vs_tag.begin(), Qfg.Cs[end_cid].ring_vs_tag.end(), i) - Qfg.Cs[end_cid].ring_vs_tag.begin();
		end2_vid = Qfg.Cs[end_cid].ring_vs[which_id];

		if (start2_vid == Qfg.Cs[end_cid].vs[0] || Qfg.Cs[start_cid].vs[0] == end2_vid){
			Qfg.Ls[i].vs.push_back(Qfg.Cs[start_cid].vs[0]);
			Qfg.Ls[i].vs.push_back(Qfg.Cs[end_cid].vs[0]);

			for (auto vid : Qfg.Ls[i].vs)adjacency_list[vid].clear();
			continue;
		}
		else if (start2_vid == end2_vid) {
			Qfg.Ls[i].vs.push_back(Qfg.Cs[start_cid].vs[0]);
			Qfg.Ls[i].vs.push_back(start2_vid);
			Qfg.Ls[i].vs.push_back(Qfg.Cs[end_cid].vs[0]);

			for (auto vid : Qfg.Ls[i].vs)adjacency_list[vid].clear();
			continue;
		}
		//sample line, compute weight of the mesh, update adjacency_list, run dijstra
		for (auto nv : quad_tree.mesh.Vs[start2_vid].neighbor_vs) adjacency_list[start2_vid].push_back(neighbor(nv, 1));
		for (auto nv : quad_tree.mesh.Vs[end2_vid].neighbor_vs) adjacency_list[end2_vid].push_back(neighbor(nv, 1));

		double sample_density = voxel_size * pow(2, STOP_EXTENT_MAX) * 0.5;
		if(!args.octree)
			sample_density = voxel_size * 0.5;

		vector<Vector3d> samples;
		int l_len = Tfg.Ls[i].vs.size() - 1;
		double len_cur = 0; int num_segment = 0;
		for (uint32_t j = 0; j < l_len; j++) {
			Vector3d v0 = mf.tri.V.col(Tfg.Ls[i].vs[j]), v1 = mf.tri.V.col(Tfg.Ls[i].vs[j + 1]);
			double dis = (v0 - v1).norm(), t = 0;
			if (dis < 1.0e-15)continue;

			while (len_cur <= num_segment * sample_density && len_cur + dis > num_segment * sample_density) {
				t = (len_cur + dis - num_segment * sample_density) / dis;
				samples.push_back(v0 + (1 - t)*(v1 - v0));
				num_segment++;
			}
			len_cur += dis;
		}
		Qfg.Ls[i].guiding_v = samples;

		MatrixXd Ps(samples.size(), 3);
		VectorXd signed_dis;
		VectorXi ids;
		MatrixXd VT, NT;
		for (uint32_t j = 0; j < samples.size(); j++) Ps.row(j) = samples[j].transpose();
		signed_distance_pseudonormal(Ps, quad_tree.TriV, quad_tree.TriF, quad_tree.tree, quad_tree.TriFN, quad_tree.TriVN, quad_tree.TriEN, quad_tree.TriEMAP, signed_dis, ids, VT, NT);
		vector<int> l_source;
		for (uint32_t j = 0; j < samples.size(); j++) {
			Vector3d &v = samples[j];
			int qid = QF_map[TF_map_reverse[ids[j]]];
			int vid = -1; double min_dis;
			for (uint32_t k = 0; k < quad_tree.mesh.Fs[qid].vs.size(); k++) {
				int vid_ = quad_tree.mesh.Fs[qid].vs[k];
				double dis = (v - quad_tree.mesh.V.col(vid_)).norm();
				if (k == 0 || min_dis > dis) { vid = vid_; min_dis = dis; }
			}
			l_source.push_back(vid);
		}
		Qfg.Ls[i].guiding_vs = l_source;
		DijkstraComputePaths(l_source, adjacency_list_w, min_distance, previous);
		for (uint32_t j = 0; j < adjacency_list.size(); j++) for (uint32_t k = 0; k < adjacency_list[j].size(); k++) {
			adjacency_list[j][k].weight = 2 * max(min_distance[j], min_distance[adjacency_list[j][k].target]);
		}

		DijkstraComputePaths(start2_vid, adjacency_list, min_distance, previous);
		vector<int> path = DijkstraGetShortestPathTo(end2_vid, previous);

		Qfg.Ls[i].vs.push_back(Qfg.Cs[start_cid].vs[0]);
		Qfg.Ls[i].vs.insert(Qfg.Ls[i].vs.end(), path.begin(), path.end());

		Qfg.Ls[i].vs.push_back(Qfg.Cs[end_cid].vs[0]);

		for (auto vid : Qfg.Ls[i].vs)adjacency_list[vid].clear();
	}
	return true;
}
bool grid_hex_meshing_bijective::reunion_circles(
	Mesh_Feature &mf_temp, vector<bool> &Corner_tag, vector<vector<uint32_t>> &circle2curve_map, Mesh_Domain &md)
{
	auto &quad_tree = md.quad_tree;	
	auto &TF_map_reverse = md.TF_map_reverse;
	auto &QV_map_reverse = md.QV_map_reverse;
	auto &QF_map = md.QF_map;
	auto &Qfg = md.Qfg;
	auto &Qfg_sur = md.Qfg_sur;
	auto &mf_quad = md.mf_quad;
	//compute Qfg
	for (uint32_t i = 0; i < Qfg.Cs.size(); i++) if (Corner_tag[i]) mf_quad.corners.push_back(Qfg.Cs[i].vs[0]);
	mf_quad.corner_curves = mf.corner_curves;
	mf_quad.curve_vs.resize(mf.curve_vs.size());
	mf_quad.curve_es.resize(mf.curve_es.size());
	mf_quad.broken_curves.resize(mf.curve_es.size());
	std::fill(mf_quad.broken_curves.begin(), mf_quad.broken_curves.end(), false);
	mf_quad.circles = mf.circles;

	mf_quad.ave_length = average_edge_length(quad_tree.mesh);
	mf_quad.tri = quad_tree.mesh;

	function<bool(Mesh &, vector<uint32_t> &, vector<uint32_t> &, const bool&)> VSES = [&](Mesh &mesh, vector<uint32_t> &vs, vector<uint32_t> &es, const bool &circle)->bool {
		int vlen = vs.size(), vlen_ = vlen;
		if(!circle) vlen--;
		es.clear();
		bool broken_curve = false;
		for (uint32_t i = 0; i < vlen; i++) {
			int v0 = vs[i], v1 = vs[(i + 1) % vlen_];
			vector<uint32_t> sharedes, es0 = mesh.Vs[v0].neighbor_es, es1 = mesh.Vs[v1].neighbor_es;
			std::sort(es0.begin(), es0.end()); std::sort(es1.begin(), es1.end());
			set_intersection(es0.begin(), es0.end(), es1.begin(), es1.end(), back_inserter(sharedes));
			if (sharedes.size()) es.push_back(sharedes[0]);
			else { 
				broken_curve = true;
			}
		}
		return broken_curve;
	};

	for (uint32_t i = 0; i<circle2curve_map.size(); i++) {
		if (circle2curve_map[i].size()==2) {
			//cout << "circular curve " << i << endl;
			vector<uint32_t> vs0, es0, vs1, es1;
			vs0 = Qfg.Ls[circle2curve_map[i][0]].vs;
			vs1 = Qfg.Ls[circle2curve_map[i][1]].vs;
			if (VSES(quad_tree.mesh, vs0, es0, false)) mf_quad.broken_curves[i] = true;
			if (VSES(quad_tree.mesh, vs1, es1, false)) mf_quad.broken_curves[i] = true;
			mf_quad.curve_vs[i] = vs0;
			mf_quad.curve_es[i] = es0;
			if (vs0[0] == vs1[0]) {
				for (uint32_t j = vs1.size() - 2; j > 1; j--) mf_quad.curve_vs[i].push_back(vs1[j]);
				reverse(es0.begin(), es0.end());
				mf_quad.curve_es[i].insert(mf_quad.curve_es[i].end(), es1.begin(), es1.end());
			}
			else if (vs0[0] == vs1[vs1.size() - 1]) {
				for (uint32_t j = 1; j <vs1.size() - 1; j++) mf_quad.curve_vs[i].push_back(vs1[j]);
				mf_quad.curve_es[i].insert(mf_quad.curve_es[i].end(), es1.begin(), es1.end());
			}
			else cout << "Bug!!!" << endl;
		}
		else {
			//cout << "non-circular curve " << i << endl;
			mf_quad.curve_vs[i] = Qfg.Ls[i].vs;
			if (VSES(quad_tree.mesh, mf_quad.curve_vs[i], mf_quad.curve_es[i], false))
				mf_quad.broken_curves[i] = true;
		}
	}

	Qfg_sur = Qfg;
	Qfg.Cs.clear(); Qfg.Ls.clear(); Qfg.Rs.clear();

	build_feature_graph(mf_quad, Qfg);

	//handle broken curves
	vector<bool> V_flag(md.mesh_subA.Vs.size(), false);
	for (int i = 0; i < mf_quad.broken_curves.size();i++) {
		if (!mf_quad.broken_curves[i]) continue;

		for (auto vid : mf_quad.curve_vs[i])V_flag[QV_map_reverse[vid]] = true;

		vector<int32_t> guiding_vs;
		for(auto lid: circle2curve_map[i]) guiding_vs.insert(guiding_vs.end(),Qfg_sur.Ls[lid].guiding_vs.begin(), Qfg_sur.Ls[lid].guiding_vs.end());
		
		for (auto vid : guiding_vs)
		{
			if(!V_flag[QV_map_reverse[vid]])
			{
				auto g_id = md.V_map_reverse[QV_map_reverse[vid]];
				tb_subdivided_cells.push_back(g_id);
			}
		}
	}
	if (tb_subdivided_cells.size()) {
		cout << "broen curves!!!  " << tb_subdivided_cells.size() << endl;
		return false;
	}

	//several curves share the same non-corner vertex
	fc.V_types.resize(md.mesh_subA.Vs.size()); fill(fc.V_types.begin(), fc.V_types.end(), Feature_V_Type::INTERIOR);
	for (auto c: Qfg.Cs) fc.V_types[QV_map_reverse[c.vs[0]]] = Feature_V_Type::CORNER;
	for (auto l : Qfg.Ls) for (auto vid : l.vs) if(fc.V_types[vid] != Feature_V_Type::INTERIOR && fc.V_types[vid] != Feature_V_Type::CORNER){
		auto g_id = md.V_map_reverse[QV_map_reverse[vid]];
		tb_subdivided_cells.push_back(g_id);
		fc.V_types[QV_map_reverse[vid]] = Feature_V_Type::LINE;
	}

	if (tb_subdivided_cells.size())return false;

	//update mf_quad
	for (auto &vid: mf_quad.corners) vid = md.V_map_reverse[QV_map_reverse[vid]];
	for(auto &vs:mf_quad.curve_vs) for (auto &vid : vs) vid = md.V_map_reverse[QV_map_reverse[vid]];
	for (uint32_t i = 0; i < mf_quad.curve_es.size(); i++) 
		VSES(md.mesh_entire, mf_quad.curve_vs[i], mf_quad.curve_es[i], mf_quad.circles[i]);

	return true;
}
bool grid_hex_meshing_bijective::patch_mapping(Mesh_Domain &md)
{
	auto &Qfg = md.Qfg;	
	auto &Qfg_sur =md.Qfg_sur;
	auto &graph_matches = md.graph_matches;
	//matching graph fg & Qfg!
	graph_matches.clear();
	graph_matches.resize(Qfg.Rs.size());
	//1. compare edges and corners
	for (auto r : Qfg.Rs) {
		//find its corresponding patches in fg
		vector<uint32_t> ls = r.ls;

		if (!ls.size()) {
			graph_matches[r.id].push_back(0);
			continue;
		}
		std::sort(ls.begin(), ls.end());
		bool found = false;
		for (auto l : ls) {
			for (auto trid : fg.Ls[l].neighbor_rs) {
				vector<uint32_t> tls = fg.Rs[trid].ls;
				if (tls.size() != ls.size()) continue;
				std::sort(tls.begin(), tls.end());
				if (std::equal(ls.begin(), ls.end(), tls.begin())) {
					graph_matches[r.id].push_back(trid);
					found = true;
				}
			}
		}
		sort(graph_matches[r.id].begin(), graph_matches[r.id].end());
		graph_matches[r.id].erase(unique(graph_matches[r.id].begin(), graph_matches[r.id].end()), graph_matches[r.id].end());
	}

	cout << "start to find bad lines!" << endl;
	//kill miss match because of shared edges! 
	vector<uint32_t> lines, regions;
	for (auto r : Qfg.Rs) {
		if (graph_matches[r.id].size())continue;
		lines.insert(lines.end(), r.ls.begin(), r.ls.end());
		regions.push_back(r.id);
		std::cout<<"missed regions: "<<r.id<<std::endl;
	}
	Feature_Graph tfg_temp = fg, tfg_temp2 = fg;
	tfg_temp.Ls.clear(); tfg_temp.Rs.clear();
	tfg_temp2.Ls.clear(); tfg_temp2.Rs.clear();
	vector<bool> r_flag(fg.Rs.size(), false);
	for (auto m : graph_matches) for (auto rid : m)r_flag[rid] = true;

	vector<uint32_t> tlines, tregions; 
	for (auto r : fg.Rs)if (!r_flag[r.id]) {
		tlines.insert(tlines.end(), r.ls.begin(), r.ls.end());
		tregions.push_back(r.id);
	}
	vector<bool> tl_flag(fg.Ls.size(), false);
	for (auto lid : tlines)if (tl_flag[lid])tfg_temp2.Ls.push_back(fg.Ls[lid]); else {
		tfg_temp.Ls.push_back(fg.Ls[lid]);
		tl_flag[lid] = true;
	}

	for (auto l : tfg_temp2.Ls)for (auto vid : Qfg_sur.Ls[l.id].guiding_vs) {
		auto g_id = md.V_map_reverse[md.QV_map_reverse[vid]];
		tb_subdivided_cells.push_back(g_id);
	}
	std::cout<<"line->tb cells: "<<tb_subdivided_cells.size()<<std::endl;
	if(tb_subdivided_cells.size()) return false;
	//kill miss match because of connecting patches
	vector<bool> V_flag(Qfg.mf.tri.Vs.size(), false);
	for (auto c : Qfg.Cs)V_flag[c.id] = true;
	for (auto l : Qfg.Ls)for (auto lvid : l.vs)V_flag[lvid] = true;
	for (auto rid : regions) {
		for (auto fid : Qfg.Rs[rid].tris)for (auto vid : Qfg.mf.tri.Fs[fid].vs)if (!V_flag[vid]) {
			auto g_id = md.V_map_reverse[md.QV_map_reverse[vid]];
			tb_subdivided_cells.push_back(g_id);
			V_flag[vid] = true;
		}
	}
	std::cout<<"patch->tb cells: "<<tb_subdivided_cells.size()<<std::endl;
	if (tb_subdivided_cells.size())return false;

	for (auto r : Qfg.Rs) {
		//find its corresponding patches in fg
		if (graph_matches[r.id].size() > 1) {
			cout << r.id << " ambiguous matching!" << endl;
			continue;
		}
	}
	cout << "patch matched!" << endl;
	return true;
}
void grid_hex_meshing_bijective::patch_trees(Mesh_Domain &md) {
	
	auto &graph_matches = md.graph_matches;
	auto &tri_Trees = md.tri_Trees;
	// build aabbtrees
	tri_Trees.clear();
	tri_Trees.resize(graph_matches.size());

	for (uint32_t j = 0; j < graph_matches.size(); j++) {
		auto &R = graph_matches[j];
		vector<bool> V_tag(mf.tri.Vs.size(), false), T_tag(mf.tri.Fs.size(), false);
		for (auto rid : R)for (auto tid : fg.Rs[rid].tris) {
			T_tag[tid] = true;
			for (auto vid : mf.tri.Fs[tid].vs)V_tag[vid] = true;
		}
		int vn = 0; vector<int> v_map(mf.tri.Vs.size(), -1);
		for (uint32_t i = 0; i < V_tag.size(); i++)if (V_tag[i])v_map[i] = vn++;
		Mesh tri_mesh;
		tri_mesh.V.resize(3, vn); vn = 0;
		for (uint32_t i = 0; i < V_tag.size(); i++)if (V_tag[i]) tri_mesh.V.col(vn++) = mf.tri.V.col(i);
		for (uint32_t i = 0; i < T_tag.size(); i++)if (T_tag[i]) {
			tri_mesh.Fs.push_back(mf.tri.Fs[i]);
			for (auto &vid : tri_mesh.Fs[tri_mesh.Fs.size() - 1].vs) vid = v_map[vid];
		}
		
		tri_Trees[j].v_map = v_map;
		build_aabb_tree(tri_mesh, tri_Trees[j]);
	}
}

//deformation
bool grid_hex_meshing_bijective::deformation(Mesh_Domain &md) {

	auto & m = md.mesh_entire;
	scaled_jacobian(m, mq);
	std::cout << "before deformation: minimum scaled J: " << mq.min_Jacobian << " average scaled J: " << mq.ave_Jacobian << endl;

	vector<bool> Huntangle_flag(m.Hs.size(), false), H_flag(m.Hs.size(), false), H_inout_tag(m.Hs.size(), true);
	vector<uint32_t> Hids;
	for (uint32_t i = 0; i < m.Hs.size(); i++)Hids.push_back(i);

	double MESHRATIO = 1;
	double LAMDA_FEATURE_PROJECTION = MESHRATIO * 1e+0;
	double LAMDA_GLUE_BOUND = MESHRATIO * 1e+6;
	double LAMDA_FEATURE_PROJECTION_BOUND = MESHRATIO * 1e+8;

	ts = Tetralize_Set();
	ts.V = m.V.transpose();
	ts.T.resize(m.Hs.size() * 8, 4);
	Vector4i t;
	for (auto &h : m.Hs) {
		for (uint32_t i = 0; i < 8; i++) {
			for (uint32_t j = 0; j < 4; j++) t[j] = h.vs[hex_tetra_table[i][j]];
			ts.T.row(h.id * 8 + i) = t;
		}
	}
	//smooth feature only
	fc = Feature_Constraints();
	fc.V_types.resize(m.Vs.size()); fill(fc.V_types.begin(), fc.V_types.end(), Feature_V_Type::INTERIOR);
	fc.V_ids.resize(m.Vs.size());
	fc.RV_type.resize(m.Vs.size());
	fill(fc.RV_type.begin(), fc.RV_type.end(), true);

	int num_corners = 0, num_lines = 0, num_regulars = 0;
	for (auto v : m.Vs) {
		auto i = v.id;
		if (v.on_medial_surface && fc.V_types[i] == Feature_V_Type::INTERIOR) {
			fc.V_types[i] = Feature_V_Type::REGULAR;
			num_regulars++;
		}
	}
	fc.ids_C.resize(num_corners); fc.C.resize(num_corners, 3);
	fc.num_a = num_lines; fc.ids_L.resize(num_lines); fc.Axa_L.resize(num_lines, 3); fc.origin_L.resize(num_lines, 3);
	fc.ids_T.resize(num_regulars); fc.normal_T.resize(num_regulars, 3); fc.dis_T.resize(num_regulars); fc.V_T.resize(num_regulars, 3);
	fc.lamda_C = fc.lamda_L = 0;
	fc.lamda_T = LAMDA_FEATURE_PROJECTION;

	ts.energy_type = SYMMETRIC_DIRICHLET;
	ts.UV = ts.V;
	ts.fc = fc;
	ts.projection = false;
	ts.global = true;
	ts.glue = false;
	ts.lamda_glue = 0;
	ts.lamda_region = 0;
	ts.record_Sequence = false;
	if (args.scaffold_type == 2 || args.scaffold_type == 3)
	{
		ts.known_value_post = true;
		//md.post_index = ts.V.rows();
		ts.post_index = md.post_index;
		ts.post_Variables.resize(ts.V.rows() - md.post_index, 3);
		for (int i = md.post_index; i< m.Vs.size(); i++)
		{
			auto &v = m.Vs[i];
			if (v.boundary)
				ts.post_Variables.row(i - md.post_index) = ts.V.row(i);
		}
	}
	ts.b.resize(0); ts.bc.resize(0, 3); ts.bc.setZero();
	ts.lamda_b = 0;

	optimization opt;
	opt.weight_opt = weight_opt;
	improve_Quality_after_Untangle_Iter_MAX = 5;
	for (uint32_t i = 0; i < improve_Quality_after_Untangle_Iter_MAX; i++) {

		compute_referenceMesh(ts.V, m.Hs, H_inout_tag, Hids, ts.RT, true);

		projection_smooth(m, fc);
		ts.fc = fc;

		opt.slim_m_opt(ts, 3, -1);

		m.V = ts.UV.transpose();
		ts.V = ts.UV;
		
		fc.lamda_T = std::min(LAMDA_FEATURE_PROJECTION * std::max(opt.engery_quality/opt.engery_soft * fc.lamda_T, 1.0), LAMDA_FEATURE_PROJECTION_BOUND);
	}

	for (auto &v : m.Vs) {
		v.v[0] = m.V(0, v.id);
		v.v[1] = m.V(1, v.id);
		v.v[2] = m.V(2, v.id);
	}
	for (auto &v : md.mesh_subA.Vs) {
		v.v = md.mesh_entire.Vs[md.V_map_reverse[v.id]].v;
		md.mesh_subA.V.col(v.id) = md.mesh_entire.V.col(md.V_map_reverse[v.id]);
	}

	scaled_jacobian(m, mq);
	
	double hausdorff_dis_threshold = 0;
	hausdorff_ratio_threshould = HR * 3;
	bool hausdorff_check = hausdorff_ratio_check(mf.tri, md.mesh_subA, hausdorff_dis_threshold);
	hausdorff_ratio_threshould = HR;
	if (!hausdorff_check) return false; else return true;
}
bool grid_hex_meshing_bijective::smoothing(Mesh_Domain &md) {

	auto & m = md.mesh_entire;
	scaled_jacobian(m, mq);
	std::cout << "before deformation: minimum scaled J: " << mq.min_Jacobian << " average scaled J: " << mq.ave_Jacobian << endl;

	vector<bool> Huntangle_flag(m.Hs.size(), false), H_flag(m.Hs.size(), false), H_inout_tag(m.Hs.size(), true);
	vector<uint32_t> Hids;
	for (uint32_t i = 0; i < m.Hs.size(); i++)Hids.push_back(i);

	ts = Tetralize_Set();
	ts.V = m.V.transpose();
	ts.T.resize(m.Hs.size() * 8, 4);
	Vector4i t;
	for (auto &h : m.Hs) {
		for (uint32_t i = 0; i < 8; i++) {
			for (uint32_t j = 0; j < 4; j++) t[j] = h.vs[hex_tetra_table[i][j]];
			ts.T.row(h.id * 8 + i) = t;
		}
	}
	//smooth feature only
	fc = Feature_Constraints();
	fc.V_types.resize(m.Vs.size()); fill(fc.V_types.begin(), fc.V_types.end(), Feature_V_Type::INTERIOR);
	fc.V_ids.resize(m.Vs.size());
	fc.RV_type.resize(m.Vs.size());
	fill(fc.RV_type.begin(), fc.RV_type.end(), true);
	fc.lamda_C = fc.lamda_L = fc.lamda_T = 0;

	ts.energy_type = SYMMETRIC_DIRICHLET;
	ts.UV = ts.V;
	ts.fc = fc;
	ts.projection = false;
	ts.global = true;
	ts.glue = false;
	ts.lamda_glue = 0;
	ts.lamda_region = 0;
	ts.record_Sequence = false;
	if (args.scaffold_type == 2 || args.scaffold_type == 3)
	{
		ts.known_value_post = true;
		//md.post_index = ts.V.rows();
		ts.post_index = md.post_index;
		ts.post_Variables.resize(ts.V.rows() - md.post_index, 3);
		for (int i = md.post_index; i< m.Vs.size(); i++)
		{
			auto &v = m.Vs[i];
			if (v.boundary)
				ts.post_Variables.row(i - md.post_index) = ts.V.row(i);
		}
	}
	ts.b.resize(0); ts.bc.resize(0, 3); ts.bc.setZero();
	ts.lamda_b = 0;

	optimization opt;
	opt.weight_opt = weight_opt;
	improve_Quality_after_Untangle_Iter_MAX = 3;
	for (uint32_t i = 0; i < improve_Quality_after_Untangle_Iter_MAX; i++) {

		compute_referenceMesh(ts.V, m.Hs, H_inout_tag, Hids, ts.RT, true);
		projection_smooth(m, fc);
		ts.fc = fc;

		opt.slim_opt_igl(ts, 5);
		m.V = ts.V.transpose();
	}

	for (auto &v : m.Vs) {
		v.v[0] = m.V(0, v.id);
		v.v[1] = m.V(1, v.id);
		v.v[2] = m.V(2, v.id);
	}
	for (auto &v : md.mesh_subA.Vs) {
		v.v = md.mesh_entire.Vs[md.V_map_reverse[v.id]].v;
		md.mesh_subA.V.col(v.id) = md.mesh_entire.V.col(md.V_map_reverse[v.id]);
	}

	return true;
}
//global padding
bool grid_hex_meshing_bijective::mesh_padding(Mesh_Domain &md) 
{
	padding_connectivity(md);

	return padding_local_untangle_geometry(md);
}
void grid_hex_meshing_bijective::padding_connectivity(Mesh_Domain &md) {

	md.mesh_subB = Mesh();
	md.mesh_subB.type = Mesh_type::Hex;
	auto & M_temp = md.mesh_subB;
	for (const auto &v : md.mesh_entire.Vs) {
		Hybrid_V v_; v_.id = M_temp.Vs.size();
		v_.v.resize(3, 0);
		for (uint32_t i = 0; i < 3; i++)v_.v[i] = md.mesh_entire.V(i, v.id);
		M_temp.Vs.push_back(v_);
	}
	for (const auto &h : md.mesh_entire.Hs) {
		Hybrid h_; h_.id = M_temp.Hs.size();
		h_.vs = h.vs;
		M_temp.Hs.push_back(h_);
	}
	//extract interface
	Mesh_Feature mf_;
	triangle_mesh_feature(mf_, md.mesh_subA);
	
	vector<vector<int>> v_maps(md.mesh_entire.Vs.size());
	vector<int> vs(2);
	for (uint32_t i = 0; i < mf_.tri.Vs.size(); i++)
	{
		int vid = md.V_map_reverse[mf_.V_map_reverse[i]];
		if (args.octree) {
			hex2Octree_map.push_back(hex2Octree_map[vid]);
			hex2Octree_map.push_back(hex2Octree_map[vid]);
		}
		Hybrid_V v;
		v.v.resize(3);

		v.id = M_temp.Vs.size();
		Vector3d nv = md.mesh_entire.V.col(vid) - mf_.ave_length * Ratio_grow * mf_.normal_V.col(i);
		for (uint32_t j = 0; j < 3; j++)v.v[j] = nv[j];

		vs[0] = v.id;
		M_temp.Vs.push_back(v);
		//second v
		v.id = M_temp.Vs.size();
		nv = md.mesh_entire.V.col(vid) + mf_.ave_length * Ratio_grow * mf_.normal_V.col(i);
		for (uint32_t j = 0; j < 3; j++)v.v[j] = nv[j];

		vs[1] = v.id;
		M_temp.Vs.push_back(v);

		v_maps[vid] = vs;
	}

	M_temp.V.resize(3, M_temp.Vs.size());
	for (uint32_t i = 0; i < M_temp.Vs.size(); i++) {
		M_temp.V(0, i) = M_temp.Vs[i].v[0];
		M_temp.V(1, i) = M_temp.Vs[i].v[1];
		M_temp.V(2, i) = M_temp.Vs[i].v[2];
	}

	for (auto &h : M_temp.Hs){
		for (auto &vid : h.vs)
			if (v_maps[vid].size()) {
				if (md.H_flag[h.id])
					vid = v_maps[vid][0];
				else
					vid = v_maps[vid][1];
			}
	}
	for (auto & f: md.mesh_entire.Fs) {
		if (f.on_medial_surface) {
			Hybrid h;
			h.vs.resize(8);

			h.id = M_temp.Hs.size();
			for (uint32_t j = 0; j < 4; j++) {
				h.vs[j] = f.vs[j];
				h.vs[j + 4] = v_maps[f.vs[j]][0];
			}
			md.H_flag.push_back(true);
			M_temp.Hs.push_back(h);
			
			h.id = M_temp.Hs.size();
			for (uint32_t j = 0; j < 4; j++) {
				h.vs[j] = f.vs[j];
				h.vs[j + 4] = v_maps[f.vs[j]][1];
			}
			md.H_flag.push_back(false);
			M_temp.Hs.push_back(h);
		}
	}

	//remove redundant vs	 
	md.mesh_entire = Mesh();
	md.mesh_entire.type = Mesh_type::Hex;

	redundentV_check(M_temp, md.mesh_entire);
	build_connectivity(md.mesh_entire);
	reorder_hex_mesh_propogation(md.mesh_entire);

	if (args.scaffold_type == 0 || args.scaffold_type == 1)
		re_indexing_connectivity(md.mesh_entire, md.H_flag, md.mesh_subA, md.V_map, md.V_map_reverse, md.H_map, md.H_map_reverse);
	else if (args.scaffold_type == 2 || args.scaffold_type == 3)
		move_boundary_vertices_back(md, hex2Octree_map);

	for (auto&f : md.mesh_entire.Fs)
	{
		if (
			(f.boundary && md.H_flag[f.neighbor_hs[0]]) ||
			!f.boundary && (md.H_flag[f.neighbor_hs[0]] != md.H_flag[f.neighbor_hs[1]])
			)
		{
			f.on_medial_surface = true;
			for (auto &vid : f.vs)
				md.mesh_entire.Vs[vid].on_medial_surface = true;
		}
	}

	feature_qmesh_udpate(md);
}
bool grid_hex_meshing_bijective::padding_local_untangle_geometry(Mesh_Domain &md) {
	
	ts = Tetralize_Set();

	vector<Hybrid> Hs_copies = md.mesh_entire.Hs;
	//multiple copy Vs
	scaled_jacobian(md.mesh_entire, mq);

	vector<bool> Huntangle_flag(md.mesh_entire.Hs.size(), false), H_flag(md.mesh_entire.Hs.size(), false);

	int nI = 0;
	for (uint32_t i = 0; i < mq.H_Js.size(); i++)if (mq.H_Js[i] < Jacobian_Bound) {
		H_flag[i] = true; nI++;
	}
	if (nI == 0) return true;
	//std::fill(H_flag.begin(), H_flag.end(), true);

	int hN = 0;
	local_region(md.mesh_entire, H_flag, Huntangle_flag, hN, ts.regionb, ts.regionbc);
	cout << "region size: " << hN << endl;
	optimization opt;
	opt.weight_opt = weight_opt;
	vector<int32_t> V_map, V_map_reverse, F_map, F_map_reverse, H_map, H_map_reverse;
	re_indexing_connectivity(md.mesh_entire, Huntangle_flag, opt.mesh, V_map, V_map_reverse, H_map, H_map_reverse);

	std::vector<Deform_V_Type> V_types(opt.mesh.Vs.size(), Deform_V_Type::Free);

	if (args.scaffold_type == 2 || args.scaffold_type == 3)
	{
		for (auto &v : opt.mesh.Vs) {
			if (v.boundary)
				V_types[v.id] = Deform_V_Type::Fixed;
		}
	}
	else
	{
		for (int i = 0; i < ts.regionb.size();i++) {
			V_types[V_map[ts.regionb[i]]] = Deform_V_Type::Fixed;
		}
	}
	Eigen::MatrixXd BV = opt.mesh.V.transpose();
	opt.assign_constraints(BV, V_types);

	if (!opt.pipeline2())
	{
		for (int i = 0; i < opt.mesh.Hs.size(); i++) {
			for (int j = 0; j < 8; j++) if (opt.mq.V_Js[i * 8 + j] < 0) {
				int rid = H_map_reverse[i];
				tb_subdivided_cells.push_back(md.mesh_entire.Hs[rid].vs[j]);
			}
		}

		return false;
	}
	else {
		for (int i = 0; i < opt.mesh.Vs.size(); i++)
		{
			int rid = V_map_reverse[i];
			md.mesh_entire.Vs[rid].v = opt.mesh.Vs[i].v;
			md.mesh_entire.V.col(rid) = opt.mesh.V.col(i);
		}

		for (auto &v : md.mesh_subA.Vs) {
			v.v = md.mesh_entire.Vs[md.V_map_reverse[v.id]].v;
			md.mesh_subA.V.col(v.id) = md.mesh_entire.V.col(md.V_map_reverse[v.id]);
		}
	}

	return true;
}
//local padding
bool grid_hex_meshing_bijective::local_padding(Mesh_Domain &md) {
	
	cout << "feature_qmesh_udpate" << endl;
	feature_qmesh_udpate(md);
	cout << "dirty_region_identification" << endl;
	dirty_region_identification(md, Dirty_Vertices);
	cout << "local connectivity" << endl;
	local_padding_connectivity_robust(md);

	feature_qmesh_udpate(md);
	patch_mapping(md);
	patch_trees(md);
	
	return padding_local_untangle_geometry(md);
}
void grid_hex_meshing_bijective::local_padding_connectivity_robust(Mesh_Domain &md) {
	
	auto &mf_quad = md.mf_quad;
	auto &Qfg = md.Qfg;
	auto &m = md.mesh_entire;
	//quads
	vector<uint32_t> qs;
	//on a flat corner 
	vector<int> E_flag(mf_quad.tri.Es.size(), -1), tE_flag(mf.tri.Es.size(), -1);
	for (const auto &l : Qfg.Ls)for (const auto &eid : l.es)E_flag[eid] = l.id;

	if (Dirty_Vertices.size()) {
		vector<bool> DV_flag(mf_quad.tri.Vs.size(), false);
		for (const auto &vid : Dirty_Vertices)DV_flag[md.QV_map[md.V_map[vid]]] = true;
		for (const auto &l : Qfg.Ls)for (const auto &eid : l.es){
			int evid0 = mf_quad.tri.Es[eid].vs[0], evid1 = mf_quad.tri.Es[eid].vs[1];
			if(DV_flag[evid0]&&DV_flag[evid1])
				E_flag[eid] = -1;
		}
	}

	for (const auto &l : fg.Ls)for (const auto &eid : l.es)tE_flag[eid] = l.id;
	for (const auto &f : mf_quad.tri.Fs) {
		int fid = md.QF_map_reverse[f.id];
		vector<vector<uint32_t>> fss(4);
		for (int i = 0; i < 4; i++)
			fss[i] = md.mesh_entire.Vs[md.V_map_reverse[md.mesh_subA.Fs[fid].vs[i]]].neighbor_fs;
		int rfid = face_from_vs(md.mesh_entire, fss);
		assert(rfid != -1);

		for (int i = 0; i < f.es.size(); i++)if (E_flag[f.es[i]] != -1) {
			int li = E_flag[f.es[i]], li_1 = E_flag[f.es[(i + 3) % 4]], li1 = E_flag[f.es[(i + 1) % 4]];
			if (li == li_1 || li == li1) {				
				qs.push_back(rfid); 
				break;
			}
			if (li_1 != -1 || li1 != -1) {
				if (li_1 != -1 && li != li_1) {
					int vid = -1;
					vector<uint32_t>vs0 = fg.Ls[li].vs, vs1 = fg.Ls[li_1].vs, sharedvs;
					sort(vs0.begin(), vs0.end()); sort(vs1.begin(), vs1.end());
					set_intersection(vs0.begin(), vs0.end(), vs1.begin(), vs1.end(), back_inserter(sharedvs));
					if (sharedvs.size())vid = sharedvs[0];
					else { cout << "BUG" << endl; system("PAUSE"); }
					vector<Vector3d> ns; vector<int> vs;
					for (auto eid : mf.tri.Vs[vid].neighbor_es) if (tE_flag[eid] == li || tE_flag[eid] == li_1) {
						uint32_t v0 = mf.tri.Es[eid].vs[0]; vs.push_back(v0);
						uint32_t v1 = mf.tri.Es[eid].vs[1]; vs.push_back(v1);
						ns.push_back((mf.tri.V.col(v0) - mf.tri.V.col(v1)).normalized());
					}
					if (vs[0] == vs[2] || vs[1] == vs[3]) ns[0] *= -1;
					double angles = acos(ns[0].dot(ns[1]));
					if (angles < 40.0 / 180 * PAI) {						
						qs.push_back(rfid); 
						break;
					}
				}
				if (li1 != -1 && li != li_1) {
					int vid = -1;
					vector<uint32_t>vs0 = fg.Ls[li].vs, vs1 = fg.Ls[li1].vs, sharedvs;
					sort(vs0.begin(), vs0.end()); sort(vs1.begin(), vs1.end());
					set_intersection(vs0.begin(), vs0.end(), vs1.begin(), vs1.end(), back_inserter(sharedvs));
					if (sharedvs.size())vid = sharedvs[0];
					else { cout << "BUG" << endl; system("PAUSE"); }
					vector<Vector3d> ns; vector<int> vs;
					for (auto eid : mf.tri.Vs[vid].neighbor_es) if (tE_flag[eid] == li || tE_flag[eid] == li1) {
						uint32_t v0 = mf.tri.Es[eid].vs[0]; vs.push_back(v0);
						uint32_t v1 = mf.tri.Es[eid].vs[1]; vs.push_back(v1);
						ns.push_back((mf.tri.V.col(v0) - mf.tri.V.col(v1)).normalized());
					}

					if (vs[0] == vs[2] || vs[1] == vs[3]) ns[0] *= -1;
					double angles = acos(ns[0].dot(ns[1]));

					if (angles < 40.0 / 180 * PAI) {
						qs.push_back(rfid); 
						break;
					}
				}
			}
		}
	}
	sort(qs.begin(), qs.end()); qs.erase(unique(qs.begin(), qs.end()), qs.end());
	//expand quads
	vector<bool> V_tag(m.Vs.size(), false);
	vector<bool> E_tag(m.Es.size(), false);
	vector<bool> F_tag(m.Fs.size(), false);
	vector<bool> H_tag(m.Hs.size(), false);
	for (auto es : mf_quad.curve_es)for (auto eid : es) {
		E_tag[eid] = true;
		for (auto vid : m.Es[eid].vs) 
			V_tag[vid] = true;
	}
	for (auto qid : qs) {
		vector<uint32_t> pool, pool_; pool.push_back(qid);
		for (uint32_t i = 0; i < 1; i++) {
			for (auto qfid : pool) {
				bool have_v = false;
				for (auto vid : m.Fs[qfid].vs)if (!V_tag[vid]) {
					for (auto nfid : m.Vs[vid].neighbor_fs) if (m.Fs[nfid].on_medial_surface) {
						F_tag[nfid] = true; pool_.push_back(nfid);
					}
					have_v = true;
				}
				//if (!have_v) 
				{
					for (auto eid : m.Fs[qfid].es)if (!E_tag[eid]) {
						for (auto nfid : m.Es[eid].neighbor_fs) if (m.Fs[nfid].on_medial_surface) {
							F_tag[nfid] = true;
							pool_.push_back(nfid);
						}
					}
				}
			}
			pool = pool_;
		}
	}
	//extract regions
	Q_regions.clear();
	int32_t search_ahead = 0;
	while (true) {
		bool foundone = false;
		for (; search_ahead < F_tag.size(); search_ahead++) if (F_tag[search_ahead] && m.Fs[search_ahead].on_medial_surface) {
			F_tag[search_ahead] = false;
			foundone = true;
			break;
		}
		if (!foundone)break;

		vector<uint32_t> region, region_;
		region.push_back(search_ahead);
		region_ = region;
		while (region_.size()) {
			vector<uint32_t> f_pool;
			for (auto f : region_)for (auto eid : m.Fs[f].es) {
				if (E_tag[eid])continue;
				for (auto nfid : m.Es[eid].neighbor_fs)if (F_tag[nfid] && m.Fs[nfid].on_medial_surface) {
					f_pool.push_back(nfid);
					F_tag[nfid] = false;
				}
			}
			region_.clear();
			if (f_pool.size()) {
				region_ = f_pool;
				region.insert(region.end(), f_pool.begin(), f_pool.end());
			}
		}
		Q_regions.push_back(region);
		search_ahead++;
	}
	//iterate region
	V_layers.clear();
	
	cout << "region total: " << Q_regions.size() << endl;
	h_io io;
	for (const auto &region : Q_regions) {
		vector<Local_V> local_vs;
		vector<int> local_fs;
		vector<int> local_hs;
		padding_arbitrary_hex_mesh_step1(md, region,local_vs, H_tag, F_tag, local_fs, local_hs);
		padding_arbitrary_hex_mesh_step2(md, region,local_vs, H_tag, local_fs);
		for (const auto &hid : local_hs) H_tag[hid] = false;

	}
	cout << "local padding done" << endl;

	m.V.resize(3, m.Vs.size());
	for (uint32_t i = 0; i < m.Vs.size(); i++) {
		m.V(0, i) = m.Vs[i].v[0];
		m.V(1, i) = m.Vs[i].v[1];
		m.V(2, i) = m.Vs[i].v[2];
	}
	Mesh M; M.type = m.type;
	M.V = m.V;
	M.Vs.resize(m.Vs.size());
	for (uint32_t j = 0; j < m.Vs.size(); j++) {
		Hybrid_V v;
		v.id = j;
		v.v = m.Vs[j].v;
		M.Vs[j] = v;
	}
	M.Hs.resize(m.Hs.size());
	for (uint32_t j = 0; j < m.Hs.size(); j++) {
		Hybrid h;
		h.id = j;
		for (auto vid : m.Hs[j].vs) {
			h.vs.push_back(vid);
			M.Vs[vid].neighbor_hs.push_back(h.id);
		}
		M.Hs[j] = h;
	}
	m = M;
	build_connectivity(m);
	reorder_hex_mesh_propogation(m);

	if (args.scaffold_type == 0 || args.scaffold_type == 1)
		re_indexing_connectivity(md.mesh_entire, md.H_flag, md.mesh_subA, md.V_map, md.V_map_reverse, md.H_map, md.H_map_reverse);
	else if (args.scaffold_type == 2 || args.scaffold_type == 3)
		move_boundary_vertices_back(md, hex2Octree_map);

	for (auto&f : md.mesh_entire.Fs)
	{
		if (
			(f.boundary && md.H_flag[f.neighbor_hs[0]]) ||
			!f.boundary && (md.H_flag[f.neighbor_hs[0]] != md.H_flag[f.neighbor_hs[1]])
			)
		{
			f.on_medial_surface = true;
			for (auto &vid : f.vs)
				md.mesh_entire.Vs[vid].on_medial_surface = true;
		}
	}
}
void grid_hex_meshing_bijective::padding_arbitrary_hex_mesh_step1(Mesh_Domain &md, const vector<uint32_t> &region,vector<Local_V> &local_vs,
vector<bool> &H_tag, vector<bool> &F_tag, vector<int> &local_fs, vector<int> &local_hs)
{
	//local hs,
	for(auto &fid:region)
	{
		auto &nhs = md.mesh_entire.Fs[fid].neighbor_hs;
		assert(nhs.size()==2);
		for(auto &nhid: nhs) {
			H_tag[nhid] = true;
			local_hs.push_back(nhid);
		}
	}
	// fs
	for(auto &hid: local_hs)for(auto &fid:md.mesh_entire.Hs[hid].fs)if (F_tag[fid])F_tag[fid] = false; else F_tag[fid] = true;
	for(auto &hid: local_hs)for(auto &fid:md.mesh_entire.Hs[hid].fs) if (F_tag[fid]) {local_fs.push_back(fid);F_tag[fid] = false;}
	// vs
	vector<uint32_t> vs;
	for (const auto &fid : local_fs){
		auto fvs = md.mesh_entire.Fs[fid].vs;
		vs.insert(vs.end(),fvs.begin(), fvs.end());
	}
	sort(vs.begin(), vs.end()); vs.erase(unique(vs.begin(), vs.end()), vs.end());
	//local vs
	for (const auto &vid : vs) {		
		vector<int> all_nhs;
		for (const auto &hid : md.mesh_entire.Vs[vid].neighbor_hs)if (H_tag[hid])all_nhs.push_back(hid);
		for (const auto &hid : all_nhs)H_tag[hid] = false;
		while(true){
			int h_ = -1;
			for (const auto &hid : all_nhs){
				if(!H_tag[hid]){ h_ = hid; H_tag[hid] = true; break;}
			}
			if(h_ == -1)
				break; 

			Local_V lv;	
			lv.boundary = true;
			lv.top = true;			
			lv.hvid = vid;		

			vector<int> hs_;
			hs_.push_back(h_);
			lv.nhs.push_back(h_);
			while (true) {
				vector<int> hs_temp;
				for (const auto&hid : hs_) {
					for (const auto &fid : md.mesh_entire.Hs[hid].fs)for (const auto &nhid : md.mesh_entire.Fs[fid].neighbor_hs)
						if (!H_tag[nhid] && find(all_nhs.begin(), all_nhs.end(), nhid) != all_nhs.end()) {
							hs_temp.push_back(nhid);
							H_tag[nhid] = true;
						}
				}
				if (hs_temp.size()) {
					hs_ = hs_temp;
					lv.nhs.insert(lv.nhs.end(), hs_.begin(), hs_.end());
				}
				else break;
			}
			lv.manifold = true;
			if(lv.nfs.size() != all_nhs.size())
				lv.manifold = true;

			lv.id = local_vs.size();		
			local_vs.push_back(lv);
		}
	}
}
void grid_hex_meshing_bijective::padding_arbitrary_hex_mesh_step2(Mesh_Domain &md, const vector<uint32_t> &region,vector<Local_V> &local_vs,
vector<bool> &H_tag, vector<int> &local_fs)
{
	map<int, vector<int>>v2lv_map;	

	for(auto &lv:local_vs) v2lv_map[lv.hvid].push_back(lv.id);

	for (auto &lv: local_vs) 
	{
		if(args.octree){	
			hex2Octree_map.push_back(hex2Octree_map[lv.hvid]);
		}
		Hybrid_V v; 
		v.v.resize(3);
		double elen = 0;
		for(auto &nvid: md.mesh_entire.Vs[lv.hvid].neighbor_vs){
			elen += (md.mesh_entire.V.col(lv.hvid) - md.mesh_entire.V.col(nvid)).norm();
		}
		elen /= md.mesh_entire.Vs[lv.hvid].neighbor_vs.size();

		v.id = md.mesh_entire.Vs.size();
		lv.id_global = v.id;

		vector<uint32_t> vs_all;
		for (const auto &nhid : lv.nhs) vs_all.insert(vs_all.end(), md.mesh_entire.Hs[nhid].vs.begin(), md.mesh_entire.Hs[nhid].vs.end());
		sort(vs_all.begin(), vs_all.end());
		vs_all.erase(unique(vs_all.begin(), vs_all.end()), vs_all.end());

		Vector3d n; n.setZero();
		for (const auto &vid : vs_all) n += md.mesh_entire.V.col(vid) - md.mesh_entire.V.col(lv.hvid);
		n.normalize();
		for (uint32_t j = 0; j < 3; j++)v.v[j] = md.mesh_entire.V.col(lv.hvid)[j] + elen * Ratio_grow  * n[j];
		md.mesh_entire.Vs.push_back(v);
	}

	Hybrid h;
	h.vs.resize(8);
	for (auto & fid: local_fs) {
		
		int hid = md.mesh_entire.Fs[fid].neighbor_hs[0];
		if(!H_tag[hid]){
			assert(md.mesh_entire.Fs[fid].neighbor_hs.size()==2);
			hid = md.mesh_entire.Fs[fid].neighbor_hs[1];
		} 
		
		if(md.H_flag[hid])
			md.H_flag.push_back(true);
		else 
			md.H_flag.push_back(false);

		for (auto &vid : md.mesh_entire.Hs[hid].vs) {
			if (v2lv_map[vid].size()) {
				int which_lvid = -1;
				for (const auto &lvid : v2lv_map[vid]) {
					auto &hs = local_vs[lvid].nhs;
					sort(hs.begin(), hs.end());
					if (find(hs.begin(), hs.end(), hid) != hs.end()) {						
						which_lvid = lvid; break;
					}
				}
				vid = local_vs[which_lvid].id_global;
			}
		}

		h.id = md.mesh_entire.Hs.size();
		auto &vs = md.mesh_entire.Fs[fid].vs;
		for (uint32_t i = 0; i < 4; i++) {
			auto &vid = vs[i];
			if (v2lv_map[vid].size()) {
				int which_lvid = -1;
				for (const auto &lvid : v2lv_map[vid]) {
					auto &hs = local_vs[lvid].nhs;
					sort(hs.begin(), hs.end());
					if (find(hs.begin(), hs.end(), hid) != hs.end()) {						
						which_lvid = lvid; break;
					}
				}
				h.vs[i] = local_vs[which_lvid].id_global;
				h.vs[i + 4] = vid;
			}
			else { cout << "BUG" << endl; }
		}

		h.id = md.mesh_entire.Hs.size();
		md.mesh_entire.Hs.push_back(h);
	}
}

void grid_hex_meshing_bijective::feature_qmesh_udpate(Mesh_Domain &md) {
	auto &mf_quad = md.mf_quad;
	auto &QV_map = md.QV_map;
	auto &QV_map_reverse = md.QV_map_reverse;
	auto &QF_map = md.QF_map;
	auto &QF_map_reverse = md.QF_map_reverse;
	auto &Qfg = md.Qfg;
	auto &hmi = md.mesh_subA;

	mf_quad.tri.type = Mesh_type::Qua;
	extract_surface_conforming_mesh(hmi, mf_quad.tri, QV_map, QV_map_reverse, QF_map, QF_map_reverse);

	//cout << "update mf_quad quad-mesh indexing" << endl;
	function<bool(Mesh &, vector<uint32_t> &, vector<uint32_t> &, const bool&)> VSES = [&](Mesh &mesh, vector<uint32_t> &vs, vector<uint32_t> &es, const bool &circle)->bool {
		int vlen = vs.size(), vlen_ = vlen;
		if (!circle) vlen--;
		es.clear();
		bool broken_curve = false;
		for (uint32_t i = 0; i < vlen; i++) {
			int v0 = vs[i], v1 = vs[(i + 1) % vlen_];
			vector<uint32_t> sharedes, es0 = mesh.Vs[v0].neighbor_es, es1 = mesh.Vs[v1].neighbor_es;
			std::sort(es0.begin(), es0.end()); std::sort(es1.begin(), es1.end());
			set_intersection(es0.begin(), es0.end(), es1.begin(), es1.end(), back_inserter(sharedes));
			if (sharedes.size()) es.push_back(sharedes[0]);
			else broken_curve = true;
		}
		return broken_curve;
	};
	for (auto &cid : mf_quad.corners)  cid = QV_map[md.V_map[cid]];
	for (auto &vs : mf_quad.curve_vs) for (auto &vid : vs) vid = QV_map[md.V_map[vid]];
	for (uint32_t i = 0; i < mf_quad.curve_es.size(); i++) VSES(mf_quad.tri, mf_quad.curve_vs[i], mf_quad.curve_es[i], mf_quad.circles[i]);
	//cout << "update Qfg" << endl;
	build_feature_graph(mf_quad, Qfg);
	vector<bool> V_flag(md.mesh_entire.Vs.size(), true);
	for (const auto & v : hmi.Vs)if (!v.boundary)V_flag[md.V_map_reverse[v.id]] = false;
	for (const auto & vs: mf_quad.curve_vs)for (const auto &vid : vs) V_flag[md.V_map_reverse[QV_map_reverse[vid]]] = false;

	vector<bool> V_tag(md.mesh_entire.Vs.size(), false);
	for (auto &R : Qfg.Rs) {
		for (auto fid : R.tris)for (auto vid : Qfg.mf.tri.Fs[fid].vs)
		{
			auto vid_ = md.V_map_reverse[QV_map_reverse[vid]];
			if (!V_tag[vid_] && V_flag[vid_]) {
				R.vs.push_back(vid_);
				V_tag[vid_] = true;
			}
		}
	}
	//cout << "update mf_quad hex-mesh indexing" << endl;
	for (auto &cid : mf_quad.corners)  cid = md.V_map_reverse[QV_map_reverse[cid]];
	for (auto &vs : mf_quad.curve_vs) for (auto &vid : vs) vid = md.V_map_reverse[QV_map_reverse[vid]];
	for (uint32_t i = 0; i < mf_quad.curve_es.size(); i++) VSES(md.mesh_entire, mf_quad.curve_vs[i], mf_quad.curve_es[i], mf_quad.circles[i]);
}
void grid_hex_meshing_bijective::dirty_region_identification(Mesh_Domain &md, std::vector<int> &Dirty_Vs) {

	auto &Qfg = md.Qfg;
	auto &mf_quad = md.mf_quad;
	auto &QV_map_reverse = md.QV_map_reverse;
	std::vector<int> C_flag(md.mesh_entire.Vs.size(), -1);

	std::vector<bool> V_flag(md.mesh_entire.Vs.size(), false);
	for (const auto & tc : fg.Cs) {

		std::fill(C_flag.begin(), C_flag.end(), -1);
		for (const auto &l : Qfg.Ls)for (const auto &vid : l.vs)C_flag[md.V_map_reverse[QV_map_reverse[vid]]] = l.id;

		std::vector<double> angles;
		int pre = -1, cur = -1;
		bool Has_Tiny = false;
		for (int i = 0; i < tc.ring_vs_tag.size(); i++) {
			if (tc.ring_vs_tag[i] >= 0) {
				if (pre == -1 && cur == -1) {
					pre = tc.ring_vs[i];
				}
				else if (pre != -1 && cur == -1) {
					cur = tc.ring_vs[i];
				}
				else if (pre != -1 && cur != -1) {
					pre = cur;
					cur = tc.ring_vs[i];
				}

				if (pre != -1 && cur != -1) {
					Vector3d l0 = (mf.tri.V.col(pre) - mf.tri.V.col(tc.vs[0])).normalized();
					Vector3d l1 = (mf.tri.V.col(cur) - mf.tri.V.col(tc.vs[0])).normalized();

					angles.push_back(std::abs(std::acos(l0.dot(l1))));
					if (angles[angles.size() - 1] < Tiny_angle_threshold) {
						Has_Tiny = true; break;
					}
				}
			}
		}
		if (Has_Tiny) {

			for (int i = 0; i < tc.ring_vs_tag.size(); i++) {
				if (tc.ring_vs_tag[i] >= 0)
					for (const auto &vid : Qfg.Ls[tc.ring_vs_tag[i]].vs)C_flag[md.V_map_reverse[QV_map_reverse[vid]]] = -1;
			}

			int qvid = Qfg.Cs[tc.id].vs[0];
			vector<int> lregion_vs;
			lregion_vs.push_back(qvid);

			int N_ring = 3;
			for (int j = 0; j < N_ring; j++) {
				vector<int> lregion_vs_;
				for (int k = 0; k < lregion_vs.size(); k++) {
					const auto & nfs = mf_quad.tri.Vs[lregion_vs[k]].neighbor_fs;
					for (const auto & fid : nfs) {
						const auto &nvs = mf_quad.tri.Fs[fid].vs;
						for (const auto &vid : nvs)
							if (C_flag[md.V_map_reverse[QV_map_reverse[vid]]] != -1)continue;
						else
							V_flag[md.V_map_reverse[QV_map_reverse[vid]]] = true;
						lregion_vs_.insert(lregion_vs_.end(), nvs.begin(), nvs.end());
					}
				}
				lregion_vs = lregion_vs_;
			}
		}
	}
	Dirty_Vs.clear();
	for (int i = 0; i < V_flag.size(); i++)if (V_flag[i])Dirty_Vs.push_back(i);
}
void grid_hex_meshing_bijective::projection_smooth(const Mesh &hmi, Feature_Constraints &fc) {

	MatrixXd Ps(fc.ids_T.size(), 3);
	int num_regulars_ = 0;
	for (int i = 0; i<fc.V_types.size(); i++)
		if (fc.V_types[i] == Feature_V_Type::REGULAR) { fc.ids_T(num_regulars_) = i; Ps.row(num_regulars_++) = hmi.V.col(i).transpose(); }

	for (int i = 0; i < fc.ids_T.size(); i++) { Ps.row(i) = hmi.V.col(fc.ids_T[i]).transpose(); }

	VectorXd signed_dis;
	VectorXi ids;
	MatrixXd V_T(fc.ids_T.size(), 3), normal_T(fc.ids_T.size(), 3);
	signed_distance_pseudonormal(Ps, tri_tree.TriV, tri_tree.TriF, tri_tree.tree, tri_tree.TriFN, tri_tree.TriVN, tri_tree.TriEN,
		tri_tree.TriEMAP, signed_dis, ids, V_T, normal_T);

	for (uint32_t j = 0; j < fc.ids_T.size(); j++) {
		fc.normal_T.row(j) = normal_T.row(j);
		fc.V_T.row(j) = V_T.row(j);
		fc.dis_T[j] = normal_T.row(j).dot(V_T.row(j));
		fc.V_ids[fc.ids_T[j]] = -1;
	}
}

bool grid_hex_meshing_bijective::feature_alignment(Mesh_Domain &md) {

	auto &m = md.mesh_entire;
	auto &m_ = md.mesh_subA;

	vector<bool> H_flag(m.Hs.size(), true);
	//hex2tet
	ts = Tetralize_Set();

	ts.V = m.V.transpose();
	ts.T.resize(m.Hs.size() * 8, 4);
	Vector4i t;
	for (auto &h : m.Hs) {
		for (uint32_t i = 0; i < 8; i++) {
			for (uint32_t j = 0; j < 4; j++) t[j] = h.vs[hex_tetra_table[i][j]];
			ts.T.row(h.id * 8 + i) = t;
		}
	}
	vector<uint32_t> Hids;
	for (uint32_t i = 0; i < m.Hs.size(); i++)Hids.push_back(i);
	
	double MESHRATIO = 1;
	double LAMDA_FEATURE_PROJECTION = MESHRATIO * args.feature_weight;//0.05
	double LAMDA_FEATURE_PROJECTION_BOUND = MESHRATIO * 1e+16;

	ts.b.resize(0); ts.bc.resize(0, 3); ts.bc.setZero();
	ts.lamda_b = MESHRATIO *1e+3;
	ts.regionb.resize(0); ts.regionbc.resize(0, 3); ts.regionbc.setZero();
	ts.lamda_region = 0;
	fc.lamda_C = fc.lamda_L = fc.lamda_T = LAMDA_FEATURE_PROJECTION;

	ts.energy_type = SYMMETRIC_DIRICHLET;
	ts.UV = ts.V;
	ts.fc = fc;
	ts.projection = false;
	ts.global = true;
	ts.glue = false;
	ts.lamda_glue = 0;
	ts.record_Sequence = false;

	if (args.scaffold_type == 2 || args.scaffold_type == 3)
	{
		ts.known_value_post = true;
		//md.post_index = ts.V.rows();
		ts.post_index = md.post_index;
		ts.post_Variables.resize(ts.V.rows() - md.post_index, 3);
		for (int i = md.post_index; i< m.Vs.size(); i++)
		{
			auto &v = m.Vs[i];
			if (v.boundary)
				ts.post_Variables.row(i - md.post_index) = ts.V.row(i);
		}
	}

	optimization opt;
	opt.weight_opt = weight_opt * 0.001;

	Mesh htri; htri.type = Mesh_type::Tri;
	extract_surface_conforming_mesh(m_, htri, md.TV_map, md.TV_map_reverse, md.TF_map, md.TF_map_reverse);
	ave_Hausdorff_dises.clear();
	ratio_ave_Hausdorff.clear();
	ratio_max_Hausdorff.clear();

	//dirty feature
	dirty_region_identification(md, Dirty_Vertices);
	//Dirty_Vertices.clear();
	dirty_local_feature_update(md, Dirty_Vertices);
	//loop
	bool Max_dis_satisified = false, Ave_dis_satisfied = false;
	improve_Quality_after_Untangle_Iter_MAX = 30;
	double energy_pre = -1;
	for (uint32_t i = 0; i < improve_Quality_after_Untangle_Iter_MAX; i++) {
		
		dirty_graph_projection(md, fc, Dirty_Vertices);

		ts.fc = fc;

		compute_referenceMesh(ts.UV, m.Hs, md.H_flag, Hids, ts.RT, true);
		opt.slim_m_opt(ts, 1, -1);
		m.V = ts.UV.transpose();
		ts.V = ts.UV;

		if (!Max_dis_satisified)
		{
			std::cout<< "energy_quality "<<opt.engery_quality<<"; soft: "<<opt.engery_soft<< " lamda_c "<< fc.lamda_C<<std::endl;
			fc.lamda_T = fc.lamda_L = fc.lamda_C = std::min(LAMDA_FEATURE_PROJECTION * std::max(opt.engery_quality/opt.engery_soft * fc.lamda_C, 1.0), LAMDA_FEATURE_PROJECTION_BOUND);
		}

		if(i<=1)
			energy_pre = opt.energy;
		else if(i>1 && energy_pre != opt.energy)
			energy_pre = opt.energy;
		else if(i>1 && energy_pre == opt.energy)
			break;

		if (stop_criterior_satisfied(md, i, mf.tri, htri, Max_dis_satisified, Ave_dis_satisfied)) {

			break;
		}
	}
	for (auto &v : m.Vs) {
		v.v[0] = m.V(0, v.id);
		v.v[1] = m.V(1, v.id);
		v.v[2] = m.V(2, v.id);
	}

	for (auto &v : md.mesh_subA.Vs) {
		v.v = md.mesh_entire.Vs[md.V_map_reverse[v.id]].v;
		md.mesh_subA.V.col(v.id) = md.mesh_entire.V.col(md.V_map_reverse[v.id]);
	}

	scaled_jacobian(m, mq);
	std::cout << "after: V, H, minimum scaled J: " << m.Vs.size() << " " << m.Hs.size() << " " << mq.min_Jacobian << " average scaled J: " << mq.ave_Jacobian << endl;
	scaled_jacobian(m_, mq);
	std::cout << "after: V, H, minimum scaled J: " << m_.Vs.size() << " " << m_.Hs.size() << " " << mq.min_Jacobian << " average scaled J: " << mq.ave_Jacobian << endl;
	mq.ave_hausdorff = ratio_ave_Hausdorff;
	mq.max_hausdorff = ratio_max_Hausdorff;
	
	double J_bound = 0.0;
	if (mq.min_Jacobian < J_bound) {
		for (int i = 0; i < m.Hs.size(); i++) {
			for (int j = 0; j<8; j++) if (mq.V_Js[i * 8 + j] < J_bound) {
				tb_subdivided_cells.push_back(m.Hs[i].vs[j]);
			}
		}
		return false;
	}

	return true;
}
void grid_hex_meshing_bijective::dirty_local_feature_update(Mesh_Domain &md, std::vector<int> &Dirty_Vs) {
	auto &hmi = md.mesh_entire;
	auto &mf_quad = md.mf_quad;

	fc.V_types.resize(hmi.Vs.size()); fill(fc.V_types.begin(), fc.V_types.end(), Feature_V_Type::INTERIOR);
	fc.V_ids.resize(hmi.Vs.size());
	fc.RV_type.resize(hmi.Vs.size());
	fill(fc.RV_type.begin(), fc.RV_type.end(), true);
	//matrices

	for (uint32_t i = 0; i < mf_quad.corners.size(); i++) {
		fc.V_types[mf_quad.corners[i]] = Feature_V_Type::CORNER;
		fc.V_ids[mf_quad.corners[i]] = mf.corners[i];
	}
	for (uint32_t i = 0; i < mf_quad.curve_vs.size(); i++)for (auto vid : mf_quad.curve_vs[i]) {
		if (fc.V_types[vid] == Feature_V_Type::CORNER)continue;
		fc.V_types[vid] = Feature_V_Type::LINE;
		fc.V_ids[vid] = i;
	}
	for (auto v : hmi.Vs) {
		auto i = v.id;
		if (v.on_medial_surface && fc.V_types[i] == Feature_V_Type::INTERIOR) {
			fc.V_types[i] = Feature_V_Type::REGULAR;
		}
	}

	for (const auto &v : Dirty_Vs) {
		fc.V_types[v] = Feature_V_Type::REGULAR;
	}

	uint32_t num_corners = 0, num_lines = 0, num_regulars = 0;
	for (const auto &type : fc.V_types)
		if (type == Feature_V_Type::CORNER)num_corners++;
		else if (type == Feature_V_Type::LINE)num_lines++;
		else if (type == Feature_V_Type::REGULAR)num_regulars++;

		fc.ids_C.resize(num_corners); fc.C.resize(num_corners, 3);
		fc.num_a = num_lines; fc.ids_L.resize(num_lines); fc.Axa_L.resize(num_lines, 3); fc.origin_L.resize(num_lines, 3);
		fc.ids_T.resize(num_regulars); fc.normal_T.resize(num_regulars, 3); fc.dis_T.resize(num_regulars); fc.V_T.resize(num_regulars, 3);
		num_corners = num_lines = num_regulars = 0;
}
void grid_hex_meshing_bijective::dirty_graph_projection(Mesh_Domain &md, Feature_Constraints &fc, std::vector<int> &Dirty_Vs) {

	auto &Qfg = md.Qfg;
	auto &hmi = md.mesh_entire;
	auto &tri_Trees = md.tri_Trees;

	uint32_t num_corners = 0, num_lines = 0, num_regulars = 0;
	for (uint32_t i = 0; i < fc.V_types.size(); i++) {
		if (fc.V_types[i] == Feature_V_Type::CORNER) {
			fc.ids_C[num_corners] = i;
			fc.C.row(num_corners) = mf.tri.V.col(fc.V_ids[i]);
			num_corners++;
		}
		else if (fc.V_types[i] == Feature_V_Type::LINE) {
			Vector3d v = hmi.V.col(i).transpose();
			uint32_t curve_id = fc.V_ids[i];
			vector<uint32_t> &curve = mf.curve_vs[curve_id];
			Vector3d tangent(1, 0, 0);
			uint32_t curve_len = curve.size();

			if (!mf.circles[curve_id]) curve_len--;

			vector<Vector3d> pvs, tangents;
			vector<pair<double, uint32_t>> dis_ids;
			Vector3d pv;
			for (uint32_t j = 0; j < curve_len; j++) {
				uint32_t pos_0 = curve[j], pos_1 = curve[(j + 1) % curve.size()];
				double t, precision_here = 1.0e1;
				point_line_projection(mf.tri.V.col(pos_0), mf.tri.V.col(pos_1), v, pv, t);
				tangent = (mf.tri.V.col(pos_1) - mf.tri.V.col(pos_0)).normalized();
				dis_ids.push_back(make_pair((v - pv).norm(), pvs.size()));
				pvs.push_back(pv);
				tangents.push_back(tangent);
			}
			sort(dis_ids.begin(), dis_ids.end());

			if (dis_ids.size()) {
				uint32_t cloestid = dis_ids[0].second;
				pv = pvs[cloestid];
				tangent = tangents[cloestid];
			}
			else {
				//brute-force search
				for (uint32_t j = 0; j < curve.size(); j++) {
					double dis = (mf.tri.V.col(curve[j]) - v).norm();
					dis_ids.push_back(make_pair(dis, j));
				}
				sort(dis_ids.begin(), dis_ids.end());

				int pos = dis_ids[0].second;
				pv = mf.tri.V.col(curve[pos]);

				curve_len = curve.size();
				if (mf.circles[curve_id] || (!mf.circles[curve_id] && pos != 0 && pos != curve_len - 1)) {
					uint32_t pos_0 = (pos - 1 + curve_len) % curve_len, pos_1 = (pos + 1) % curve_len;
					tangent += (mf.tri.V.col(curve[pos]) - mf.tri.V.col(curve[pos_0])).normalized();
					tangent += (mf.tri.V.col(curve[pos_1]) - mf.tri.V.col(curve[pos])).normalized();
				}
				else if (!mf.circles[curve_id] && pos == 0) {
					uint32_t pos_1 = (pos + 1) % curve_len;
					tangent += (mf.tri.V.col(curve[pos_1]) - mf.tri.V.col(curve[pos])).normalized();
				}
				else if (!mf.circles[curve_id] && pos == curve_len - 1) {
					uint32_t pos_0 = (pos - 1 + curve_len) % curve_len;
					tangent += (mf.tri.V.col(curve[pos]) - mf.tri.V.col(curve[pos_0])).normalized();
				}
				tangent.normalize();
			}
			fc.ids_L[num_lines] = i;
			fc.origin_L.row(num_lines) = pv;
			fc.Axa_L.row(num_lines) = tangent;
			num_lines++;
		}
	}
	//cout << "graph_projection patch projection" << endl;
	vector<bool> V_flag(hmi.Vs.size(), false);
	num_regulars = 0;

	for (const auto &vid : Dirty_Vs) V_flag[vid] = true;
	std::cout<<"dirty_vs size: "<<Dirty_Vs.size()<<std::endl;
	if (Dirty_Vs.size()) {
		MatrixXd Ps(Dirty_Vs.size(), 3);
		int num_regulars_ = 0;
		for (const auto &vid : Dirty_Vs) { fc.ids_T(num_regulars + num_regulars_) = vid; Ps.row(num_regulars_++) = hmi.V.col(vid).transpose(); }

		VectorXd signed_dis;
		VectorXi ids;
		MatrixXd V_T(Dirty_Vs.size(), 3), normal_T(Dirty_Vs.size(), 3);
		signed_distance_pseudonormal(Ps, tri_tree.TriV, tri_tree.TriF, tri_tree.tree, tri_tree.TriFN, tri_tree.TriVN, tri_tree.TriEN,
			tri_tree.TriEMAP, signed_dis, ids, V_T, normal_T);

		for (uint32_t j = 0; j < Dirty_Vs.size(); j++) {
			fc.normal_T.row(num_regulars + j) = normal_T.row(j);
			fc.V_T.row(num_regulars + j) = V_T.row(j);
			fc.dis_T[num_regulars + j] = normal_T.row(j).dot(V_T.row(j));
			fc.V_ids[Dirty_Vs[j]] = -1;
		}

		num_regulars = Dirty_Vs.size();
	}

	for (uint32_t i = 0; i < Qfg.Rs.size(); i++) {
		auto &R = Qfg.Rs[i];
		vector<int> vs;
		for (const auto &vid : R.vs)if (!V_flag[vid]) vs.push_back(vid);

		if (!vs.size()) {
			continue;
		}

		MatrixXd Ps(vs.size(), 3);
		int num_regulars_ = 0;
		for (auto vid : vs) { fc.ids_T(num_regulars + num_regulars_) = vid; Ps.row(num_regulars_++) = hmi.V.col(vid).transpose(); }

		VectorXd signed_dis;
		VectorXi ids;
		MatrixXd V_T(vs.size(), 3), normal_T(vs.size(), 3);
		signed_distance_pseudonormal(Ps, tri_Trees[i].TriV, tri_Trees[i].TriF, tri_Trees[i].tree, tri_Trees[i].TriFN, tri_Trees[i].TriVN, tri_Trees[i].TriEN,
			tri_Trees[i].TriEMAP, signed_dis, ids, V_T, normal_T);

		for (uint32_t j = 0; j < vs.size(); j++) {
			fc.normal_T.row(num_regulars + j) = normal_T.row(j);
			fc.V_T.row(num_regulars + j) = V_T.row(j);
			fc.dis_T[num_regulars + j] = normal_T.row(j).dot(V_T.row(j));
			fc.V_ids[vs[j]] = -1;
		}
		num_regulars += vs.size();
	}
}
void grid_hex_meshing_bijective::tet_assembling(const Mesh &m, const vector<Hybrid> &Hs_copies, const vector<bool> &H_flag, const vector<bool> &Huntangle_flag, bool local) {
	vector<Hybrid> hs = m.Hs;
	if(local)  hs = Hs_copies;

	Vector4i t;
	int count = 0;
	for (auto &h : hs) if ((local && Huntangle_flag[h.id]) || !local) {
		for (uint32_t i = 0; i < 8; i++) {
			for (uint32_t j = 0; j < 4; j++) t[j] = h.vs[hex_tetra_table[i][j]];
			ts.T.row(count * 8 + i) = t;
		}
		count++;

		if (H_flag[h.id]) {
			MatrixXd C(8, 3), C_;
			hex2cuboid(ts.V, m.Hs[h.id].vs, C, false);

			MatrixXd X(8, 3), Y(8, 3);
			for (uint32_t i = 0; i < 8; i++) {
				X.row(i) = C.row(i);
				Y.row(i) = m.V.col(m.Hs[h.id].vs[i]);
			}
			MatrixXd R; VectorXd T;
			double scale;
			igl::procrustes(X, Y, true, false, scale, R, T);
			R *= scale;
			C_ = (C * R).rowwise() + T.transpose();
			C = C_;

			for (uint32_t i = 0; i < 8; i++) ts.V.row(h.vs[i]) = C.row(i);
		}
	}
}
void grid_hex_meshing_bijective::local_region(const Mesh &m, const vector<bool> &HQ_flag, vector<bool> &H_flag, int &hn, VectorXi &b, MatrixXd &bc) {
	H_flag = HQ_flag;
	for (int i = 0; i < region_Size_untangle; i++) {
		std::vector<bool> H_flag_(H_flag.size(),false);
		for (uint32_t j = 0; j < H_flag.size(); j++) if(H_flag[j]){
			for (const auto &vid : m.Hs[j].vs)for (const auto &nhid : m.Vs[vid].neighbor_hs)H_flag_[nhid] = true;
		}
		H_flag = H_flag_;
	}

	hn = 0;
	for (const auto &g : H_flag)if (g)hn++;

	vector<bool> F_flag(m.Fs.size(), false);
	for (const auto f : m.Fs)if (!f.boundary) {
		if ((H_flag[f.neighbor_hs[0]] && !H_flag[f.neighbor_hs[1]]) || (H_flag[f.neighbor_hs[1]] && !H_flag[f.neighbor_hs[0]]))
			F_flag[f.id] = true;
	}

	vector<bool> V_flag(m.Vs.size(), false);
	vector<uint32_t> vi;vector<Vector3d> vs;
	for (const auto &f : m.Fs)if (F_flag[f.id]) for (const auto &vid : f.vs)if (!V_flag[vid]) {
		V_flag[vid] = true;
		vi.push_back(vid); vs.push_back(m.V.col(vid).transpose());
	}
	b.resize(vi.size());
	bc.resize(vi.size(), 3);
	for (int i = 0; i < vi.size(); i++) {
		b[i] = vi[i];
		bc.row(i) = vs[i];
	}
}
bool grid_hex_meshing_bijective::stop_criterior_satisfied(Mesh_Domain &md, const int iter_after_untangle, const Mesh &tm0, Mesh &tm1, bool &max_dis_satisified, bool & ave_dis_satisfied) {
	
	double bbox_diagonal, max_hausdorff_dis, ave_hausdorff_dis;
	auto &hmi = md.mesh_entire;
	for (int i = 0; i < md.TV_map_reverse.size(); i++)tm1.V.col(i) = hmi.V.col(md.V_map_reverse[md.TV_map_reverse[i]]);

	compute(tm0, tm1, bbox_diagonal, max_hausdorff_dis, ave_hausdorff_dis);
	
	cout << "ave_hausdorff_dis: " << ave_hausdorff_dis << endl;

	ave_Hausdorff_dises.push_back(ave_hausdorff_dis);

	if (iter_after_untangle >= start_ave_hausdorff_count_Iter){
		double ratio_max = max_hausdorff_dis / bbox_diagonal;
		int cur_ind = (int) ave_Hausdorff_dises.size() - 1;
		if (cur_ind < 1) return false;

		double ratio_ave = (ave_Hausdorff_dises[cur_ind - 1] - ave_Hausdorff_dises[cur_ind]) / ave_Hausdorff_dises[cur_ind];

		cout << "ratio_max: " << ratio_max << endl;
		cout << "ratio_ave: " << ratio_ave << endl;

		max_HR = ratio_max;

		ratio_ave_Hausdorff.push_back(ratio_ave);
		ratio_max_Hausdorff.push_back(ratio_max);

		if (ratio_max < 0.8*HR) max_dis_satisified = true;
		else max_dis_satisified = false;
		if (ratio_ave < 0.8*STOP_AVE_HAUSDORFF_THRESHOLD) ave_dis_satisfied = true;
		else ave_dis_satisfied = false;
		if (ratio_ave < STOP_AVE_HAUSDORFF_THRESHOLD && ratio_max < HR) return true;
		if (ave_dis_satisfied);// return true;

	}
	return false;
}

void grid_hex_meshing_bijective::geomesh2mesh(GEO::Mesh &gm, Mesh &m) {
	m.Vs.clear(); m.Es.clear(); m.Fs.clear(); m.Hs.clear();
	m.Vs.resize(gm.vertices.nb());
	m.Fs.resize(gm.facets.nb());
	m.Hs.resize(gm.cells.nb());
	for (uint32_t i = 0; i < m.Vs.size(); i++) {
		Hybrid_V v;
		v.id = i;
		v.v.push_back(gm.vertices.point_ptr(i)[0]);
		v.v.push_back(gm.vertices.point_ptr(i)[1]);
		v.v.push_back(gm.vertices.point_ptr(i)[2]);
		m.Vs[i] = v;
	}
	m.V.resize(3, m.Vs.size());
	for (uint32_t i = 0; i < m.Vs.size(); i++) {
		m.V(0, i) = m.Vs[i].v[0];
		m.V(1, i) = m.Vs[i].v[1];
		m.V(2, i) = m.Vs[i].v[2];
	}

	if (m.type == Mesh_type::Tri) {
		for (uint32_t i = 0; i < m.Fs.size(); i++) {
			Hybrid_F f;
			f.id = i;
			f.vs.resize(gm.facets.nb_vertices(i));
			for (uint32_t j = 0; j < f.vs.size(); j++) {
				f.vs[j] = gm.facets.vertex(i, j);
			}
			m.Fs[i] = f;
		}
	}
	else if (m.type == Mesh_type::Hex) {
		int vmap[8] = { 0,1,3,2,4,5,7,6 };
		for (uint32_t i = 0; i < m.Hs.size(); i++) {
			Hybrid h;
			h.id = i;
			h.vs.resize(8);
			for (uint32_t j = 0; j < 8; j++) {
				h.vs[j] = gm.cells.vertex(i, vmap[j]);
				m.Vs[gm.cells.vertex(i, j)].neighbor_hs.push_back(i);
			}
			m.Hs[i] = h;
		}
	}
}
void grid_hex_meshing_bijective::build_aabb_tree(Mesh &tmi, Treestr &a_tree,bool is_tri) {
	a_tree.TriV = tmi.V.transpose();
	a_tree.TriF.resize(tmi.Fs.size(), 3);

	for (uint32_t i = 0; i < tmi.Fs.size(); i++) {
		a_tree.TriF(i, 0) = tmi.Fs[i].vs[0];
		a_tree.TriF(i, 1) = tmi.Fs[i].vs[1];
		a_tree.TriF(i, 2) = tmi.Fs[i].vs[2];
	}
	if (!is_tri) {
		// Precompute signed distance AABB tree
		a_tree.tree.init(a_tree.TriV, a_tree.TriF);
		// Precompute vertex,edge and face normals
		igl::per_face_normals(a_tree.TriV, a_tree.TriF, a_tree.TriFN);
		igl::per_vertex_normals(a_tree.TriV, a_tree.TriF, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, a_tree.TriFN, a_tree.TriVN);
		igl::per_edge_normals(a_tree.TriV, a_tree.TriF, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, a_tree.TriFN, a_tree.TriEN, a_tree.TriE, a_tree.TriEMAP);
	}
}
bool grid_hex_meshing_bijective::hausdorff_ratio_check(Mesh &m0, Mesh &m1) {

	std::function<void(Mesh &, Mesh &, vector<bool> &, int &) > re_indexing = [&](Mesh &M, Mesh &m, vector<bool> &V_flag, int & N)->void {
			m.V.resize(3, N); N = 0; vector<int> v_map(M.Vs.size(), 0);
			for (uint32_t i = 0; i < V_flag.size(); i++) if (V_flag[i]) { m.V.col(N) = M.V.col(i); v_map[i] = N++; }
			for (auto f : M.Fs) {
				if (!f.boundary) continue;
				vector<uint32_t> bvs; for (auto vid : f.vs)if (V_flag[vid]) bvs.push_back(v_map[vid]);
				if (bvs.size() == 3) {
					Hybrid_F hf; hf.vs = bvs;
					m.Fs.push_back(hf);
				}
				else if (bvs.size() == 4) {
					Hybrid_F hf;
					hf.vs.push_back(bvs[0]);
					hf.vs.push_back(bvs[1]);
					hf.vs.push_back(bvs[2]);
					m.Fs.push_back(hf);
					hf.vs.clear();
					hf.vs.push_back(bvs[2]);
					hf.vs.push_back(bvs[3]);
					hf.vs.push_back(bvs[0]);
					m.Fs.push_back(hf);
				}
			}
		};
		vector<bool> V_flag; int bvN = 0, bbvN = 0;

		Mesh Mglobal;
		V_flag.resize(m1.Vs.size()); std::fill(V_flag.begin(), V_flag.end(), false);
		bvN = 0;
		for (uint32_t i = 0; i < m1.Vs.size(); i++) if (m1.Vs[i].boundary) { V_flag[i] = true; bvN++; }
		re_indexing(m1, Mglobal, V_flag, bvN);

		if (!compute(m0, Mglobal, hausdorff_ratio, hausdorff_ratio_threshould)) {
			cout << "too large hausdorff distance" << endl;
			return false;
		}
		return true;
}
bool grid_hex_meshing_bijective::hausdorff_ratio_check(Mesh &m0, Mesh &m1, double & hausdorff_dis_threshold) {
	std::function<void(Mesh &, Mesh &, vector<bool> &, int &) > re_indexing = [&](Mesh &M, Mesh &m, vector<bool> &V_flag, int & N)->void {
		m.V.resize(3, N); N = 0; vector<int> v_map(M.Vs.size(), 0);
		for (uint32_t i = 0; i < V_flag.size(); i++) if (V_flag[i]) { m.V.col(N) = M.V.col(i); v_map[i] = N++; }
		for (auto f : M.Fs) {
			if (!f.boundary) continue;
			vector<uint32_t> bvs; for (auto vid : f.vs)if (V_flag[vid]) bvs.push_back(v_map[vid]);
			if (bvs.size() == 3) {
				Hybrid_F hf; hf.vs = bvs;
				m.Fs.push_back(hf);
			}
			else if (bvs.size() == 4) {
				Hybrid_F hf;
				hf.vs.push_back(bvs[0]);
				hf.vs.push_back(bvs[1]);
				hf.vs.push_back(bvs[2]);
				m.Fs.push_back(hf);
				hf.vs.clear();
				hf.vs.push_back(bvs[2]);
				hf.vs.push_back(bvs[3]);
				hf.vs.push_back(bvs[0]);
				m.Fs.push_back(hf);
			}
		}
	};
	vector<bool> V_flag; int bvN = 0, bbvN = 0;

	Mesh Mglobal;
	V_flag.resize(m1.Vs.size()); std::fill(V_flag.begin(), V_flag.end(), false);
	bvN = 0;
	for (uint32_t i = 0; i < m1.Vs.size(); i++) if (m1.Vs[i].boundary) { V_flag[i] = true; bvN++; }
	re_indexing(m1, Mglobal, V_flag, bvN);

	if (!compute(m0, Mglobal, hausdorff_ratio, hausdorff_ratio_threshould, hausdorff_dis_threshold)) {
		cout << "too large hausdorff distance" << endl;
		return false;
	}
	return true;
}
grid_hex_meshing_bijective::~grid_hex_meshing_bijective()
{
}
