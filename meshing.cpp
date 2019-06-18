#include "meshing.h"
#include "grid_meshing/grid_hex_meshing.h"
using namespace std;

bool meshing::processing(string &path)
{
	//GEO::Mesh gtmesh;
	Mesh tmesh;
	tmesh.type = Mesh_type::Tri;
	GEO::mesh_load(path + ".obj", M_i);
	geomesh2mesh(M_i, tmesh);	

	if (!tmesh.Vs.size() || !tmesh.Fs.size())
	{
		std::cout << "empty file" << std::endl;
		return false;
	}
	build_connectivity(tmesh);

	if (!manifoldness_closeness_check(tmesh))
	{
		std::cout << "non-manifold model" << std::endl;
		return false;
	}

	mf.read_from_file = false;
	if (io.read_feature_Graph_FGRAPH(mf, path + ".fgraph")) {
		mf.read_from_file = true;
	}

	std::vector<Mesh> components;
	decompose_components(tmesh, components);	
	
	std::vector<Mesh> results(components.size());
	
	for(int i=0;i<components.size();i++)
	{
		std::cout << "components "<<i+1<<" of "<<components.size() << endl;
		Eigen::MatrixXd T, IT;
		forward_positioning(components[i], T);
		mesh2geomesh(components[i], M_i);
		//preprocessing features
		feature(components[i], path);
		pipeline(results[i]);
		back_positioning(results[i], T);
	}
	//compose all the components
	mesho.type = Mesh_type::Hex;
	combine_components(results, mesho);
	return true;
}
bool meshing::feature(Mesh &mesh, string &path)
{
	mf.angle_threshold = 0;	
	if (!mf.read_from_file) 
	{
		std::cout << "no feature file, detect based on angle" << std::endl;
		mf.angle_threshold = 0;	
		mf.orphan_curve = true;
		mf.orphan_curve_single = true;
	}

	mf.tri = mesh;
	triangle_mesh_feature(mf);
	build_feature_graph(mf, fg);

	if(!mf.read_from_file)
	{
		;
	}

	return true;
}
bool meshing::pipeline(Mesh &hmesh)
{
	grid_hex_meshing_bijective hex_meshing;
	if(args.Hausdorff_ratio_t != 0)
		hex_meshing.HR = args.Hausdorff_ratio_t;
	
	if (hex_meshing.HR <= 0) {
		cout << "please provide a reasonable hausdorff distance ratio > 0" << endl;
		return false;
	}

	hex_meshing.MD.mesh_entire.type = Mesh_type::Hex;
	hex_meshing.MD.mesh_subA.type = Mesh_type::Hex;
	hex_meshing.MD.mesh_subB.type = Mesh_type::Hex;

	hex_meshing.STOP_EXTENT_MIN = hex_meshing.STOP_EXTENT_MAX = 15;
	if(args.edge_length_ratio != 0)
		hex_meshing.STOP_EXTENT_MIN = hex_meshing.STOP_EXTENT_MAX = args.edge_length_ratio;

	hex_meshing.weight_opt = args.weight_opt;
	hex_meshing.pipeline("");

	hmesh = hex_meshing.MD.mesh_subA;
	meshob = hex_meshing.MD.mesh_subB;
	mqo = hex_meshing.mq;
	
	return true;
}

void meshing::forward_positioning(Mesh &m, Eigen::MatrixXd &T)
{
	T.resize(4,4);
	T.setIdentity();
	Eigen::Vector3d min_corner = m.V.rowwise().minCoeff();
	Eigen::Vector3d max_corner = m.V.rowwise().maxCoeff();

	Eigen::Vector3d extent = max_corner - min_corner;
	Eigen::Vector3d center = (max_corner + min_corner) * 0.5;
	double volume_aabb = extent[0]*extent[1]*extent[2];

	Eigen::MatrixXd TP;
	Eigen::VectorXd S;
	if(!PCA_BBOX(m.V, TP, S))
		return;
	
	double volume_oobb = S[0]*S[1]*S[2];

	if(volume_oobb >= volume_aabb)
	{
		std::cout<<"AABB"<<std::endl;
		double s = extent.maxCoeff();
		T(2,2) = T(1,1) = T(0,0) = 1.0/s; 
		T(0, 3) = -center[0];
		T(1, 3) = -center[1];
		T(2, 3) = -center[2];
	}
	else
	{
		std::cout<<"PCA"<<std::endl;
		double s = S.maxCoeff();
		Eigen::Matrix3d S_matrix;
		S_matrix.setZero();
		S_matrix(2,2) = S_matrix(1,1) = S_matrix(0,0) = 1.0/s; 
		T = TP;
		T.block(0,0,3,3) = S_matrix*TP.block(0,0,3,3);
	}
	Eigen::MatrixXd V = m.V;
	V.row(0).array() += T(0, 3);
	V.row(1).array() += T(1, 3);
	V.row(2).array() += T(2, 3);

	m.V = T.block(0, 0, 3, 3)*V;

	for (int i = 0; i < m.V.cols(); i++) {
		m.Vs[i].v[0] = m.V(0, i);
		m.Vs[i].v[1] = m.V(1, i);
		m.Vs[i].v[2] = m.V(2, i);
	}
}
void meshing::back_positioning(Mesh &m, MatrixXd &T)
{
	if(!T.isIdentity())
	{
		Eigen::MatrixXd V = m.V;
		m.V = T.block(0, 0, 3, 3).inverse()*V;
		m.V.row(0).array() -= T(0, 3);
		m.V.row(1).array() -= T(1, 3);
		m.V.row(2).array() -= T(2, 3);

		for (int i = 0; i < m.V.cols(); i++) {
			m.Vs[i].v[0] = m.V(0, i);
			m.Vs[i].v[1] = m.V(1, i);
			m.Vs[i].v[2] = m.V(2, i);
		}
	}
}
void meshing::geomesh2mesh(GEO::Mesh &gm, Mesh &m) {
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
void meshing::mesh2geomesh(Mesh &m, GEO::Mesh &gm) {
	gm.clear();
	gm.vertices.create_vertices((int) m.Vs.size());
	for (int i = 0; i < (int) gm.vertices.nb(); ++i) {
		GEO::vec3 &p = gm.vertices.point(i);
		p[0] = m.Vs[i].v[0];
		p[1] = m.Vs[i].v[1];
		p[2] = m.Vs[i].v[2];
	}
	if (m.type == Mesh_type::Tri) {
		gm.facets.create_triangles((int) m.Fs.size());

		for (int c = 0; c < (int) gm.facets.nb(); ++c) {
			for (int lv = 0; lv < 3; ++lv) {
				gm.facets.set_vertex(c, lv, m.Fs[c].vs[lv]);
			}
		}
	}
	gm.facets.connect();
	gm.cells.connect();
	if(gm.cells.nb() != 0 && gm.facets.nb() == 0) {
		gm.cells.compute_borders();
	}

}
