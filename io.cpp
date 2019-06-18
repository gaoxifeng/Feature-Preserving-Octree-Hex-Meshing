#include "io.h"
#include <igl/remove_duplicate_vertices.h>
using namespace std;

bool h_io::read_hybrid_mesh_VTK(Mesh &hmi, std::string path){
	std::ifstream ff(path, std::ios::in);
	if (ff.fail()) return false;
	char s[1024], sread[1024], sread2[1024];
	uint32_t vnum, hnum;	double x, y, z;

	bool find = false; uint32_t lines = 0;
	while (!find)
	{
		ff.getline(s, 1023);
		if (sscanf(s, "%s %d %s", &sread, &vnum, &sread2) == 3 && (strcmp(sread, "POINTS") == 0))
			find = true;
		if (++lines>10)
			throw std::runtime_error("cannot find head of VTK!");
	}
	hmi.V.resize(3, vnum);
	hmi.V.setZero();
	hmi.Vs.resize(vnum);
	for (uint32_t i = 0; i<vnum; i++)
	{
		ff.getline(s, 1023);
		sscanf(s, "%lf %lf %lf", &x, &y, &z);

		hmi.V(0, i) = x;
		hmi.V(1, i) = y;
		hmi.V(2, i) = z;

		Hybrid_V v;
		v.id = i; v.boundary = false;
		hmi.Vs[i] = v;
	}

	find = false;
	while (!find)
	{
		uint32_t temp_int;
		ff.getline(s, 1023);
		if (sscanf(s, "%s %d %d", &sread, &hnum, &temp_int) == 3 && (strcmp(sread, "CELLS") == 0))
			find = true;
	}
	hmi.Hs.resize(hnum); 
	Hybrid h;
	uint32_t a, b, c, d, e, f, g, m, o, p, q;
	for (uint32_t i = 0; i<hnum; i++)
	//while (ff.getline(s, 1023))
	{
		ff.getline(s, 1023);
		if (sscanf(s, "%d %d %d %d %d", &vnum, &a, &b, &c, &d) == 5 && vnum == 4) {
			h.vs.resize(4);
			//a--;b--;c--;d--;//temporarily
			h.vs[0] = a;
			h.vs[1] = b;
			h.vs[2] = c;
			h.vs[3] = d;
		}
		else if (sscanf(s, "%d %d %d %d %d %d", &vnum, &a, &b, &c, &d, &e) == 6 && vnum == 5) {
			h.vs.resize(5);
			h.vs[0] = a;
			h.vs[1] = b;
			h.vs[2] = c;
			h.vs[3] = d;
			h.vs[4] = e;
		}
		else if (sscanf(s, "%d %d %d %d %d %d %d", &vnum, &a, &b, &c, &d, &e, &f) == 7 && vnum == 6)
		{
			h.vs.resize(6);
			h.vs[0] = a;
			h.vs[1] = b;
			h.vs[2] = c;
			h.vs[3] = d;
			h.vs[4] = e;
			h.vs[5] = f;
		}
		else if (sscanf(s, "%d %d %d %d %d %d %d %d %d", &vnum, &a, &b, &c, &d, &e, &f, &g, &m) == 9 && vnum == 8)
		{
			h.vs.resize(8);
			h.vs[0] = a;
			h.vs[1] = b;
			h.vs[2] = c;
			h.vs[3] = d;
			h.vs[4] = e;
			h.vs[5] = f;
			h.vs[6] = g;
			h.vs[7] = m;
			//std::reverse(h.vs.begin(), h.vs.begin() + 4);
			//std::reverse(h.vs.begin() + 4, h.vs.end());
		}
		else { 
			std::cout << "Wrong format of input file!" << endl; system("PAUSE"); 
		}

		h.id = i; hmi.Hs[h.id] = h;
		for (uint32_t i = 0; i < h.vs.size(); i++) hmi.Vs[h.vs[i]].neighbor_hs.push_back(h.id);
	}
	ff.close();
	return true;
}
void h_io::write_hybrid_mesh_VTK(Mesh &hmi, std::string path)
{
	std::fstream f (path, std::ios::out);

	f << "# vtk DataFile Version 2.0" << std::endl << "mesh vtk data - converted from .off" << std::endl;
	f << "ASCII" << std::endl;
	f << "DATASET UNSTRUCTURED_GRID" << std::endl;

	//f << "POINTS " << hmi.Vs.size() << " double" << std::endl;
	//for (uint32_t i = 0; i<hmi.V.cols(); i++)
	//	f << hmi.V(0, i) << " " << hmi.V(1, i) << " " << hmi.V(2, i) << std::endl;
	f << "POINTS " << hmi.V.cols() << " double" << std::endl;
	/*for (const auto &v:hmi.Vs)
		f << v.v[0] << " " << v.v[1] << " " << v.v[2] << std::endl;*/
	for (int i = 0; i<hmi.V.cols(); i++)
		f << hmi.V(0, i) << " " << hmi.V(1, i) << " " << hmi.V(2, i) << std::endl;

	if (hmi.type == Mesh_type::Tri || hmi.type == Mesh_type::Qua) {
		uint32_t vnum = hmi.Fs[0].vs.size();
		f << "CELLS " << hmi.Fs.size() << " " << hmi.Fs.size() * (vnum + 1) << std::endl;

		for (uint32_t i = 0; i < hmi.Fs.size(); i++) {
			f << " " << vnum << " ";
			for (uint32_t j = 0; j < vnum; j++) f << hmi.Fs[i].vs[j] << " ";
			f << std::endl;
		}
		f << "CELL_TYPES " << hmi.Fs.size() << std::endl;
		for (uint32_t i = 0; i < hmi.Fs.size(); i++)
			if (hmi.type == 0) f << 5 << std::endl; else f << 9 << std::endl;
	}else if (hmi.type == Mesh_type::Hyb) {
		uint32_t vnum = 0;
		for (auto f : hmi.Fs)vnum += f.vs.size() + 1;
		f << "CELLS " << hmi.Fs.size() << " " << vnum << std::endl;

		for (auto ff : hmi.Fs){
			f << " " << ff.vs.size() << " ";
			for(auto vid: ff.vs) f << vid << " ";
			f << std::endl;
		}
		f << "CELL_TYPES " << hmi.Fs.size() << std::endl;
		for (uint32_t i = 0; i < hmi.Fs.size(); i++)
			f << 7 << std::endl;
	}
	else {
		uint32_t vnum = hmi.Hs[0].vs.size();
		f << "CELLS " << hmi.Hs.size() << " " << hmi.Hs.size() * (vnum + 1) << std::endl;

		for (uint32_t i = 0; i < hmi.Hs.size(); i++)
		{
			f << " " << vnum << " ";
			for (uint32_t j = 0; j < vnum; j++)
				f << hmi.Hs[i].vs[j] << " ";
			/*f << hmi.Hs[i].vs[3] << " ";
			f << hmi.Hs[i].vs[2] << " ";
			f << hmi.Hs[i].vs[1] << " ";
			f << hmi.Hs[i].vs[0] << " ";
			f << hmi.Hs[i].vs[7] << " ";
			f << hmi.Hs[i].vs[6] << " ";
			f << hmi.Hs[i].vs[5] << " ";
			f << hmi.Hs[i].vs[4] << " ";*/
			f<< std::endl;
		}
		f << "CELL_TYPES " << hmi.Hs.size() << std::endl;
		for (uint32_t i = 0; i < hmi.Hs.size(); i++)
			if(hmi.type== Mesh_type::Tet)
				f << 10 << std::endl;
			else
				f << 12 << std::endl;
	}


	f << "POINT_DATA " << hmi.Vs.size() << std::endl;
	f << "SCALARS fixed int" << std::endl;
	f << "LOOKUP_TABLE default" << std::endl;
	for (uint32_t i = 0; i<hmi.Vs.size(); i++) {
		if (hmi.Vs[i].boundary) f << "1" << std::endl; else f << "0" << std::endl;
	}
	f.close();
}
bool h_io::read_hybrid_mesh_MESH(Mesh &hmi, std::string path)
{
	hmi.Vs.clear(); hmi.Hs.clear();
	std::fstream f(path, std::ios::in);
	if (f.fail()) return false;
	char s[1024], sread[1024];
	int vnum, hnum;	double x, y, z;

	int find = false;
	while (!find)
	{
		f.getline(s, 1023);
		if (sscanf(s, "%s%d", &sread, &vnum) == 2 && (strcmp(sread, "Vertices") == 0))
		{
			find = true;
		}
		else if(sscanf(s, "%s", &sread) == 1 && (strcmp(sread, "Vertices") == 0))
		{
			find = true;
			f.getline(s, 1023);
			sscanf(s, "%d", &vnum);
		}
	}
	hmi.V.resize(3, vnum);
	hmi.V.setZero();
	hmi.Vs.resize(vnum);

	for (int i = 0; i<vnum; i++)
	{
		f.getline(s, 1023);
		sscanf(s, "%lf %lf %lf", &x, &y, &z);

		hmi.V(0, i) = x;
		hmi.V(1, i) = y;
		hmi.V(2, i) = z;

		Hybrid_V v;
		v.id = i; v.boundary = false;
		hmi.Vs[i] = v;
	}
	find = false;
	bool hex_found = false, tet_found = false;
	while (!find)
	{
		int temp_int;
		f.getline(s, 1023);
		if (sscanf(s, "%s%d", &sread, &hnum) == 2 && (strcmp(sread, "Hexahedra") == 0))
		{
			hex_found = true;
			find = true;
		}
		else if (sscanf(s, "%s", &sread) == 1 && (strcmp(sread, "Hexahedra") == 0))
		{
			f.getline(s, 1023);
			sscanf(s, "%d", &hnum);
			hex_found = true;
			find = true;
		}
		else if (sscanf(s, "%s", &sread) == 1 && (strcmp(sread, "Tetrahedra") == 0))
		{
			f.getline(s, 1023);
			sscanf(s, "%d", &hnum);
			tet_found = true;
			find = true;
		}
	}

	if (hex_found)hmi.type = Mesh_type::Hex;else if(tet_found) hmi.type = Mesh_type::Tet;

	hmi.Hs.resize(hnum);
	Hybrid h;
	int32_t a, b, c, d, e, ff, g, m, o, p, q;

	int hid = 0;
	for (int i = 0; i<hnum; i++)
	{
		f.getline(s, 1023);
		int a, b, c, d, e, f, g, m;
		if (hex_found) {
			//sscanf(s,"%d %d %d %d %d %d %d %d %d",&vnum,&a,&b,&c,&d,&e,&f,&g,&m);
			sscanf(s, "%d %d %d %d %d %d %d %d %d", &a, &b, &c, &d, &e, &ff, &g, &m, &vnum);

			a--; b--; c--; d--; e--; ff--; g--; m--;

			h.vs.resize(8);
			h.vs[0] = a;
			h.vs[1] = b;
			h.vs[2] = c;
			h.vs[3] = d;
			h.vs[4] = e;
			h.vs[5] = ff;
			h.vs[6] = g;
			h.vs[7] = m;
		}
		else if (tet_found) {
			//sscanf(s,"%d %d %d %d %d %d %d %d %d",&vnum,&a,&b,&c,&d,&e,&f,&g,&m);
			sscanf(s, "%d %d %d %d %d", &a, &b, &c, &d, &e);

			a--; b--; c--; d--;

			h.vs.resize(4);
			h.vs[0] = a;
			h.vs[1] = b;
			h.vs[2] = c;
			h.vs[3] = d;
		}

		h.id = i; hmi.Hs[h.id] = h;
		for (uint32_t i = 0; i < h.vs.size(); i++) hmi.Vs[h.vs[i]].neighbor_hs.push_back(h.id);
	}

	f.close();
	return true;
}
void h_io::write_hybrid_mesh_MESH(Mesh &hmi, std::string path)
{
	std::fstream f(path, std::ios::out);

	f << "MeshVersionFormatted 1" << std::endl;
	f << "Dimension 3" << std::endl;
	f << "Vertices" << " " << hmi.V.cols() << std::endl;
	for (int i = 0; i<hmi.V.cols(); i++)
		f<< hmi.V(0, i) << " " << hmi.V(1, i) << " " << hmi.V(2, i) << " " << 0 << std::endl;

	if (hmi.type == Mesh_type::Tri || hmi.type == Mesh_type::HSur) {
		f<< "Triangles" << endl;
		f<< hmi.Fs.size() << std::endl;

		for (int i = 0; i<hmi.Fs.size(); i++){
			f<< hmi.Fs[i].vs[0] + 1 << " " << hmi.Fs[i].vs[1] + 1 << " " << hmi.Fs[i].vs[2] + 1 << " " << 0 << std::endl;
		}
	}else if(hmi.type == Mesh_type::Hex) {
		f<< "Hexahedra" << endl;
		f<< hmi.Hs.size() << std::endl;

		for (int i = 0; i < hmi.Hs.size(); i++) {
			for (auto vid : hmi.Hs[i].vs)
				f<< vid + 1 << " ";
			f<<0<<  endl;
		}
	}
	f << "End";
	f.close();
}

void h_io::read_hybrid_mesh_OBJ(Mesh &hmi, string path) {
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	igl::readOBJ(path, V, F);

	hmi.Vs.resize(V.rows());
	hmi.V = V.transpose();
	int i = 0;
	for (auto &v: hmi.Vs) {
		v.id = i;
		v.v.resize(3);
		v.v[0] = V(v.id, 0);
		v.v[1] = V(v.id, 1);
		v.v[2] = V(v.id, 2);
		v.boundary = false;
		hmi.Vs[i] = v;
		i++;
	}
	hmi.Fs.resize(F.rows());
	i = 0;
	for (auto &f : hmi.Fs) {
		f.id = i;
		f.vs.resize(F.row(i).size());
		for (uint32_t j = 0; j < f.vs.size(); j++) {
			f.vs[j] = F(i, j);
			hmi.Vs[f.vs[j]].neighbor_fs.push_back(i);
		}
		hmi.Fs[i] = f;
		i++;
	}

}
void h_io::write_hybrid_mesh_OBJ(Mesh &hmi, std::string path) {
	std::fstream f(path, std::ios::out);
	if (hmi.V.cols()) {
		for (int i = 0; i<hmi.V.cols(); i++)
			f << "v " << hmi.V(0, i) << " " << hmi.V(1, i) << " " << hmi.V(2, i) << std::endl;
	}
	else {
		for (int i = 0; i<hmi.Vs.size(); i++)
			f << "v " << hmi.Vs[i].v[0] << " " << hmi.Vs[i].v[1] << " " << hmi.Vs[i].v[2] << std::endl;
	}

	for (int i = 0; i < hmi.Fs.size(); i++) {
		f << "f";
		for (int j = 0; j < hmi.Fs[i].vs.size(); j++)
			f << " " << hmi.Fs[i].vs[j] + 1;
		f << endl;
	}
	f.close();
}

void h_io::write_mesh_OBJ(Eigen::MatrixXd &V, Eigen::MatrixXi &F, bool col, std::string path) {
	std::fstream f(path, std::ios::out);

	if (col) {
		for (int i = 0; i < V.cols(); i++) {
			f << "v ";
			for (int j = 0; j<V.rows(); j++)
				f << V(j, i) <<" ";
			f << std::endl;
		}
		for (int i = 0; i < F.cols(); i++) {
			f << "f";
			for (int j = 0; j < F.rows(); j++)
				f << " " << F(j, i) + 1;
			f << endl;
		}
	}
	else {
		for (int i = 0; i < V.rows(); i++) {
			f << "v ";
			for (int j = 0; j<V.cols(); j++)
				f << V(i, j) << " ";
			f << std::endl;
		}
		for (int i = 0; i < F.rows(); i++) {
			f << "f";
			for (int j = 0; j < F.cols(); j++)
				f << " " << F(i, j) + 1;
			f << endl;
		}
	}
	f.close();
}

bool h_io::read_feature_Graph_FGRAPH(Mesh_Feature &mf, string path) {
	std::fstream f(path, std::ios::in);
	if (f.fail()) return false;
	char s[1024]; int cnum = 0, edgenum=0;
	f.getline(s, 1023);
	sscanf(s, "%lf %i %i", &mf.angle_threshold, &mf.orphan_curve, &mf.orphan_curve_single);
	f.getline(s, 1023);
	sscanf(s, "%i %i", &cnum, &edgenum);
	mf.IN_corners.resize(cnum);
	for (int i = 0; i < cnum; i++) {
		f.getline(s, 1023);
		sscanf(s, "%i", &(mf.IN_corners[i]));
	}
	mf.IN_v_pairs.resize(edgenum);
	for (int i = 0; i < edgenum; i++) {
		f.getline(s, 1023);
		int v0 = -1, v1 = -1;
		sscanf(s, "%i %i", &v0, &v1);
		mf.IN_v_pairs[i].push_back(v0);
		mf.IN_v_pairs[i].push_back(v1);
	}
	f.close();
	return true;
}
void h_io::write_feature_Graph_FGRAPH(Mesh_Feature &mf, string path) {
	std::fstream f(path, std::ios::out);
	int n = 0;
	for (auto e : mf.E_feature_flag)if (e)n++;

	f << mf.angle_threshold << " " << mf.orphan_curve << " " << mf.orphan_curve_single << endl;
	f << mf.corners.size() << " "<< n << endl;
	for(auto cid: mf.corners)f << cid << std::endl;
	for (const auto &e : mf.tri.Es)if(mf.E_feature_flag[e.id]) f << e.vs[0]<<" " << e.vs[1] << std::endl;

	f.close();
}