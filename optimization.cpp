#include "optimization.h"
#include "igl/procrustes.h"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

bool optimization::pipeline() {
	V_Boundary_flag.clear(); V_Boundary_flag.resize(mesh.Vs.size(), false);
	BV.resize(mesh.Vs.size(), 3); BV.setZero();
	for (auto v : mesh.Vs) {
		V_Boundary_flag[v.id] = v.boundary;
		BV.row(v.id) = mesh.V.col(v.id);
	}


	if (gluing()) recovering();
	else cout << "optimization failure" << endl;
	return true;
};


bool optimization::fix_boundary() {

	//quality
	scaled_jacobian(mesh, mq);
	std::cout << "before: minimum scaled J: " << mq.min_Jacobian << " average scaled J: " << mq.ave_Jacobian << endl;
	Tetralize_Set ts;
//****************one at a time****************//
int32_t nv = 0;
vector<uint32_t> V_ranges; V_ranges.reserve(mesh.Vs.size());
	for (auto v : mesh.Vs) {
		nv += v.neighbor_hs.size();
		V_ranges.push_back(nv - 1);
	}

	ts.V_ranges = V_ranges;
	ts.V.resize(nv, 3);
	//T_
	ts.T.resize(mesh.Hs.size() * 8, 4);
	Vector4i t;

	for (auto h : mesh.Hs) {

		for (uint32_t i = 0; i < 8; i++) {
			for (uint32_t j = 0; j < 4; j++) t[j] = V_ranges[h.vs[hex_tetra_table[i][j]]];
			ts.T.row(h.id * 8 + i) = t;
		}
		bool have_boundary_v = false;
		for (uint32_t i = 0; i < 8; i++) {
			if (V_Boundary_flag[h.vs[i]]) {

				have_boundary_v = true;
			}
		}
		if (have_boundary_v)
		{
			Tetralize_Set ts_i;
			ts_i.V_ranges = ts.V_ranges;
			ts_i.T.resize(8, 4);
			for (uint32_t i = 0; i < 8; i++) {
				for (uint32_t j = 0; j < 4; j++) t[j] = hex_tetra_table[i][j];
				ts_i.T.row(i) = t;
			}
			ts_i.V.resize(8, 3);
			for (uint32_t i = 0; i < 8; i++) {
				ts_i.V(i, 0) = hex_ref_shape[i][0];
				ts_i.V(i, 1) = hex_ref_shape[i][1];
				ts_i.V(i, 2) = hex_ref_shape[i][2];
			}
			//ts_i.V = mesh.V.transpose();
			uint32_t bvn = 0;
			for (uint32_t i = 0; i < 8; i++) if (V_Boundary_flag[h.vs[i]]) bvn++;
			ts_i.b.resize(bvn);
			ts_i.bc.resize(bvn, 3); bvn = 0;
			for (uint32_t i = 0; i < 8; i++) if (V_Boundary_flag[h.vs[i]]) {
				ts_i.b[bvn] = i;

				Vector3d rand_v = Vector3d::Random();
				ts_i.bc.row(bvn++) = BV.row(h.vs[i]);
				cout << "which v " << i << endl;
			}
			cout << "boundary num " << bvn << endl;

			ts_i.global = false;
			ts_i.projection = true;
			ts_i.lamda_projection = 1e+2;
			ts_i.glue = false;
			ts_i.lamda_glue = 1e5;
			slim_m_opt(ts_i, -1, 0);

			double hex_minJ = 1, average_minJ = 0;
			for (uint32_t i = 0; i<ts_i.T.rows(); i++)
			{
				VectorXd c0 = ts_i.V.row(ts_i.T(i, 0));
				VectorXd c1 = ts_i.V.row(ts_i.T(i, 1));
				VectorXd c2 = ts_i.V.row(ts_i.T(i, 2));
				VectorXd c3 = ts_i.V.row(ts_i.T(i, 3));

				double jacobian_value = a_jacobian(c0, c1, c2, c3);

				if (hex_minJ>jacobian_value) hex_minJ = jacobian_value;
				average_minJ += jacobian_value;
			}
			for (uint32_t i = 0; i < 8; i++) {
				ts.V.row(V_ranges[h.vs[i]]) = ts_i.V.row(i);
				V_ranges[h.vs[i]]--;
			}

			cout << "relocate cube " << h.id << endl;
		}
		else {
			for (uint32_t i = 0; i < 8; i++) {
				ts.V(V_ranges[h.vs[i]], 0) = hex_ref_shape[i][0];
				ts.V(V_ranges[h.vs[i]], 1) = hex_ref_shape[i][1];
				ts.V(V_ranges[h.vs[i]], 2) = hex_ref_shape[i][2];
				V_ranges[h.vs[i]]--;
			}
		}
	}


	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) {
		int start = 0, end = ts.V_ranges[i];
		if (i != 0) start = ts.V_ranges[i - 1] + 1;
		Vector3d v; v.setZero();
		for (uint32_t j = start; j <= end; j++) v += ts.V.row(j);
		v /= (end - start + 1);
		mesh.V.col(i) = v;
	}

	Mesh mesh_; mesh_.type = Mesh_type::Tet;
	mesh_.V = ts.V.transpose();
	mesh_.Hs.resize(ts.T.rows());
	for (uint32_t i = 0; i < ts.T.rows(); i++) {
		Hybrid h;
		h.vs.resize(4);
		for (uint32_t j = 0; j < 4; j++)h.vs[j] = ts.T(i, j);
		mesh_.Hs[i] = h;
	}

	double hex_minJ = 1, average_minJ = 0;
	for (uint32_t i = 0; i<ts.T.rows(); i++)
	{
		VectorXd c0 = ts.V.row(ts.T(i, 0));
		VectorXd c1 = ts.V.row(ts.T(i, 1));
		VectorXd c2 = ts.V.row(ts.T(i, 2));
		VectorXd c3 = ts.V.row(ts.T(i, 3));

		double jacobian_value = a_jacobian(c0, c1, c2, c3);

		if (hex_minJ>jacobian_value) hex_minJ = jacobian_value;
		average_minJ += jacobian_value;

		if (jacobian_value < 0) {
			cout << "negative jacobian " << i << " " << i / 8 << " " << i % 8 << endl;

		}
	}
	scaled_jacobian(mesh, mq);
	cout << "after: minimum scaled J: " << mq.min_Jacobian << " average scaled J: " << mq.ave_Jacobian << endl;
	if (hex_minJ < 0) return false;

//===================================mapping===================================//
	vector<int32_t> mappedV(ts.V.rows(), -1), mappedV_reverse(ts.V.rows(), -1);
	nv = 0;
	for (uint32_t i = 0; i < ts.V_ranges.size();i++) {
		if (V_Boundary_flag[i]) continue;
		int start = 0, end = ts.V_ranges[i];
		if (i != 0) start = ts.V_ranges[i - 1] + 1;
		for (uint32_t j = start; j < end + 1; j++) {
			mappedV[j] = nv;
			mappedV_reverse[nv++] = j;
		}
	}

	ts.known_value_post = true; ts.post_index = nv;
	ts.post_Variables.resize(ts.V.rows() - nv, 3);

	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) {
		if (V_Boundary_flag[i]) {
			int start = 0, end = ts.V_ranges[i];
			if (i != 0) start = ts.V_ranges[i - 1] + 1;
			for (uint32_t j = start; j < end + 1; j++) {
				mappedV[j] = nv;
				mappedV_reverse[nv] = j;
				ts.post_Variables.row(nv++ - ts.post_index) = BV.row(i);
			}
		}
	}

	V_ranges = ts.V_ranges;
	//reorder variables
	MatrixXd tempV = ts.V;
	nv = 0; int id = 0;
	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) if (!V_Boundary_flag[i]) {
		int start = 0, end = V_ranges[i];
		if (i != 0) start = V_ranges[i - 1] + 1;
		nv += end - start + 1;
		ts.V_ranges[id++] = nv - 1;
		//cout << ts.V_ranges[id-1] << endl;
	}
	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) if (V_Boundary_flag[i]) {
		int start = 0, end = V_ranges[i];
		if (i != 0) start = V_ranges[i - 1] + 1;
		nv += end - start + 1;
		ts.V_ranges[id++] = nv - 1;
		//cout << ts.V_ranges[id - 1] << endl;
	}
	for (uint32_t i = 0; i < ts.V.rows(); i++) ts.V.row(i) = tempV.row(mappedV_reverse[i]);
	for (uint32_t i = 0; i < ts.T.rows(); i++)for (uint32_t j = 0; j < 4; j++) ts.T(i, j) = mappedV[ts.T(i, j)];

	ts.global = false;
	ts.projection = false;
	ts.lamda_projection = 1e+5;
	ts.glue = true;
	ts.known_value_post = true;
	ts.lamda_glue = 2e1;
	slim_m_opt(ts, -1, 1);
	tempV = ts.V;

	ts.V_ranges = V_ranges;
	for (uint32_t i = 0; i < ts.V.rows(); i++) ts.V.row(i) = tempV.row(mappedV[i]);
	for (uint32_t i = 0; i < ts.T.rows(); i++)for (uint32_t j = 0; j < 4; j++) ts.T(i, j) = mappedV_reverse[ts.T(i, j)];

	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) {
		int start = 0, end = ts.V_ranges[i];
		if (i != 0) start = ts.V_ranges[i - 1] + 1;
		Vector3d v; v.setZero();
		for (uint32_t j = start; j <= end; j++) v += ts.V.row(j);
		v /= (end - start + 1);
		mesh.V.col(i) = v;
	}

	//Mesh mesh_; 
	mesh_.type = Mesh_type::Tet;
	mesh_.V = ts.V.transpose();
	mesh_.Hs.resize(ts.T.rows());
	for (uint32_t i = 0; i < ts.T.rows(); i++) {
		Hybrid h;
		h.vs.resize(4);
		for (uint32_t j = 0; j < 4; j++)h.vs[j] = ts.T(i, j);
		mesh_.Hs[i] = h;
	}

	hex_minJ = 1, average_minJ = 0;
	for (uint32_t i = 0; i<ts.T.rows(); i++)
	{
		VectorXd c0 = ts.V.row(ts.T(i, 0));
		VectorXd c1 = ts.V.row(ts.T(i, 1));
		VectorXd c2 = ts.V.row(ts.T(i, 2));
		VectorXd c3 = ts.V.row(ts.T(i, 3));

		double jacobian_value = a_jacobian(c0, c1, c2, c3);

		if (hex_minJ>jacobian_value) hex_minJ = jacobian_value;
		average_minJ += jacobian_value;

		if (jacobian_value < 0) {
			cout << "negative jacobian " << i << " " << i / 8 << " " << i % 8 << endl;

		}
	}

	scaled_jacobian(mesh, mq);
	cout << "after: minimum scaled J: " << mq.min_Jacobian << " average scaled J: " << mq.ave_Jacobian << endl;
	return true;
}

bool optimization::gluing() {
	//quality
	scaled_jacobian(mesh, mq);
	std::cout << "before: minimum scaled J: " << mq.min_Jacobian << " average scaled J: " << mq.ave_Jacobian << endl;

	Tetralize_Set ts;
	tetralize_mesh(ts, mesh);
	ts.global = false;
	ts.projection = false;
	ts.lamda_projection = 1.e+10;
	ts.glue = true;
	ts.lamda_glue = 2e1;
	slim_m_opt(ts, -1, 1);

	for (uint32_t i = 0; i < ts.V_ranges.size();i++) {
		int start = 0, end = ts.V_ranges[i];
		if (i != 0) start = ts.V_ranges[i - 1] + 1;
		Vector3d v; v.setZero();
		for (uint32_t j = start; j <= end; j++) v += ts.V.row(j);
		v /= (end - start + 1);
		mesh.V.col(i) = v;
	}


	char path[300];
	sprintf(path, "%s%s", path_out, "_glue.vtk");
	io.write_hybrid_mesh_VTK(mesh, path);

	Mesh mesh_; mesh_.type = Mesh_type::Tet;
	mesh_.V = ts.V.transpose();
	mesh_.Hs.resize(ts.T.rows());
	for (uint32_t i = 0; i < ts.T.rows(); i++) {
		Hybrid h;
		h.vs.resize(4);
		for (uint32_t j = 0; j < 4; j++)h.vs[j] = ts.T(i, j);
		mesh_.Hs[i] = h;
	}

	double hex_minJ = 1, average_minJ=0;
	for (uint32_t i = 0; i<ts.T.rows(); i++)
	{
		VectorXd c0 = ts.V.row(ts.T(i,0));
		VectorXd c1 = ts.V.row(ts.T(i,1));
		VectorXd c2 = ts.V.row(ts.T(i,2));
		VectorXd c3 = ts.V.row(ts.T(i,3));

		double jacobian_value = a_jacobian(c0, c1, c2, c3);

		if (hex_minJ>jacobian_value) hex_minJ = jacobian_value;
		average_minJ += jacobian_value;
	}

	scaled_jacobian(mesh, mq);
	cout << "after: minimum scaled J: " << mq.min_Jacobian << " average scaled J: " << mq.ave_Jacobian << endl;
	if (mq.min_Jacobian < 0) return false;

	return true;
}
bool optimization::tetralize_mesh(Tetralize_Set &ts, Mesh &mesh_r) {

	vector<uint32_t> V_ranges; V_ranges.reserve(mesh_r.Vs.size());
	//re-index
	int32_t nv = 0;
	for (auto v : mesh_r.Vs) {
		nv += v.neighbor_hs.size();
		V_ranges.push_back(nv - 1);
	}

	ts.V_ranges = V_ranges;
	ts.V.resize(nv, 3);
	//T_
	ts.T.resize(mesh_r.Hs.size() * 8, 4);
	Vector4i t;
	for (auto h : mesh_r.Hs) {
		for (uint32_t i = 0; i < 8; i++) {
			for (uint32_t j = 0; j < 4; j++) t[j] = V_ranges[h.vs[hex_tetra_table[i][j]]];
			ts.T.row(h.id * 8 + i) = t;
		}
		for (uint32_t i = 0; i < 8; i++) {
			ts.V(V_ranges[h.vs[i]], 0) = hex_ref_shape[i][0];
			ts.V(V_ranges[h.vs[i]], 1) = hex_ref_shape[i][1];
			ts.V(V_ranges[h.vs[i]], 2) = hex_ref_shape[i][2];
			V_ranges[h.vs[i]]--;
		}
	}
	//boundary constraints
	int32_t bvn = 0;
	for (uint32_t i = 0; i < V_Boundary_flag.size(); i++) if (V_Boundary_flag[i]) {
		int start = 0, end = ts.V_ranges[i];
		if (i != 0) start = ts.V_ranges[i - 1] + 1;
		bvn += (end + 1 - start);
	}
	ts.b.resize(bvn);
	ts.bc.resize(bvn, 3); bvn = 0;
	for (uint32_t i = 0; i < V_Boundary_flag.size(); i++) if (V_Boundary_flag[i]) {

		int start = 0, end = ts.V_ranges[i];
		if (i != 0) start = ts.V_ranges[i - 1] + 1;
		for (uint32_t j = start; j <= end; j++) {
			ts.b[bvn] = j;
			ts.bc.row(bvn++) = BV.row(i);
		}
	}
	return true;
}
bool optimization::recovering() {

	Tetralize_Set ts;
	tetralize_meshO(ts, mesh);
	ts.global = false;
	ts.projection = true;
	ts.lamda_projection = 1.e+10;
	ts.glue = false;
	slim_m_opt(ts, -1, 0);

	mesh.V = ts.V.transpose();

	scaled_jacobian(mesh, mq);
	cout << "after: minimum scaled J: " << mq.min_Jacobian << " average scaled J: " << mq.ave_Jacobian << endl;

	if (mq.min_Jacobian < 0) return false;

	return true;
}
bool optimization::tetralize_meshO(Tetralize_Set &ts, Mesh &mesh_r) {

	ts.V = mesh_r.V.transpose();
	//T_
	ts.T.resize(mesh_r.Hs.size() * 8, 4);
	Vector4i t;
	for (auto h : mesh_r.Hs) {
		for (uint32_t i = 0; i < 8; i++) {
			for (uint32_t j = 0; j < 4; j++) t[j] = h.vs[hex_tetra_table[i][j]];
			ts.T.row(h.id * 8 + i) = t;
		}
	}
	int32_t bvn = 0;
	for (auto b_flag : V_Boundary_flag) if (b_flag) bvn++;
	ts.b.resize(bvn);
	ts.bc.resize(bvn, 3); bvn = 0;
	for (uint32_t i = 0; i < V_Boundary_flag.size(); i++) if (V_Boundary_flag[i]) {
		ts.b[bvn] = i;
		ts.bc.row(bvn++) = BV.row(i);
	}

	return true;
}

bool optimization::pipeline2() {
	scaled_jacobian(mesh, mq);
	std::cout << "before: minimum scaled J: " << mq.min_Jacobian << " average scaled J: " << mq.ave_Jacobian << endl;

	if (mesh.type == Mesh_type::Hex) {
		rigidDeformCubes();

		if(!glueInternalNodes())
		{
			return false;
		}
	}
	else if (mesh.type == Mesh_type::Tri) {
		rigidDeformTriangles();
		glueInternalNodesTriangles();
	}

	return true;
}; 
void optimization::boundary_info(){
	V_Boundary_flag.clear(); V_Boundary_flag.resize(mesh.Vs.size(), false);
	BV.resize(mesh.Vs.size(), 3); BV.setZero();
	for (auto v : mesh.Vs) {
		V_Boundary_flag[v.id] = v.boundary;
		//V_Boundary_flag[v.id] = false;
	}
	//V_Boundary_flag[0] = true;
	//V_Boundary_flag[1] = true;
	BV = mesh.V.transpose();

	cout << BV.row(0) << endl;
	cout << BV.row(1) << endl;
}
void optimization::assign_constraints(MatrixXd &CV, vector<Deform_V_Type> &Type) {
	V_Boundary_flag.clear(); V_Boundary_flag.resize(Type.size(), false);
	BV = CV;
	for (uint32_t i = 0; i < Type.size();i++) {
		if(Type[i] != Deform_V_Type::Free) V_Boundary_flag[i] = true;
	}
}

void optimization::rigidDeformCubes(){

	Cubes.clear();

	for (auto h : mesh.Hs) {
		bool have_boundary_v = false;
		vector<int32_t> bvs;
		for (uint32_t i = 0; i < 8; i++) {
			if (V_Boundary_flag[h.vs[i]]) {
				bvs.push_back(i);
				have_boundary_v = true;
			}
		}
		MatrixXd C(8, 3);
		for (uint32_t i = 0; i < 8; i++) {
			C(i, 0) = hex_ref_shape[i][0];
			C(i, 1) = hex_ref_shape[i][1];
			C(i, 2) = hex_ref_shape[i][2];
		}
		Cubes.push_back(C);
		if (!bvs.size()) {
			MatrixXd X(8, 3), Y(8, 3);
			for (uint32_t i = 0; i < 8; i++) {
				X.row(i) = C.row(i);
				Y.row(i) = BV.row(h.vs[i]);
			}
			MatrixXd R; VectorXd T;
			double scale;
			igl::procrustes(X, Y, true, false, scale, R, T);
			R *= scale;
			Cubes[Cubes.size() - 1] = (C * R).rowwise() + T.transpose();
		}
		else if (bvs.size()) {
			MatrixXd X(8, 3);
			for (uint32_t i = 0; i < 8; i++) {
				X.row(i) = BV.row(h.vs[i]);
			}
			Cubes[Cubes.size() - 1] = X;
		}
	}
}
void optimization::freeDeformCubes() {

	Tetralize_Set ts_i;
	ts_i.T.resize(8, 4);
	Vector4i t;
	for (uint32_t i = 0; i < 8; i++) {
		for (uint32_t j = 0; j < 4; j++) t[j] = hex_tetra_table[i][j];
		ts_i.T.row(i) = t;
	}

	for (auto h : mesh.Hs) {
		bool have_boundary_v = false;
		vector<int32_t> bvs;
		for (uint32_t i = 0; i < 8; i++) {
			if (V_Boundary_flag[h.vs[i]]) {
				bvs.push_back(i);
				have_boundary_v = true;
			}
		}
		if (bvs.size() < 3) continue;

		ts_i.V = Cubes[h.id];

		ts_i.b.resize(bvs.size());
		ts_i.bc.resize(bvs.size(), 3); int bvn = 0;
		for (auto bvid : bvs) {
			ts_i.b[bvn] = bvid;
			ts_i.bc.row(bvn++) = BV.row(h.vs[bvid]);
		}
		ts_i.UV = ts_i.V;
		ts_i.glue = false;
		ts_i.global = false;
		ts_i.projection = true;
		ts_i.lamda_projection = 1e+6;
		slim_m_opt(ts_i, -1, 0, true);

		for (auto bvid : bvs) ts_i.UV.row(bvid) = BV.row(h.vs[bvid]);

		double hex_minJ = 1, average_minJ = 0;
		for (uint32_t i = 0; i < ts_i.T.rows(); i++)
		{
			VectorXd c0 = ts_i.UV.row(ts_i.T(i, 0));
			VectorXd c1 = ts_i.UV.row(ts_i.T(i, 1));
			VectorXd c2 = ts_i.UV.row(ts_i.T(i, 2));
			VectorXd c3 = ts_i.UV.row(ts_i.T(i, 3));

			double jacobian_value = a_jacobian(c0, c1, c2, c3);

			if (hex_minJ > jacobian_value) hex_minJ = jacobian_value;
			average_minJ += jacobian_value;
		}
		Cubes[h.id] = ts_i.UV;
	}
}
bool optimization::glueInternalNodes() {

	Tetralize_Set ts;
	vector<uint32_t> V_ranges; V_ranges.reserve(mesh.Vs.size());
	//re-index
	int32_t nv = 0;
	for (const auto &v : mesh.Vs) {
		nv += v.neighbor_hs.size();
		V_ranges.push_back(nv - 1);
	}

	ts.V_ranges = V_ranges;
	ts.V.resize(nv, 3);
	//T_
	ts.T.resize(mesh.Hs.size() * 8, 4);
	Vector4i t;
	for (const auto &h : mesh.Hs) {
		for (uint32_t i = 0; i < 8; i++) {
			for (uint32_t j = 0; j < 4; j++) t[j] = V_ranges[h.vs[hex_tetra_table[i][j]]];
			ts.T.row(h.id * 8 + i) = t;
		}
		for (uint32_t i = 0; i < 8; i++) {
			ts.V.row(V_ranges[h.vs[i]]) = Cubes[h.id].row(i);
			V_ranges[h.vs[i]]--;
		}
	}

	std::vector<bool> H_inout_tag(mesh.Hs.size(), true);
	std::vector<uint32_t> Hids;
	for (uint32_t i = 0; i <mesh.Hs.size(); i++)Hids.push_back(i);

	compute_referenceMesh(BV, mesh.Hs, H_inout_tag, Hids, ts.RT, false);

	//===================================mapping===================================//
	ts.mappedV.resize(ts.V.rows(), -1);
	ts.mappedV_reverse.resize(ts.V.rows(), -1);
	nv = 0;
	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) {
		if (V_Boundary_flag[i]) continue;
		int start = 0, end = ts.V_ranges[i];
		if (i != 0) start = ts.V_ranges[i - 1] + 1;
		for (uint32_t j = start; j < end + 1; j++) {
			ts.mappedV[j] = nv;
			ts.mappedV_reverse[nv++] = j;
		}
	}

	ts.known_value_post = true; ts.post_index = nv;
	ts.post_Variables.resize(ts.V.rows() - nv, 3);

	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) {
		if (V_Boundary_flag[i]) {
			int start = 0, end = ts.V_ranges[i];
			if (i != 0) start = ts.V_ranges[i - 1] + 1;

			for (uint32_t j = start; j < end + 1; j++) {
				ts.mappedV[j] = nv;
				ts.mappedV_reverse[nv] = j;
				ts.post_Variables.row(nv - ts.post_index) = BV.row(i);
				nv++;
			}
		}
	}

	V_ranges = ts.V_ranges;
	//reorder variables
	MatrixXd tempV = ts.V;
	nv = 0; int id = 0;
	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) if (!V_Boundary_flag[i]) {
		int start = 0, end = V_ranges[i];
		if (i != 0) start = V_ranges[i - 1] + 1;
		nv += end - start + 1;
		ts.V_ranges[id++] = nv - 1;
	}
	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) if (V_Boundary_flag[i]) {
		int start = 0, end = V_ranges[i];
		if (i != 0) start = V_ranges[i - 1] + 1;
		nv += end - start + 1;
		ts.V_ranges[id++] = nv - 1;
	}
	for (uint32_t i = 0; i < ts.V.rows(); i++) ts.V.row(i) = tempV.row(ts.mappedV_reverse[i]);
	for (uint32_t i = 0; i < ts.T.rows(); i++)for (uint32_t j = 0; j < 4; j++) ts.T(i, j) = ts.mappedV[ts.T(i, j)];

	ts.UV = ts.V;
	ts.O_Vranges = V_ranges;
	ts.global = false;
	ts.projection = false;
	ts.lamda_projection = 1e+3;
	ts.glue = true;
	ts.record_Sequence = true;
	ts.known_value_post = true;
	ts.lamda_glue = 1e4;
	ts.m = mesh;
	slim_m_opt(ts, -1, 1, true);
	
	ts.V = ts.UV;
	tempV = ts.V;

	ts.V_ranges = V_ranges;
	for (uint32_t i = 0; i < ts.V.rows(); i++) ts.V.row(i) = tempV.row(ts.mappedV[i]);
	for (uint32_t i = 0; i < ts.T.rows(); i++)for (uint32_t j = 0; j < 4; j++) ts.T(i, j) = ts.mappedV_reverse[ts.T(i, j)];

	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) {
		int start = 0, end = ts.V_ranges[i];
		if (i != 0) start = ts.V_ranges[i - 1] + 1;
		Vector3d v; v.setZero();
		for (uint32_t j = start; j <= end; j++) v += ts.V.row(j);
		v /= (end - start + 1);
		mesh.V.col(i) = v;
	}

	scaled_jacobian(mesh, mq);

	cout << "after: minimum scaled J: " << mq.min_Jacobian << " average scaled J: " << mq.ave_Jacobian << endl;
	if (mq.min_Jacobian < 0)
		return false;
	return true;
}

void optimization::rigidDeformTriangles() {
	Triangles.clear();

	for (auto f : mesh.Fs) {
		bool have_boundary_v = false;
		vector<int32_t> bvs;
		for (uint32_t i = 0; i < 3; i++) {
			if (V_Boundary_flag[f.vs[i]]) {
				bvs.push_back(i);
				have_boundary_v = true;
			}
		}
		MatrixXd Tri(3, 2);
		for (uint32_t i = 0; i < 3; i++) {
			Tri(i, 0) = tri_ref_shape[i][0];
			Tri(i, 1) = tri_ref_shape[i][1];
		}
		Triangles.push_back(Tri);
		if (!bvs.size()) {
			MatrixXd X(3, 2), Y(3, 2);
			for (uint32_t i = 0; i < 3; i++) {
				X.row(i) = Tri.row(i);
				int column = 0;
				for(uint32_t j=0;j<3;j++) if(mesh.plane!=j)
					Y(i, column++) = BV(f.vs[i], j);
			}
			MatrixXd R; VectorXd T;
			double scale;
			igl::procrustes(X, Y, true, false, scale, R, T);
			R *= scale;
			Triangles[Triangles.size() - 1] = (Tri * R).rowwise() + T.transpose();

		}
		else if (bvs.size() >= 2) {
			MatrixXd X(bvs.size(), 2), Y(bvs.size(), 2);
			for (uint32_t i = 0; i < bvs.size(); i++) {
				X.row(i) = Tri.row(bvs[i]);
				int column = 0;
				for (uint32_t j = 0; j<3; j++) if (mesh.plane != j)
					Y(i, column++) = BV(f.vs[bvs[i]], j);
			}
			MatrixXd R; VectorXd T;
			double scale;
			igl::procrustes(X, Y, true, false, scale, R, T);
			R *= scale;
			Triangles[Triangles.size() - 1] = (Tri * R).rowwise() + T.transpose();
		}
		else  if (bvs.size() < 2) {
			Float len = 0; Vector3d v0 = mesh.V.col(f.vs[bvs[0]]), v1;
			if (bvs.size() == 1) {
				int32_t nbv = 0, v0id = f.vs[bvs[0]];
				for (uint32_t i = 0; i < mesh.Vs[v0id].neighbor_vs.size(); i++) {
					uint32_t vid = mesh.Vs[f.vs[bvs[0]]].neighbor_vs[i];
					if (V_Boundary_flag[vid]) {
						len += (v0 - mesh.V.col(vid)).norm();
						nbv++;
					}
				}
				len /= nbv;
				Tri *= len;
			}
			RowVector2d T;
			int column = 0;
			for (int32_t j = 0; j < 3; j++) {
				if (mesh.plane != j) {
					T(column) = BV(f.vs[bvs[0]], j) - Tri(bvs[0], column);
					column++;
				}
			}

			Triangles[Triangles.size() - 1] = Tri.rowwise() + T;
		}
	}
}
bool optimization::glueInternalNodesTriangles() {
	
	Tetralize_Set ts;
	vector<uint32_t> V_ranges; V_ranges.reserve(mesh.Vs.size());
	//re-index
	int32_t nv = 0;
	for (auto v : mesh.Vs) {
		nv += v.neighbor_fs.size();
		V_ranges.push_back(nv - 1);
	}

	ts.V_ranges = V_ranges;
	ts.V.resize(nv, 3); ts.V.setZero();
	ts.UV.resize(nv, 2);
	//T_
	ts.T.resize(mesh.Fs.size() * 1, 3);
	
	for (auto f : mesh.Fs) {
		for (uint32_t i = 0; i < 3; i++) {
			ts.UV.row(V_ranges[f.vs[i]]) = Triangles[f.id].row(i);
			V_ranges[f.vs[i]]--;

			ts.T(f.id, i) = f.vs[i];
		}
	}
	//===================================mapping===================================//
	ts.mappedV.resize(ts.UV.rows(), -1);
	ts.mappedV_reverse.resize(ts.UV.rows(), -1);
	nv = 0;
	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) {
		if (V_Boundary_flag[i]) continue;
		int start = 0, end = ts.V_ranges[i];
		if (i != 0) start = ts.V_ranges[i - 1] + 1;
		for (uint32_t j = start; j < end + 1; j++) {
			ts.mappedV[j] = nv;
			ts.mappedV_reverse[nv++] = j;
		}
	}

	ts.known_value_post = true; ts.post_index = nv;
	ts.post_Variables.resize(ts.V.rows() - nv, 2);

	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) {
		if (V_Boundary_flag[i]) {
			int start = 0, end = ts.V_ranges[i];
			if (i != 0) start = ts.V_ranges[i - 1] + 1;

			for (uint32_t j = start; j < end + 1; j++) {
				ts.mappedV[j] = nv;
				ts.mappedV_reverse[nv] = j;
				int column = 0;
				for (uint32_t k = 0; k < 3; k++) if (mesh.plane != k)
					ts.post_Variables(nv - ts.post_index, column++) = BV(i, k);
				nv++;
			}
		}
	}

	V_ranges = ts.V_ranges;
	//reorder variables
	MatrixXd tempV = ts.UV;
	nv = 0; int id = 0;
	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) if (!V_Boundary_flag[i]) {
		int start = 0, end = V_ranges[i];
		if (i != 0) start = V_ranges[i - 1] + 1;
		nv += end - start + 1;
		ts.V_ranges[id++] = nv - 1;
	}
	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) if (V_Boundary_flag[i]) {
		int start = 0, end = V_ranges[i];
		if (i != 0) start = V_ranges[i - 1] + 1;
		nv += end - start + 1;
		ts.V_ranges[id++] = nv - 1;
	}
	for (uint32_t i = 0; i < ts.V.rows(); i++) ts.UV.row(i) = tempV.row(ts.mappedV_reverse[i]);
	for (uint32_t i = 0; i < ts.T.rows(); i++)for (uint32_t j = 0; j < 3; j++) ts.T(i, j) = ts.mappedV[ts.T(i, j)];

	for (uint32_t i = 0; i < ts.V.rows(); i++) ts.V.row(i).segment(0,2) = ts.UV.row(i);

	ts.O_Vranges = V_ranges;
	ts.global = false;
	ts.projection = false;
	ts.lamda_projection = 1e+3;
	ts.glue = true;
	ts.record_Sequence = false;
	ts.known_value_post = true;
	ts.lamda_glue = 5e3;
	slim_m_opt(ts, -1, 1);

	for (uint32_t i = 0; i < ts.V.rows(); i++) ts.V.row(i).segment(0, 2) = ts.UV.row(i);

	tempV = ts.V;

	ts.V_ranges = V_ranges;
	for (uint32_t i = 0; i < ts.V.rows(); i++) ts.V.row(i) = tempV.row(ts.mappedV[i]);
	for (uint32_t i = 0; i < ts.T.rows(); i++)for (uint32_t j = 0; j < 3; j++) ts.T(i, j) = ts.mappedV_reverse[ts.T(i, j)];

	for (uint32_t i = 0; i < ts.V_ranges.size(); i++) {
		int start = 0, end = ts.V_ranges[i];
		if (i != 0) start = ts.V_ranges[i - 1] + 1;
		Vector2d v; v.setZero();
		for (uint32_t j = start; j <= end; j++) v += ts.V.row(j);
		v /= (end - start + 1);

		int column = 0;
		for (uint32_t k = 0; k < 3; k++) //if (mesh.plane != k)
			mesh.V(k, i) = v(column++);
	}

	Mesh mesh_;
	mesh_.type = Mesh_type::Tri;
	mesh_.V = ts.V.transpose();
	int column = 0;
	for (uint32_t k = 0; k < 3; k++) if (mesh.plane != k)
		mesh_.V.row(k) = ts.V.row(column++);

	double hex_minJ = 1, average_minJ = 0;
	for (uint32_t i = 0; i<ts.T.rows(); i++)
	{
		VectorXd c0 = ts.V.row(ts.T(i, 0));
		VectorXd c1 = ts.V.row(ts.T(i, 1));
		VectorXd c2 = ts.V.row(ts.T(i, 2));

		double jacobian_value = a_jacobian(c0, c1, c2);

		if (hex_minJ>jacobian_value) hex_minJ = jacobian_value;
		average_minJ += jacobian_value;

		if (jacobian_value < 0) {
			cout << "negative jacobian " << i << " " << i / 8 << " " << i % 8 << endl;
		}
	}
	cout << "hex_minJ average_minJ: " << hex_minJ << " " << average_minJ / ts.T.rows() << endl;

	return true;
}


void optimization::slim_m_opt(Tetralize_Set &ts, const uint32_t iter, const int type, bool verbose) {
	SLIMData sData;

	Timer<> timer;
	//timer.beginStage("prefactoring ");
	sData.stop_threshold = 1.e-7;
	sData.soft_const_p = 1e1;
	sData.exp_factor = 5.0;
	sData.weight_opt = weight_opt;
	slim_precompute(ts.V, ts.T, ts.UV, sData, ts.energy_type, ts);
	
	slim_solve2(sData, iter, type, verbose);

	ts.UV = sData.V_o;
	energy = sData.energy;
	engery_quality = sData.engery_quality;
	engery_soft = sData.engery_soft;
}
void optimization::scaff_slim_m_opt(Tetralize_Set &ts, const uint32_t iter, Mesh_Domain &md, const int type) {
	SLIMData sData;

	Timer<> timer;
	//timer.beginStage("prefactoring ");
	sData.stop_threshold = 1.e-7;
	sData.soft_const_p = 1e1;
	sData.exp_factor = 5.0;
	slim_precompute(ts.V, ts.T, ts.UV, sData, ts.energy_type, ts);

	for (int i = 0; i < md.H_flag.size(); i++)
	{
		if (md.H_flag[i] == 0)
		{
			for (int j = 0; j < 8; j++)
				sData.M[i * 8 + j] *= 1;
		}
	}

	auto &s = sData;
	s.WGL_M.resize(s.dim * s.dim * s.f_n);
	for (int i = 0; i < s.dim * s.dim; i++)
		for (int j = 0; j < s.f_n; j++)
			s.WGL_M(i * s.f_n + j) = s.M(j);

	s.mesh_area = s.M.sum();

	slim_solve(sData, iter, type, false);

	ts.UV = sData.V_o;
	//ts.V = sData.V_o;
	energy = sData.energy;
}
void optimization::slim_opt(Tetralize_Set &ts, const uint32_t iter, bool loop) {
	igl::SLIMData sData;

	Timer<> timer;
	timer.beginStage("prefactoring ");
	sData.soft_const_p = ts.lamda_projection;
	sData.exp_factor = 5.0;

	timer.endStage("prefactoring");

	timer.beginStage("solve ");
	if (loop) {
		Mesh mesh_ = mesh;
		mesh_.V = ts.V.transpose();
		for (uint32_t i = 0; i < iter; i++) {
			slim_solve(sData, 1);
			ts.V = sData.V_o;

			for (uint32_t i = 0; i < ts.V_ranges.size(); i++) {
				int start = 0, end = ts.V_ranges[i];
				if (i != 0) start = ts.V_ranges[i - 1] + 1;
				Vector3d v; v.setZero();
				for (uint32_t j = start; j <= end; j++) v += ts.V.row(j);
				v /= (end - start + 1);
				mesh_.V.col(i) = v;
			}
			scaled_jacobian(mesh_, mq);
			cout << "after: minimum scaled J: " << mq.min_Jacobian << " average scaled J: " << mq.ave_Jacobian << endl;
		}

	}else slim_solve(sData, iter);

	timer.endStage("solve");

	ts.V = sData.V_o;
}
void optimization::slim_opt_igl(Tetralize_Set &ts, const uint32_t iter) {
	igl::SLIMData sData;
	//cout << "localize_ts" << endl;
	sData.weight_opt = weight_opt;
	sData.soft_const_p = ts.lamda_b;//lamda_b not adapted for simplification pipeline yet
	sData.exp_factor = 5.0;
	sData.lamda_C = ts.fc.lamda_C;
	sData.lamda_T = ts.fc.lamda_T;
	sData.lamda_L = ts.fc.lamda_L;
	sData.lamda_region = ts.lamda_region;
	sData.lamda_glue = ts.lamda_glue;
	sData.Vgroups = ts.Vgroups;

	Timer<> timer;
	//timer.beginStage("prefactoring ");

	igl::SLIMData::SLIM_ENERGY energy = igl::SLIMData::SYMMETRIC_DIRICHLET;// igl::SLIMData::CONFORMAL;

	if (ts.projection)
		slim_precompute(ts.V, ts.T, ts.V, sData, energy, ts.s, ts.sc,
			ts.fc.ids_C, ts.fc.C,
			ts.fc.ids_L, ts.fc.Axa_L, ts.fc.origin_L,
			ts.fc.ids_T, ts.fc.normal_T, ts.fc.dis_T, ts.regionb, ts.regionbc, ts.projection, ts.global, ts.RT);
	else
		slim_precompute(ts.V, ts.T, ts.V, sData, energy, ts.b, ts.bc,
			ts.fc.ids_C, ts.fc.C,
			ts.fc.ids_L, ts.fc.Axa_L, ts.fc.origin_L,
			ts.fc.ids_T, ts.fc.normal_T, ts.fc.dis_T, ts.regionb, ts.regionbc, ts.projection, ts.global, ts.RT);
	//timer.endStage("end prefactoring");

	//timer.beginStage("solve ");

	slim_solve(sData, iter);
	ts.V = sData.V_o;
	//timer.endStage("end solve");
}