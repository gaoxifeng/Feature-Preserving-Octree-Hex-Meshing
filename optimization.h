#pragma once
#include "global_functions.h"
#include <algorithm>
#include "igl/slim.h"
#include "slim_m.h"
#include "timer.h"
class optimization
{

public:
	optimization() {};
	~optimization() {};

	bool pipeline();

	bool fix_boundary();
	bool gluing();
	bool tetralize_mesh(Tetralize_Set &ts, Mesh &mesh_r);
	bool recovering();
	bool tetralize_meshO(Tetralize_Set &ts, Mesh &mesh_r);

	bool pipeline2();
	void boundary_info();
	void assign_constraints(MatrixXd &CV, vector<Deform_V_Type> &Type);
	void rigidDeformCubes();
	void freeDeformCubes();
	bool glueInternalNodes();

	void rigidDeformTriangles();
	bool glueInternalNodesTriangles();

	void slim_m_opt(Tetralize_Set &ts, const uint32_t iter, const int type, bool verbose=false);
	void scaff_slim_m_opt(Tetralize_Set &ts, const uint32_t iter, Mesh_Domain &md, const int type);
	void slim_opt(Tetralize_Set &ts, const uint32_t iter, bool loop);
	void slim_opt_igl(Tetralize_Set &ts, const uint32_t iter);
public:
	h_io io;
	Mesh mesh;
	Mesh_Quality mq;
	
	vector<bool> V_Boundary_flag;
	MatrixXd BV;
	vector<MatrixXd> Cubes;
	vector<MatrixXd> Triangles;

	double energy=-1;
	double engery_quality = -1;
	double engery_soft = -1;

	double weight_opt = 1;
};

