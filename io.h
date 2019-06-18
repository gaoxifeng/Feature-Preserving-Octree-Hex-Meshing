#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include "global_types.h"
#include "global_functions.h"
#include <igl/readOBJ.h>
#include <igl/readSTL.h>
#include <igl/readWRL.h>

class h_io
{
private:
	int counter;
public:
	h_io(void) { counter = -1; }; ~h_io(void) {};

	bool read_hybrid_mesh_VTK(Mesh &hmi, std::string path);
	void write_hybrid_mesh_VTK(Mesh &hmi, std::string path);

	bool read_hybrid_mesh_MESH(Mesh &hmi, std::string path);
	void write_hybrid_mesh_MESH(Mesh &hmi, std::string path);

	void read_hybrid_mesh_OBJ(Mesh &hmi, std::string path);
	void write_hybrid_mesh_OBJ(Mesh &hmi, std::string path);

	void write_mesh_OBJ(Eigen::MatrixXd &V,Eigen::MatrixXi &F, bool col, std::string path);

	bool read_feature_Graph_FGRAPH(Mesh_Feature &mf, std::string path);
	void write_feature_Graph_FGRAPH(Mesh_Feature &mf, std::string path);
};

