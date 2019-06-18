#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include "global_types.h"
#include "global_functions.h"
#include "grid_meshing/grid_hex_meshing.h"

class meshing
{
public:
	meshing(void) {}; ~meshing(void) {};
	bool processing(string &path);
	bool pipeline(Mesh &hmesh);
	bool feature(Mesh &mesh, string &path);

	void forward_positioning(Mesh &m, Eigen::MatrixXd &T);
	void back_positioning(Mesh &m, Eigen::MatrixXd &T);
	void geomesh2mesh(GEO::Mesh &gm, Mesh &m);
	void mesh2geomesh(Mesh &m, GEO::Mesh &gm);

	Mesh mesho, meshob;
	Mesh_Quality mqo;
private:
	h_io io;
};

