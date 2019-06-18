#include "grid_meshing/grid_hex_meshing.h"
#include "meshing.h"
#include "optimization.h"
#include "timer.h"
#include "CLI11.h"

h_io io;

int main(int argc, char* argv[])
{
	CLI::App app{ "RobustHexMeshing" };
	app.add_option("--ch", args.choice, "functionality choice")->required();
	app.add_option("--in", args.input, "Input");
	app.add_option("--out", args.output, "Output mesh.");
	app.add_option("--o", args.octree, "Octree meshing.");
	app.add_option("--n", args.num_cell, "num_cells for voxel meshing.");
	app.add_option("--h", args.Hausdorff_ratio_t, "Hausdorff_ratio_t");
	app.add_option("--e", args.edge_length_ratio, "edge_length_ratio");
	app.add_option("--w", args.weight_opt, "weight_opt");
	app.add_option("--fw", args.feature_weight, "feature_weight_opt");
	app.add_option("--r", args.pca_oobb, "pca_oobb");
	app.add_option("--s", args.scaffold_type, "scaffold_type");
	app.add_option("--f", args.Hard_Feature, "Hard_Feature");
	app.add_option("--Iter", args.Iteration_Base, "optimization Iteration_Base");
	try {
		app.parse(argc, argv);
	}
	catch (const CLI::ParseError &e) {
		return app.exit(e);
	}

	GEO::initialize();

	int nprocess = -1;
	tbb::task_scheduler_init init(nprocess == -1 ? tbb::task_scheduler_init::automatic : nprocess);
	Eigen::initParallel();
	int n = 1;
	Eigen::setNbThreads(n);
	if (args.choice == "GRID") {
		string in_format = "", path = "";
		size_t last_slash_idx = args.input.rfind('.');
		in_format = args.input.substr(last_slash_idx);
		path = args.input.substr(0, last_slash_idx);
		if (!(in_format == ".obj" || in_format == ".OBJ")) {
			cout << "not support such input format!" << endl;
			return 0;
		}
				
		meshing m;

		Timer<> timer;
		timer.beginStage("START MESHING");
		cout << endl;

		if(!m.processing(path))
			return false;

		timer.endStage("END MESHING");
		std::cout << "TIMING: " << timer.value() << "ms" << endl;

		m.mqo.timings = timer.value();

		if (args.output.empty()) {
			string patho;
			patho = path + ".mesh";
			std::cout << "writing mesh to patho " << patho << endl;
			io.write_hybrid_mesh_MESH(m.mesho, patho);
			patho = path + ".vtk";
			std::cout << "writing mesh to patho " << patho << endl;
			io.write_hybrid_mesh_VTK(m.mesho, patho);

			 patho = path + "_subB.mesh";
			 std::cout << "writing mesh to patho " << patho << endl;
			 io.write_hybrid_mesh_MESH(m.meshob, patho);
			 patho = path + "_subB.vtk";
			 std::cout << "writing mesh to patho " << patho << endl;
			 io.write_hybrid_mesh_VTK(m.meshob, patho);
		}
		else {
			string out_format = "";
			last_slash_idx = args.output.rfind('.');
			out_format = args.output.substr(last_slash_idx);
			if (out_format == ".vtk" || out_format == ".VTK") io.write_hybrid_mesh_VTK(m.mesho, args.output);
			else if (out_format == ".mesh" || out_format == ".MESH"){
				path = args.output.substr(0, last_slash_idx);
				string patho;
				patho = path + ".mesh";
				std::cout << "writing mesh to patho " << patho << endl;
				io.write_hybrid_mesh_MESH(m.mesho, patho);
				patho = path + ".vtk";
				std::cout << "writing mesh to patho " << patho << endl;
				io.write_hybrid_mesh_VTK(m.mesho, patho);

				patho = path + "_subB.mesh";
				std::cout << "writing mesh to patho " << patho << endl;
				io.write_hybrid_mesh_MESH(m.meshob, patho);
				patho = path + "_subB.vtk";
				std::cout << "writing mesh to patho " << patho << endl;
				io.write_hybrid_mesh_VTK(m.meshob, patho);
			} 
			else return false;
		}
	}
	return 0;
}