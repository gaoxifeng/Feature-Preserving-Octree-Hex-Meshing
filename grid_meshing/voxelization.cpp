#pragma once
#include "voxelization.h"
bool endswith(std::string const &str, std::string const &ending) {
	if (str.length() >= ending.length()) {
		return (0 == str.compare(str.length() - ending.length(), ending.length(), ending));
	}
	else {
		return false;
	}
}



////////////////////////////////////////////////////////////////////////////////

namespace Layout {

	GEO::vec3i index3_from_index(int idx, GEO::vec3i size) {
		return GEO::vec3i(
			idx % size[0],
			(idx / size[0]) % size[1],
			(idx / size[0]) / size[1]
		);
	}

	int index_from_index3(GEO::vec3i vx, GEO::vec3i size) {
		return (vx[2] * size[1] + vx[1]) * size[0] + vx[0];
	}

}

namespace GEO {

	bool filename_has_supported_extension(const std::string &filename) {
		std::vector<std::string> extensions;
		GEO::MeshIOHandlerFactory::list_creators(extensions);
		for (auto &ext : extensions) {
			if (endswith(filename, ext)) {
				return true;
			}
		}
		return false;
	}
}
// -----------------------------------------------------------------------------


////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// NOTE: Function `point_in_triangle_2d` comes from SDFGen by Christopher Batty.
// https://github.com/christopherbatty/SDFGen/blob/master/makelevelset3.cpp
////////////////////////////////////////////////////////////////////////////////

// calculate twice signed area of triangle (0,0)-(x1,y1)-(x2,y2)
// return an SOS-determined sign (-1, +1, or 0 only if it's a truly degenerate triangle)
int orientation(
	double x1, double y1, double x2, double y2, double &twice_signed_area)
{
	twice_signed_area = y1*x2 - x1*y2;
	if (twice_signed_area>0) return 1;
	else if (twice_signed_area<0) return -1;
	else if (y2>y1) return 1;
	else if (y2<y1) return -1;
	else if (x1>x2) return 1;
	else if (x1<x2) return -1;
	else return 0; // only true when x1==x2 and y1==y2
}

// -----------------------------------------------------------------------------

// robust test of (x0,y0) in the triangle (x1,y1)-(x2,y2)-(x3,y3)
// if true is returned, the barycentric coordinates are set in a,b,c.
bool point_in_triangle_2d(
	double x0, double y0, double x1, double y1,
	double x2, double y2, double x3, double y3,
	double &a, double &b, double &c)
{
	x1 -= x0; x2 -= x0; x3 -= x0;
	y1 -= y0; y2 -= y0; y3 -= y0;
	int signa = orientation(x2, y2, x3, y3, a);
	if (signa == 0) return false;
	int signb = orientation(x3, y3, x1, y1, b);
	if (signb != signa) return false;
	int signc = orientation(x1, y1, x2, y2, c);
	if (signc != signa) return false;
	double sum = a + b + c;
	geo_assert(sum != 0); // if the SOS signs match and are nonzero, there's no way all of a, b, and c are zero.
	a /= sum;
	b /= sum;
	c /= sum;
	return true;
}

// -----------------------------------------------------------------------------



////////////////////////////////////////////////////////////////////////////////

void compute_sign(const GEO::Mesh &M, const GEO::MeshFacetsAABB &aabb_tree,
	OctreeGrid &octree, GEO::vec3 origin, double spacing)
{
	Eigen::VectorXf & inside = octree.cellAttributes.create<float>("inside");
	inside.resize(octree.numCells());
	inside.setZero();

	try {
		GEO::ProgressTask task("Ray marching", 100);

		GEO::vec3 min_corner, max_corner;
		GEO::get_bbox(M, &min_corner[0], &max_corner[0]);

		GEO::parallel_for([&](int cellId) {
			auto cell_xyz_min = octree.cellCornerPos(cellId, OctreeGrid::CORNER_X0_Y0_Z0);
			auto extent = octree.cellExtent(cellId);

			GEO::Box box;
			box.xyz_min[0] = origin[0] + spacing * cell_xyz_min[0];
			box.xyz_min[1] = origin[1] + spacing * cell_xyz_min[1];
			box.xyz_max[0] = box.xyz_min[0] + spacing * extent;
			box.xyz_max[1] = box.xyz_min[1] + spacing * extent;
			box.xyz_min[2] = min_corner[2] - spacing;
			box.xyz_max[2] = max_corner[2] + spacing;

			GEO::vec3 center(
				box.xyz_min[0] + 0.5 * spacing * extent,
				box.xyz_min[1] + 0.5 * spacing * extent,
				origin[2] + spacing * cell_xyz_min[2] + 0.5 * spacing * extent
			);

			std::vector<std::pair<double, int>> inter;
			auto action = [&M, &inter, &center](GEO::index_t f) {
				double z;
				if (int s = intersect_ray_z(M, f, center, z)) {
					inter.emplace_back(z, s);
				}
			};
			aabb_tree.compute_bbox_facet_bbox_intersections(box, action);
			std::sort(inter.begin(), inter.end());

			std::vector<double> reduced;
			for (int i = 0, s = 0; i < inter.size(); ++i) {
				const int ds = inter[i].second;
				s += ds;
				if ((s == -1 && ds < 0) || (s == 0 && ds > 0)) {
					reduced.push_back(inter[i].first);
				}
			}

			int num_before = 0;
			for (double z : reduced) {
				if (z < center[2]) { ++num_before; }
			}
			if (num_before % 2 == 1) {
				inside(cellId) = 1.0;
			}
		}, 0, octree.numCells());
	}
	catch (const GEO::TaskCanceled&) {
		// Do early cleanup
	}
}

////////////////////////////////////////////////////////////////////////////////
// -----------------------------------------------------------------------------

void paraview_dump(std::string &filename, const VoxelGrid<num_t> &voxels) {
	GEO::vec3i size = voxels.grid_size();

	std::string extension = GEO::FileSystem::extension(filename);
	std::string basename = GEO::FileSystem::base_name(filename, true);
	if (!extension.empty() && extension[0] != '.') {
		extension = "." + extension;
	}
	std::string outname = filename.substr(0, filename.size() - extension.size());
	std::ofstream metafile(outname + ".mhd");
	metafile << "ObjectType = Image\nNDims = 3\n"
		<< "DimSize = " << size[0] << " " << size[1] << " " << size[2] << "\n"
		<< "ElementType = MET_CHAR\nElementDataFile = " + basename + ".raw\n";
	metafile.close();

	std::ofstream rawfile(outname + ".raw", std::ios::binary);
	rawfile.write(reinterpret_cast<const char*>(voxels.rawbuf()), voxels.num_voxels() * sizeof(num_t));
	rawfile.close();
}

// -----------------------------------------------------------------------------

void triangle_mesh_dump(std::string &filename, const VoxelGrid<num_t> &voxels) {
	using GEO::vec3i;

	vec3i cell_size = voxels.grid_size();
	vec3i node_size = cell_size + vec3i(1, 1, 1);
	int num_cells = voxels.num_voxels();
	int num_nodes = node_size[0] * node_size[1] * node_size[2];

	// Create triangle list from voxel grid
	GEO::vector<GEO::index_t> triangles;
	for (int idx = 0; idx < num_cells; ++idx) {
		vec3i pos = voxels.index3_from_index(idx);

		// Skip empty voxels
		if (voxels.at(idx) == num_t(0)) { continue; }

		// Define corner index
		std::array<GEO::index_t, 8> corners;
		corners[0] = Layout::index_from_index3(pos + vec3i(0, 0, 0), node_size);
		corners[1] = Layout::index_from_index3(pos + vec3i(1, 0, 0), node_size);
		corners[2] = Layout::index_from_index3(pos + vec3i(1, 1, 0), node_size);
		corners[3] = Layout::index_from_index3(pos + vec3i(0, 1, 0), node_size);
		corners[4] = Layout::index_from_index3(pos + vec3i(0, 0, 1), node_size);
		corners[5] = Layout::index_from_index3(pos + vec3i(1, 0, 1), node_size);
		corners[6] = Layout::index_from_index3(pos + vec3i(1, 1, 1), node_size);
		corners[7] = Layout::index_from_index3(pos + vec3i(0, 1, 1), node_size);

		// Subroutine to emit a facet quad
		auto check_facet = [&](int axis, int delta, int v1, int v2, int v3, int v4) {
			// Compute neigh voxel position
			vec3i neigh = pos;
			neigh[axis] += delta;

			// Check whether neigh voxel is empty
			bool neigh_is_empty = false;
			if (neigh[axis] < 0 || neigh[axis] >= cell_size[axis]) {
				neigh_is_empty = true;
			}
			else {
				int neigh_idx = voxels.index_from_index3(neigh);
				neigh_is_empty = (voxels.at(neigh_idx) == num_t(0));
			}

			// If neigh voxel is empty, emit triangle strips
			if (neigh_is_empty) {
				triangles.insert(triangles.end(), { corners[v1], corners[v2], corners[v3] });
				triangles.insert(triangles.end(), { corners[v3], corners[v2], corners[v4] });
			}
		};

		// Check adjacency and emit facets
		check_facet(0, -1, 0, 4, 3, 7); // left facet
		check_facet(0, 1, 2, 6, 1, 5); // right facet
		check_facet(1, -1, 1, 5, 0, 4); // front facet
		check_facet(1, 1, 3, 7, 2, 6); // back facet
		check_facet(2, -1, 1, 0, 2, 3); // lower facet
		check_facet(2, 1, 4, 5, 7, 6); // upper facet
	}

	// Assign vertex id (and remap triangle list)
	int num_vertices = 0;
	std::vector<int> node_id(num_nodes, -1);
	for (GEO::index_t &c : triangles) {
		if (node_id[c] == -1) {
			node_id[c] = num_vertices++;
		}
		c = node_id[c];
	}

	// Create Geogram mesh
	GEO::Mesh M;
	M.vertices.create_vertices(num_vertices);
	for (int v = 0; v < num_nodes; ++v) {
		if (node_id[v] != -1) {
			vec3i pos = Layout::index3_from_index(v, node_size);
			M.vertices.point(node_id[v]) = GEO::vec3(pos);
		}
	}
	M.facets.assign_triangle_mesh(triangles, true);

	// Connect facets
	M.facets.connect();

	// Rescale to unit box, and set min corner to 0
	// TODO: Add option to normalize, or output original real-world positions
	GEO::vec3 min_corner, max_corner;
	GEO::get_bbox(M, &min_corner[0], &max_corner[0]);
	GEO::vec3 extent = max_corner - min_corner;
	double scaling = std::max(extent[0], std::max(extent[1], extent[2]));
	for (int v = 0; v < M.vertices.nb(); ++v) {
		M.vertices.point(v) = (M.vertices.point(v) - min_corner) / scaling;
	}

	// Save mesh
	GEO::mesh_save(M, filename);
}

// -----------------------------------------------------------------------------

void volume_mesh_dump(std::string &filename, GEO::Mesh &mesh, const VoxelGrid<num_t> &voxels) {
	using GEO::vec3i;

	vec3i cell_dims = voxels.grid_size();
	vec3i node_dims = cell_dims + vec3i(1, 1, 1);
	int num_nodes = node_dims[0] * node_dims[1] * node_dims[2];

	auto delta = [](int i) {
		return vec3i((i & 1) ^ ((i >> 1) & 1), (i >> 1) & 1, (i >> 2) & 1);
	};
	auto inv_delta = [](vec3i u) {
		return (u[1] ? 4 * u[2] + 3 - u[0] : 4 * u[2] + u[0]);
	};
	auto cell_corner_id = [&](int cell_id, int corner_id) {
		auto pos = Layout::index3_from_index(cell_id, cell_dims);
		pos += delta(corner_id);
		return Layout::index_from_index3(pos, node_dims);
	};

	mesh.vertices.create_vertices(num_nodes);
	for (int idx = 0; idx < num_nodes; ++idx) {
		vec3i posi = Layout::index3_from_index(idx, node_dims);
		GEO::vec3 pos(posi[0], posi[1], posi[2]);
		mesh.vertices.point(idx) = voxels.origin() + pos * voxels.spacing();
	}

	for (int cell_id = 0; cell_id < voxels.num_voxels(); ++cell_id) {
		if (voxels.at(cell_id) >= 0.5) {
			vec3i diff[8] = {
				vec3i(0,0,0), vec3i(1,0,0), vec3i(0,1,0), vec3i(1,1,0),
				vec3i(0,0,1), vec3i(1,0,1), vec3i(0,1,1), vec3i(1,1,1)
			};
			int v[8];
			for (GEO::index_t lv = 0; lv < 8; ++lv) {
				int corner_id = inv_delta(diff[lv]);
				v[lv] = cell_corner_id(cell_id, corner_id);
			}
			mesh.cells.create_hex(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
		}
	}

	mesh.cells.compute_borders();
	mesh.cells.connect();
	mesh.vertices.remove_isolated();

	//GEO::mesh_save(mesh, filename);
}

// -----------------------------------------------------------------------------


////////////////////////////////////////////////////////////////////////////////

//// https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
//static unsigned next_pow2(unsigned x) {
//	x -= 1;
//	x |= (x >> 1);
//	x |= (x >> 2);
//	x |= (x >> 4);
//	x |= (x >> 8);
//	x |= (x >> 16);
//	return x + 1;
//}

void compute_octree(const GEO::Mesh &M, GEO::Mesh &mo, const GEO::MeshFacetsAABB &aabb_tree,
	const std::string &filename, GEO::vec3 min_corner, GEO::vec3 extent,
	double spacing, int padding, bool graded, bool paired)
{
	GEO::vec3 origin = min_corner - padding * spacing * GEO::vec3(1, 1, 1);
	Eigen::Vector3i grid_size(
		next_pow2(std::ceil(extent[0] / spacing) + 2 * padding),
		next_pow2(std::ceil(extent[1] / spacing) + 2 * padding),
		next_pow2(std::ceil(extent[2] / spacing) + 2 * padding)
	);

	OctreeGrid octree(grid_size);

	// Subdivide cells
	auto should_subdivide = [&](int x, int y, int z, int extent) {
		if (extent == 1) { return false; }
		GEO::Box box;
		box.xyz_min[0] = origin[0] + spacing * x;
		box.xyz_min[1] = origin[1] + spacing * y;
		box.xyz_min[2] = origin[2] + spacing * z;
		box.xyz_max[0] = box.xyz_min[0] + spacing * extent;
		box.xyz_max[1] = box.xyz_min[1] + spacing * extent;
		box.xyz_max[2] = box.xyz_min[2] + spacing * extent;
		bool has_triangles = false;
		auto action = [&has_triangles](int id) { has_triangles = true; };
		aabb_tree.compute_bbox_facet_bbox_intersections(box, action);
		return has_triangles;
	};
	octree.subdivide(should_subdivide, graded, paired);

	// Compute inside/outside info
	compute_sign(M, aabb_tree, octree, origin, spacing);

	// Export
	Eigen::Vector3d o(origin[0], origin[1], origin[2]);
	Eigen::Vector3d s(spacing, spacing, spacing);
	GEO::Logger::out("OctreeGrid") << "Creating volume mesh..." << std::endl;
	octree.createMesh(mo, o, s);
}
