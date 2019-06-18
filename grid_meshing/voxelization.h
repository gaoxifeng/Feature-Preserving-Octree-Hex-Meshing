#pragma once
#include "octree.h"
#include <geogram/basic/file_system.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/progress.h>
#include <geogram/basic/stopwatch.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_geometry.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_AABB.h>
#include <geogram/numerics/predicates.h>
#include <algorithm>
#include <array>
#include <iterator>

////////////////////////////////////////////////////////////////////////////////

bool endswith(std::string const &str, std::string const &ending);
template <typename Scalar, size_t Rows>
inline std::ostream& operator<<(std::ostream &out, std::array<Scalar, Rows> v) {
	out << "{";
	if (!v.empty()) {
		std::copy(v.begin(), v.end() - 1, std::ostream_iterator<Scalar>(out, "; "));
		out << v.back();
	}
	out << "}";
	return out;
}
namespace Layout {
	GEO::vec3i index3_from_index(int idx, GEO::vec3i size);
	int index_from_index3(GEO::vec3i vx, GEO::vec3i size);
}
namespace GEO {
	bool filename_has_supported_extension(const std::string &filename);
}

////////////////////////////////////////////////////////////////////////////////

template<typename T>
class VoxelGrid {
private:
	// Member data
	std::vector<T> m_data;
	GEO::vec3      m_origin;
	double         m_spacing; // voxel size (in mm)
	GEO::vec3i     m_grid_size;

public:
	// Interface
	VoxelGrid(GEO::vec3 origin, GEO::vec3 extent, double voxel_size, int padding);

	GEO::vec3i grid_size() const { return m_grid_size; }
	int num_voxels() const { return m_grid_size[0] * m_grid_size[1]  * m_grid_size[2]; }

	GEO::vec3 origin() const { return m_origin; }
	double spacing() const { return m_spacing; }

	GEO::vec3i index3_from_index(int idx) const { return Layout::index3_from_index(idx, m_grid_size); }
	int index_from_index3(GEO::vec3i vx) const { return Layout::index_from_index3(vx, m_grid_size); }

	GEO::vec3 voxel_center(int x, int y, int z) const;

	const T at(int idx) const { return m_data[idx]; }
	T & at(int idx) { return m_data[idx]; }
	const T * rawbuf() const { return m_data.data(); }
	T * raw_layer(int z) { return m_data.data() + z * m_grid_size[1] * m_grid_size[0]; }
};
template<typename T>
VoxelGrid<T>::VoxelGrid(GEO::vec3 origin, GEO::vec3 extent, double spacing, int padding)
	: m_origin(origin)
	, m_spacing(spacing)
{
	m_origin -= padding * spacing * GEO::vec3(1, 1, 1);
	m_grid_size[0] = (int)std::ceil(extent[0] / spacing) + 2 * padding;
	m_grid_size[1] = (int)std::ceil(extent[1] / spacing) + 2 * padding;
	m_grid_size[2] = (int)std::ceil(extent[2] / spacing) + 2 * padding;
	GEO::Logger::out("Voxels") << "Grid size: "
		<< m_grid_size[0] << " x " << m_grid_size[1] << " x " << m_grid_size[2] << std::endl;
	m_data.assign(m_grid_size[0] * m_grid_size[1] * m_grid_size[2], T(0));
}

template<typename T>
GEO::vec3 VoxelGrid<T>::voxel_center(int x, int y, int z) const {
	GEO::vec3 pos;
	pos[0] = (x + 0.5) * m_spacing;
	pos[1] = (y + 0.5) * m_spacing;
	pos[2] = (z + 0.5) * m_spacing;
	return pos + m_origin;
}

////////////////////////////////////////////////////////////////////////////////

template<typename T>
class DexelGrid {
private:
	// Member data
	std::vector<std::vector<T> > m_data;
	GEO::vec3      m_origin;
	double         m_spacing; // voxel size (in mm)
	GEO::vec2i     m_grid_size;

public:
	// Interface
	DexelGrid(GEO::vec3 origin, GEO::vec3 extent, double voxel_size, int padding);

	GEO::vec2i grid_size() const { return m_grid_size; }
	int num_dexels() const { return m_grid_size[0] * m_grid_size[1]; }

	GEO::vec3 origin() const { return m_origin; }
	double spacing() const { return m_spacing; }

	GEO::vec2 dexel_center(int x, int y) const;

	const std::vector<T> & at(int x, int y) const { return m_data[x + m_grid_size[0]*y]; }
	std::vector<T> & at(int x, int y) { return m_data[x + m_grid_size[0]*y]; }
};
template<typename T>
DexelGrid<T>::DexelGrid(GEO::vec3 origin, GEO::vec3 extent, double spacing, int padding)
	: m_origin(origin)
	, m_spacing(spacing)
{
	m_origin -= padding * spacing * GEO::vec3(1, 1, 1);
	m_grid_size[0] = (int)std::ceil(extent[0] / spacing) + 2 * padding;
	m_grid_size[1] = (int)std::ceil(extent[1] / spacing) + 2 * padding;
	GEO::Logger::out("Voxels") << "Grid size: "
		<< m_grid_size[0] << " x " << m_grid_size[1] << std::endl;
	m_data.assign(m_grid_size[0] * m_grid_size[1], std::vector<T>(0));
}

template<typename T>
GEO::vec2 DexelGrid<T>::dexel_center(int x, int y) const {
	GEO::vec2 pos;
	pos[0] = (x + 0.5) * m_spacing;
	pos[1] = (y + 0.5) * m_spacing;
	return pos + GEO::vec2(m_origin[0], m_origin[1]);
}

////////////////////////////////////////////////////////////////////////////////
// NOTE: Function `point_in_triangle_2d` comes from SDFGen by Christopher Batty.
// https://github.com/christopherbatty/SDFGen/blob/master/makelevelset3.cpp
////////////////////////////////////////////////////////////////////////////////

// calculate twice signed area of triangle (0,0)-(x1,y1)-(x2,y2)
// return an SOS-determined sign (-1, +1, or 0 only if it's a truly degenerate triangle)
int orientation(
	double x1, double y1, double x2, double y2, double &twice_signed_area);
// -----------------------------------------------------------------------------

// robust test of (x0,y0) in the triangle (x1,y1)-(x2,y2)-(x3,y3)
// if true is returned, the barycentric coordinates are set in a,b,c.
bool point_in_triangle_2d(
	double x0, double y0, double x1, double y1,
	double x2, double y2, double x3, double y3,
	double &a, double &b, double &c);
// -----------------------------------------------------------------------------
// \brief Computes the (approximate) orientation predicate in 2d.
// \details Computes the sign of the (approximate) signed volume of
//  the triangle p0, p1, p2
// \param[in] p0 first vertex of the triangle
// \param[in] p1 second vertex of the triangle
// \param[in] p2 third vertex of the triangle
// \retval POSITIVE if the triangle is oriented positively
// \retval ZERO if the triangle is flat
// \retval NEGATIVE if the triangle is oriented negatively
// \todo check whether orientation is inverted as compared to
//   Shewchuk's version.
inline GEO::Sign orient_2d_inexact(GEO::vec2 p0, GEO::vec2 p1, GEO::vec2 p2) {
	double a11 = p1[0] - p0[0];
	double a12 = p1[1] - p0[1];

	double a21 = p2[0] - p0[0];
	double a22 = p2[1] - p0[1];

	double Delta = GEO::det2x2(
		a11, a12,
		a21, a22
	);

	return GEO::geo_sgn(Delta);
}

/**
* @brief      { Intersect a vertical ray with a triangle }
*
* @param[in]  M     { Mesh containing the triangle to intersect }
* @param[in]  f     { Index of the facet to intersect }
* @param[in]  q     { Query point (only XY coordinates are used) }
* @param[out] z     { Intersection }
*
* @return     { {-1,0,1} depending on the sign of the intersection. }
*/
template<int X = 0, int Y = 1, int Z = 2>
int intersect_ray_z(const GEO::Mesh &M, GEO::index_t f, const GEO::vec3 &q, double &z) {
	using namespace GEO;

	index_t c = M.facets.corners_begin(f);
	const vec3& p1 = Geom::mesh_vertex(M, M.facet_corners.vertex(c++));
	const vec3& p2 = Geom::mesh_vertex(M, M.facet_corners.vertex(c++));
	const vec3& p3 = Geom::mesh_vertex(M, M.facet_corners.vertex(c));

	double u, v, w;
	if (point_in_triangle_2d(
		q[X], q[Y], p1[X], p1[Y], p2[X], p2[Y], p3[X], p3[Y], u, v, w))
	{
		z = u*p1[Z] + v*p2[Z] + w*p3[Z];
		auto sign = orient_2d_inexact(vec2(p1[X], p1[Y]), vec2(p2[X], p2[Y]), vec2(p3[X], p3[Y]));
		switch (sign) {
		case GEO::POSITIVE: return 1;
		case GEO::NEGATIVE: return -1;
		default: return 0;
		}
	}

	return 0;
}
// -----------------------------------------------------------------------------

template<typename T>
void compute_sign(const GEO::Mesh &M,
	const GEO::MeshFacetsAABB &aabb_tree, VoxelGrid<T> &voxels)
{
	const GEO::vec3i size = voxels.grid_size();

	try {
		GEO::ProgressTask task("Ray marching", 100);

		GEO::vec3 min_corner, max_corner;
		GEO::get_bbox(M, &min_corner[0], &max_corner[0]);

		const GEO::vec3 origin = voxels.origin();
		const double spacing = voxels.spacing();

		GEO::parallel_for([&](int y) {
			if (GEO::Thread::current()->id() == 0) {
				task.progress((int)(100.0 * y / size[1] * GEO::Process::number_of_cores()));
			}
			for (int x = 0; x < size[0]; ++x) {
				GEO::vec3 center = voxels.voxel_center(x, y, 0);

				GEO::Box box;
				box.xyz_min[0] = box.xyz_max[0] = center[0];
				box.xyz_min[1] = box.xyz_max[1] = center[1];
				box.xyz_min[2] = min_corner[2] - spacing;
				box.xyz_max[2] = max_corner[2] + spacing;

				std::vector<std::pair<double, int>> inter;
				auto action = [&M, &inter, &center](GEO::index_t f) {
					double z;
					if (int s = intersect_ray_z(M, f, center, z)) {
						inter.emplace_back(z, s);
					}
				};
				aabb_tree.compute_bbox_facet_bbox_intersections(box, action);
				std::sort(inter.begin(), inter.end());

				for (int z = 0, s = 0, i = 0; z < size[2]; ++z) {
					GEO::vec3 center = voxels.voxel_center(x, y, z);
					for (; i < inter.size() && inter[i].first < center[2]; ++i) {
						s += inter[i].second;
					}
					const int idx = voxels.index_from_index3(GEO::vec3i(x, y, z));
					voxels.at(idx) = T(s < 0 ? 1 : 0);
				}
			}
		}, 0, size[1]);
	}
	catch (const GEO::TaskCanceled&) {
		// Do early cleanup
	}
}
// -----------------------------------------------------------------------------

template<typename T>
void compute_sign(const GEO::Mesh &M,
	const GEO::MeshFacetsAABB &aabb_tree, DexelGrid<T> &dexels)
{
	const GEO::vec2i size = dexels.grid_size();

	try {
		GEO::ProgressTask task("Ray marching", 100);

		GEO::vec3 min_corner, max_corner;
		GEO::get_bbox(M, &min_corner[0], &max_corner[0]);

		const GEO::vec3 origin = dexels.origin();
		const double spacing = dexels.spacing();

		GEO::parallel_for([&](int y) {
			if (GEO::Thread::current()->id() == 0) {
				// task.progress((int) (100.0 * y / size[1] * GEO::Process::number_of_cores()));
			}
			for (int x = 0; x < size[0]; ++x) {
				GEO::vec2 center_xy = dexels.dexel_center(x, y);
				GEO::vec3 center(center_xy[0], center_xy[1], 0);

				GEO::Box box;
				box.xyz_min[0] = box.xyz_max[0] = center[0];
				box.xyz_min[1] = box.xyz_max[1] = center[1];
				box.xyz_min[2] = min_corner[2] - spacing;
				box.xyz_max[2] = max_corner[2] + spacing;

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

				dexels.at(x, y).resize(reduced.size());
				std::copy_n(reduced.begin(), reduced.size(), dexels.at(x, y).begin());
			}
		}, 0, size[1]);
	}
	catch (const GEO::TaskCanceled&) {
		// Do early cleanup
	}
}

// -----------------------------------------------------------------------------

void compute_sign(const GEO::Mesh &M, const GEO::MeshFacetsAABB &aabb_tree,
	OctreeGrid &octree, GEO::vec3 origin, double spacing);
////////////////////////////////////////////////////////////////////////////////

typedef unsigned char num_t;

// -----------------------------------------------------------------------------

void paraview_dump(std::string &filename, const VoxelGrid<num_t> &voxels);
// -----------------------------------------------------------------------------

void triangle_mesh_dump(std::string &filename, const VoxelGrid<num_t> &voxels);
// -----------------------------------------------------------------------------

void volume_mesh_dump(std::string &filename, GEO::Mesh &mesh, const VoxelGrid<num_t> &voxels);
// -----------------------------------------------------------------------------

template<typename T>
void dexel_dump(std::string &filename, const DexelGrid<T> &dexels) {
	using GEO::vec3;

	GEO::Mesh mesh;

	for (int y = 0; y < dexels.grid_size()[1]; ++y) {
		for (int x = 0; x < dexels.grid_size()[0]; ++x) {
			for (int i = 0; 2 * i < dexels.at(x, y).size(); ++i) {
				GEO::vec3 xyz_min, xyz_max;
				xyz_min[0] = dexels.origin()[0] + x * dexels.spacing();
				xyz_min[1] = dexels.origin()[1] + y * dexels.spacing();
				xyz_min[2] = dexels.at(x, y)[2 * i + 0];
				xyz_max[0] = dexels.origin()[0] + (x + 1) * dexels.spacing();
				xyz_max[1] = dexels.origin()[1] + (y + 1) * dexels.spacing();
				xyz_max[2] = dexels.at(x, y)[2 * i + 1];
				vec3 diff[8] = {
					vec3(0,0,0), vec3(1,0,0), vec3(0,1,0), vec3(1,1,0),
					vec3(0,0,1), vec3(1,0,1), vec3(0,1,1), vec3(1,1,1)
				};
				int v = mesh.vertices.nb();
				for (int lv = 0; lv < 8; ++lv) {
					for (int d = 0; d < 3; ++d) {
						diff[lv][d] = xyz_min[d] + diff[lv][d] * (xyz_max[d] - xyz_min[d]);
					}
					diff[lv] += dexels.origin();
					mesh.vertices.create_vertex(&diff[lv][0]);
				}
				mesh.cells.create_hex(v, v + 1, v + 2, v + 3, v + 4, v + 5, v + 6, v + 7);
			}
		}
	}

	mesh.cells.compute_borders();
	mesh.cells.connect();
	mesh.vertices.remove_isolated();

	GEO::mesh_save(mesh, filename);
}
////////////////////////////////////////////////////////////////////////////////

// https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
static unsigned next_pow2(unsigned x) {
	x -= 1;
	x |= (x >> 1);
	x |= (x >> 2);
	x |= (x >> 4);
	x |= (x >> 8);
	x |= (x >> 16);
	return x + 1;
}

void compute_octree(const GEO::Mesh &M, GEO::Mesh &mo, const GEO::MeshFacetsAABB &aabb_tree,
	const std::string &filename, GEO::vec3 min_corner, GEO::vec3 extent,
	double spacing, int padding, bool graded, bool paired);