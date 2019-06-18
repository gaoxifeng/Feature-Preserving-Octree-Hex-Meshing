#pragma once

////////////////////////////////////////////////////////////////////////////////
#include <Eigen/Dense>
////////////////////////////////////////////////////////////////////////////////

namespace Math {

	inline bool isPowerOfTwo(int x) {
		return (x > 0) && !(x & (x - 1));
	}

	inline int nextPowerOfTwo(int x) {
		// http://stackoverflow.com/questions/364985/algorithm-for-finding-the-smallest-power-of-two-thats-greater-or-equal-to-a-giv
		if (x < 0) { return 0; }
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return x+1;
	}

} // namespace Math

////////////////////////////////////////////////////////////////////////////////
struct Layout2D {

	static int toIndex(Eigen::Vector2i x, Eigen::Vector2i dims) {
		return x[0] + dims[0] * x[1];
	}

	static Eigen::Vector2i toGrid(int i, Eigen::Vector2i dims) {
		return Eigen::Vector2i(
			i % dims[0],
			i / dims[0]
		);
	}

	/*
	static int toint(Eigen::Vector3i x, Eigen::Vector3i dims) {
	return x[2] + dims[2] * (x[1] + dims[1] * x[0]);
	}

	static Eigen::Vector3i toVector(int i, Eigen::Vector3i dims) {
	return {{
	(i / dims[2]) / dims[1],
	(i / dims[2]) % dims[1],
	i % dims[2],
	}};
	}
	*/

	static bool isValid(Eigen::Vector2i x, Eigen::Vector2i dims) {
		return (x.array() >= 0).all() && (x.array() < dims.array()).all();
	}

	static bool isValid(int i, Eigen::Vector2i dims) {
		return isValid(toGrid(i, dims), dims);
	}

	static Eigen::Vector2i clamp(Eigen::Vector2i x, Eigen::Vector2i dims) {
		return Eigen::Vector2i(
			std::max(0, std::min(x[0], dims[0] - 1)),
			std::max(0, std::min(x[1], dims[1] - 1))
		);
	}

};

struct Layout3D {

	static int toIndex(Eigen::Vector3i x, Eigen::Vector3i dims) {
		return x[0] + dims[0] * (x[1] + dims[1] * x[2]);
	}

	static Eigen::Vector3i toGrid(int i, Eigen::Vector3i dims) {
		return Eigen::Vector3i(
			i % dims[0],
			(i / dims[0]) % dims[1],
			(i / dims[0]) / dims[1]
		);
	}

/*
	static int toint(Eigen::Vector3i x, Eigen::Vector3i dims) {
		return x[2] + dims[2] * (x[1] + dims[1] * x[0]);
	}

	static Eigen::Vector3i toVector(int i, Eigen::Vector3i dims) {
		return {{
			(i / dims[2]) / dims[1],
			(i / dims[2]) % dims[1],
			i % dims[2],
		}};
	}
*/

	static bool isValid(Eigen::Vector3i x, Eigen::Vector3i dims) {
		return (x.array() >= 0).all() && (x.array() < dims.array()).all();
	}

	static bool isValid(int i, Eigen::Vector3i dims) {
		return isValid(toGrid(i, dims), dims);
	}

	static Eigen::Vector3i clamp(Eigen::Vector3i x, Eigen::Vector3i dims) {
		return Eigen::Vector3i(
			std::max(0, std::min(x[0], dims[0]-1)),
			std::max(0, std::min(x[1], dims[1]-1)),
			std::max(0, std::min(x[2], dims[2]-1))
		);
	}

};

////////////////////////////////////////////////////////////////////////////////
namespace Quad {

	// Map local corner index (between 0 and 3) to the actual corner of the unit
	// cube [0,1]^3. The corners are obtained in this order:
	// (0,0), (1,0), (1,1), (0,1)
	inline Eigen::Vector2i delta(int i) {
		Eigen::Vector2i v;
		if (i == 0) { v(0) = 0; v(1) = 0; }
		else if(i==1) { v(0) = 1; v(1) = 0; }
		else if (i == 2) { v(0) = 1; v(1) = 1; }
		else if (i == 3) { v(0) = 0; v(1) = 1; }

		return v;
	}

	// Inverse mapping (delta -> corner index)
	inline int invDelta(Eigen::Vector2i u) {
		if (u(1) == 0) { return u(0); }
		else { return 3 - u(0); }
	}

} // namespace Cube

namespace Cube {

	// Map local corner index (between 0 and 7) to the actual corner of the unit
	// cube [0,1]^3. The corners are obtained in this order:
	// (0,0,0), (0,1,0), (1,1,0), (1,0,0), (0,0,1), (0,1,1), (1,1,1), (1,0,1)
	inline Eigen::Vector3i delta(int i) {
		return Eigen::Vector3i((i & 1) ^ ((i >> 1) & 1), (i >> 1) & 1, (i >> 2) & 1);
	}

	// Inverse mapping (delta -> corner index)
	inline int invDelta(Eigen::Vector3i u) {
		if (u[1]) {
			return 4*u[2] + 3 - u[0];
		} else {
			return 4*u[2] + u[0];
		}
	}

} // namespace Cube
