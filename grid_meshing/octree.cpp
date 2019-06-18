////////////////////////////////////////////////////////////////////////////////
#include "octree.h"
#include "common.h"
#include <geogram/basic/logger.h>
#include <unsupported/Eigen/SparseExtra>
#include <algorithm>
#include <random>
#include <stack>
#include <queue>
////////////////////////////////////////////////////////////////////////////////

#define oct_assert assert
#define oct_debug assert

////////////////////////////////////////////////////////////////////////////////

OctreeGrid::OctreeGrid(Eigen::Vector3i fineCellGridSize, int maxNodeGuess, int maxCellGuess)
	: m_NodeGridSize(fineCellGridSize.array() + 1)
	, m_CellGridSize(fineCellGridSize)
	, m_NumRootCells(0)
{
	m_Nodes.reserve(maxNodeGuess);
	m_Cells.reserve(maxCellGuess);

	// Some sanity checks
	oct_assert(Math::isPowerOfTwo(fineCellGridSize[0]));
	oct_assert(Math::isPowerOfTwo(fineCellGridSize[1]));
	oct_assert(Math::isPowerOfTwo(fineCellGridSize[2]));

	// Max depth depends on fineCellGridSize
	int minFineCellSize = m_CellGridSize.minCoeff();
	m_MaxDepth = 0;
	while ( m_MaxDepth < 64 && (1 << m_MaxDepth) < minFineCellSize) { ++m_MaxDepth; }
	// logger_debug("OctreeGrid", "MaxDepth: %s", m_MaxDepth);

	// Create root cells
	createRootCells();
}
void OctreeGrid::OctreeGrid_initialize(Eigen::Vector3i fineCellGridSize, int maxNodeGuess, int maxCellGuess) {
	m_NodeGridSize = fineCellGridSize.array() + 1;
	m_CellGridSize = fineCellGridSize;
	m_NumRootCells = 0;

	m_Nodes.reserve(maxNodeGuess);
	m_Cells.reserve(maxCellGuess);

	// Some sanity checks
	oct_assert(Math::isPowerOfTwo(fineCellGridSize[0]));
	oct_assert(Math::isPowerOfTwo(fineCellGridSize[1]));
	oct_assert(Math::isPowerOfTwo(fineCellGridSize[2]));

	// Max depth depends on fineCellGridSize
	int minFineCellSize = m_CellGridSize.minCoeff();
	m_MaxDepth = 0;
	while (m_MaxDepth < 64 && (1 << m_MaxDepth) < minFineCellSize) { ++m_MaxDepth; }
	// logger_debug("OctreeGrid", "MaxDepth: %s", m_MaxDepth);

	// Create root cells
	createRootCells();
}
// -----------------------------------------------------------------------------

// Create root cells and connect their nodes accordingly
void OctreeGrid::createRootCells() {
	// Clear current octree
	m_Nodes.clear();
	m_Cells.clear();

	// Anisotropic grids: we need multiple root cells
	int minFineCellSize = m_CellGridSize.minCoeff();
	Eigen::Vector3i coarseCellGridSize = m_CellGridSize / minFineCellSize;
	Eigen::Vector3i coarseNodeGridSize = coarseCellGridSize.array() + 1;

	// Create coarse grid nodes and set up adjacency relations
	for (int i = 0; i < coarseNodeGridSize.prod(); ++i) {
		Node newNode;
		Eigen::Vector3i coarsePos = Layout3D::toGrid(i, coarseNodeGridSize);
		newNode.position = (1 << m_MaxDepth) * coarsePos;
		for (int axis = 0; axis < 3; ++axis) {
			for (int c = 0; c < 2; ++c) {
				Eigen::Vector3i q = coarsePos;
				q[axis] += (c ? 1 : -1);
				q = Layout3D::clamp(q, coarseNodeGridSize);
				int j = Layout3D::toIndex(q, coarseNodeGridSize);
				newNode.neighNodeId[2*axis+c] = (j != i ? j : -1);
			}
		}
		m_Nodes.emplace_back(newNode);
	}

	// Create coarse grid cells
	for (int e = 0; e < coarseCellGridSize.prod(); ++e) {
		Cell newCell;
		Eigen::Vector3i lowerCorner = Layout3D::toGrid(e, coarseCellGridSize);
		for (int k = 0; k < 8; ++k) {
			Eigen::Vector3i currentCorner = lowerCorner + Cube::delta(k);
			int i = Layout3D::toIndex(currentCorner, coarseNodeGridSize);
			newCell.setCorner(k, i);
		}
		m_Cells.emplace_back(newCell);
	}
	m_NumRootCells = (int) m_Cells.size();

	// Link adjacent cells together
	for (int i = 0; i < coarseCellGridSize.prod(); ++i) {
		Eigen::Vector3i coarsePos = Layout3D::toGrid(i, coarseCellGridSize);
		for (int axis = 0; axis < 3; ++axis) {
			for (int c = 0; c < 2; ++c) {
				Eigen::Vector3i q = coarsePos;
				q[axis] += (c ? 1 : -1);
				q = Layout3D::clamp(q, coarseCellGridSize);
				int j = Layout3D::toIndex(q, coarseCellGridSize);
				m_Cells[i].neighCellId[2*axis+c] = (j != i ? j : -1);
			}
		}
	}
}

// -----------------------------------------------------------------------------


////////////////////////////////////////////////////////////////////////////////
// Public accessors
////////////////////////////////////////////////////////////////////////////////

Eigen::Vector3d OctreeGrid::cellCenterPos(int cellId) const {
	return cellCornerPos(cellId, 0).cast<double>().array()
		+ 0.5 * cellExtent(cellId);
}

// -----------------------------------------------------------------------------

// Position of a cell corner
Eigen::Vector3i OctreeGrid::cellCornerPos(int cellId, int localCornerId) const {
	return m_Nodes[m_Cells[cellId].corner(localCornerId)].position;
}

// -----------------------------------------------------------------------------

// Size of a cell
int OctreeGrid::cellExtent(int cellId) const {
	return cellCornerPos(cellId, 1)[0] - cellCornerPos(cellId, 0)[0];
}

////////////////////////////////////////////////////////////////////////////////

// Returns true iff the octree is 2:1 graded
bool OctreeGrid::is2to1Graded() const {
	for (int i = 0; i < numCells(); ++i) {
		if (cellIsLeaf(i) && !cellIs2to1Graded(i)) {
			return false;
		}
	}
	return true;
}

// Return true iff the cell cellId is 2:1 graded
bool OctreeGrid::cellIs2to1Graded(int cellId) const {
	const Cell &cell = m_Cells[cellId];
	const int v0 = cell.corner(CORNER_X0_Y0_Z0);
	const int v1 = cell.corner(CORNER_X1_Y0_Z0);
	const int v2 = cell.corner(CORNER_X1_Y1_Z0);
	const int v3 = cell.corner(CORNER_X0_Y1_Z0);
	const int v4 = cell.corner(CORNER_X0_Y0_Z1);
	const int v5 = cell.corner(CORNER_X1_Y0_Z1);
	const int v6 = cell.corner(CORNER_X1_Y1_Z1);
	const int v7 = cell.corner(CORNER_X0_Y1_Z1);
	auto testEdge = [this] (int a, int b, int axis) {
		return (nextNode(a, axis) == b || nextNode(a, axis) == prevNode(b, axis));
	};
	return testEdge(v0, v1, X) && testEdge(v3, v2, X) && testEdge(v4, v5, X) && testEdge(v7, v6, X)
		&& testEdge(v0, v3, Y) && testEdge(v1, v2, Y) && testEdge(v4, v7, Y) && testEdge(v5, v6, Y)
		&& testEdge(v0, v4, Z) && testEdge(v1, v5, Z) && testEdge(v3, v7, Z) && testEdge(v2, v6, Z);
}

// -----------------------------------------------------------------------------

// Return true iff the octree is paired
bool OctreeGrid::isPaired() const {
	for (int i = 0; i < numCells(); ++i) {
		if (!cellIsPaired(i)) {
			return false;
		}
	}
	return true;
}

// Return true iff the cell cellId is paired (its children are either all leaves, or all internal nodes)
bool OctreeGrid::cellIsPaired(int cellId) const {
	if (cellIsLeaf(cellId)) {
		return true;
	} else {
		const int firstChild = m_Cells[cellId].firstChild;
		const bool allLeaf = cellIsLeaf(firstChild);
		for (int k = 1; k < 8; ++k) {
			if (cellIsLeaf(firstChild + k) != allLeaf) {
				return false;
			}
		}
		return true;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Sudivision routines
////////////////////////////////////////////////////////////////////////////////

// Create a new double-link adjacency relation along axis
void OctreeGrid::createNodeLinks(int node1, int node2, int axis) {
	oct_debug(node1 != -1 && node2 != -1);
	if (nextNode(node1, axis) == -1) {
		oct_debug(prevNode(node2, axis) == -1);
		m_Nodes[node1].setNext(axis, node2);
		m_Nodes[node2].setPrev(axis, node1);
		for (int c = 0; c < 3; ++c) {
			if (c == axis) { continue; }
			oct_debug(m_Nodes[node1].position[c] == m_Nodes[node2].position[c]);
		}
	}
}

// -----------------------------------------------------------------------------

// Update double-linked list of ajdacent nodes along axis
void OctreeGrid::updateNodeLinks(int node1, int node2, int mid, int axis) {
	oct_debug(node1 != -1 && node2 != -1);
	if (nextNode(node1, axis) == node2) {
		oct_debug(prevNode(node2, axis) == node1);
		m_Nodes[node1].setNext(axis, mid);
		m_Nodes[node2].setPrev(axis, mid);
		m_Nodes[mid].setNext(axis, node2);
		m_Nodes[mid].setPrev(axis, node1);
		for (int c = 0; c < 3; ++c) {
			if (c == axis) { continue; }
			oct_debug(m_Nodes[mid].position[c] == m_Nodes[node1].position[c]);
			oct_debug(m_Nodes[mid].position[c] == m_Nodes[node2].position[c]);
		}
	}
}

// -----------------------------------------------------------------------------

// Update double-linked list of ajdacent cells along axis
void OctreeGrid::updateCellLinks(int cell1, int cell2, int axis) {
	oct_debug(cell1 != -1 && cell2 != -1);
	oct_debug(cellExtent(cell1) == cellExtent(cell2));
	m_Cells[cell1].setNext(axis, cell2);
	m_Cells[cell2].setPrev(axis, cell1);
	oct_debug(cellCornerPos(cell1, 0)[axis] + cellExtent(cell1) == cellCornerPos(cell2, 0)[axis]);
}

// -----------------------------------------------------------------------------

// Update links between the descendants of adjacents cells along axis
void OctreeGrid::updateSubcellLinks(int cell1, int cell2, int axis) {
	if (cell1 == -1 || cell2 == -1) { return; }
	oct_debug(axis == X || axis == Y || axis == Z);
	const int ax1 = (axis == X ? Y : X);
	const int ax2 = (axis == Z ? Y : Z);
	const int offset1 = m_Cells[cell1].firstChild;
	const int offset2 = m_Cells[cell2].firstChild;
	Eigen::Vector3i delta;
	for (int i = 0; i < 4; ++i) {
		delta[ax1] = i%2;
		delta[ax2] = i/2;
		delta[axis] = 1;
		int subcell1 = offset1 + Cube::invDelta(delta);
		delta[axis] = 0;
		int subcell2 = offset2 + Cube::invDelta(delta);
		if (offset1 == -1 && offset2 != -1) {
			m_Cells[subcell2].setPrev(axis, cell1);
			updateSubcellLinks(cell1, subcell2, axis);
		} else if (offset1 != -1 && offset2 == -1) {
			m_Cells[subcell1].setNext(axis, cell2);
			updateSubcellLinks(subcell1, cell2, axis);
		} else if (offset1 != -1 && offset2 != -1) {
			updateCellLinks(subcell1, subcell2, axis);
			updateSubcellLinks(subcell1, subcell2, axis);
		}
	}
}

// -----------------------------------------------------------------------------

// Update links between the direct children of a cell along axis
void OctreeGrid::updateSubcellLinks(int cell, int axis) {
	oct_debug(cell != -1);
	oct_debug(axis == X || axis == Y || axis == Z);
	const int ax1 = (axis == X ? Y : X);
	const int ax2 = (axis == Z ? Y : Z);
	const int offset = m_Cells[cell].firstChild;
	Eigen::Vector3i delta;
	for (int i = 0; i < 4; ++i) {
		delta[ax1] = i%2;
		delta[ax2] = i/2;
		delta[axis] = 0;
		int subcell1 = offset + Cube::invDelta(delta);
		delta[axis] = 1;
		int subcell2 = offset + Cube::invDelta(delta);
		updateCellLinks(subcell1, subcell2, axis);
	}
}

////////////////////////////////////////////////////////////////////////////////

// Retrieve the index of the midnode of an edge, if it exists
int OctreeGrid::getMidEdgeNode(int node1, int node2, int axis) const {
	const Node &n1 = m_Nodes[node1];
	const Node &n2 = m_Nodes[node2];
	oct_debug(n1.position[axis] < n2.position[axis]);
	if (n1.next(axis) == node2) {
		oct_debug(n2.prev(axis) == node1);
		return -1;
	} else {
		const int c1 = n1.position[axis];
		const int c2 = n2.position[axis];
		const int c3 = (c1 + c2) / 2;
		oct_debug((c1 + c2) % 2 == 0);
		int m1 = n1.next(axis);
		int m2 = n2.prev(axis);
		while (m_Nodes[m1].position[axis] != c3 && m_Nodes[m2].position[axis] != c3) {
			m1 = nextNode(m1, axis);
			m2 = prevNode(m2, axis);
			oct_debug(m1 != -1 && m2 != -1);
			oct_debug(m1 != node2 && m2 != node1);
		}
		if (m_Nodes[m1].position[axis] == c3) {
			return m1;
		} else {
			return m2;
		}
	}
}

// -----------------------------------------------------------------------------

// Add a node at the middle of an edge
int OctreeGrid::addMidEdgeNode(int node1, int node2, int axis) {
	oct_debug(node1 != -1 && node2 != -1);
	oct_debug(nextNode(node1, axis) == node2);
	oct_debug(prevNode(node2, axis) == node1);

	// Setup node position
	Node newNode;
	newNode.position = m_Nodes[node1].position;
	newNode.position[axis] = (m_Nodes[node1].position[axis] + m_Nodes[node2].position[axis]) / 2;
	oct_debug((m_Nodes[node1].position[axis] + m_Nodes[node2].position[axis]) % 2 == 0);

	// Setup node adjacency
	int newId = (int) m_Nodes.size();
	m_Nodes.emplace_back(newNode);
	updateNodeLinks(node1, node2, newId, axis);
	return newId;
}

// -----------------------------------------------------------------------------

int OctreeGrid::splitEdge(int node1, int node2, int axis) {
	oct_debug(node1 != -1 && node2 != -1);
	for (int c = 0; c < 3; ++c) {
		if (c == axis) { continue; }
		oct_debug(m_Nodes[node1].position[c] == m_Nodes[node2].position[c]);
	}
	int id = getMidEdgeNode(node1, node2, axis);
	if (id == -1) {
		return addMidEdgeNode(node1, node2, axis);
	} else {
		return id;
	}
}

////////////////////////////////////////////////////////////////////////////////

// Subdivide a face along a given axis
int OctreeGrid::splitFace(int node1, int node2, int node3, int node4, int normalAxis) {
	oct_debug(node1 != -1 && node2 != -1 && node3 != -1 && node4 != -1);
	oct_debug(normalAxis == X || normalAxis == Y || normalAxis == Z);
	const int ax1 = (normalAxis == X ? Y : X);
	const int ax2 = (normalAxis == Z ? Y : Z);

	// Start by splitting edges of the face
	int mid1 = splitEdge(node1, node2, ax1);
	int mid2 = splitEdge(node3, node4, ax1);
	int mid3 = splitEdge(node1, node3, ax2);
	int mid4 = splitEdge(node2, node4, ax2);

	// Connect central face node to adj nodes on ax1 and ax2
	createNodeLinks(mid1, mid2, ax2);
	createNodeLinks(mid3, mid4, ax1);
	int mid5 = splitEdge(mid1, mid2, ax2);
	updateNodeLinks(mid3, mid4, mid5, ax1);

	return mid5;
}

////////////////////////////////////////////////////////////////////////////////

/*

Bottom face:
       x──────x──────x
      ╱      ╱      ╱
     ╱      ╱      ╱
    x──────x──────x
   ╱      ╱      ╱
  ╱      ╱      ╱
 x──────x──────x


Left face:
       x
      ╱│
     ╱ │
    x  │
   ╱│  x
  ╱ │ ╱│
 x  │╱ │
 │  x  │
 │ ╱│  x
 │╱ │ ╱
 x  │╱
 │  x
 │ ╱
 │╱
 x


Big cube:
       x──────x──────x
      ╱┆     ╱      ╱│
     ╱ ┆    ╱      ╱ │
    x┄┄┼┄┄┄x┄┄┄┄┄┄x  │
   ╱   x  ╱   x  ╱┆  x
  ╱    ┆ ╱      ╱ ┆ ╱│
 x─────┼x──────x  ┆╱ │
 │  x  ┆┆  o   │  x  │
 │     x┼┄┄┄┄┄x┼┄⌿┼┄┄x
 │    ╱ ┆      │╱ ┆ ╱
 x┄┄┄⌿┄┄x┄┄┄┄┄┄x  ┆╱
 │  x   ┆  x   │  x
 │ ╱    ┆      │ ╱
 │╱     ┆      │╱
 x──────x──────x


Edge nodes:
       x─────e76─────x
      ╱┆     ╱      ╱│
     ╱ ┆    ╱      ╱ │
   e47┄┼┄┄┄x┄┄┄┄┄e56 │
   ╱  e37 ╱      ╱┆ e26
  ╱    ┆ ╱      ╱ ┆ ╱│
 x─────e45─────x  ┆╱ │
 │     ┆┆      │  x  │
 │     x┼┄┄┄┄e32┄⌿┼┄┄x
 │    ╱ ┆      │╱ ┆ ╱
e04┄┄⌿┄┄x┄┄┄┄┄e15 ┆╱
 │ e03  ┆      │ e12
 │ ╱    ┆      │ ╱
 │╱     ┆      │╱
 x─────e01─────x

Corner nodes:
      v7──────x─────v6
      ╱┆     ╱      ╱│
     ╱ ┆    ╱      ╱ │
    x┄┄┼┄┄┄x┄┄┄┄┄┄x  │
   ╱   x  ╱      ╱┆  x
  ╱    ┆ ╱      ╱ ┆ ╱│
v4─────┼x─────v5  ┆╱ │
 │     ┆┆      │  x  │
 │    v3┼┄┄┄┄┄x┼┄⌿┼┄v2
 │    ╱ ┆      │╱ ┆ ╱
 x┄┄┄⌿┄┄x┄┄┄┄┄┄x  ┆╱
 │  x   ┆      │  x
 │ ╱    ┆      │ ╱
 │╱     ┆      │╱
v0──────x─────v1

Face nodes:
      v7──────x─────v6
      ╱┆     ╱      ╱│
     ╱ ┆    ╱      ╱ │
    x┄┄┼┄┄f5┄┄┄┄┄┄x  │
   ╱   x  ╱  f3  ╱┆  x
  ╱    ┆ ╱      ╱ ┆ ╱│
v4─────┼x─────v5  ┆╱ │
 │ f0  ┆┆      │ f1  │
 │    v3┼┄┄┄┄┄x┼┄⌿┼┄v2
 │    ╱ ┆      │╱ ┆ ╱
 x┄┄┄⌿┄f2┄┄┄┄┄┄x  ┆╱
 │  x   ┆ f4   │  x
 │ ╱    ┆      │ ╱
 │╱     ┆      │╱
v0──────x─────v1

*/


// Subdivide a cell and add the subcells to the octree
int OctreeGrid::splitCell(int cellId, bool graded, bool paired) {
	oct_debug(cellId != -1);
	oct_debug(cellIsLeaf(cellId));
	const Cell &cell = m_Cells[cellId];

	// Retrieve cell corners
	const int v0 = cell.corner(CORNER_X0_Y0_Z0);
	const int v1 = cell.corner(CORNER_X1_Y0_Z0);
	const int v2 = cell.corner(CORNER_X1_Y1_Z0);
	const int v3 = cell.corner(CORNER_X0_Y1_Z0);
	const int v4 = cell.corner(CORNER_X0_Y0_Z1);
	const int v5 = cell.corner(CORNER_X1_Y0_Z1);
	const int v6 = cell.corner(CORNER_X1_Y1_Z1);
	const int v7 = cell.corner(CORNER_X0_Y1_Z1);

	// Start by splitting incident faces
	const int f0 = splitFace(v0, v3, v4, v7, X);
	const int f1 = splitFace(v1, v2, v5, v6, X);
	const int f2 = splitFace(v0, v1, v4, v5, Y);
	const int f3 = splitFace(v3, v2, v7, v6, Y);
	const int f4 = splitFace(v0, v1, v3, v2, Z);
	const int f5 = splitFace(v4, v5, v7, v6, Z);

	// Then connect the middle points of the faces
	createNodeLinks(f0, f1, X);
	createNodeLinks(f2, f3, Y);
	createNodeLinks(f4, f5, Z);
	const int c0 = splitEdge(f0, f1, X);
	updateNodeLinks(f2, f3, c0, Y);
	updateNodeLinks(f4, f5, c0, Z);

	// Retrieve midpoint of cell edges
	const int e01 = getMidEdgeNode(v0, v1, X);
	const int e32 = getMidEdgeNode(v3, v2, X);
	const int e45 = getMidEdgeNode(v4, v5, X);
	const int e76 = getMidEdgeNode(v7, v6, X);
	const int e03 = getMidEdgeNode(v0, v3, Y);
	const int e12 = getMidEdgeNode(v1, v2, Y);
	const int e47 = getMidEdgeNode(v4, v7, Y);
	const int e56 = getMidEdgeNode(v5, v6, Y);
	const int e04 = getMidEdgeNode(v0, v4, Z);
	const int e15 = getMidEdgeNode(v1, v5, Z);
	const int e26 = getMidEdgeNode(v2, v6, Z);
	const int e37 = getMidEdgeNode(v3, v7, Z);

	// Create a new cell for each subvolume
	const int offset = (int) m_Cells.size();
	m_Cells.resize(m_Cells.size() + 8);
	m_Cells[offset+CORNER_X0_Y0_Z0].cornerNodeId = {{v0, e01, f4, e03, e04, f2, c0, f0}};
	m_Cells[offset+CORNER_X1_Y0_Z0].cornerNodeId = {{e01, v1, e12, f4, f2, e15, f1, c0}};
	m_Cells[offset+CORNER_X1_Y1_Z0].cornerNodeId = {{f4, e12, v2, e32, c0, f1, e26, f3}};
	m_Cells[offset+CORNER_X0_Y1_Z0].cornerNodeId = {{e03, f4, e32, v3, f0, c0, f3, e37}};
	m_Cells[offset+CORNER_X0_Y0_Z1].cornerNodeId = {{e04, f2, c0, f0, v4, e45, f5, e47}};
	m_Cells[offset+CORNER_X1_Y0_Z1].cornerNodeId = {{f2, e15, f1, c0, e45, v5, e56, f5}};
	m_Cells[offset+CORNER_X1_Y1_Z1].cornerNodeId = {{c0, f1, e26, f3, f5, e56, v6, e76}};
	m_Cells[offset+CORNER_X0_Y1_Z1].cornerNodeId = {{f0, c0, f3, e37, e47, f5, e76, v7}};

	// Update link to child cell
	m_Cells[cellId].firstChild = offset;

	// Update cell adjacency relations
	for (int axis = 0; axis < 3; ++axis) {
		updateSubcellLinks(cellId, axis);
		updateSubcellLinks(cellId, nextCell(cellId, axis), axis);
		updateSubcellLinks(prevCell(cellId, axis), cellId, axis);
	}

	// Ensure proper 2:1 grading
	if (graded) {
		makeCellGraded(cellId, paired);
	}

	// Ensure children are either all leaves, or all internal cells
	if (paired) {
		makeCellPaired(cellId, graded);

		if (cellId < m_NumRootCells) {
			// Special case for root cells: if one gets split, then we need to split all root cells
			for (int c = 0; c < m_NumRootCells; ++c) {
				if (cellIsLeaf(c)) { splitCell(c, graded, paired); }
			}
		} else {
			// Ensure sibling cells are also properly split
			int firstSibling = m_NumRootCells + 8 * ((cellId - m_NumRootCells) / 8);
			for (int c = firstSibling; c < firstSibling + 8; ++c) {
				if (cellIsLeaf(c)) { splitCell(c, graded, paired); }
			}
		}

	}

	return c0;
}

////////////////////////////////////////////////////////////////////////////////

// Make the cell 2:1 graded
bool OctreeGrid::makeCellGraded(int cellId, bool paired) {
	bool splitOccured = false;
	for (int ax1 = 0; ax1 < 3; ++ax1) {
		for (int d1 = 0; d1 < 2; ++d1) {
			// Neighboring cells along a face
			if (adjCell(cellId, ax1, d1) != -1) {
				while (adjCell(adjCell(cellId, ax1, d1), ax1, 1-d1) != cellId) {
					oct_debug(cellExtent(adjCell(cellId, ax1, d1)) > cellExtent(cellId));
					splitCell(adjCell(cellId, ax1, d1), true, paired);
					splitOccured = true;
				}
				// Neighboring cell along an edge
				for (int ax2 = 0; ax2 < 3; ++ax2) {
					if (ax1 == ax2) { continue; }
					const int c1 = adjCell(cellId, ax1, d1);
					for (int d2 = 0; d2 < 2; ++d2) {
						if (adjCell(c1, ax2, d2) != -1) {
							while (adjCell(adjCell(c1, ax2, d2), ax2, 1-d2) != c1) {
								oct_debug(cellExtent(adjCell(c1, ax2, d2)) > cellExtent(c1));
								splitCell(adjCell(c1, ax2, d2), true, paired);
								splitOccured = true;
							}
						}
					}
				}
			}
		}
	}
	return splitOccured;
}

// -----------------------------------------------------------------------------

// Make the cell Paired (its children are either all leaves, or all internal nodes)
bool OctreeGrid::makeCellPaired(int cellId, bool graded) {
	if (!cellIsPaired(cellId)) {
		const int firstChild = m_Cells[cellId].firstChild;
		for (int k = 0; k < 8; ++k) {
			if (cellIsLeaf(firstChild + k)) {
				splitCell(firstChild + k, graded, true);
			}
		}
		return true;
	}
	return false;
}

////////////////////////////////////////////////////////////////////////////////

// Traverse the leaf cells recursively and split them according to the predicate function
void OctreeGrid::subdivide(std::function<bool(int, int, int, int)> predicate,
	bool graded, bool paired, int maxCells)
{
	std::queue<int> pending;
	for (int i = 0; i < (int) m_Cells.size(); ++i) {
		if (cellIsLeaf(i)) {
			pending.push(i);
		}
	}

	int numNodesBefore = numNodes();
	int numCellsBefore = numCells();
	int numSubdivided = 0;
	if (maxCells < 0) {
		maxCells = std::numeric_limits<int>::max();
	}
	while (!pending.empty() && numCells() + 8 <= maxCells) {
		int id = pending.front();
		pending.pop();
		int extent = cellExtent(id);
		auto pos = cellCornerPos(id, 0);
		if (predicate(pos[0], pos[1], pos[2], extent)) {
			if (extent == 1) {
				std::cerr << "[OctreeGrid] Cannot subdivide cell of length 1." << std::endl;
			} else {
				if (cellIsLeaf(id)) {
					splitCell(id, graded, paired);
				}
				for (int k = 0; k < 8; ++k) {
					pending.push(m_Cells[id].firstChild + k);
				}
			}
		}
	}

	// Resize attribute vectors
	nodeAttributes.resize(numNodes());
	cellAttributes.resize(numCells());

	GEO::Logger::out("OctreeGrid") << "Subdivide has split " << numSubdivided << " cells\n";
	GEO::Logger::out("OctreeGrid") << "Num nodes: " << numNodesBefore << " -> " << numNodes() << "\n";
	GEO::Logger::out("OctreeGrid") << "Num cells: " << numCellsBefore << " -> " << numCells() << std::endl;
}
void OctreeGrid::subdivide(std::function<bool(int, int, int, int)> predicate, std::vector<int> &tb_subdivided_cells,
	bool graded, bool paired, int maxCells) {
	std::queue<int> pending;
	for (auto cid: tb_subdivided_cells) if (cellIsLeaf(cid)) pending.push(cid);

	int numNodesBefore = numNodes();
	int numCellsBefore = numCells();
	int numSubdivided = 0;
	if (maxCells < 0) {
		maxCells = std::numeric_limits<int>::max();
	}
	while (!pending.empty() && numCells() + 8 <= maxCells) {
		int id = pending.front();
		pending.pop();
		int extent = cellExtent(id);
		auto pos = cellCornerPos(id, 0);
		if (predicate(pos[0], pos[1], pos[2], extent)) {
			if (extent == 1) {
				std::cerr << "[OctreeGrid] Cannot subdivide cell of length 1." << std::endl;
			}
			else {
				if (cellIsLeaf(id)) {
					splitCell(id, graded, paired);
				}
				//for (int k = 0; k < 8; ++k) {
				//	pending.push(m_Cells[id].firstChild + k);
				//}
			}
		}
	}

	// Resize attribute vectors
	nodeAttributes.resize(numNodes());
	cellAttributes.resize(numCells());

	GEO::Logger::out("OctreeGrid") << "Subdivide has split " << numSubdivided << " cells\n";
	GEO::Logger::out("OctreeGrid") << "Num nodes: " << numNodesBefore << " -> " << numNodes() << "\n";
	GEO::Logger::out("OctreeGrid") << "Num cells: " << numCellsBefore << " -> " << numCells() << std::endl;
}
////////////////////////////////////////////////////////////////////////////////
// Mesh export
////////////////////////////////////////////////////////////////////////////////

// Initialize a new geogram mesh corresponding to the current grid
void OctreeGrid::createMesh(
	GEO::Mesh &mesh, const Eigen::Vector3d &origin, const Eigen::Vector3d &spacing) const
{
	mesh.clear(false, false);

	// logger_debug("OctreeGrid", "createMesh(): Allocate vertices and cells");

	// Create the mesh of regular grid
	mesh.vertices.create_vertices(numNodes());
	for (int idx = 0; idx < numNodes(); ++idx) {
		Eigen::Vector3d pos = origin + nodePos(idx).cast<double>().cwiseProduct(spacing);
		mesh.vertices.point(idx) = GEO::vec3(pos[0], pos[1], pos[2]);
	}

	// Count num of leaf cells
	int numLeaves = 0;
	for (int c = 0; c < numCells(); ++c) {
		if (cellIsLeaf(c)) { ++numLeaves; }
	}
	GEO::index_t firstCube = mesh.cells.create_hexes(numLeaves);
	for (int q = 0, c = 0; q < numCells(); ++q) {
		if (!cellIsLeaf(q)) {
			continue;
		}
		Eigen::Vector3i diff[8] = {
			{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0},
			{0,0,1}, {1,0,1}, {0,1,1}, {1,1,1}
		};
		for (GEO::index_t lv = 0; lv < 8; ++lv) {
			int cornerId = Cube::invDelta(diff[lv]);
			int v = cellCornerId(q, cornerId);
			mesh.cells.set_vertex(firstCube + c, lv, v);
		}
		++c;
	}

	// logger_debug("OctreeGrid", "createMesh(): Connecting cells");
	//GEO::Logger::out("OctreeGrid") << "Computing borders..." << std::endl;
	//mesh.cells.compute_borders();
	//GEO::Logger::out("OctreeGrid") << "Connecting cells..." << std::endl;
	//mesh.cells.connect();

	// logger_debug("OctreeGrid", "createMesh(): Creating attributes...");
	updateMeshAttributes(mesh);
}

////////////////////////////////////////////////////////////////////////////////

// Shortcut macro to make life easier
#define CHECK_TYPE(T, id, name, grid, mesh, cth)                   \
	do {                                                           \
		if ((id) == std::type_index(typeid(T))) {                  \
			setGeogramAttribute<T>((name), (grid), (mesh), (cth)); \
			return;                                                \
		}                                                          \
	} while (0)

#define CHECK_ALL_TYPE(id, name, grid, mesh, cth)        \
	do {                                                 \
		CHECK_TYPE(unsigned, id, name, grid, mesh, cth); \
		CHECK_TYPE(int, id, name, grid, mesh, cth);      \
		CHECK_TYPE(float, id, name, grid, mesh, cth);    \
		CHECK_TYPE(double, id, name, grid, mesh, cth);   \
	} while (0)

////////////////////////////////////////////////////////////////////////////////

namespace {

// -----------------------------------------------------------------------------

template<typename T>
void setGeogramAttribute(const std::string &name, const OctreeGrid &grid, GEO::Mesh &mesh,
	const std::vector<int> &cellToHex)
{
	typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorT;

	const VectorT &gridCellAttr = grid.cellAttributes.get<T>(name);
	GEO::Attribute<T> meshCellAttr(mesh.cells.attributes(), name);

	for (size_t q = 0; q < cellToHex.size(); ++q) {
		if (cellToHex[q] != -1) {
			meshCellAttr[cellToHex[q]] = gridCellAttr(q);
		}
	}
}

// -----------------------------------------------------------------------------

void setGeogramAttribute(const std::string &name, const OctreeGrid &grid, GEO::Mesh &mesh,
	const std::vector<int> &cellToHex)
{
	std::type_index id = grid.cellAttributes.type(name);
	CHECK_ALL_TYPE(id, name, grid, mesh, cellToHex);
}

// -----------------------------------------------------------------------------

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

// Update attributes of a geogram mesh according to the current grid
void OctreeGrid::updateMeshAttributes(GEO::Mesh &mesh) const {
	// Map octree cell to hex in the final mesh (keeping only the leaves)
	std::vector<int> cellToHex(numCells(), -1);
	for (int q = 0, c = 0; q < numCells(); ++q) {
		if (cellIsLeaf(q)) {
			cellToHex[q] = c++;
		}
	}
	for (auto name : cellAttributes.keys()) {
		setGeogramAttribute(name, *this, mesh, cellToHex);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Debug stuff
////////////////////////////////////////////////////////////////////////////////

void OctreeGrid::assertIsValid() {
	// Check node adjacency relations
	for (int node1 = 0; node1 < (int) m_Nodes.size(); ++node1) {
		for (int axis = 0; axis < 3; ++axis) {
			int node0 = prevNode(node1, axis);
			int node2 = nextNode(node1, axis);
			if (m_Nodes[node1].position[axis] == 0) {
				oct_debug(node0 == -1);
			} else if (node0 != -1) {
				oct_debug(nextNode(node0, axis) == node1);
			}
			if (m_Nodes[node1].position[axis] == m_CellGridSize[axis]) {
				oct_debug(node2 == -1);
			} else if (node2 != -1) {
				oct_debug(prevNode(node2, axis) == node1);
			}
		}
	}
	// Check cell adjacency relations
	for (int cell1 = 0; cell1 < (int) m_Cells.size(); ++cell1) {
		for (int axis = 0; axis < 3; ++axis) {
			int cell0 = prevCell(cell1, axis);
			int cell2 = nextCell(cell1, axis);
			if (cellCornerPos(cell1, CORNER_X0_Y0_Z0)[axis] == 0) {
				oct_debug(cell0 == -1);
			} else {
				oct_debug(cell0 != -1);
				if (cellExtent(cell1) == cellExtent(cell0)) {
					oct_debug(nextCell(cell0, axis) == cell1);
				}
			}
			if (cellCornerPos(cell1, CORNER_X1_Y1_Z1)[axis] == m_CellGridSize[axis]) {
				oct_debug(cell2 == -1);
			} else {
				oct_debug(cell2 != -1);
				if (cellExtent(cell1) == cellExtent(cell2)) {
					oct_debug(prevCell(cell2, axis) == cell1);
				}
			}
		}
	}
}

// -----------------------------------------------------------------------------

void OctreeGrid::testSubdivideRandom(bool graded, bool paired) {
	bool bfs = false;
	std::vector<std::pair<int, int> > leaves, next;

	// Init: start with root cells
	leaves.reserve(m_Cells.size());
	for (int i = 0; i < (int) m_Cells.size(); ++i) {
		leaves.emplace_back(0, i);
	}

	std::default_random_engine gen;
	std::uniform_real_distribution<double> distr(0, 1);

	assertIsValid();
	int counter = 0;
	int nextCheck = (int) leaves.size();
	while (!leaves.empty()) {
		std::uniform_int_distribution<int> take(0, (int) leaves.size() - 1);
		int i = take(gen);
		int depth = leaves[i].first;
		int id = leaves[i].second;
		std::swap(leaves[i], leaves.back());
		leaves.pop_back();
		if (depth < m_MaxDepth) {
			if (distr(gen) > 0.2 && cellIsLeaf(id)) {
				int newId = (int) m_Cells.size();
				splitCell(id, graded, paired);
				for (int k = 0; k < 8; ++k) {
					if (bfs) {
						next.emplace_back(depth + 1, newId++);
					} else {
						leaves.emplace_back(depth + 1, newId++);
					}
				}
			}
		}
		++counter;
		if (!bfs && counter == nextCheck) {
			assertIsValid();
			nextCheck = (int) leaves.size();
			counter = 0;
		}
		if (leaves.empty()) {
			assertIsValid();
			std::swap(leaves, next);
		}
	}
	assertIsValid();

	//std::shuffle(m_Cells.begin(), m_Cells.end(), std::default_random_engine());

	// logger_debug("Octree", "Graded %s", is2to1Graded());
	// logger_debug("Octree", "Build ok");
	std::cout << "Graded: " << is2to1Graded() << std::endl;
	std::cout << "Paired: " << isPaired() << std::endl;

	// logger_debug("Octree", "Num nodes: %s", m_Nodes.size());
	// logger_debug("Octree", "Num cells: %s", m_Cells.size());

	if (graded) { oct_assert(is2to1Graded()); }
	if (paired) { oct_assert(isPaired()); }
}
