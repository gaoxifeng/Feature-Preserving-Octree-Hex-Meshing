#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "attributes.h"
#include "geogram/mesh/mesh.h"
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <vector>
#include <array>
#include <utility>
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief      Class for octree grid.
 */
struct Node {
public:
	// Data members
	std::array<int, 6> neighNodeId;
	Eigen::Vector3i    position;

public:
	// Default constructor, sets neighbor ids to -1
	Node() {
		std::fill(neighNodeId.begin(), neighNodeId.end(), -1);
	};

	// Accessors
	int  prev(int axis) const { return neighNodeId[2 * axis]; }
	int  next(int axis) const { return neighNodeId[2 * axis + 1]; }
	void setPrev(int axis, int id) { neighNodeId[2 * axis] = id; }
	void setNext(int axis, int id) { neighNodeId[2 * axis + 1] = id; }
};
/////////////////
// Octree Cell //
/////////////////

struct Cell {
public:
	// Data members
	int firstChild;
	std::array<int, 8> cornerNodeId;
	std::array<int, 6> neighCellId;

public:
	// Default constructor, sets firstChild and neighbors to -1
	Cell() : firstChild(-1) {
		std::fill(neighCellId.begin(), neighCellId.end(), -1);
	};

	// Accessors
	int  corner(int localId) const { return cornerNodeId[localId]; }
	void setCorner(int localId, int value) { cornerNodeId[localId] = value; }

	// Adjacent cells
	int  adj(int axis, int dir) const { return neighCellId[2 * axis + dir]; }
	int  prev(int axis) const { return neighCellId[2 * axis]; }
	int  next(int axis) const { return neighCellId[2 * axis + 1]; }
	void setPrev(int axis, int id) { neighCellId[2 * axis] = id; }
	void setNext(int axis, int id) { neighCellId[2 * axis + 1] = id; }
};
class OctreeGrid {

public:
	/////////////////////////////
	// Public member variables //
	/////////////////////////////

	WA::AttributeManager nodeAttributes;
	WA::AttributeManager cellAttributes;

public:
	///////////////////////
	// Numbering aliases //
	///////////////////////

	// Axis
	enum : int {
		X = 0,
		Y = 1,
		Z = 2,
	};

	// Cell corners
	enum : int {
		CORNER_X0_Y0_Z0 = 0,
		CORNER_X1_Y0_Z0 = 1,
		CORNER_X1_Y1_Z0 = 2,
		CORNER_X0_Y1_Z0 = 3,
		CORNER_X0_Y0_Z1 = 4,
		CORNER_X1_Y0_Z1 = 5,
		CORNER_X1_Y1_Z1 = 6,
		CORNER_X0_Y1_Z1 = 7,
	};

	typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;

	//////////////////////
	// Member variables //
	//////////////////////

	// Maximum index of a node/cell in the grid
	Eigen::Vector3i m_NodeGridSize;
	Eigen::Vector3i m_CellGridSize;

	// Max octree depth
	int m_MaxDepth;

	// Number of root cells
	int m_NumRootCells;

	// Octree cells and nodes
	std::vector<Node> m_Nodes;
	std::vector<Cell> m_Cells;

public:
	/////////////////
	// Constructor //
	/////////////////
	OctreeGrid() {};
	// Build an empty octree with a single root cell
	OctreeGrid(Eigen::Vector3i fineCellGridSize, int maxNodeGuess = 0, int maxCellGuess = 0);
	void OctreeGrid_initialize(Eigen::Vector3i fineCellGridSize, int maxNodeGuess = 0, int maxCellGuess = 0);

	// Create root cells and connect their nodes accordingly
	void createRootCells();

public:
	//////////////////////
	// Public accessors //
	//////////////////////

	// Maximum depth of a cell
	int maxDepth() const { return m_MaxDepth; }

	// Dimension of the grid
	int dimension() const { return 3; }

	// Number of nodes
	int numNodes() const { return (int) m_Nodes.size(); }

	// Number of cells
	int numCells() const { return (int) m_Cells.size(); }

	// Node position
	Eigen::Vector3i nodePos(int nodeId) const {
		assert(nodeId != -1); return m_Nodes[nodeId].position;
	}

	// Cell center position
	Eigen::Vector3d cellCenterPos(int cellId) const;

	// Position of a cell corner
	Eigen::Vector3i cellCornerPos(int cellId, int localCornerId) const;

	// Return the id of the corner nodes of a cell
	int cellCornerId(int cellId, int cornerId) const { return m_Cells[cellId].corner(cornerId); }

	// Size of a cell
	int cellExtent(int cellId) const;

	// Return true iff the cell has no children
	bool cellIsLeaf(int cellId) const { assert(cellId != -1); return m_Cells[cellId].firstChild == -1; }

	// Returns true iff the octree is 2:1 graded
	bool is2to1Graded() const;

	// Return true iff the leaf cell cellId is 2:1 graded
	bool cellIs2to1Graded(int cellId) const;

	// Return true iff the octree is paired
	bool isPaired() const;

	// Return true iff the cell cellId is paired (its children are either all leaves, or all internal nodes)
	bool cellIsPaired(int cellId) const;

private:
	///////////////////////
	// Private accessors //
	///////////////////////

	// Prev node along axis
	int prevNode(int nodeId, int axis) const { assert(nodeId != -1); return m_Nodes[nodeId].prev(axis); }

	// Next node along axis
	int nextNode(int nodeId, int axis) const { assert(nodeId != -1); return m_Nodes[nodeId].next(axis); }

	// Prev cell along axis
	int prevCell(int cellId, int axis) const { assert(cellId != -1); return m_Cells[cellId].prev(axis); }

	// Next cell along axis
	int nextCell(int cellId, int axis) const { assert(cellId != -1); return m_Cells[cellId].next(axis); }

	// Next cell along axis in direction dir (\in {0, 1})
	int adjCell(int cellId, int axis, int dir) const { assert(cellId != -1); return m_Cells[cellId].adj(axis, dir); }

private:
	/////////////////////////
	// Sudivision routines //
	/////////////////////////

	// Create a new double-link adjacency relation along axis
	void createNodeLinks(int node1, int node2, int axis);

	// Update double-linked list of ajdacent nodes along axis
	void updateNodeLinks(int node1, int node2, int mid, int axis);

	// Update double-linked list of ajdacent cells along axis
	void updateCellLinks(int cell1, int cell2, int axis);

	// Update links between the descendants of adjacents cells along axis
	void updateSubcellLinks(int cell1, int cell2, int axis);

	// Update links between the direct children of a cell along axis
	void updateSubcellLinks(int cell, int axis);

	// Retrieve the index of the midnode of an edge, if it exists
	int getMidEdgeNode(int node1, int node2, int axis) const;

	// Add a node at the middle of an edge
	int addMidEdgeNode(int node1, int node2, int axis);

	// Subdivide an edge along a given axis
	// @return     { id of the new node in the middle of the edge }
	int splitEdge(int node1, int node2, int axis);

	// Subdivide a face along a given axis
	// @return     { id of the new node in the middle of the face }
	int splitFace(int node1, int node2, int node3, int node4, int normalAxis);

	// Subdivide a cell (graded: impose a 2:1 cell size grading)
	// @return     { id of the new node in the middle of the cell }
	int splitCell(int cellId, bool graded, bool paired);

	// Make the cell 2:1 graded
	// @return     { true if a subdivision occurred }
	bool makeCellGraded(int cellId, bool paired);

	// Make the cell paired (its children are either all leaves, or all internal nodes)
	// @return     { true if a subdivision occurred }
	bool makeCellPaired(int cellId, bool graded);

public:
	// Traverse the leaf cells recursively and split them according to the predicate function
	void subdivide(std::function<bool(int, int, int, int)> predicate,
		bool graded = false, bool paired = false, int maxCells = -1);
	void subdivide(std::function<bool(int, int, int, int)> predicate, std::vector<int> &tb_subdivided_cells,
		bool graded = false, bool paired = false, int maxCells = -1);
public:
	/////////////////
	// Mesh export //
	/////////////////

	// Initialize a new geogram mesh corresponding to the current grid
	void createMesh(GEO::Mesh &mesh, const Eigen::Vector3d &origin, const Eigen::Vector3d &spacing) const;

	// Update attributes of a geogram mesh according to the current grid
	void updateMeshAttributes(GEO::Mesh &mesh) const;

public:
	///////////
	// Debug //
	///////////

	// Test whether the tree is in a valid configuration
	void assertIsValid();

	// Subdivide a cell into 8 child cells
	void testSubdivideRandom(bool graded, bool paired);
};

