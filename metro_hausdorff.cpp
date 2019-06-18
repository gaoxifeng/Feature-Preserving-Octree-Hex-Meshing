#include "metro_hausdorff.h"
////////////////// Command line Flags and parameters
// -----------------------------------------------------------------------------------------------
// simple aux function that compute the name for the file containing the stored computations
std::string SaveFileName(const std::string &filename)
{
	int pos = filename.find_last_of('.', filename.length());
	std::string fileout = filename.substr(0, pos) + "_metro.ply";
	return fileout;
}

int compute(Mesh &mesh0, Mesh & mesh1, double &hausdorff_ratio, double &hausdorff_ratio_threshold){
#define MSG_ERR_MESH_LOAD               "error loading the input meshes.\n"
#define MSG_ERR_INVALID_OPTION          "unable to parse option '%s'\n"
#define MSG_ERR_FILE_OPEN               "unable to open the output file.'n"
#define MSG_ERR_UNKNOWN_FORMAT          "unknown file format '%s'.\n"

	// global constants
#define NO_SAMPLES_PER_FACE             10
#define N_SAMPLES_EDGE_TO_FACE_RATIO    0.1
#define BBOX_FACTOR                     0.1
#define INFLATE_PERCENTAGE			    0.02
#define MIN_SIZE					    125		/* 125 = 5^3 */
#define N_HIST_BINS                     256
#define PRINT_EVERY_N_ELEMENTS          1000

	bool NumberOfSamples = false;
	bool SamplesPerAreaUnit = false;
	bool CleaningFlag = false;

	CMesh                 S1, S2;
	float                ColorMin = 0, ColorMax = 0;
	double                dist1_max, dist2_max;
	unsigned long         n_samples_target, elapsed_time;
	double								n_samples_per_area_unit;
	int                   flags;

	// default parameters
	flags = SamplingFlags::VERTEX_SAMPLING |
		SamplingFlags::EDGE_SAMPLING |
		SamplingFlags::FACE_SAMPLING |
		SamplingFlags::SIMILAR_SAMPLING;
	
	if (!(flags & SamplingFlags::USE_HASH_GRID) && !(flags & SamplingFlags::USE_AABB_TREE) && !(flags & SamplingFlags::USE_OCTREE))
		flags |= SamplingFlags::USE_STATIC_GRID;

	flags &= ~SamplingFlags::EDGE_SAMPLING;
	flags &= ~SamplingFlags::FACE_SAMPLING;

	// load input meshes.
	//OpenMesh("",S1);
	//OpenMesh(argv[2],S2);
	//string S1NewName = SaveFileName(argv[1]);
	//string S2NewName = SaveFileName(argv[2]);
	//tri::io::Importer<CMesh>::Open(m, filename);
	S1.vert.resize(mesh0.V.cols());
	for (int i = 0; i < mesh0.V.cols(); i++) {
		CVertex v; 
		v.P()[0] = mesh0.V(0, i);
		v.P()[1] = mesh0.V(1, i);
		v.P()[2] = mesh0.V(2, i);
		S1.vert[i] = v;
	}
	S1.face.resize(mesh0.Fs.size());
	//Allocator<CMesh>::AddFaces(S1, mesh0.Fs.size());
	for (int i = 0; i < mesh0.Fs.size(); i++) {
		CFace f;
		f.V(0) = &(S1.vert[mesh0.Fs[i].vs[0]]);
		f.V(1) = &(S1.vert[mesh0.Fs[i].vs[1]]);
		f.V(2) = &(S1.vert[mesh0.Fs[i].vs[2]]);
		S1.face[i] = f;
	}

	S2.vert.resize(mesh1.V.cols());
	for (int i = 0; i < mesh1.V.cols(); i++) {
		CVertex v;
		v.P()[0] = mesh1.V(0, i);
		v.P()[1] = mesh1.V(1, i);
		v.P()[2] = mesh1.V(2, i);
		S2.vert[i] = v;
	}
	S2.face.resize(mesh1.Fs.size());
	//Allocator<CMesh>::AddFaces(S1, mesh0.Fs.size());
	for (int i = 0; i < mesh1.Fs.size(); i++) {
		CFace f;
		f.V(0) = &(S2.vert[mesh1.Fs[i].vs[0]]);
		f.V(1) = &(S2.vert[mesh1.Fs[i].vs[1]]);
		f.V(2) = &(S2.vert[mesh1.Fs[i].vs[2]]);
		S2.face[i] = f;
	}

	S1.vn = S1.vert.size();
	S1.fn = S1.face.size();

	S2.vn = S2.vert.size();
	S2.fn = S2.face.size();

	if (!NumberOfSamples && !SamplesPerAreaUnit) {
		NumberOfSamples = true;
		//n_samples_target = max(S1.vn, S2.vn);// take 10 samples per face
		n_samples_target = 10 * max(S1.fn, S2.fn);// take 10 samples per face
	}

	// compute face information
	tri::UpdateComponentEP<CMesh>::Set(S1);
	tri::UpdateComponentEP<CMesh>::Set(S2);

	// set bounding boxes for S1 and S2
	tri::UpdateBounding<CMesh>::Box(S1);
	tri::UpdateBounding<CMesh>::Box(S2);

	// set Bounding Box.
	Box3<CMesh::ScalarType>    bbox, tmp_bbox_M1 = S1.bbox, tmp_bbox_M2 = S2.bbox;
	bbox.Add(S1.bbox);
	bbox.Add(S2.bbox);
	bbox.Offset(bbox.Diag()*0.02);
	S1.bbox = bbox;
	S2.bbox = bbox;

	// initialize time info.
	int t0 = clock();

	Sampling<CMesh> ForwardSampling(S1, S2);
	Sampling<CMesh> BackwardSampling(S2, S1);

	// print mesh info.
	//printf("Mesh info:\n");
	//printf(" M1: '%s'\n\tvertices  %7i\n\tfaces     %7i\n\tarea      %12.4f\n", argv[1], S1.vn, S1.fn, ForwardSampling.GetArea());
	//printf("\tbbox (%7.4f %7.4f %7.4f)-(%7.4f %7.4f %7.4f)\n", tmp_bbox_M1.min[0], tmp_bbox_M1.min[1], tmp_bbox_M1.min[2], tmp_bbox_M1.max[0], tmp_bbox_M1.max[1], tmp_bbox_M1.max[2]);
	//printf("\tbbox diagonal %f\n", (float)tmp_bbox_M1.Diag());
	//printf(" M2: '%s'\n\tvertices  %7i\n\tfaces     %7i\n\tarea      %12.4f\n", argv[2], S2.vn, S2.fn, BackwardSampling.GetArea());
	//printf("\tbbox (%7.4f %7.4f %7.4f)-(%7.4f %7.4f %7.4f)\n", tmp_bbox_M2.min[0], tmp_bbox_M2.min[1], tmp_bbox_M2.min[2], tmp_bbox_M2.max[0], tmp_bbox_M2.max[1], tmp_bbox_M2.max[2]);
	//printf("\tbbox diagonal %f\n", (float)tmp_bbox_M2.Diag());

	// Forward distance.
	//printf("\nForward distance (M1 -> M2):\n");
	ForwardSampling.SetFlags(flags);
	if (NumberOfSamples){
		ForwardSampling.SetSamplesTarget(n_samples_target);
		n_samples_per_area_unit = ForwardSampling.GetNSamplesPerAreaUnit();
	}
	else{
		ForwardSampling.SetSamplesPerAreaUnit(n_samples_per_area_unit);
		n_samples_target = ForwardSampling.GetNSamplesTarget();
	}
	//printf("target # samples      : %lu\ntarget # samples/area : %f\n", n_samples_target, n_samples_per_area_unit);
	ForwardSampling.Hausdorff();
	dist1_max = ForwardSampling.GetDistMax();
	//printf("\ndistances:\n  max  : %f (%f  wrt bounding box diagonal)\n", (float)dist1_max, (float)dist1_max / bbox.Diag());
	//printf("  mean : %f\n", ForwardSampling.GetDistMean());
	//printf("  RMS  : %f\n", ForwardSampling.GetDistRMS());
	//printf("# vertex samples %9lu\n", ForwardSampling.GetNVertexSamples());
	//printf("# edge samples   %9lu\n", ForwardSampling.GetNEdgeSamples());
	//printf("# area samples   %9lu\n", ForwardSampling.GetNAreaSamples());
	//printf("# total samples  %9lu\n", ForwardSampling.GetNSamples());
	//printf("# samples per area unit: %f\n\n", ForwardSampling.GetNSamplesPerAreaUnit());

	// Backward distance.
	//printf("\nBackward distance (M2 -> M1):\n");
	BackwardSampling.SetFlags(flags);
	if (NumberOfSamples){
		BackwardSampling.SetSamplesTarget(n_samples_target);
		n_samples_per_area_unit = BackwardSampling.GetNSamplesPerAreaUnit();
	}
	else{
		BackwardSampling.SetSamplesPerAreaUnit(n_samples_per_area_unit);
		n_samples_target = BackwardSampling.GetNSamplesTarget();
	}
	//printf("target # samples      : %lu\ntarget # samples/area : %f\n", n_samples_target, n_samples_per_area_unit);
	BackwardSampling.Hausdorff();
	dist2_max = BackwardSampling.GetDistMax();
	//printf("\ndistances:\n  max  : %f (%f  wrt bounding box diagonal)\n", (float)dist2_max, (float)dist2_max / bbox.Diag());
	//printf("  mean : %f\n", BackwardSampling.GetDistMean());
	//printf("  RMS  : %f\n", BackwardSampling.GetDistRMS());
	//printf("# vertex samples %9lu\n", BackwardSampling.GetNVertexSamples());
	//printf("# edge samples   %9lu\n", BackwardSampling.GetNEdgeSamples());
	//printf("# area samples   %9lu\n", BackwardSampling.GetNAreaSamples());
	//printf("# total samples  %9lu\n", BackwardSampling.GetNSamples());
	//printf("# samples per area unit: %f\n\n", BackwardSampling.GetNSamplesPerAreaUnit());

	// compute time info.
	elapsed_time = clock() - t0;
	int n_total_sample = ForwardSampling.GetNSamples() + BackwardSampling.GetNSamples();
	double mesh_dist_max = max(dist1_max, dist2_max);

	hausdorff_ratio = (float)mesh_dist_max / bbox.Diag();
	//hausdorff_ratio_threshold /= bbox.Diag();
	//printf("\nHausdorff distance: %f (%f  wrt bounding box diagonal)\n", (float)mesh_dist_max, hausdorff_ratio);
	printf("%f  wrt bounding box diagonal\n", hausdorff_ratio);
	/*printf("  Computation time  : %d ms\n", (int)(1000.0*elapsed_time / CLOCKS_PER_SEC));
	printf("  # samples/second  : %f\n\n", (float)n_total_sample / ((float)elapsed_time / CLOCKS_PER_SEC));*/

	if (hausdorff_ratio > hausdorff_ratio_threshold) return false;
	return true;
}
int compute(Mesh &mesh0, Mesh & mesh1, double &hausdorff_ratio, double &hausdorff_ratio_threshold, double &hausdorff_dis_threshold) {
#define MSG_ERR_MESH_LOAD               "error loading the input meshes.\n"
#define MSG_ERR_INVALID_OPTION          "unable to parse option '%s'\n"
#define MSG_ERR_FILE_OPEN               "unable to open the output file.'n"
#define MSG_ERR_UNKNOWN_FORMAT          "unknown file format '%s'.\n"

	// global constants
#define NO_SAMPLES_PER_FACE             10
#define N_SAMPLES_EDGE_TO_FACE_RATIO    0.1
#define BBOX_FACTOR                     0.1
#define INFLATE_PERCENTAGE			    0.02
#define MIN_SIZE					    125		/* 125 = 5^3 */
#define N_HIST_BINS                     256
#define PRINT_EVERY_N_ELEMENTS          1000

	bool NumberOfSamples = false;
	bool SamplesPerAreaUnit = false;
	bool CleaningFlag = false;

	CMesh                 S1, S2;
	float                ColorMin = 0, ColorMax = 0;
	double                dist1_max, dist2_max;
	unsigned long         n_samples_target, elapsed_time;
	double								n_samples_per_area_unit;
	int                   flags;

	// default parameters
	flags = SamplingFlags::VERTEX_SAMPLING |
		SamplingFlags::EDGE_SAMPLING |
		SamplingFlags::FACE_SAMPLING |
		SamplingFlags::SIMILAR_SAMPLING;

	if (!(flags & SamplingFlags::USE_HASH_GRID) && !(flags & SamplingFlags::USE_AABB_TREE) && !(flags & SamplingFlags::USE_OCTREE))
		flags |= SamplingFlags::USE_STATIC_GRID;

	flags &= ~SamplingFlags::EDGE_SAMPLING;
	flags &= ~SamplingFlags::FACE_SAMPLING;

	// load input meshes.
	//OpenMesh("",S1);
	//OpenMesh(argv[2],S2);
	//string S1NewName = SaveFileName(argv[1]);
	//string S2NewName = SaveFileName(argv[2]);
	//tri::io::Importer<CMesh>::Open(m, filename);
	S1.vert.resize(mesh0.V.cols());
	for (int i = 0; i < mesh0.V.cols(); i++) {
		CVertex v;
		v.P()[0] = mesh0.V(0, i);
		v.P()[1] = mesh0.V(1, i);
		v.P()[2] = mesh0.V(2, i);
		S1.vert[i] = v;
	}
	S1.face.resize(mesh0.Fs.size());
	//Allocator<CMesh>::AddFaces(S1, mesh0.Fs.size());
	for (int i = 0; i < mesh0.Fs.size(); i++) {
		CFace f;
		f.V(0) = &(S1.vert[mesh0.Fs[i].vs[0]]);
		f.V(1) = &(S1.vert[mesh0.Fs[i].vs[1]]);
		f.V(2) = &(S1.vert[mesh0.Fs[i].vs[2]]);
		S1.face[i] = f;
	}

	S2.vert.resize(mesh1.V.cols());
	for (int i = 0; i < mesh1.V.cols(); i++) {
		CVertex v;
		v.P()[0] = mesh1.V(0, i);
		v.P()[1] = mesh1.V(1, i);
		v.P()[2] = mesh1.V(2, i);
		S2.vert[i] = v;
	}
	S2.face.resize(mesh1.Fs.size());
	//Allocator<CMesh>::AddFaces(S1, mesh0.Fs.size());
	for (int i = 0; i < mesh1.Fs.size(); i++) {
		CFace f;
		f.V(0) = &(S2.vert[mesh1.Fs[i].vs[0]]);
		f.V(1) = &(S2.vert[mesh1.Fs[i].vs[1]]);
		f.V(2) = &(S2.vert[mesh1.Fs[i].vs[2]]);
		S2.face[i] = f;
	}

	S1.vn = S1.vert.size();
	S1.fn = S1.face.size();

	S2.vn = S2.vert.size();
	S2.fn = S2.face.size();

	if (!NumberOfSamples && !SamplesPerAreaUnit) {
		NumberOfSamples = true;
		//n_samples_target = max(S1.vn, S2.vn);// take 10 samples per face
		n_samples_target = 10 * max(S1.fn, S2.fn);// take 10 samples per face
	}

	// compute face information
	tri::UpdateComponentEP<CMesh>::Set(S1);
	tri::UpdateComponentEP<CMesh>::Set(S2);

	// set bounding boxes for S1 and S2
	tri::UpdateBounding<CMesh>::Box(S1);
	tri::UpdateBounding<CMesh>::Box(S2);

	// set Bounding Box.
	Box3<CMesh::ScalarType>    bbox, tmp_bbox_M1 = S1.bbox, tmp_bbox_M2 = S2.bbox;
	bbox.Add(S1.bbox);
	bbox.Add(S2.bbox);
	bbox.Offset(bbox.Diag()*0.02);
	S1.bbox = bbox;
	S2.bbox = bbox;

	// initialize time info.
	int t0 = clock();

	Sampling<CMesh> ForwardSampling(S1, S2);
	Sampling<CMesh> BackwardSampling(S2, S1);
	// Forward distance.
	//printf("\nForward distance (M1 -> M2):\n");
	ForwardSampling.SetFlags(flags);
	if (NumberOfSamples) {
		ForwardSampling.SetSamplesTarget(n_samples_target);
		n_samples_per_area_unit = ForwardSampling.GetNSamplesPerAreaUnit();
	}
	else {
		ForwardSampling.SetSamplesPerAreaUnit(n_samples_per_area_unit);
		n_samples_target = ForwardSampling.GetNSamplesTarget();
	}
	//printf("target # samples      : %lu\ntarget # samples/area : %f\n", n_samples_target, n_samples_per_area_unit);
	ForwardSampling.Hausdorff();
	dist1_max = ForwardSampling.GetDistMax();
	// Backward distance.
	//printf("\nBackward distance (M2 -> M1):\n");
	BackwardSampling.SetFlags(flags);
	if (NumberOfSamples) {
		BackwardSampling.SetSamplesTarget(n_samples_target);
		n_samples_per_area_unit = BackwardSampling.GetNSamplesPerAreaUnit();
	}
	else {
		BackwardSampling.SetSamplesPerAreaUnit(n_samples_per_area_unit);
		n_samples_target = BackwardSampling.GetNSamplesTarget();
	}
	//printf("target # samples      : %lu\ntarget # samples/area : %f\n", n_samples_target, n_samples_per_area_unit);
	BackwardSampling.Hausdorff();
	dist2_max = BackwardSampling.GetDistMax();
	// compute time info.
	elapsed_time = clock() - t0;
	int n_total_sample = ForwardSampling.GetNSamples() + BackwardSampling.GetNSamples();
	double mesh_dist_max = max(dist1_max, dist2_max);

	//cout << "mesh_dist_max bbox.Diag(): " << mesh_dist_max << "; " << bbox.Diag() << endl;
	hausdorff_ratio = (float)mesh_dist_max / bbox.Diag();
	hausdorff_dis_threshold = hausdorff_ratio_threshold * bbox.Diag();
	//hausdorff_ratio_threshold /= bbox.Diag();
	//printf("\nHausdorff distance: %f (%f  wrt bounding box diagonal)\n", (float)mesh_dist_max, hausdorff_ratio);
	printf("%f  wrt bounding box diagonal\n", hausdorff_ratio);
	/*printf("  Computation time  : %d ms\n", (int)(1000.0*elapsed_time / CLOCKS_PER_SEC));
	printf("  # samples/second  : %f\n\n", (float)n_total_sample / ((float)elapsed_time / CLOCKS_PER_SEC));*/

	double max_ave_dis_ratio = (std::max(BackwardSampling.GetDistMean(), ForwardSampling.GetDistMean()))/ bbox.Diag();
	printf("%f  average wrt bounding box diagonal\n", max_ave_dis_ratio);

	//if (hausdorff_ratio > hausdorff_ratio_threshold || max_ave_dis_ratio>0.5*hausdorff_ratio_threshold) return false;
	if (hausdorff_ratio > hausdorff_ratio_threshold) return false;
	return true;
}
void compute(const Mesh &mesh0, const Mesh &mesh1, double & bbox_diagonal, double & max_hausdorff_dis, double & ave_hausdorff_dis) {
#define MSG_ERR_MESH_LOAD               "error loading the input meshes.\n"
#define MSG_ERR_INVALID_OPTION          "unable to parse option '%s'\n"
#define MSG_ERR_FILE_OPEN               "unable to open the output file.'n"
#define MSG_ERR_UNKNOWN_FORMAT          "unknown file format '%s'.\n"

	// global constants
#define NO_SAMPLES_PER_FACE             10
#define N_SAMPLES_EDGE_TO_FACE_RATIO    0.1
#define BBOX_FACTOR                     0.1
#define INFLATE_PERCENTAGE			    0.02
#define MIN_SIZE					    125		/* 125 = 5^3 */
#define N_HIST_BINS                     256
#define PRINT_EVERY_N_ELEMENTS          1000

	bool NumberOfSamples = false;
	bool SamplesPerAreaUnit = false;
	bool CleaningFlag = false;

	CMesh                 S1, S2;
	float                ColorMin = 0, ColorMax = 0;
	double                dist1_max, dist2_max;
	unsigned long         n_samples_target, elapsed_time;
	double								n_samples_per_area_unit;
	int                   flags;

	// default parameters
	flags = SamplingFlags::VERTEX_SAMPLING |
		SamplingFlags::EDGE_SAMPLING |
		SamplingFlags::FACE_SAMPLING |
		SamplingFlags::SIMILAR_SAMPLING;

	if (!(flags & SamplingFlags::USE_HASH_GRID) && !(flags & SamplingFlags::USE_AABB_TREE) && !(flags & SamplingFlags::USE_OCTREE))
		flags |= SamplingFlags::USE_STATIC_GRID;

	flags &= ~SamplingFlags::EDGE_SAMPLING;
	flags &= ~SamplingFlags::FACE_SAMPLING;

	// load input meshes.
	//OpenMesh("",S1);
	//OpenMesh(argv[2],S2);
	//string S1NewName = SaveFileName(argv[1]);
	//string S2NewName = SaveFileName(argv[2]);
	//tri::io::Importer<CMesh>::Open(m, filename);
	S1.vert.resize(mesh0.V.cols());
	for (int i = 0; i < mesh0.V.cols(); i++) {
		CVertex v;
		v.P()[0] = mesh0.V(0, i);
		v.P()[1] = mesh0.V(1, i);
		v.P()[2] = mesh0.V(2, i);
		S1.vert[i] = v;
	}
	S1.face.resize(mesh0.Fs.size());
	//Allocator<CMesh>::AddFaces(S1, mesh0.Fs.size());
	for (int i = 0; i < mesh0.Fs.size(); i++) {
		CFace f;
		f.V(0) = &(S1.vert[mesh0.Fs[i].vs[0]]);
		f.V(1) = &(S1.vert[mesh0.Fs[i].vs[1]]);
		f.V(2) = &(S1.vert[mesh0.Fs[i].vs[2]]);
		S1.face[i] = f;
	}

	S2.vert.resize(mesh1.V.cols());
	for (int i = 0; i < mesh1.V.cols(); i++) {
		CVertex v;
		v.P()[0] = mesh1.V(0, i);
		v.P()[1] = mesh1.V(1, i);
		v.P()[2] = mesh1.V(2, i);
		S2.vert[i] = v;
	}
	S2.face.resize(mesh1.Fs.size());
	//Allocator<CMesh>::AddFaces(S1, mesh0.Fs.size());
	for (int i = 0; i < mesh1.Fs.size(); i++) {
		CFace f;
		f.V(0) = &(S2.vert[mesh1.Fs[i].vs[0]]);
		f.V(1) = &(S2.vert[mesh1.Fs[i].vs[1]]);
		f.V(2) = &(S2.vert[mesh1.Fs[i].vs[2]]);
		S2.face[i] = f;
	}

	S1.vn = S1.vert.size();
	S1.fn = S1.face.size();

	S2.vn = S2.vert.size();
	S2.fn = S2.face.size();

	if (!NumberOfSamples && !SamplesPerAreaUnit) {
		NumberOfSamples = true;
		//n_samples_target = max(S1.vn, S2.vn);// take 10 samples per face
		n_samples_target = 10 * max(S1.fn, S2.fn);// take 10 samples per face
	}

	// compute face information
	tri::UpdateComponentEP<CMesh>::Set(S1);
	tri::UpdateComponentEP<CMesh>::Set(S2);

	// set bounding boxes for S1 and S2
	tri::UpdateBounding<CMesh>::Box(S1);
	tri::UpdateBounding<CMesh>::Box(S2);

	// set Bounding Box.
	Box3<CMesh::ScalarType>    bbox, tmp_bbox_M1 = S1.bbox, tmp_bbox_M2 = S2.bbox;
	bbox.Add(S1.bbox);
	bbox.Add(S2.bbox);
	bbox.Offset(bbox.Diag()*0.02);
	S1.bbox = bbox;
	S2.bbox = bbox;

	// initialize time info.
	int t0 = clock();

	Sampling<CMesh> ForwardSampling(S1, S2);
	Sampling<CMesh> BackwardSampling(S2, S1);
	// Forward distance.
	//printf("\nForward distance (M1 -> M2):\n");
	ForwardSampling.SetFlags(flags);
	if (NumberOfSamples) {
		ForwardSampling.SetSamplesTarget(n_samples_target);
		n_samples_per_area_unit = ForwardSampling.GetNSamplesPerAreaUnit();
	}
	else {
		ForwardSampling.SetSamplesPerAreaUnit(n_samples_per_area_unit);
		n_samples_target = ForwardSampling.GetNSamplesTarget();
	}
	//printf("target # samples      : %lu\ntarget # samples/area : %f\n", n_samples_target, n_samples_per_area_unit);
	ForwardSampling.Hausdorff();
	dist1_max = ForwardSampling.GetDistMax();
	// Backward distance.
	//printf("\nBackward distance (M2 -> M1):\n");
	BackwardSampling.SetFlags(flags);
	if (NumberOfSamples) {
		BackwardSampling.SetSamplesTarget(n_samples_target);
		n_samples_per_area_unit = BackwardSampling.GetNSamplesPerAreaUnit();
	}
	else {
		BackwardSampling.SetSamplesPerAreaUnit(n_samples_per_area_unit);
		n_samples_target = BackwardSampling.GetNSamplesTarget();
	}
	//printf("target # samples      : %lu\ntarget # samples/area : %f\n", n_samples_target, n_samples_per_area_unit);
	BackwardSampling.Hausdorff();
	dist2_max = BackwardSampling.GetDistMax();
	// compute time info.
	elapsed_time = clock() - t0;
	int n_total_sample = ForwardSampling.GetNSamples() + BackwardSampling.GetNSamples();
	
	bbox_diagonal = bbox.Diag();
	max_hausdorff_dis = max(dist1_max, dist2_max);
	ave_hausdorff_dis = max(BackwardSampling.GetDistMean(), ForwardSampling.GetDistMean());
}