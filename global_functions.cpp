#include "global_functions.h"
#include "global_types.h"
#include <unordered_set>
#include "igl/bounding_box_diagonal.h"
#include <igl/embree/reorient_facets_raycast.h>
//for signed distance
#include <igl/edge_lengths.h>
#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/signed_distance.h>
#include <Eigen/Sparse>

//===================================mesh connectivities===================================
void build_connectivity(Mesh &hmi) {
	hmi.Es.clear(); if (hmi.Hs.size() &&hmi.type!= Mesh_type::Hyb) hmi.Fs.clear();
	//either hex or tri
	if (hmi.type == Mesh_type::Tri || hmi.type == Mesh_type::Qua|| hmi.type == Mesh_type::HSur) {
		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> temp;
		temp.reserve(hmi.Fs.size() * 3);
		for (uint32_t i = 0; i < hmi.Fs.size(); ++i) {
			int vn = hmi.Fs[i].vs.size();
			for (uint32_t j = 0; j < vn; ++j) {
				uint32_t v0 = hmi.Fs[i].vs[j], v1 = hmi.Fs[i].vs[(j + 1) % vn];
				if (v0 > v1) std::swap(v0, v1);
				temp.push_back(std::make_tuple(v0, v1, i, j));
			}
			hmi.Fs[i].es.resize(vn);
		}
		std::sort(temp.begin(), temp.end());
		hmi.Es.reserve(temp.size() / 2);
		uint32_t E_num = 0;
		Hybrid_E e; e.boundary = true; e.vs.resize(2);
		for (uint32_t i = 0; i < temp.size(); ++i) {
			if (i == 0 || (i != 0 && (std::get<0>(temp[i]) != std::get<0>(temp[i - 1]) ||
				std::get<1>(temp[i]) != std::get<1>(temp[i - 1])))) {
				e.id = E_num; E_num++;
				e.vs[0] = std::get<0>(temp[i]);
				e.vs[1] = std::get<1>(temp[i]);
				hmi.Es.push_back(e);
			}
			else if (i != 0 && (std::get<0>(temp[i]) == std::get<0>(temp[i - 1]) &&
				std::get<1>(temp[i]) == std::get<1>(temp[i - 1])))
				hmi.Es[E_num - 1].boundary = false;

			hmi.Fs[std::get<2>(temp[i])].es[std::get<3>(temp[i])] = E_num - 1;
		}
		//boundary
		for (auto &v : hmi.Vs) v.boundary = false;
		for (uint32_t i = 0; i < hmi.Es.size(); ++i)
			if (hmi.Es[i].boundary) {
				hmi.Vs[hmi.Es[i].vs[0]].boundary = hmi.Vs[hmi.Es[i].vs[1]].boundary = true;
			}
	}
	else if (hmi.type == Mesh_type::Tet) {
		std::vector<std::vector<uint32_t>> total_fs(hmi.Hs.size() * 4);
		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> tempF(hmi.Hs.size() * 4);
		std::vector<uint32_t> vs(3);
		for (uint32_t i = 0; i < hmi.Hs.size(); ++i) {
			for (short j = 0; j < 4; j++) {
				for (short k = 0; k < 3; k++) vs[k] = hmi.Hs[i].vs[tet_faces[j][k]];
				uint32_t id = 4 * i + j;
				total_fs[id] = vs;
				std::sort(vs.begin(), vs.end());
				tempF[id] = std::make_tuple(vs[0], vs[1], vs[2], id, i, j);
			}
			hmi.Hs[i].fs.resize(4);
		}
		std::sort(tempF.begin(), tempF.end());
		hmi.Fs.reserve(tempF.size() / 3);
		Hybrid_F f; f.boundary = true;
		uint32_t F_num = 0;
		for (uint32_t i = 0; i < tempF.size(); ++i) {
			if (i == 0 || (i != 0 &&
				(std::get<0>(tempF[i]) != std::get<0>(tempF[i - 1]) || std::get<1>(tempF[i]) != std::get<1>(tempF[i - 1]) ||
					std::get<2>(tempF[i]) != std::get<2>(tempF[i - 1])))) {
				f.id = F_num; F_num++;
				f.vs = total_fs[std::get<3>(tempF[i])];
				hmi.Fs.push_back(f);
			}
			else if (i != 0 && (std::get<0>(tempF[i]) == std::get<0>(tempF[i - 1]) && std::get<1>(tempF[i]) == std::get<1>(tempF[i - 1]) &&
				std::get<2>(tempF[i]) == std::get<2>(tempF[i - 1])))
				hmi.Fs[F_num - 1].boundary = false;

			hmi.Hs[std::get<4>(tempF[i])].fs[std::get<5>(tempF[i])] = F_num - 1;
		}

		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> temp(hmi.Fs.size() * 3);
		for (uint32_t i = 0; i < hmi.Fs.size(); ++i) {
			for (uint32_t j = 0; j < 3; ++j) {
				uint32_t v0 = hmi.Fs[i].vs[j], v1 = hmi.Fs[i].vs[(j + 1) % 3];
				if (v0 > v1) std::swap(v0, v1);
				temp[3 * i + j] = std::make_tuple(v0, v1, i, j);
			}
			hmi.Fs[i].es.resize(3);
		}
		std::sort(temp.begin(), temp.end());
		hmi.Es.reserve(temp.size() / 2);
		uint32_t E_num = 0;
		Hybrid_E e; e.boundary = false; e.vs.resize(2);
		for (uint32_t i = 0; i < temp.size(); ++i) {
			if (i == 0 || (i != 0 && (std::get<0>(temp[i]) != std::get<0>(temp[i - 1]) ||
				std::get<1>(temp[i]) != std::get<1>(temp[i - 1])))) {
				e.id = E_num; E_num++;
				e.vs[0] = std::get<0>(temp[i]);
				e.vs[1] = std::get<1>(temp[i]);
				hmi.Es.push_back(e);
			}
			hmi.Fs[std::get<2>(temp[i])].es[std::get<3>(temp[i])] = E_num - 1;
		}
		//boundary
		for (auto &v : hmi.Vs) v.boundary = false;
		for (uint32_t i = 0; i < hmi.Fs.size(); ++i)
			if (hmi.Fs[i].boundary) for (uint32_t j = 0; j < 3; ++j) {
				uint32_t eid = hmi.Fs[i].es[j];
				hmi.Es[eid].boundary = true;
				hmi.Vs[hmi.Es[eid].vs[0]].boundary = hmi.Vs[hmi.Es[eid].vs[1]].boundary = true;
			}
	}
	else if(hmi.type == Mesh_type::Hex) {

		std::vector<std::vector<uint32_t>> total_fs(hmi.Hs.size() * 6);
		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>> tempF(hmi.Hs.size() * 6);
		std::vector<uint32_t> vs(4);
		for (uint32_t i = 0; i < hmi.Hs.size(); ++i) {
			for (short j = 0; j < 6; j++){
				for (short k = 0; k < 4; k++) vs[k] = hmi.Hs[i].vs[hex_face_table[j][k]];
				uint32_t id = 6 * i + j;
				total_fs[id] = vs;
				std::sort(vs.begin(), vs.end());
				tempF[id] = std::make_tuple(vs[0], vs[1], vs[2], vs[3], id, i, j);
			}
			hmi.Hs[i].fs.resize(6);
		}
		std::sort(tempF.begin(), tempF.end());
		hmi.Fs.reserve(tempF.size() / 3);
		Hybrid_F f; f.boundary = true;
		uint32_t F_num = 0;
		for (uint32_t i = 0; i < tempF.size(); ++i) {
			if (i == 0 || (i != 0 &&
				(std::get<0>(tempF[i]) != std::get<0>(tempF[i - 1]) || std::get<1>(tempF[i]) != std::get<1>(tempF[i - 1]) ||
					std::get<2>(tempF[i]) != std::get<2>(tempF[i - 1]) || std::get<3>(tempF[i]) != std::get<3>(tempF[i - 1])))) {
				f.id = F_num; F_num++;
				f.vs = total_fs[std::get<4>(tempF[i])];
				hmi.Fs.push_back(f);
			}
			else if (i != 0 && (std::get<0>(tempF[i]) == std::get<0>(tempF[i - 1]) && std::get<1>(tempF[i]) == std::get<1>(tempF[i - 1]) &&
				std::get<2>(tempF[i]) == std::get<2>(tempF[i - 1]) && std::get<3>(tempF[i]) == std::get<3>(tempF[i - 1])))
				hmi.Fs[F_num - 1].boundary = false;

			hmi.Hs[std::get<5>(tempF[i])].fs[std::get<6>(tempF[i])] = F_num - 1;
		}

		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> temp(hmi.Fs.size() * 4);
		for (uint32_t i = 0; i < hmi.Fs.size(); ++i) {
			for (uint32_t j = 0; j < 4; ++j) {
				uint32_t v0 = hmi.Fs[i].vs[j], v1 = hmi.Fs[i].vs[(j + 1) % 4];
				if (v0 > v1) std::swap(v0, v1);
				temp[4 * i + j] = std::make_tuple(v0, v1, i, j);
			}
			hmi.Fs[i].es.resize(4);
		}
		std::sort(temp.begin(), temp.end());
		hmi.Es.reserve(temp.size() / 2);
		uint32_t E_num = 0;
		Hybrid_E e; e.boundary = false; e.vs.resize(2);
		for (uint32_t i = 0; i < temp.size(); ++i) {
			if (i == 0 || (i != 0 && (std::get<0>(temp[i]) != std::get<0>(temp[i - 1]) ||
				std::get<1>(temp[i]) != std::get<1>(temp[i - 1])))) {
				e.id = E_num; E_num++;
				e.vs[0] = std::get<0>(temp[i]);
				e.vs[1] = std::get<1>(temp[i]);
				hmi.Es.push_back(e);
			}
			hmi.Fs[std::get<2>(temp[i])].es[std::get<3>(temp[i])] = E_num - 1;
		}
		//boundary
		for (auto &v : hmi.Vs) v.boundary = false;
		for (uint32_t i = 0; i < hmi.Fs.size(); ++i)
			if (hmi.Fs[i].boundary) for (uint32_t j = 0; j < 4; ++j) {
				uint32_t eid = hmi.Fs[i].es[j];
				hmi.Es[eid].boundary = true;
				hmi.Vs[hmi.Es[eid].vs[0]].boundary = hmi.Vs[hmi.Es[eid].vs[1]].boundary = true;
			}
	}
	else if(hmi.type == Mesh_type::Hyb) {

		for (auto h : hmi.Hs)for (auto fid : h.fs)hmi.Fs[fid].neighbor_hs.push_back(h.id);
		for (auto &f : hmi.Fs)if (f.neighbor_hs.size() == 2)f.boundary = false; else f.boundary = true;
		for (auto &f : hmi.Fs) f.neighbor_hs.clear();

		std::vector<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> temp;
		for (auto &f: hmi.Fs) {
			for (uint32_t j = 0; j < f.vs.size(); ++j) {
				uint32_t v0 = f.vs[j], v1 = f.vs[(j + 1) % f.vs.size()];
				if (v0 > v1) std::swap(v0, v1);
				temp.push_back(std::make_tuple(v0, v1, f.id, j));
			}
			hmi.Fs[f.id].es.resize(f.vs.size());
		}
		std::sort(temp.begin(), temp.end());
		hmi.Es.reserve(temp.size() / 2);
		uint32_t E_num = 0;
		Hybrid_E e; e.boundary = false; e.vs.resize(2);
		for (uint32_t i = 0; i < temp.size(); ++i) {
			if (i == 0 || (i != 0 && (std::get<0>(temp[i]) != std::get<0>(temp[i - 1]) ||
				std::get<1>(temp[i]) != std::get<1>(temp[i - 1])))) {
				e.id = E_num; E_num++;
				e.vs[0] = std::get<0>(temp[i]);
				e.vs[1] = std::get<1>(temp[i]);
				hmi.Es.push_back(e);
			}
			hmi.Fs[std::get<2>(temp[i])].es[std::get<3>(temp[i])] = E_num - 1;
		}
		//boundary
		for (auto &v : hmi.Vs) v.boundary = false;
		for (auto f:hmi.Fs)
			if (f.boundary) for (uint32_t j = 0; j < f.es.size(); ++j) {
				uint32_t eid = f.es[j];
				hmi.Es[eid].boundary = true;
				hmi.Vs[hmi.Es[eid].vs[0]].boundary = hmi.Vs[hmi.Es[eid].vs[1]].boundary = true;
			}
	}
	//f_nhs;
	for (auto &f : hmi.Fs)f.neighbor_hs.clear();
	for (uint32_t i = 0; i < hmi.Hs.size(); i++) {
		for (uint32_t j = 0; j < hmi.Hs[i].fs.size(); j++) hmi.Fs[hmi.Hs[i].fs[j]].neighbor_hs.push_back(i);
	}
	//e_nfs, v_nfs
	for (auto &e : hmi.Es) e.neighbor_fs.clear();
	for (auto &v : hmi.Vs) v.neighbor_fs.clear();
	for (uint32_t i = 0; i < hmi.Fs.size(); i++) {
		for (uint32_t j = 0; j < hmi.Fs[i].es.size(); j++) hmi.Es[hmi.Fs[i].es[j]].neighbor_fs.push_back(i);
		for (uint32_t j = 0; j < hmi.Fs[i].vs.size(); j++) hmi.Vs[hmi.Fs[i].vs[j]].neighbor_fs.push_back(i);
	}
	//v_nes, v_nvs
	for (auto &v : hmi.Vs) {
		v.neighbor_es.clear();
		v.neighbor_vs.clear();
	}
	for (uint32_t i = 0; i < hmi.Es.size(); i++) {
		uint32_t v0 = hmi.Es[i].vs[0], v1 = hmi.Es[i].vs[1];
		hmi.Vs[v0].neighbor_es.push_back(i);
		hmi.Vs[v1].neighbor_es.push_back(i);
		hmi.Vs[v0].neighbor_vs.push_back(v1);
		hmi.Vs[v1].neighbor_vs.push_back(v0);
	}
	//e_nhs
	for (auto &e : hmi.Es) e.neighbor_hs.clear();
	for (uint32_t i = 0; i < hmi.Es.size(); i++) {
		std::vector<uint32_t> nhs;
		for (uint32_t j = 0; j < hmi.Es[i].neighbor_fs.size(); j++) {
			uint32_t nfid = hmi.Es[i].neighbor_fs[j];
			nhs.insert(nhs.end(), hmi.Fs[nfid].neighbor_hs.begin(), hmi.Fs[nfid].neighbor_hs.end());
		}
		std::sort(nhs.begin(), nhs.end()); nhs.erase(std::unique(nhs.begin(), nhs.end()), nhs.end());
		hmi.Es[i].neighbor_hs = nhs;
	}
	//v_nhs
	for (auto &v : hmi.Vs) v.neighbor_hs.clear();
	for (uint32_t i = 0; i < hmi.Hs.size(); i++) {
		for (uint32_t j = 0; j < hmi.Hs[i].vs.size(); j++) hmi.Vs[hmi.Hs[i].vs[j]].neighbor_hs.push_back(i);
	}
}
void topology_info(Mesh &mesh, Frame &frame, Mesh_Topology & mt) {

//==================hex-mesh==================//
	for (int i = 0; i < mesh.Vs.size(); i++) if (!mesh.Vs[i].neighbor_hs.size()) cout << "Mesh Topology ERROR double check!" << endl;
	//surface_euler, volume_euler;
	uint32_t num_boundary_v = 0, num_boundary_e = 0, num_boundary_f = 0;
	for (int i = 0; i<mesh.Vs.size(); i++) if (mesh.Vs[i].boundary) num_boundary_v++;
	for (int i = 0; i<mesh.Es.size(); i++) if (mesh.Es[i].boundary) num_boundary_e++;
	for (int i = 0; i<mesh.Fs.size(); i++) if (mesh.Fs[i].boundary) num_boundary_f++;
	mt.surface_euler = num_boundary_v + num_boundary_f - num_boundary_e;
	mt.genus = (2 - mt.surface_euler) / 2;
	mt.volume_euler = mesh.Vs.size() + mesh.Fs.size() - mesh.Es.size() - mesh.Hs.size();
	mt.euler_problem = false;
	if (mt.surface_euler != 2 * mt.volume_euler) { cout << "Surface euler not equal to volume euler" << endl; mt.euler_problem = true; }
	//surface_manifoldness;
	mt.manifoldness_problem = false;
	mt.surface_manifoldness = true;
	vector<short> E_flag(mesh.Es.size(), 0); vector<short> V_flag(mesh.Vs.size(), 0);
	for (uint32_t i = 0; i<mesh.Vs.size(); i++){//topology disk
		if (!mesh.Vs[i].boundary) continue;
		vector<vector<uint32_t>> fes;
		for (auto fid : mesh.Vs[i].neighbor_fs) {
			if (mesh.Fs[fid].boundary) fes.push_back(mesh.Fs[fid].es);
		}
		if (!disk_polygon(mesh, frame, fes, E_flag, V_flag, true)) {
			mt.surface_manifoldness = false;
			mt.manifoldness_problem = true;
			break;
		}
	}
	//volume_manifoldness;
	mt.volume_manifoldness = true;
	vector<bool> F_flag(mesh.Fs.size(), false);
	vector<vector<uint32_t>> F_nvs;
//==================frame==================//
	for (int i = 0; i < frame.FVs.size(); i++) if (!frame.FVs[i].neighbor_fhs.size()) cout << "Frame Topology ERROR double check!" << endl;
	//surface_euler, volume_euler;
	num_boundary_v = num_boundary_e = num_boundary_f = 0;
	for (int i = 0; i<frame.FVs.size(); i++) if (frame.FVs[i].boundary) num_boundary_v++;
	for (int i = 0; i<frame.FEs.size(); i++) if (frame.FEs[i].boundary) num_boundary_e++;
	for (int i = 0; i<frame.FFs.size(); i++) if (frame.FFs[i].boundary) num_boundary_f++;
	mt.frame_surface_euler = num_boundary_v + num_boundary_f - num_boundary_e;
	mt.frame_genus = (2 - mt.frame_surface_euler) / 2;
	mt.frame_volume_euler = frame.FVs.size() + frame.FFs.size() - frame.FEs.size() - frame.FHs.size();
	mt.frame_euler_problem = false;
	if (mt.frame_surface_euler != 2 * mt.frame_volume_euler) { cout << "Frame surface euler not equal to volume euler" << endl; mt.frame_euler_problem = true; }
	//surface_manifoldness;
	mt.frame_manifoldness_problem = false;
	mt.frame_surface_manifoldness = true;
	E_flag.resize(frame.FEs.size()); V_flag.resize(frame.FVs.size());
	for (uint32_t i = 0; i<frame.FVs.size(); i++) {//topology disk
		if (!frame.FVs[i].boundary) continue;
		vector<vector<uint32_t>> fes;
		for (auto fid : frame.FVs[i].neighbor_ffs) {
			if (frame.FFs[fid].boundary) fes.push_back(frame.FFs[fid].es);
		}
		if (!disk_polygon(mesh, frame, fes, E_flag, V_flag, false)) {
			mt.frame_surface_manifoldness = false;
			mt.frame_manifoldness_problem = true;
			break;
		}
	}
	//volume_manifoldness;
	mt.frame_volume_manifoldness = true;
	F_flag.resize(frame.FFs.size()); E_flag.resize(frame.FEs.size()); V_flag.resize(frame.FVs.size());
	F_nvs.resize(frame.FVs.size());
	for (uint32_t i = 0; i<frame.FVs.size(); i++) {//topology sphere
		vector<vector<uint32_t>> pfs;
		for (auto hid : frame.FVs[i].neighbor_fhs) pfs.push_back(frame.FHs[hid].fs);

		if (!sphere_polyhedral(mesh, frame, F_nvs, pfs, F_flag, E_flag, V_flag, false)) {
			mt.frame_volume_manifoldness = false;
			mt.frame_manifoldness_problem = true;
			break;
		}
	}
}
bool manifoldness_closeness_check(Mesh &mesh)
{
	if(mesh.type == Mesh_type::Tri)
	{
		for(auto &e:mesh.Es)
		{
			if(e.neighbor_fs.size()!=2)
				return false;
		}

		std::vector<short> E_flag(mesh.Es.size(), 0), V_flag(mesh.Vs.size(), 0);
		Frame frame;
		for(auto &v:mesh.Vs)
		{
			std::vector<std::vector<uint32_t>> fes;
			for(auto &fid: v.neighbor_fs)	
				fes.push_back(mesh.Fs[fid].es);
			if(!disk_polygon(mesh, frame, fes, E_flag, V_flag, true))
				return false;
		}
	}

	return true;
}
bool disk_polygon(Mesh &mesh, Frame &frame, vector<vector<uint32_t>> &fes, vector<short> &E_flag, vector<short> &V_flag, const bool &Ismesh) {
	vector<uint32_t> pes;
	for (int i = 0; i < fes.size(); i++) {
		for (int j = 0; j < fes[i].size(); j++)
			if (E_flag[fes[i][j]]) E_flag[fes[i][j]] = false;
			else E_flag[fes[i][j]] = true;
	}
	for (int i = 0; i < fes.size(); i++)
		for (int j = 0; j < fes[i].size(); j++)
			if (E_flag[fes[i][j]]) { pes.push_back(fes[i][j]); E_flag[fes[i][j]] = false; }
	//test nvs for each v
	function<void(vector<uint32_t> &, const uint32_t &)> two_vs = [&](vector<uint32_t> &vs, const uint32_t & eid) {
		if (Ismesh) { vs = mesh.Es[eid].vs; }
		else { vs = frame.FEs[eid].vs; }
	};

	vector<uint32_t> vs;
	for (uint32_t i = 0; i < pes.size(); ++i) {
		two_vs(vs, pes[i]);
		V_flag[vs[0]]++; V_flag[vs[1]]++;
		if (V_flag[vs[0]] > 2 || V_flag[vs[1]] > 2) {
			for (uint32_t j = 0; j < pes.size(); ++j) {
				two_vs(vs, pes[j]);
				V_flag[vs[0]] = V_flag[vs[1]] = 0;
			}
			return false;
		}
	}
	for (uint32_t i = 0; i < pes.size(); ++i) {
		two_vs(vs, pes[i]);
		V_flag[vs[0]] = V_flag[vs[1]] = 0;
	}
	//extract the polygon	
	if (!pes.size()) return false;
	vector<uint32_t> pvs;
	pvs.reserve(pes.size());
	vector<bool> e_flag(pes.size(), false);
	two_vs(vs, pes[0]);
	pvs.insert(pvs.end(), vs.begin(),vs.end());
	e_flag[0] = true;
	uint32_t start_v = pvs[1];
	for (uint32_t i = 2; i < pes.size(); i++) {
		for (uint32_t j = 1; j < pes.size(); j++) {
			if (!e_flag[j]) {

				two_vs(vs, pes[j]);
				if (vs[0] == start_v) {
					e_flag[j] = true;
					pvs.push_back(vs[1]);
					start_v = vs[1];
					break;
				}
				else if (vs[1] == start_v) {
					e_flag[j] = true;
					pvs.push_back(vs[0]);
					start_v = vs[0];
					break;
				}
			}
		}
	}
	if (pvs.size() != pes.size()) return false;
	return true;
}
bool sphere_polyhedral(Mesh &mesh, Frame &frame, vector<vector<uint32_t>> &F_nvs, vector<vector<uint32_t>> &pfs, vector<bool> &F_flag, vector<short> &E_flag, vector<short> &V_flag, const bool &Ismesh) {

	vector<uint32_t> pf;
	for (int i = 0; i < pfs.size(); i++)
		for (int j = 0; j < pfs[i].size(); j++)
			if (F_flag[pfs[i][j]]) F_flag[pfs[i][j]] = false;
			else F_flag[pfs[i][j]] = true;
	for (int i = 0; i < pfs.size(); i++)
		for (int j = 0; j < pfs[i].size(); j++)
			if (F_flag[pfs[i][j]]) { pf.push_back(pfs[i][j]); F_flag[pfs[i][j]] = false; }
	//test each e whether non-manifold
	function<void(vector<uint32_t> &, const uint32_t &)> four_es = [&](vector<uint32_t> &es, const uint32_t & fid) {
		if (Ismesh) { es = mesh.Fs[fid].es; }
		else { es = frame.FFs[fid].es; }
	};
	function<void(vector<uint32_t> &, const uint32_t &)> four_vs = [&](vector<uint32_t> &vs, const uint32_t & fid) {
		if (Ismesh) { vs = mesh.Fs[fid].vs; }
		else { vs = frame.FFs[fid].vs; }
	};
	bool non_simple = false;
	vector<uint32_t> es;
	for (uint32_t i = 0; i < pf.size(); ++i) {
		four_es(es, pf[i]);
		for (auto eid : es) {
			E_flag[eid]++;
			if (E_flag[eid] > 2) non_simple = true;
		}
		if (non_simple) {
			for (uint32_t k = 0; k < pf.size(); ++k) {
				four_es(es, pf[k]);
				for (auto eid : es) E_flag[eid] = false;
			}
			return false;
		}
	}
	for (uint32_t k = 0; k < pf.size(); ++k){
		four_es(es, pf[k]);
		for (auto eid : es) if(E_flag[eid]!=2) non_simple = true;
	}
	for (uint32_t k = 0; k < pf.size(); ++k) {
		four_es(es, pf[k]);
		for (auto eid : es) E_flag[eid] = false;
	}
	if (non_simple) return false;
	//test each v whether non-manifold
	std::vector<uint32_t> vs_set;
	for (uint32_t i = 0; i < pf.size(); ++i) {
		std::vector<uint32_t> vs;
		four_vs(vs, pf[i]);
		for (auto vid : vs) {
			F_nvs[vid].push_back(i);
			if (!V_flag[vid]) {vs_set.push_back(vid); V_flag[vid] = true;}
		}
	}
	for (auto vid: vs_set)  V_flag[vid] = false;

	for (auto vid: vs_set) {
		sort(F_nvs[vid].begin(), F_nvs[vid].end());
		F_nvs[vid].erase(std::unique(F_nvs[vid].begin(), F_nvs[vid].end()), F_nvs[vid].end());

		vector<vector<uint32_t>> fes;
		for (auto fid : F_nvs[vid])
			if(Ismesh) fes.push_back(mesh.Fs[fid].es); else fes.push_back(frame.FFs[fid].es);

		if (!disk_polygon(mesh, frame, fes, E_flag, V_flag, Ismesh)){
			non_simple = true; break;
		}
	}
	for (auto vid : vs_set) F_nvs[vid].clear();

	if (non_simple) return false;
	return true;
}
bool comp_topology(Mesh_Topology & mt0, Mesh_Topology & mt1) {

	if (mt1.manifoldness_problem) return false;
	if (mt0.genus != mt1.genus) return false;
	if (mt0.frame_genus != mt1.frame_genus) return false;
	return true;
}
bool redundentV_check(Mesh &meshI, Mesh &meshO) {
	bool redundentV = false;
	build_connectivity(meshI);
	vector<int> V_tag(meshI.Vs.size(), -1);
	int vI = 0;
	for (auto v : meshI.Vs) {
		if (!v.neighbor_hs.size() || !v.neighbor_fs.size()) { redundentV = true; continue; }
		V_tag[v.id] = vI++;
	}
	meshO.type = meshI.type;
	meshO.V.resize(3, vI);
	for (int i = 0; i < V_tag.size();i++) {
		if (V_tag[i] == -1) continue;
		Hybrid_V vo;
		vo.id = meshO.Vs.size(); 
		vo.v = meshI.Vs[i].v;
		vo.boundary = false;

		meshO.Vs.push_back(vo);
		meshO.V.col(V_tag[i]) = meshI.V.col(i);
	}
	meshO.Hs = meshI.Hs;
	Hybrid h;
	if (meshO.type == Mesh_type::Hex)h.vs.resize(8);
	else if (meshO.type == Mesh_type::Tet)h.vs.resize(4);

	for (auto &h:meshO.Hs) {
		for (uint32_t j = 0; j < h.vs.size(); j++) h.vs[j] = V_tag[h.vs[j]];
		for (uint32_t i = 0; i < h.vs.size(); i++) meshO.Vs[h.vs[i]].neighbor_hs.push_back(h.id);
	}
	return redundentV;
}
double average_edge_length(const Mesh &mesh) {
	double ave_length = 0;
	int N = 0;
	double min_len = std::numeric_limits<double>::max(), max_len =std::numeric_limits<double>::min();
	for (uint32_t i = 0; i < mesh.Fs.size(); i++) {
		auto &vs = mesh.Fs[i].vs;
		for (uint32_t j = 0; j < vs.size(); j++) 
		{
			double len = (mesh.V.col(vs[j]) - mesh.V.col(vs[(j + 1) % vs.size()])).norm();
			ave_length += len;

			if(len > max_len)
				max_len = len;
			if(len < min_len)
				min_len = len;
		}
		N += vs.size();
	}
	if (N == 0) { cout << "No Faces" << endl; system("PAUSE"); }
	ave_length /= N;
	std::cout<<"max_len, min_len: "<<max_len<<" "<<min_len<<std::endl;

	return ave_length;
}
void edge_length(const Mesh &mesh, Mesh_Quality &mq)
{
	int N = mesh.Es.size();
	mq.ELens.resize(N);
	double min_len = std::numeric_limits<double>::max(), max_len =std::numeric_limits<double>::min();
	for (uint32_t i = 0; i < mesh.Es.size(); i++) {
		auto &vs = mesh.Es[i].vs;
		mq.ELens[i] = (mesh.V.col(vs[0]) - mesh.V.col(vs[1])).norm();
	}
	std::sort(mq.ELens.begin(), mq.ELens.end());
	
}
void compute_volume(Mesh &mesh, Mesh_Quality &mq)
{
	mq.H_mineLens.resize(mesh.Hs.size());
	mq.H_maxeLens.resize(mesh.Hs.size());

	for (auto &e : mesh.Es) 
	{
		for(auto &hid : e.neighbor_hs)
			mesh.Hs[hid].es.push_back(e.id);
	}

	Eigen::MatrixXd V = mesh.V.transpose();
	mq.Vols.resize(mesh.Hs.size());
	for (const auto &h : mesh.Hs) {
		vector<MatrixXd> vout;
		double vol = 0;
		hex2tet24(V, h.vs, vol);
		mq.Vols[h.id] = vol;

		double min_len = std::numeric_limits<double>::max(), max_len =std::numeric_limits<double>::min();		
		for (auto eid: h.es) {
			auto &vs = mesh.Es[eid].vs;
			double len = (mesh.V.col(vs[0]) - mesh.V.col(vs[1])).norm();
			if(min_len > len)
				min_len = len;
			if(max_len < len)
				max_len = len;	
		}
		mq.H_mineLens[h.id] = min_len;
		mq.H_maxeLens[h.id] = max_len;
	}
}
void area_volume(const Mesh &mesh, double &volume) {
	Float res = 0;
	Vector3d ori; ori.setZero();
	for (auto f : mesh.Fs) {
		auto &fvs = f.vs;
		Vector3d center; center.setZero(); for (auto vid : fvs) center += mesh.V.col(vid); center /= fvs.size();

		for (uint32_t j = 0; j < fvs.size(); j++) {
			Vector3d x = mesh.V.col(fvs[j]) - ori, y = mesh.V.col(fvs[(j + 1) % fvs.size()]) - ori, z = center - ori;
			res += -((x[0] * y[1] * z[2] + x[1] * y[2] * z[0] + x[2] * y[0] * z[1]) - (x[2] * y[1] * z[0] + x[1] * y[0] * z[2] + x[0] * y[2] * z[1]));
		}
	}
	volume =std::abs( res/6);
}
double global_boundary_ratio(Mesh &mesh, bool medial_surface) {
	int N = 0;
	for (const auto & v: mesh.Vs) {
		if(medial_surface &&v.on_medial_surface)
			N++;
		else if (!medial_surface && v.boundary)N++;
	}
	double ratio = 0;
	if(N>0)ratio = mesh.Vs.size() / double(N);
	cout << "Mesh Ratio: " << ratio << endl;
	return ratio;
}
void re_indexing_connectivity(Mesh &hmi, MatrixXi &H){
	hmi.Vs.clear();
	hmi.Es.clear();
	hmi.Fs.clear();
	hmi.Hs.clear();
	//Vs
	hmi.Vs.resize(hmi.V.cols());
	for (uint32_t i = 0; i<hmi.V.cols(); i++)
	{
		Hybrid_V v;
		v.id = i; v.boundary = false;
		hmi.Vs[i] = v;
	}
	//Hs
	hmi.Hs.resize(H.cols());
	Hybrid h; ;
	if (hmi.type == Mesh_type::Hex)h.vs.resize(8);
	else if (hmi.type == Mesh_type::Tet)h.vs.resize(4);

	for (uint32_t i = 0; i < H.cols(); i++) {
		for (uint32_t j = 0; j < H.rows(); j++) h.vs[j] = H(j, i);
		h.id = i; hmi.Hs[h.id] = h;
		for (uint32_t i = 0; i < h.vs.size(); i++) hmi.Vs[h.vs[i]].neighbor_hs.push_back(h.id);
	}
//Es, Fs, and their connectivities
	build_connectivity(hmi);
}
void re_indexing_connectivity(Mesh &hmi, vector<bool> &H_flag,Mesh &Ho, vector<int32_t> &V_map, vector<int32_t> &V_map_reverse, vector<int32_t> &H_map, vector<int32_t> &H_map_reverse) {
	V_map.clear();
	V_map_reverse.clear(); H_map.clear();
	H_map_reverse.clear();
	Ho = Mesh();
	Ho.type = Mesh_type::Hex;
	vector<bool> v_tag_(hmi.Vs.size(), false);

	for (uint32_t j = 0; j < H_flag.size(); j++) if (H_flag[j]) { for (auto hvid : hmi.Hs[j].vs) v_tag_[hvid] = true; }
	V_map.resize(v_tag_.size(), -1);
	for (uint32_t j = 0; j < v_tag_.size(); j++) {
		if (!v_tag_[j]) continue;
		Hybrid_V v;
		v.id = Ho.Vs.size(); V_map[j] = v.id; V_map_reverse.push_back(j);
		v.v = hmi.Vs[j].v;
		Ho.Vs.push_back(v);
	}
	Ho.V.resize(3, Ho.Vs.size());
	for (uint32_t j = 0; j < Ho.Vs.size(); j++) {
		Ho.V(0, j) = Ho.Vs[j].v[0];
		Ho.V(1, j) = Ho.Vs[j].v[1];
		Ho.V(2, j) = Ho.Vs[j].v[2];
	}
	for (uint32_t j = 0; j < H_flag.size(); j++)if (H_flag[j]) {
		Hybrid h;
		h.id = Ho.Hs.size();
		for (auto vid : hmi.Hs[j].vs) {
			h.vs.push_back(V_map[vid]);
			//Ho.Vs[V_map[vid]].neighbor_hs.push_back(h.id);
		}
		Ho.Hs.push_back(h);
		H_map_reverse.push_back(j);
	}
	build_connectivity(Ho);
}
void re_indexing_connectivity_sur(Mesh &mi, vector<bool> &f_flag, Mesh &mo, vector<int32_t> &V_map, vector<int32_t> &V_map_reverse, vector<int32_t> &f_map, vector<int32_t> &f_map_reverse) {
	V_map.clear();
	V_map_reverse.clear(); f_map.clear();
	f_map_reverse.clear();
	mo.Vs.clear(); mo.Es.clear(); mo.Fs.clear(); mo.Hs.clear(); mo.V.resize(0, 0);
	mo.type = mi.type;
	vector<bool> v_tag_(mi.Vs.size(), false);

	for (uint32_t j = 0; j < f_flag.size(); j++) if (f_flag[j]) { for (auto fvid : mi.Fs[j].vs) v_tag_[fvid] = true; }
	V_map.resize(v_tag_.size(), -1);
	for (uint32_t j = 0; j < v_tag_.size(); j++) {
		if (!v_tag_[j]) continue;
		Hybrid_V v;
		v.id = mo.Vs.size(); V_map[j] = v.id; V_map_reverse.push_back(j);
		v.v = mi.Vs[j].v;
		mo.Vs.push_back(v);
	}
	mo.V.resize(3, mo.Vs.size());
	for (uint32_t j = 0; j < mo.Vs.size(); j++) {
		mo.V(0, j) = mo.Vs[j].v[0];
		mo.V(1, j) = mo.Vs[j].v[1];
		mo.V(2, j) = mo.Vs[j].v[2];
	}
	for (uint32_t j = 0; j < f_flag.size(); j++)if (f_flag[j]) {
		Hybrid_F f;
		f.id = mo.Fs.size();
		for (auto vid : mi.Fs[j].vs) {
			f.vs.push_back(V_map[vid]);
			//mo.Vs[V_map[vid]].neighbor_hs.push_back(f.id);
		}
		mo.Fs.push_back(f);
		f_map_reverse.push_back(j);
	}
	build_connectivity(mo);

}

void decompose_components(Mesh &mesh, std::vector<Mesh> &components)
{
	if(mesh.type == Mesh_type::Tri || mesh.type == Mesh_type::HSur || mesh.type == Mesh_type::Qua)
	{
		std::vector<bool> f_tag(mesh.Fs.size(), false);
		std::vector<std::vector<int>> Groups;
		while (true) {
			std::vector<int> group;
			int id = -1;
			for (int i = 0; i < f_tag.size(); i++)if (!f_tag[i]) { id = i; break; }
			if (id == -1)break;			
			std::queue<int> q;
			q.push(id);
			while (!q.empty()) {
				auto fid = q.front();
				q.pop();

				if (!f_tag[fid]) {
					group.push_back(fid);
					f_tag[fid] = true;
				}
				else
					continue;
				for(auto &vid: mesh.Fs[fid].vs)
				{
					for(auto &vfid: mesh.Vs[vid].neighbor_fs)
						if(!f_tag[vfid]) q.push(vfid);
				}
			}
			Groups.push_back(group);
		}
		//direction
		//cout << "correct orientation " << endl;
		components.resize(Groups.size());
		if(components.size() == 1)
		{
			components[0] =mesh;
			return;
		}
		for (int i =0;i< Groups.size(); i++) {
			auto &group = Groups[i];
			std::fill(f_tag.begin(), f_tag.end(), false);
			for(auto &tag: group)
				f_tag[tag] = true;

			std::vector<int32_t> V_map, V_map_reverse, f_map, f_map_reverse;
			re_indexing_connectivity_sur(mesh, f_tag, components[i], V_map, V_map_reverse, f_map, f_map_reverse);
		}
	}
}
void combine_components(std::vector<Mesh> &components, Mesh &mesh)
{
	if(!components.size())
		return;
	if(components.size() == 1)
	{
		mesh = components[0];
		return;
	}
	if(mesh.type == Mesh_type::Hex || mesh.type == Mesh_type::Tet || mesh.type == Mesh_type::Hyb)
	{
		mesh.type = components[0].type;
		for(auto & c: components)
		{
			std::vector<int> v_map(c.Vs.size(), -1);
			for(auto &v: c.Vs)
			{
				Hybrid_V v_;
				v_.id = mesh.Vs.size();
				v_.v = v.v;
				mesh.Vs.push_back(v_);
				v_map[v.id] = v_.id;
			}
			for(auto &h: c.Hs)
			{
				Hybrid h_;
				h_.id = mesh.Hs.size();
				for(const auto &vid: h.vs)
					h_.vs.push_back(v_map[vid]);
				mesh.Hs.push_back(h_);
			}
		}
		mesh.V.resize(3, mesh.Vs.size());
		for (uint32_t j = 0; j < mesh.Vs.size(); j++) {
			mesh.V(0, j) = mesh.Vs[j].v[0];
			mesh.V(1, j) = mesh.Vs[j].v[1];
			mesh.V(2, j) = mesh.Vs[j].v[2];
		}
	}
}

void move_boundary_vertices_back(Mesh_Domain &md, std::vector<int> &hex2Octree_map)
{
	std::vector<int> mappedV(md.mesh_entire.Vs.size(), -1);
	std::vector<int> mappedV_reverse(md.mesh_entire.Vs.size(), -1);
	Mesh mi; mi.type = Mesh_type::Hex;

	std::vector<int> hex2Octree_map_;
	int nv =0;
	for (const auto &v: md.mesh_entire.Vs) {
		if (v.boundary) continue;		
		Hybrid_V hv;
		hv.id = nv;
		hv.boundary = v.boundary;
		hv.v = v.v;
		mi.Vs.push_back(hv);
		
		mappedV[v.id] = nv;
		mappedV_reverse[nv++] = v.id;
		hex2Octree_map_.push_back(hex2Octree_map[v.id]);
	}
	md.known_value_post = true; md.post_index = nv;
	for (const auto &v: md.mesh_entire.Vs) {
		if (!v.boundary) continue;		
		Hybrid_V hv;
		hv.id = nv;
		hv.boundary = v.boundary;
		hv.v = v.v;
		mi.Vs.push_back(hv);
		
		mappedV[v.id] = nv;
		mappedV_reverse[nv++] = v.id;
		hex2Octree_map_.push_back(hex2Octree_map[v.id]);
	}
	for (auto &h: md.mesh_entire.Hs) {
		for(auto &vid: h.vs)
			vid = mappedV[vid];
	}
	mi.V.resize(3, mi.Vs.size());
	for (uint32_t j = 0; j < mi.Vs.size(); j++) {
		mi.V(0, j) = mi.Vs[j].v[0];
		mi.V(1, j) = mi.Vs[j].v[1];
		mi.V(2, j) = mi.Vs[j].v[2];
	}

	hex2Octree_map.swap(hex2Octree_map_);

	//md.post_index = mi.Vs.size(); 
	md.mesh_entire.Vs = mi.Vs;
	md.mesh_entire.V = mi.V;
	build_connectivity(md.mesh_entire);
	reorder_hex_mesh_propogation(md.mesh_entire);
	re_indexing_connectivity(md.mesh_entire, md.H_flag, md.mesh_subA, md.V_map, md.V_map_reverse, md.H_map, md.H_map_reverse);
}
void refine_surface_mesh(const Mesh &meshi, Mesh &mesho, int iter) {

	mesho = meshi;

	for (int i = 0; i < iter; i++) {
		Mesh M_;
		M_.type = Mesh_type::HSur;

		vector<int> E2V(mesho.Es.size()), F2V(mesho.Fs.size());

		int vn = 0;
		for (const auto &v : mesho.Vs) {
			Hybrid_V v_;
			v_.id = vn++;
			v_.v.resize(3);
			for (int j = 0; j < 3; j++)v_.v[j] = mesho.V(j, v.id);
			M_.Vs.push_back(v_);
		}

		for (const auto &e : mesho.Es) {
			Hybrid_V v;
			v.id = vn++;
			v.v.resize(3);

			Vector3d center;
			center.setZero();
			for (auto vid : e.vs) center += mesho.V.col(vid);
			center /= e.vs.size();
			for (int j = 0; j < 3; j++)v.v[j] = center[j];

			M_.Vs.push_back(v);
			E2V[e.id] = v.id;
		}
		for (const auto &f : mesho.Fs) {
			Hybrid_V v;
			v.id = vn++;
			v.v.resize(3);

			Vector3d center;
			center.setZero();
			for (auto vid : f.vs) center += mesho.V.col(vid);
			center /= f.vs.size();
			for (int j = 0; j < 3; j++)v.v[j] = center[j];

			M_.Vs.push_back(v);
			F2V[f.id] = v.id;
		}
		//new elements
		uint32_t F_num = 0;

		Hybrid_F F; F.boundary = true;
		std::vector<uint32_t> vs(4);
		for (const auto & f : mesho.Fs) {
			auto &fvs = f.vs;
			auto &fes = f.es;
			int fvn = fvs.size();
			for (uint32_t j = 0; j < fvn; j++) {
				vs[0] = E2V[fes[(j - 1 + fvn) % fvn]];
				vs[1] = fvs[j];
				vs[2] = E2V[fes[j]];
				vs[3] = F2V[f.id];

				F.id = F_num; F_num++;
				F.vs = vs;
				M_.Fs.push_back(F);
			}
		}

		M_.V.resize(3, M_.Vs.size());
		for (const auto &v : M_.Vs)for (int j = 0; j < 3; j++)M_.V(j, v.id) = v.v[j];

		build_connectivity(M_);
		orient_surface_mesh(M_);
		build_connectivity(M_);
		mesho = M_;
	}
}
void triangulation(Mesh &meshi, Mesh &mesho) {
	mesho.Vs.clear(); mesho.Es.clear(); mesho.Fs.clear(); mesho.Hs.clear();
	mesho.type = Mesh_type::Tri;

	vector<bool> V_tag(meshi.Vs.size(), false);

	for (const auto &f : meshi.Fs) for (int i = 2; i < f.vs.size(); i++) {

		Hybrid_F hf;
		hf.vs.push_back(f.vs[0]);
		hf.vs.push_back(f.vs[(i-1)%f.vs.size()]);
		hf.vs.push_back(f.vs[i]);
		mesho.Fs.push_back(hf);
	}
	mesho.Vs = meshi.Vs;
	mesho.V = meshi.V;
	//orient direction
	build_connectivity(mesho);
	orient_surface_mesh(mesho);
}
void extract_surface_mesh(Mesh &meshi, Mesh &mesho) {
	mesho.Vs.clear(); mesho.Es.clear(); mesho.Fs.clear(); mesho.Hs.clear();
	mesho.type = Mesh_type::Tri;

	vector<bool> V_tag(meshi.Vs.size(), false);
	vector<int32_t> V_map(meshi.Vs.size(), -1), V_map_reverse;

	for (auto f : meshi.Fs) if (f.boundary) {
		for (auto vid : f.vs) V_tag[vid] = true;
		
		if (f.vs.size() == 3) {
			Hybrid_F hf; hf.vs = f.vs;
			mesho.Fs.push_back(hf);
		}
		else if (f.vs.size() == 4) {
			Hybrid_F hf;
			hf.vs.push_back(f.vs[0]);
			hf.vs.push_back(f.vs[1]);
			hf.vs.push_back(f.vs[2]);
			mesho.Fs.push_back(hf);
			hf.vs.clear();
			hf.vs.push_back(f.vs[2]);
			hf.vs.push_back(f.vs[3]);
			hf.vs.push_back(f.vs[0]);
			mesho.Fs.push_back(hf);
		}

	}
	//re-indexing
	uint32_t newV_id = 0;
	for (uint32_t i = 0; i < V_tag.size(); i++) if (V_tag[i]) {
		V_map[i] = newV_id++; V_map_reverse.push_back(i);
	}
	mesho.V.resize(3, newV_id);
	for (uint32_t i = 0; i < V_tag.size(); i++) if (V_tag[i]) {
		Hybrid_V v;
		v.id = mesho.Vs.size(); mesho.Vs.push_back(v);
		mesho.V.col(v.id) = meshi.V.col(i);
	}
	for (uint32_t i = 0; i < mesho.Fs.size(); i++) for (uint32_t j = 0; j < 3; j++) mesho.Fs[i].vs[j] = V_map[mesho.Fs[i].vs[j]];
	//orient direction
	build_connectivity(mesho);
	orient_surface_mesh(mesho);
}
void extract_surface_conforming_mesh(Mesh &meshi, Mesh &mesho, vector<int32_t> &V_map, vector<int32_t> &V_map_reverse, vector<int32_t> &F_map, vector<int32_t> &F_map_reverse) {
	mesho.Vs.clear(); mesho.Es.clear(); mesho.Fs.clear(); mesho.Hs.clear();
	
	vector<bool> V_tag(meshi.Vs.size(), false);
	F_map.resize(meshi.Fs.size()); fill(F_map.begin(), F_map.end(), -1);
	V_map.resize(meshi.Vs.size()); fill(V_map.begin(), V_map.end(), -1);
	V_map_reverse.clear(); F_map_reverse.clear();
	for (auto f : meshi.Fs) if (f.boundary) {
		for (auto vid : f.vs) V_tag[vid] = true;

		if (mesho.type == Mesh_type::Qua) {
			Hybrid_F hf; hf.vs = f.vs;
			hf.id = mesho.Fs.size();
			mesho.Fs.push_back(hf);
			F_map[f.id] = hf.id;
			F_map_reverse.push_back(f.id);
		}
		else if (mesho.type == Mesh_type::Tri) {
			Hybrid_F hf;
			hf.id = mesho.Fs.size();
			hf.vs.push_back(f.vs[0]);
			hf.vs.push_back(f.vs[1]);
			hf.vs.push_back(f.vs[2]);
			mesho.Fs.push_back(hf);
			F_map[f.id] = hf.id;
			F_map_reverse.push_back(f.id);
			hf.id = mesho.Fs.size();
			hf.vs.clear();
			hf.vs.push_back(f.vs[2]);
			hf.vs.push_back(f.vs[3]);
			hf.vs.push_back(f.vs[0]);
			mesho.Fs.push_back(hf);
			F_map_reverse.push_back(f.id);
		}
	}
	//re-indexing
	uint32_t newV_id = 0;
	for (uint32_t i = 0; i < V_tag.size(); i++) if (V_tag[i]) {
		V_map[i] = newV_id++; V_map_reverse.push_back(i);
	}
	mesho.V.resize(3, newV_id);
	for (uint32_t i = 0; i < V_tag.size(); i++) if (V_tag[i]) {
		Hybrid_V v;
		v.id = mesho.Vs.size(); mesho.Vs.push_back(v);
		mesho.V.col(v.id) = meshi.V.col(i);
	}
	for (auto &f: mesho.Fs) for (auto &vid:f.vs) vid = V_map[vid];
	//orient direction
	build_connectivity(mesho);
	orient_surface_mesh(mesho);
	build_connectivity(mesho);
}
void  orient_surface_mesh(Mesh &hmi) {

	vector<bool> flag(hmi.Fs.size(), true);
	flag[0] = false;

	std::queue<uint32_t> pf_temp; pf_temp.push(0);
	while (!pf_temp.empty()) {
		uint32_t fid = pf_temp.front(); pf_temp.pop();
		for (auto eid : hmi.Fs[fid].es) for (auto nfid : hmi.Es[eid].neighbor_fs) {
			if (!flag[nfid]) continue;
			uint32_t v0 = hmi.Es[eid].vs[0], v1 = hmi.Es[eid].vs[1];
			int32_t v0_pos = std::find(hmi.Fs[fid].vs.begin(), hmi.Fs[fid].vs.end(), v0) - hmi.Fs[fid].vs.begin();
			int32_t v1_pos = std::find(hmi.Fs[fid].vs.begin(), hmi.Fs[fid].vs.end(), v1) - hmi.Fs[fid].vs.begin();

			if ((v0_pos + 1) % hmi.Fs[fid].vs.size() != v1_pos) swap(v0, v1);

			int32_t v0_pos_ = std::find(hmi.Fs[nfid].vs.begin(), hmi.Fs[nfid].vs.end(), v0) - hmi.Fs[nfid].vs.begin();
			int32_t v1_pos_ = std::find(hmi.Fs[nfid].vs.begin(), hmi.Fs[nfid].vs.end(), v1) - hmi.Fs[nfid].vs.begin();

			if ((v0_pos_ + 1) % hmi.Fs[nfid].vs.size() == v1_pos_) std::reverse(hmi.Fs[nfid].vs.begin(), hmi.Fs[nfid].vs.end());

			pf_temp.push(nfid); flag[nfid] = false;
		}
	}
	Float res = 0;
	Vector3d ori; ori.setZero();
	for (auto f : hmi.Fs) {
		auto &fvs = f.vs;
		Vector3d center; center.setZero(); for (auto vid : fvs) center += hmi.V.col(vid); center /= fvs.size();

		for (uint32_t j = 0; j < fvs.size(); j++) {
			Vector3d x = hmi.V.col(fvs[j]) - ori, y = hmi.V.col(fvs[(j + 1) % fvs.size()]) - ori, z = center - ori;
			res += -((x[0] * y[1] * z[2] + x[1] * y[2] * z[0] + x[2] * y[0] * z[1]) - (x[2] * y[1] * z[0] + x[1] * y[0] * z[2] + x[0] * y[2] * z[1]));
		}
	}
	if (res > 0) {
		for (uint32_t i = 0; i < hmi.Fs.size(); i++) std::reverse(hmi.Fs[i].vs.begin(), hmi.Fs[i].vs.end());
	}
}
void  orient_triangle_mesh_acw(Mesh &hmi) {

	Eigen::MatrixXd V = hmi.V.transpose();
	Eigen::MatrixXi FF, F(hmi.Fs.size(), 3);
	for (uint32_t i = 0; i < F.rows(); i++)
		for (uint32_t j = 0; j < 3; j++)F(i, j) = hmi.Fs[i].vs[j];
	Eigen::VectorXi I, C;
	igl::embree::reorient_facets_raycast(V, F, F.rows() * 100, 10, 0, false, false, I, C);
	// apply reorientation
	FF.conservativeResize(F.rows(), F.cols());
	for (int i = 0; i < I.rows(); i++) {
		if (I(i)) FF.row(i) = (F.row(i).reverse()).eval();
		else FF.row(i) = F.row(i);
	}
	for (uint32_t i = 0; i < FF.rows(); i++)
		for (uint32_t j = 0; j < 3; j++)hmi.Fs[i].vs[j] = FF(i, j);
}
void orient_triangle_mesh(Mesh &hmi) {
	vector<vector<Float>> p(hmi.V.cols()); vector<vector<uint32_t>> f(hmi.Fs.size());
	for (uint32_t i = 0; i<hmi.V.cols(); i++){
		vector<Float> pv;
		pv.push_back(hmi.V(0,i));
		pv.push_back(hmi.V(1,i));
		pv.push_back(hmi.V(2,i));
		p[i] = pv;
	}
	for (uint32_t i = 0; i < hmi.Fs.size(); i++) f[i] = hmi.Fs[i].vs;

	std::map<std::set<uint32_t>, vector<uint32_t>> edge_2_neb_tri;
	std::set<vector<uint32_t>> direct_edges;
	std::set<std::set<uint32_t>> no_direct_edges;
	vector<bool> whe_tri_in;

	vector<vector<uint32_t>> nf;

	nf = f;

	for (int i = 0; i<f.size(); i++)
	{
		std::set<uint32_t> xa, xb, xc;

		xa.insert(f[i][0]);
		xa.insert(f[i][1]);

		xb.insert(f[i][1]);
		xb.insert(f[i][2]);

		xc.insert(f[i][2]);
		xc.insert(f[i][0]);

		whe_tri_in.push_back(false);

		vector<uint32_t> ct;

		ct.push_back(i);

		if (edge_2_neb_tri.find(xa) == edge_2_neb_tri.end())
		{
			edge_2_neb_tri.insert(std::pair<std::set<uint32_t>, vector<uint32_t>>(xa, ct));
		}
		else
		{
			edge_2_neb_tri[xa].push_back(i);
		}

		if (edge_2_neb_tri.find(xb) == edge_2_neb_tri.end())
		{
			edge_2_neb_tri.insert(std::pair<std::set<uint32_t>, vector<uint32_t>>(xb, ct));
		}
		else
		{
			edge_2_neb_tri[xb].push_back(i);
		}

		if (edge_2_neb_tri.find(xc) == edge_2_neb_tri.end())
		{
			edge_2_neb_tri.insert(std::pair<std::set<uint32_t>, vector<uint32_t>>(xc, ct));
		}
		else
		{
			edge_2_neb_tri[xc].push_back(i);
		}
	}

	std::set<uint32_t> xa, xb, xc;
	vector<uint32_t> ya, yb, yc;

	xa.insert(f[0][0]);
	xa.insert(f[0][1]);

	xb.insert(f[0][1]);
	xb.insert(f[0][2]);

	xc.insert(f[0][2]);
	xc.insert(f[0][0]);

	ya.push_back(f[0][0]);
	ya.push_back(f[0][1]);

	yb.push_back(f[0][1]);
	yb.push_back(f[0][2]);

	yc.push_back(f[0][2]);
	yc.push_back(f[0][0]);

	no_direct_edges.insert(xa);
	no_direct_edges.insert(xb);
	no_direct_edges.insert(xc);

	direct_edges.insert(ya);
	direct_edges.insert(yb);
	direct_edges.insert(yc);

	whe_tri_in[0] = true;

	std::queue<uint32_t> queue_loop;

	for (uint32_t i = 0; i<2; i++)
	{
		if (!whe_tri_in[edge_2_neb_tri[xa][i]])
		{
			queue_loop.push(edge_2_neb_tri[xa][i]);

			whe_tri_in[edge_2_neb_tri[xa][i]] = true;
		}

		if (!whe_tri_in[edge_2_neb_tri[xb][i]])
		{
			queue_loop.push(edge_2_neb_tri[xb][i]);

			whe_tri_in[edge_2_neb_tri[xb][i]] = true;
		}

		if (!whe_tri_in[edge_2_neb_tri[xc][i]])
		{
			queue_loop.push(edge_2_neb_tri[xc][i]);

			whe_tri_in[edge_2_neb_tri[xc][i]] = true;
		}
	}

	while (!queue_loop.empty())
	{
		xa.clear();
		xb.clear();
		xc.clear();

		ya.clear();
		yb.clear();
		yc.clear();

		uint32_t c;

		c = queue_loop.front();

		xa.insert(f[c][0]);
		xa.insert(f[c][1]);

		xb.insert(f[c][1]);
		xb.insert(f[c][2]);

		xc.insert(f[c][2]);
		xc.insert(f[c][0]);

		ya.push_back(f[c][0]);
		ya.push_back(f[c][1]);

		yb.push_back(f[c][1]);
		yb.push_back(f[c][2]);

		yc.push_back(f[c][2]);
		yc.push_back(f[c][0]);

		uint32_t cnt, ct;

		cnt = 0;
		ct = 0;

		if (no_direct_edges.find(xa) != no_direct_edges.end())
		{
			cnt++;
		}

		if (no_direct_edges.find(xb) != no_direct_edges.end())
		{
			cnt++;
		}

		if (no_direct_edges.find(xc) != no_direct_edges.end())
		{
			cnt++;
		}

		if (direct_edges.find(ya) != direct_edges.end())
		{
			ct++;
		}

		if (direct_edges.find(yb) != direct_edges.end())
		{
			ct++;
		}

		if (direct_edges.find(yc) != direct_edges.end())
		{
			ct++;
		}

		if (cnt == 0)
		{
			std::cout << "Error in triangle direction solving!" << std::endl;
			exit(0);
		}

		if (ct != 0)
		{
			ya.clear();
			yb.clear();
			yc.clear();

			ya.push_back(f[c][1]);
			ya.push_back(f[c][0]);

			yb.push_back(f[c][2]);
			yb.push_back(f[c][1]);

			yc.push_back(f[c][0]);
			yc.push_back(f[c][2]);

			nf[c][0] = f[c][1];
			nf[c][1] = f[c][0];

		}
		else
		{
			// do nothing
		}

		no_direct_edges.insert(xa);
		no_direct_edges.insert(xb);
		no_direct_edges.insert(xc);

		direct_edges.insert(ya);
		direct_edges.insert(yb);
		direct_edges.insert(yc);

		for (uint32_t i = 0; i<2; i++)
		{
			if (!whe_tri_in[edge_2_neb_tri[xa][i]])
			{
				queue_loop.push(edge_2_neb_tri[xa][i]);

				whe_tri_in[edge_2_neb_tri[xa][i]] = true;
			}

			if (!whe_tri_in[edge_2_neb_tri[xb][i]])
			{
				queue_loop.push(edge_2_neb_tri[xb][i]);

				whe_tri_in[edge_2_neb_tri[xb][i]] = true;
			}

			if (!whe_tri_in[edge_2_neb_tri[xc][i]])
			{
				queue_loop.push(edge_2_neb_tri[xc][i]);

				whe_tri_in[edge_2_neb_tri[xc][i]] = true;
			}
		}

		queue_loop.pop();
	}

	Float res = 0;

	vector<Float> ori(3, 0);

	for (uint32_t i = 0; i<nf.size(); i++)
		res += uctet(ori, p[nf[i][0]], p[nf[i][1]], p[nf[i][2]]);

	if (res > 0)
	{
		uint32_t tmi;

		for (uint32_t i = 0; i<nf.size(); i++)
		{
			tmi = nf[i][0];
			nf[i][0] = nf[i][1];
			nf[i][1] = tmi;
		}
	}

	for (uint32_t i = 0; i < hmi.Fs.size(); i++)
		hmi.Fs[i].vs = nf[i];
}
void  orient_triangle_mesh(MatrixXd &Tri_V, MatrixXi &Tri_F) {

	vector<vector<Float>> p(Tri_V.cols()); vector<vector<uint32_t>> f(Tri_F.rows());
	for (uint32_t i = 0; i<Tri_V.cols(); i++) {
		vector<Float> pv;
		pv.push_back(Tri_V(0,i));
		pv.push_back(Tri_V(1,i));
		pv.push_back(Tri_V(2,i));
		p[i] = pv;
	}
	for (uint32_t i = 0; i < Tri_F.rows(); i++)
		for (uint32_t j = 0; j < 3; j++)
			f[i].push_back(Tri_F(i, j));

	std::map<std::set<uint32_t>, vector<uint32_t>> edge_2_neb_tri;
	std::set<vector<uint32_t>> direct_edges;
	std::set<std::set<uint32_t>> no_direct_edges;
	vector<bool> whe_tri_in;

	vector<vector<uint32_t>> nf;

	nf = f;

	for (int i = 0; i<f.size(); i++)
	{
		std::set<uint32_t> xa, xb, xc;

		xa.insert(f[i][0]);
		xa.insert(f[i][1]);

		xb.insert(f[i][1]);
		xb.insert(f[i][2]);

		xc.insert(f[i][2]);
		xc.insert(f[i][0]);

		whe_tri_in.push_back(false);

		vector<uint32_t> ct;

		ct.push_back(i);

		if (edge_2_neb_tri.find(xa) == edge_2_neb_tri.end())
		{
			edge_2_neb_tri.insert(std::pair<std::set<uint32_t>, vector<uint32_t>>(xa, ct));
		}
		else
		{
			edge_2_neb_tri[xa].push_back(i);
		}

		if (edge_2_neb_tri.find(xb) == edge_2_neb_tri.end())
		{
			edge_2_neb_tri.insert(std::pair<std::set<uint32_t>, vector<uint32_t>>(xb, ct));
		}
		else
		{
			edge_2_neb_tri[xb].push_back(i);
		}

		if (edge_2_neb_tri.find(xc) == edge_2_neb_tri.end())
		{
			edge_2_neb_tri.insert(std::pair<std::set<uint32_t>, vector<uint32_t>>(xc, ct));
		}
		else
		{
			edge_2_neb_tri[xc].push_back(i);
		}
	}

	std::set<uint32_t> xa, xb, xc;
	vector<uint32_t> ya, yb, yc;

	xa.insert(f[0][0]);
	xa.insert(f[0][1]);

	xb.insert(f[0][1]);
	xb.insert(f[0][2]);

	xc.insert(f[0][2]);
	xc.insert(f[0][0]);

	ya.push_back(f[0][0]);
	ya.push_back(f[0][1]);

	yb.push_back(f[0][1]);
	yb.push_back(f[0][2]);

	yc.push_back(f[0][2]);
	yc.push_back(f[0][0]);

	no_direct_edges.insert(xa);
	no_direct_edges.insert(xb);
	no_direct_edges.insert(xc);

	direct_edges.insert(ya);
	direct_edges.insert(yb);
	direct_edges.insert(yc);

	whe_tri_in[0] = true;

	std::queue<uint32_t> queue_loop;

	for (uint32_t i = 0; i<2; i++)
	{
		if (!whe_tri_in[edge_2_neb_tri[xa][i]])
		{
			queue_loop.push(edge_2_neb_tri[xa][i]);

			whe_tri_in[edge_2_neb_tri[xa][i]] = true;
		}

		if (!whe_tri_in[edge_2_neb_tri[xb][i]])
		{
			queue_loop.push(edge_2_neb_tri[xb][i]);

			whe_tri_in[edge_2_neb_tri[xb][i]] = true;
		}

		if (!whe_tri_in[edge_2_neb_tri[xc][i]])
		{
			queue_loop.push(edge_2_neb_tri[xc][i]);

			whe_tri_in[edge_2_neb_tri[xc][i]] = true;
		}
	}

	while (!queue_loop.empty())
	{
		xa.clear();
		xb.clear();
		xc.clear();

		ya.clear();
		yb.clear();
		yc.clear();

		uint32_t c;

		c = queue_loop.front();

		xa.insert(f[c][0]);
		xa.insert(f[c][1]);

		xb.insert(f[c][1]);
		xb.insert(f[c][2]);

		xc.insert(f[c][2]);
		xc.insert(f[c][0]);

		ya.push_back(f[c][0]);
		ya.push_back(f[c][1]);

		yb.push_back(f[c][1]);
		yb.push_back(f[c][2]);

		yc.push_back(f[c][2]);
		yc.push_back(f[c][0]);

		uint32_t cnt, ct;

		cnt = 0;
		ct = 0;

		if (no_direct_edges.find(xa) != no_direct_edges.end())
		{
			cnt++;
		}

		if (no_direct_edges.find(xb) != no_direct_edges.end())
		{
			cnt++;
		}

		if (no_direct_edges.find(xc) != no_direct_edges.end())
		{
			cnt++;
		}

		if (direct_edges.find(ya) != direct_edges.end())
		{
			ct++;
		}

		if (direct_edges.find(yb) != direct_edges.end())
		{
			ct++;
		}

		if (direct_edges.find(yc) != direct_edges.end())
		{
			ct++;
		}

		if (cnt == 0)
		{
			std::cout << "Error in triangle direction solving!" << std::endl;
			exit(0);
		}

		if (ct != 0)
		{
			ya.clear();
			yb.clear();
			yc.clear();

			ya.push_back(f[c][1]);
			ya.push_back(f[c][0]);

			yb.push_back(f[c][2]);
			yb.push_back(f[c][1]);

			yc.push_back(f[c][0]);
			yc.push_back(f[c][2]);

			nf[c][0] = f[c][1];
			nf[c][1] = f[c][0];

		}
		else
		{
			// do nothing
		}

		no_direct_edges.insert(xa);
		no_direct_edges.insert(xb);
		no_direct_edges.insert(xc);

		direct_edges.insert(ya);
		direct_edges.insert(yb);
		direct_edges.insert(yc);

		for (uint32_t i = 0; i<2; i++)
		{
			if (!whe_tri_in[edge_2_neb_tri[xa][i]])
			{
				queue_loop.push(edge_2_neb_tri[xa][i]);

				whe_tri_in[edge_2_neb_tri[xa][i]] = true;
			}

			if (!whe_tri_in[edge_2_neb_tri[xb][i]])
			{
				queue_loop.push(edge_2_neb_tri[xb][i]);

				whe_tri_in[edge_2_neb_tri[xb][i]] = true;
			}

			if (!whe_tri_in[edge_2_neb_tri[xc][i]])
			{
				queue_loop.push(edge_2_neb_tri[xc][i]);

				whe_tri_in[edge_2_neb_tri[xc][i]] = true;
			}
		}

		queue_loop.pop();
	}

	Float res = 0;

	vector<Float> ori(3, 0);

	for (uint32_t i = 0; i<nf.size(); i++)
		res += uctet(ori, p[nf[i][0]], p[nf[i][1]], p[nf[i][2]]);

	if (res > 0)
	{
		uint32_t tmi;

		for (uint32_t i = 0; i<nf.size(); i++)
		{
			tmi = nf[i][0];
			nf[i][0] = nf[i][1];
			nf[i][1] = tmi;
		}
	}

	for (uint32_t i = 0; i < Tri_F.rows(); i++)
		for (uint32_t j = 0; j < 3; j++)
			Tri_F(i, j) = nf[i][j];
}
Float uctet(vector<Float> a, vector<Float> b, vector<Float> c, vector<Float> d) {
	Float res = 0;
	vector<Float> x, y, z;

	for (uint32_t i = 0; i<3; i++){
		x.push_back(b[i] - a[i]);
		y.push_back(c[i] - a[i]);
		z.push_back(d[i] - a[i]);
	}
	res = -((x[0] * y[1] * z[2] + x[1] * y[2] * z[0] + x[2] * y[0] * z[1]) - (x[2] * y[1] * z[0] + x[1] * y[0] * z[2] + x[0] * y[2] * z[1]));

	return res;
}

void extract_boundary_from_surface_mesh(const Mesh &meshi, Mesh &mesho) {
	if (mesho.type == Mesh_type::line) {

		int vnum = 0;
		for (const auto &v_ : meshi.Vs) if (v_.boundary)vnum++;
		vector<int> V2v(meshi.Vs.size(), -1);
		mesho.V.resize(3, vnum);
		vnum = 0;
		for (const auto &v_:meshi.Vs) if(v_.boundary){
			Hybrid_V v; v.v.resize(3);
			v.v[0] = v_.v[0];
			v.v[1] = v_.v[1];
			v.v[2] = 0; // v_.v[2];
			v.id = vnum++;
			v.boundary = v_.boundary;
			mesho.Vs.push_back(v);

			mesho.V(0, v.id) = v.v[0];
			mesho.V(1, v.id) = v.v[1];
			mesho.V(2, v.id) = v.v[2];

			V2v[v_.id] = v.id;
		}
		for (const auto &e_:meshi.Es) if(e_.boundary){

			Hybrid_E e;
			e.vs.push_back(V2v[e_.vs[0]]);
			e.vs.push_back(V2v[e_.vs[1]]);
			e.id = mesho.Es.size();
			e.boundary = true;
			mesho.Es.push_back(e);
		}
	}

}

int face_from_vs(Mesh &m, vector<vector<uint32_t>> &fss)
{
	vector<unordered_set<uint32_t>> ofss(fss.size());
	if (fss.size() < 1)return -1;

	int i = 0;
	for (auto &fs : fss)
	{
		for (auto &fid : fs)
			ofss[i].insert(fid);
		i++;
	}

	for (auto &fid : ofss[0])
	{
		int found = true;
		i = 1;
		for (; i < ofss.size(); i++)
		{
			if (!ofss[i].count(fid)) {
				found = false;
				break;
			}
		}
		if (found)
			return fid;
	}

	return -1;
}

void loops_from_es(const std::vector<Hybrid_E> &es, std::vector<std::vector<int>> &loops) {

	int max_v = 0;
	for (const auto &e : es) {
		if (e.vs[0] > max_v) max_v = e.vs[0];
		if (e.vs[1] > max_v) max_v = e.vs[1];
	}

	std::vector<bool> v_flag(max_v + 1, false), e_flag(es.size(), false);
	while (true) {
		std::vector<int> loop;
		int eid = -1;
		for (int i = 0; i < es.size();i++)if (!e_flag[i]) {
			eid = i; break;
		}
		if (eid == -1)break;

		loop.push_back(es[eid].vs[0]);
		loop.push_back(es[eid].vs[1]);
		v_flag[loop[0]] = true;
		v_flag[loop[1]] = true;

		e_flag[eid] = true;
		uint32_t start_v = loop[1];
		while(true){
			bool found = false;
			for (uint32_t j = 0; j < es.size(); j++) {
				if (!e_flag[j]) {
					if (es[j].vs[0] == start_v) {
						e_flag[j] = true;
						if(!v_flag[es[j].vs[1]])
							loop.push_back(es[j].vs[1]);
						start_v = es[j].vs[1];
						found = true;
						break;
					}
					else if (es[j].vs[1] == start_v) {
						e_flag[j] = true;
						if (!v_flag[es[j].vs[0]])
							loop.push_back(es[j].vs[0]);
						start_v = es[j].vs[0];
						found = true;
						break;
					}
				}
			}
			if (!found)break;
		}
		loops.push_back(loop);
	}
}
void direction_correction(std::vector<std::vector<int>> &loops, Mesh &m) {
	//interior, exterior judgement is missing

	for (int i = 0; i < loops.size(); i++) {
		double area = 0;
		for (int j = 0; j < loops[i].size(); j++) {
			int v_0 = loops[i][(j - 1 + loops[i].size()) % loops[i].size()], v0 = loops[i][j], v1 = loops[i][(j + 1)%loops[i].size()];
			Eigen::Vector3d e0 = m.V.col(v0) - m.V.col(v_0);
			Eigen::Vector3d e1 = m.V.col(v1) - m.V.col(v0);
			area += (e1[0] - e0[0])*(e1[1] + e0[1]);
		}
		if (area < 0)
			std::reverse(loops[i].begin(), loops[i].end());
	}
}


void face_soup_info(Mesh &hmi, vector<uint32_t> &fs_soup, vector<bool> &E_tag, vector<uint32_t> &bvs) {
	vector<uint32_t> bes;
	int f = -1;
	for (auto fid : fs_soup)
		for (auto eid : hmi.Fs[fid].es)
			if (E_tag[eid])E_tag[eid] = false;else E_tag[eid] = true;
	for (auto fid : fs_soup)
		for (auto eid : hmi.Fs[fid].es)
			if (E_tag[eid]) {
				if(f==-1) f = fid;
				E_tag[eid] = false; bes.push_back(eid);
			}
	//ordering bes
	if (!bes.size()) { cout << "BUG: no boundary es!" << endl; system("PAUSE"); }
	bvs.clear();
	bvs.reserve(bes.size());
	vector<bool> e_flag(bes.size(), false);
	bvs.push_back(hmi.Es[bes[0]].vs[0]);
	bvs.push_back(hmi.Es[bes[0]].vs[1]);
	e_flag[0] = true;
	uint32_t start_v = bvs[1];
	for (uint32_t i = 2; i < bes.size(); i++) {
		for (uint32_t j = 1; j < bes.size(); j++) {
			if (!e_flag[j]) {
				if (hmi.Es[bes[j]].vs[0] == start_v) {
					e_flag[j] = true;
					bvs.push_back(hmi.Es[bes[j]].vs[1]);
					start_v = hmi.Es[bes[j]].vs[1];
					break;
				}
				else if (hmi.Es[bes[j]].vs[1] == start_v) {
					e_flag[j] = true;
					bvs.push_back(hmi.Es[bes[j]].vs[0]);
					start_v = hmi.Es[bes[j]].vs[0];
					break;
				}
			}
		}
	}
	if (bvs.size() != bes.size()) { cout << "BUG: vs != es!" << endl; }//system("PAUSE");}
	//judge direction
	bool correct = true;
	for (int i = 0; i < hmi.Fs[f].vs.size(); i++)
		if (hmi.Fs[f].vs[i] == bvs[0])
			if (hmi.Fs[f].vs[(i+1)% hmi.Fs[f].vs.size()] != bvs[1])
				correct = false;
	if (!correct) std::reverse(bvs.begin(), bvs.end());
}
void face_soup_info(Mesh &hmi, vector<uint32_t> &fs_soup, vector<bool> &E_tag, vector<uint32_t> &bes, vector<uint32_t> &bvs) {
	bes.clear();
	int f = -1;
	for (auto fid : fs_soup)
		for (auto eid : hmi.Fs[fid].es)
			if (E_tag[eid])E_tag[eid] = false; else E_tag[eid] = true;
	for (auto fid : fs_soup)
		for (auto eid : hmi.Fs[fid].es)
			if (E_tag[eid]) {
				if (f == -1) f = fid;
				E_tag[eid] = false; bes.push_back(eid);
			}
	//ordering bes
	if (!bes.size()) { cout << "BUG: no boundary es!" << endl; system("PAUSE"); }
	bvs.clear();
	bvs.reserve(bes.size());
	vector<bool> e_flag(bes.size(), false);
	bvs.push_back(hmi.Es[bes[0]].vs[0]);
	bvs.push_back(hmi.Es[bes[0]].vs[1]);
	e_flag[0] = true;
	uint32_t start_v = bvs[1];
	for (uint32_t i = 2; i < bes.size(); i++) {
		for (uint32_t j = 1; j < bes.size(); j++) {
			if (!e_flag[j]) {
				if (hmi.Es[bes[j]].vs[0] == start_v) {
					e_flag[j] = true;
					bvs.push_back(hmi.Es[bes[j]].vs[1]);
					start_v = hmi.Es[bes[j]].vs[1];
					break;
				}
				else if (hmi.Es[bes[j]].vs[1] == start_v) {
					e_flag[j] = true;
					bvs.push_back(hmi.Es[bes[j]].vs[0]);
					start_v = hmi.Es[bes[j]].vs[0];
					break;
				}
			}
		}
	}
	if (bvs.size() != bes.size()) { cout << "BUG: vs != es!" << endl; }// system("PAUSE");}
	//judge direction
	bool correct = true;
	for (int i = 0; i < hmi.Fs[f].vs.size(); i++)
		if (hmi.Fs[f].vs[i] == bvs[0])
			if (hmi.Fs[f].vs[(i + 1) % hmi.Fs[f].vs.size()] != bvs[1])
				correct = false;
	if (!correct) std::reverse(bvs.begin(), bvs.end());
}
void face_soup_roll_region(Mesh &hmi, uint32_t sfid, vector<bool> &E_tag, vector<uint32_t> &bes, vector<uint32_t> &fs, vector<uint32_t> &bvs) {
	vector<uint32_t> bes_ = bes;
	for (auto eid : bes)E_tag[eid] = true;
	bes.clear(); bvs.clear();
	int  svid = -1, cvid = -1, pre_eid = -1;
	for (auto eid : hmi.Fs[sfid].es) {
		if (E_tag[eid]) { svid = hmi.Es[eid].vs[0]; bvs.push_back(hmi.Es[eid].vs[0]); cvid = hmi.Es[eid].vs[1]; bes.push_back(eid); pre_eid = eid; break; }
	}
	fs.clear();
	fs.push_back(sfid);
	while (cvid != svid) {
		bvs.push_back(cvid);
		int eid = -1; 
		do {
			vector<uint32_t> &es = hmi.Fs[sfid].es;
			for (auto neid : hmi.Vs[cvid].neighbor_es)if (neid != pre_eid && find(es.begin(), es.end(), neid) != es.end()) {
				eid = neid; break;
			}
			pre_eid = eid;
			if (E_tag[eid])break;

			for (auto nfid : hmi.Es[eid].neighbor_fs)if (hmi.Fs[nfid].boundary && nfid != sfid) {sfid = nfid; break;}
			fs.push_back(sfid);
		} while (!E_tag[eid]);
		bes.push_back(eid);
		if (hmi.Es[eid].vs[0] == cvid)cvid = hmi.Es[eid].vs[1];else cvid = hmi.Es[eid].vs[0];
	}
	for (auto eid : bes_)E_tag[eid] = false;

}
void face_soup_expand_region(Mesh &hmi, vector<uint32_t> &fs_soup, vector<bool> &E_tag, vector<bool> &F_tag, vector<uint32_t> &bes) {
	for (auto eid : bes)E_tag[eid] = true;
	for (auto fid : fs_soup)F_tag[fid] = true;
	vector<uint32_t> fs_soup_ = fs_soup;
	while (true) {
		vector<uint32_t> fs_temp;
		for (auto fid : fs_soup_) {
			for (auto eid : hmi.Fs[fid].es)if (!E_tag[eid]) {
				for (auto nfid : hmi.Es[eid].neighbor_fs)if (hmi.Fs[nfid].boundary && !F_tag[nfid]) {
					fs_temp.push_back(nfid);
					F_tag[nfid] = true;
				}
			}
		}
		fs_soup_.clear();
		if (fs_temp.size()) {
			fs_soup.insert(fs_soup.end(), fs_temp.begin(), fs_temp.end());
			fs_soup_ = fs_temp;
			fs_temp.clear();
		}
		else break;
	}
	for (auto eid : bes)E_tag[eid] = false;
	for (auto fid : fs_soup)F_tag[fid] = false;
}
void DijkstraComputePaths(vertex_t source, const adjacency_list_t &adjacency_list, std::vector<weight_t> &min_distance, std::vector<vertex_t> &previous){
	
	int n = adjacency_list.size();
	min_distance.clear();
	min_distance.resize(n, max_weight);
	min_distance[source] = 0;
	previous.clear();
	previous.resize(n, -1);
	// we use greater instead of less to turn max-heap into min-heap
	std::priority_queue<weight_vertex_pair_t,
		std::vector<weight_vertex_pair_t>,
		std::greater<weight_vertex_pair_t> > vertex_queue;
	vertex_queue.push(std::make_pair(min_distance[source], source));

	while (!vertex_queue.empty()){
		weight_t dist = vertex_queue.top().first;
		vertex_t u = vertex_queue.top().second;
		vertex_queue.pop();
		// Because we leave old copies of the vertex in the priority queue
		// (with outdated higher distances), we need to ignore it when we come
		// across it again, by checking its distance against the minimum distance
		if (dist > min_distance[u]) continue;
		// Visit each edge exiting u
		const std::vector<neighbor> &neighbors = adjacency_list[u];
		for (std::vector<neighbor>::const_iterator neighbor_iter = neighbors.begin();
			neighbor_iter != neighbors.end();
			neighbor_iter++){
			vertex_t v = neighbor_iter->target;
			weight_t weight = neighbor_iter->weight;
			weight_t distance_through_u = dist + weight;
			if (distance_through_u < min_distance[v]) {
				min_distance[v] = distance_through_u;
				previous[v] = u;
				vertex_queue.push(std::make_pair(min_distance[v], v));
			}
		}
	}
}
void DijkstraComputePaths(vector<vertex_t> &source, const adjacency_list_t &adjacency_list, std::vector<weight_t> &min_distance, std::vector<vertex_t> &previous) {

	int n = adjacency_list.size();
	min_distance.clear();
	min_distance.resize(n, max_weight);
	for(auto s:source) min_distance[s] = 0;
	previous.clear();
	previous.resize(n, -1);
	// we use greater instead of less to turn max-heap into min-heap
	std::priority_queue<weight_vertex_pair_t,
		std::vector<weight_vertex_pair_t>,
		std::greater<weight_vertex_pair_t> > vertex_queue;
	for(auto s:source) vertex_queue.push(std::make_pair(min_distance[s], s));

	while (!vertex_queue.empty()) {
		weight_t dist = vertex_queue.top().first;
		vertex_t u = vertex_queue.top().second;
		vertex_queue.pop();
		// Because we leave old copies of the vertex in the priority queue
		// (with outdated higher distances), we need to ignore it when we come
		// across it again, by checking its distance against the minimum distance
		if (dist > min_distance[u]) continue;
		// Visit each edge exiting u
		const std::vector<neighbor> &neighbors = adjacency_list[u];
		for (std::vector<neighbor>::const_iterator neighbor_iter = neighbors.begin();
			neighbor_iter != neighbors.end();
			neighbor_iter++) {
			vertex_t v = neighbor_iter->target;
			weight_t weight = neighbor_iter->weight;
			weight_t distance_through_u = dist + weight;
			if (distance_through_u < min_distance[v]) {
				min_distance[v] = distance_through_u;
				previous[v] = u;
				vertex_queue.push(std::make_pair(min_distance[v], v));
			}
		}
	}
}
std::vector<vertex_t> DijkstraGetShortestPathTo(vertex_t vertex, const std::vector<vertex_t> &previous){
	std::vector<vertex_t> path;
	for (; vertex != -1; vertex = previous[vertex]) path.push_back(vertex);
	std::reverse(path.begin(), path.end());
	return path;
}

bool curve_parameterization(const vector<Eigen::Vector3d> &l0, const vector<Eigen::Vector3d> &l1,
	vector<Eigen::Vector3d> &lo, bool circle, bool uniform) {

	if (l0.size() < 2 || l1.size() < 2)return false;

	double len0 = 0, len1 = 0;
	std::vector<double> ratio0(l0.size()), ratio1(l1.size());
	ratio1[0] = ratio0[0] = 0;
	//ratio0
	if (uniform) {
		double step = 1.0 / (l0.size() - 1);
		for (int j = 1; j < l0.size(); j++)ratio0[j] = j*step;
	}
	else {
		for (int j = 1; j < l0.size(); j++) {
			double dis = (l0[j - 1] - l0[j]).norm();
			len0 += dis;
			ratio0[j] = len0;
		}
		for (int j = 0; j < ratio0.size(); j++) ratio0[j] /= len0;
	}
	if (!circle){
		//ratio1
		for (int j = 1; j < l1.size(); j++) {
			double dis = (l1[j - 1] - l1[j]).norm();
			len1 += dis;
			ratio1[j] = len1;
		}
		for (int j = 0; j < ratio1.size(); j++) ratio1[j] /= len1;

		//lo
		lo.resize(l0.size());
		lo[0] = l1[0];
		lo[lo.size() - 1] = l1[l1.size() - 1];

		for (int j = 1; j < l0.size() - 1; j++) {
			Vector3d v;
			bool found = false;
			for (int k = 0; k < ratio1.size(); k++) {
				if (ratio1[k] == ratio1[k + 1])continue;
				if (ratio0[j] >= ratio1[k] && ratio0[j] <= ratio1[k + 1]) {
					double r = (ratio0[j] - ratio1[k]) / (ratio1[k + 1] - ratio1[k]);
					v = l1[k] + r*(l1[k + 1] - l1[k]);
					found = true;
					break;
				}
			}
			assert(found);
			lo[j] = v;
		}
	}
	else {
		//start p
		Vector3d v = l0[0], pv;
		vector<Vector3d> pvs;
		vector<pair<double, uint32_t>> dis_ids;

		for (uint32_t j = 0; j < l1.size(); j++) {			
			double t, precision_here = 1.0e1;
			point_line_projection(l1[j], l1[(j + 1) % l1.size()], v, pv, t);
			dis_ids.push_back(make_pair((v - pv).norm(), j));
			pvs.push_back(pv);			
		}
		std::sort(dis_ids.begin(), dis_ids.end());
		assert(dis_ids.size());
		uint32_t closetid = dis_ids[0].second;
		pv = pvs[closetid];
		//l1_
		vector<Eigen::Vector3d> l1_;
		l1_.push_back(pv);
		for (int i = 0; i < l1.size(); i++) {
			l1_.push_back(l1[(i+closetid)%l1.size()]);
		}
		l1_.push_back(pv);
		//ratio1
		ratio1.resize(l1_.size());
		for (int j = 1; j < l1_.size(); j++) {
			double dis = (l1_[j - 1] - l1_[j]).norm();
			len1 += dis;
			ratio1[j] = len1;
		}
		for (int j = 0; j < ratio1.size(); j++) ratio1[j] /= len1;

		//lo
		lo.resize(l0.size());
		lo[0] = l1_[0];

		for (int j = 1; j < l0.size(); j++) {
			Vector3d v;
			bool found = false;
			for (int k = 0; k < ratio1.size(); k++) {
				if (ratio1[k] == ratio1[k + 1])continue;
				if (ratio0[j] >= ratio1[k] && ratio0[j] <= ratio1[k + 1]) {
					double r = (ratio0[j] - ratio1[k]) / (ratio1[k + 1] - ratio1[k]);
					v = l1_[k] + r*(l1_[k + 1] - l1_[k]);
					found = true;
					break;
				}
			}
			assert(found);
			lo[j] = v;
		}
	}
	return true;
}
//===================================mesh quality==========================================
void reorder_quad_mesh_propogation(Mesh &mi) {
	//connected components
	Mesh_Quality mq1, mq2;
	orient_surface_mesh(mi);
	//direction
	Mesh m1;
	m1.type = Mesh_type::Qua;
	m1.V = mi.V;
	scaled_jacobian(mi, mq1);
	cout << "m1 jacobian " << mq1.min_Jacobian<<" "<<mq1.ave_Jacobian << endl;
	if (mq1.min_Jacobian < 0) {
		m1.Fs = mi.Fs;
		for (auto &f : m1.Fs) { std::swap(f.vs[1], f.vs[3]);}
		scaled_jacobian(m1, mq2);
		cout << "m2 jacobian " << mq2.min_Jacobian << " " << mq2.ave_Jacobian << endl;
		if (mq2.ave_Jacobian > mq1.ave_Jacobian) {
			for (auto &f : m1.Fs) mi.Fs[f.id] = f;
		}
	}
}

void reorder_hex_mesh(Mesh &hmi) {
	for (auto &h:hmi.Hs) {
		double vol = 0;
		vector<Float> vols;
		for (uint32_t j = 0; j<8; j++)
		{
			uint32_t v0, v1, v2, v3;
			v0 = hex_tetra_table[j][0]; v1 = hex_tetra_table[j][1];
			v2 = hex_tetra_table[j][2]; v3 = hex_tetra_table[j][3];

			Vector3d c0 = hmi.V.col(h.vs[v0]);
			Vector3d c1 = hmi.V.col(h.vs[v1]);
			Vector3d c2 = hmi.V.col(h.vs[v2]);
			Vector3d c3 = hmi.V.col(h.vs[v3]);

			vols.push_back(a_jacobian_nonscaled(c0, c1, c2, c3));
			vol += vols[j];
		}
		if (vol < 0) {
			auto vs = h.vs;
			h.vs[0] = vs[3];
			h.vs[1] = vs[2];
			h.vs[2] = vs[1];
			h.vs[3] = vs[0];
			h.vs[4] = vs[7];
			h.vs[5] = vs[6];
			h.vs[6] = vs[5];
			h.vs[7] = vs[4];
		}
	}
}
void reorder_hex_mesh_propogation(Mesh &hmi) {
	//connected components
	vector<bool> H_tag(hmi.Hs.size(), false);
	vector<vector<uint32_t>> Groups;
	while (true) {
		vector<uint32_t> group;
		int shid = -1;
		for (uint32_t i = 0; i < H_tag.size(); i++)if (!H_tag[i]) { shid = i; break; }
		if (shid == -1)break;
		group.push_back(shid); H_tag[shid] = true;
		vector<uint32_t> group_ = group;
		while (true) {
			vector<uint32_t> pool;
			for (auto hid : group_) {
				vector<vector<uint32_t>> Fvs(6), fvs_sorted;
				for (uint32_t i = 0; i < 6; i++)for (uint32_t j = 0; j < 4; j++) Fvs[i].push_back(hmi.Hs[hid].vs[hex_face_table[i][j]]);
				fvs_sorted = Fvs;
				for (auto &vs : fvs_sorted)sort(vs.begin(), vs.end());

				for (auto fid : hmi.Hs[hid].fs)if (!hmi.Fs[fid].boundary) {
					int nhid = hmi.Fs[fid].neighbor_hs[0];
					if (nhid == hid) nhid = hmi.Fs[fid].neighbor_hs[1];

					if (!H_tag[nhid]) { 
						pool.push_back(nhid); H_tag[nhid] = true; 
					
						vector<uint32_t> fvs = hmi.Fs[fid].vs;
						sort(fvs.begin(), fvs.end());

						int f_ind = -1;
						for (uint32_t i = 0; i < 6; i++) if (std::equal(fvs.begin(), fvs.end(), fvs_sorted[i].begin())) {
							f_ind = i; break;
						}
						vector<uint32_t> hvs = hmi.Hs[nhid].vs;
						hmi.Hs[nhid].vs.clear();
						vector<uint32_t> topvs = Fvs[f_ind];
						std::reverse(topvs.begin(), topvs.end());
						vector<uint32_t> bottomvs;
						for (uint32_t i = 0; i < 4; i++) {
							for (auto nvid : hmi.Vs[topvs[i]].neighbor_vs) if (nvid != topvs[(i + 3) % 4] && nvid != topvs[(i + 1) % 4]
								&& std::find(hvs.begin(), hvs.end(), nvid) != hvs.end()) {
								bottomvs.push_back(nvid); break;
							}
						}
						hmi.Hs[nhid].vs = topvs;
						hmi.Hs[nhid].vs.insert(hmi.Hs[nhid].vs.end(), bottomvs.begin(), bottomvs.end());
					}
				}
			}
			if (pool.size()) {
				group_ = pool;
				group.insert(group.end(),pool.begin(), pool.end());
				pool.size();
			}
			else break;
		}
		Groups.push_back(group);
	}
	//direction
	//cout << "correct orientation " << endl;
	for (auto group : Groups) {
		Mesh_Quality mq1, mq2;
		Mesh m1, m2;
		m2.type = m1.type = Mesh_type::Hex;

		m2.V = m1.V = hmi.V;
		for (auto hid : group)m1.Hs.push_back(hmi.Hs[hid]);
		scaled_jacobian(m1, mq1);
		//cout << "m1 jacobian " << mq1.min_Jacobian<<" "<<mq1.ave_Jacobian << endl;
		if (mq1.min_Jacobian > 0) continue;
		m2.Hs = m1.Hs;
		for (auto &h : m2.Hs) { swap(h.vs[1], h.vs[3]); swap(h.vs[5], h.vs[7]);}
		scaled_jacobian(m2, mq2);
		//cout << "m2 jacobian " << mq2.min_Jacobian << " " << mq2.ave_Jacobian << endl;
		if (mq2.ave_Jacobian > mq1.ave_Jacobian) {
			for (auto &h : m2.Hs) hmi.Hs[h.id] = h;
		}
	}
}
bool scaled_jacobian(Mesh &hmi, Mesh_Quality &mq)
{
	if (hmi.type == Mesh_type::Hex) {
		mq.ave_Jacobian = 0;
		mq.min_Jacobian = 1;
		mq.deviation_Jacobian = 0;
		mq.V_Js.resize(hmi.Hs.size() * 8); mq.V_Js.setZero();
		mq.H_Js.resize(hmi.Hs.size()); mq.H_Js.setZero();

		int in = 0;
		for (uint32_t i = 0; i<hmi.Hs.size(); i++)
		{
			double hex_minJ = 1;
			for (uint32_t j = 0; j<8; j++)
			{
				uint32_t v0, v1, v2, v3;
				v0 = hex_tetra_table[j][0]; v1 = hex_tetra_table[j][1];
				v2 = hex_tetra_table[j][2]; v3 = hex_tetra_table[j][3];

				Vector3d c0 = hmi.V.col(hmi.Hs[i].vs[v0]);
				Vector3d c1 = hmi.V.col(hmi.Hs[i].vs[v1]);
				Vector3d c2 = hmi.V.col(hmi.Hs[i].vs[v2]);
				Vector3d c3 = hmi.V.col(hmi.Hs[i].vs[v3]);

				double jacobian_value = a_jacobian(c0, c1, c2, c3);

				if (hex_minJ>jacobian_value) hex_minJ = jacobian_value;

				uint32_t id = 8 * i + j;
				mq.V_Js[id] = jacobian_value;
			}
			mq.H_Js[i] = hex_minJ;
			mq.ave_Jacobian += hex_minJ;
			if (mq.min_Jacobian > hex_minJ) mq.min_Jacobian = hex_minJ;

			if(hex_minJ<0)
			{
				//std::cout<<"id "<<i<<std::endl;
				in++;
			}

		}
		mq.ave_Jacobian /= hmi.Hs.size();
		for (int i = 0; i < mq.H_Js.size(); i++)
			mq.deviation_Jacobian += (mq.H_Js[i] - mq.ave_Jacobian)*(mq.H_Js[i] - mq.ave_Jacobian);
		mq.deviation_Jacobian/= hmi.Hs.size();

		if(in>0)
			std::cout<<"flipped elements: "<<in<<std::endl;
	}
	else if (hmi.type == Mesh_type::Qua) {//2D planar only!
		mq.ave_Jacobian = 0;
		mq.min_Jacobian = 1;
		mq.deviation_Jacobian = 0;
		mq.V_Js.resize(hmi.Fs.size() * 4); mq.V_Js.setZero();
		mq.H_Js.resize(hmi.Fs.size()); mq.H_Js.setZero();

		for (uint32_t i = 0; i<hmi.Fs.size(); i++){
			double hex_minJ = 1;
			for (uint32_t j = 0; j<4; j++){
				Vector3d c0 = hmi.V.col(hmi.Fs[i].vs[j]);
				Vector3d c1 = hmi.V.col(hmi.Fs[i].vs[(j + 1) % 4]);
				Vector3d c2 = hmi.V.col(hmi.Fs[i].vs[(j + 3) % 4]);
				Vector3d n0 = (c1 - c0).normalized();
				Vector3d n1 = (c2 - c0).normalized();
				double jacobian_value = n0[0]*n1[1] - n0[1] * n1[0];

				if (hex_minJ>jacobian_value) hex_minJ = jacobian_value;

				uint32_t id = 4 * i + j;
				mq.V_Js[id] = jacobian_value;
			}
			mq.H_Js[i] = hex_minJ;
			mq.ave_Jacobian += hex_minJ;
			if (mq.min_Jacobian > hex_minJ) mq.min_Jacobian = hex_minJ;

		}
		mq.ave_Jacobian /= hmi.Fs.size();
		for (int i = 0; i < mq.H_Js.size(); i++)
			mq.deviation_Jacobian += (mq.H_Js[i] - mq.ave_Jacobian)*(mq.H_Js[i] - mq.ave_Jacobian);
		mq.deviation_Jacobian /= hmi.Fs.size();
	}
	else if (hmi.type == Mesh_type::Tet) {
		mq.ave_Jacobian = 0;
		mq.min_Jacobian = 1;
		mq.deviation_Jacobian = 0;
		mq.V_Js.resize(hmi.Hs.size() * 4); mq.V_Js.setZero();
		mq.H_Js.resize(hmi.Hs.size()); mq.H_Js.setZero();

		for (uint32_t i = 0; i<hmi.Hs.size(); i++)
		{
			Vector3d c0 = hmi.V.col(hmi.Hs[i].vs[0]);
			Vector3d c1 = hmi.V.col(hmi.Hs[i].vs[1]);
			Vector3d c2 = hmi.V.col(hmi.Hs[i].vs[2]);
			Vector3d c3 = hmi.V.col(hmi.Hs[i].vs[3]);

			double jacobian_value = a_jacobian(c0, c1, c2, c3);

			mq.H_Js[i] = jacobian_value;
			mq.ave_Jacobian += jacobian_value;
			if (mq.min_Jacobian > jacobian_value) mq.min_Jacobian = jacobian_value;

		}
		mq.ave_Jacobian /= hmi.Hs.size();
		for (int i = 0; i < mq.H_Js.size(); i++)
			mq.deviation_Jacobian += (mq.H_Js[i] - mq.ave_Jacobian)*(mq.H_Js[i] - mq.ave_Jacobian);
		mq.deviation_Jacobian /= hmi.Hs.size();
	}
	else return false;

	return true;
}

double a_jacobian(Vector3d &v0, Vector3d &v1, Vector3d &v2, Vector3d &v3)
{
	Matrix3d Jacobian;
	
	Jacobian.col(0) = (v1 - v0) * .5;
	Jacobian.col(1) = (v2 - v0) * .5;
	Jacobian.col(2) = (v3 - v0) * .5;


	double norm1 = Jacobian.col(0).norm();
	double norm2 = Jacobian.col(1).norm();
	double norm3 = Jacobian.col(2).norm();

	double scaled_jacobian = Jacobian.determinant();
	if (std::abs(norm1) < Precision || std::abs(norm2) < Precision || std::abs(norm3) < Precision){
		std::cout << "Potential Bug, check!" << endl; //system("PAUSE");
		return scaled_jacobian;
	}
	scaled_jacobian /= norm1*norm2*norm3;
	return scaled_jacobian;
}
Float a_jacobian_nonscaled(Vector3d &v0, Vector3d &v1, Vector3d &v2, Vector3d &v3) {
	Matrix3d Jacobian;

	Jacobian.col(0) = v1 - v0;
	Jacobian.col(1) = v2 - v0;
	Jacobian.col(2) = v3 - v0;


	double norm1 = Jacobian.col(0).norm();
	double norm2 = Jacobian.col(1).norm();
	double norm3 = Jacobian.col(2).norm();

	return Jacobian.determinant();
}
double a_jacobian(VectorXd &v0, VectorXd &v1, VectorXd &v2, VectorXd &v3)
{
	Matrix3d Jacobian;

	Jacobian.col(0) = v1 - v0;
	Jacobian.col(1) = v2 - v0;
	Jacobian.col(2) = v3 - v0;
	//Jacobian.col(0) = v0 - v3;
	//Jacobian.col(1) = v1 - v3;
	//Jacobian.col(2) = v2 - v3;


	double norm1 = Jacobian.col(0).norm();
	double norm2 = Jacobian.col(1).norm();
	double norm3 = Jacobian.col(2).norm();

	double scaled_jacobian = Jacobian.determinant();
	if (std::abs(norm1) < Precision || std::abs(norm2) < Precision || std::abs(norm3) < Precision) {
		return scaled_jacobian = 0;
		std::cout << "Potential Bug, check!" << endl; system("PAUSE");
	}
	scaled_jacobian /= norm1*norm2*norm3;
	return scaled_jacobian;
}
double a_jacobian(VectorXd &v0, VectorXd &v1, VectorXd &v2)
{
	Matrix2d Jacobian;

	//Jacobian.col(0) = v1 - v0;
	//Jacobian.col(1) = v2 - v0;
	//Jacobian.col(2) = v3 - v0;
	Jacobian.col(0) = v0 - v2;
	Jacobian.col(1) = v1 - v2;


	double norm1 = Jacobian.col(0).norm();
	double norm2 = Jacobian.col(1).norm();

	double scaled_jacobian = Jacobian.determinant();
	if (std::abs(norm1) < Precision || std::abs(norm2) < Precision) {
		return scaled_jacobian = 0;
		std::cout << "Potential Bug, check!" << endl; system("PAUSE");
	}
	scaled_jacobian /= norm1*norm2;
	return scaled_jacobian;
}
//===================================feature v tags==========================================
bool quad_mesh_feature(Mesh_Feature &mf) {
	Mesh &mesh = mf.tri;
	mesh.type = Mesh_type::Qua;
	if (!mesh.Fs.size()) return false;

	mf.V_map_reverse.clear();
	mf.V_map.clear();
	//connectivity
	build_connectivity(mesh);

	//normals
	mf.normal_Tri.resize(3, mesh.Fs.size()); mf.normal_Tri.setZero();
	mf.normal_V.resize(3, mesh.Vs.size()); mf.normal_V.setZero();
	mf.Tcenters.clear(); mf.Tcenters.resize(mesh.Fs.size());
	for (uint32_t i = 0; i < mesh.Fs.size(); i++) {
		const auto &vs = mesh.Fs[i].vs;

		Vector3d c; c.setZero();
		for (auto vid : vs)c += mesh.V.col(vid);
		c /= 4;
		mf.Tcenters[i] = c;

		mf.normal_Tri.col(i).setZero();
		for (uint32_t j = 0; j < vs.size(); j++) {
			Vector3d vec0 = mesh.V.col(vs[j]) - c;
			Vector3d vec1 = mesh.V.col(vs[(j+1)%vs.size()]) - c;
			mf.normal_Tri.col(i) += (vec0.cross(vec1)).normalized();
		}
		mf.normal_Tri.col(i) /= 4;
		mf.normal_V.col(vs[0]) += mf.normal_Tri.col(i);
		mf.normal_V.col(vs[1]) += mf.normal_Tri.col(i);
		mf.normal_V.col(vs[2]) += mf.normal_Tri.col(i);
		mf.normal_V.col(vs[3]) += mf.normal_Tri.col(i);
	}
	for (uint32_t i = 0; i<mf.normal_V.cols(); ++i)
		if (mf.normal_V.col(i) != Vector3d::Zero()) mf.normal_V.col(i) = mf.normal_V.col(i).normalized();
	mf.ave_length = 0;
	for (uint32_t i = 0; i < mesh.Es.size(); i++) {
		uint32_t v0 = mesh.Es[i].vs[0];
		uint32_t v1 = mesh.Es[i].vs[1];

		mf.ave_length += (mesh.V.col(v0) - mesh.V.col(v1)).norm();
	}
	mf.ave_length /= mesh.Es.size();

	return true;
}
bool triangle_mesh_feature(Mesh_Feature &mf) {

	Mesh &mesh = mf.tri;
	mesh.type = Mesh_type::Tri; 
	if (!mesh.Fs.size()) return false;
	
	mf.V_map_reverse.clear();
	mf.V_map.clear();
	//connectivity
	build_connectivity(mesh);

	//normals
	mf.normal_Tri.resize(3, mesh.Fs.size()); mf.normal_Tri.setZero();
	mf.normal_V.resize(3, mesh.Vs.size()); mf.normal_V.setZero();
	mf.Tcenters.clear(); mf.Tcenters.resize(mesh.Fs.size());
	for (uint32_t i = 0; i < mesh.Fs.size(); i++) {
		const auto &vs = mesh.Fs[i].vs;
		Vector3d vec0 = mesh.V.col(vs[1]) - mesh.V.col(vs[0]);
		Vector3d vec1 = mesh.V.col(vs[2]) - mesh.V.col(vs[0]);

		mf.normal_Tri.col(i) = (vec0.cross(vec1)).normalized();
		mf.normal_V.col(vs[0]) += mf.normal_Tri.col(i);
		mf.normal_V.col(vs[1]) += mf.normal_Tri.col(i);
		mf.normal_V.col(vs[2]) += mf.normal_Tri.col(i);

		mf.Tcenters[i].setZero();
		mf.Tcenters[i] += mesh.V.col(vs[0]);
		mf.Tcenters[i] += mesh.V.col(vs[1]);
		mf.Tcenters[i] += mesh.V.col(vs[2]);
		mf.Tcenters[i] /= 3;
	}
	for (uint32_t i = 0; i<mf.normal_V.cols(); ++i)
		if (mf.normal_V.col(i) != Vector3d::Zero()) mf.normal_V.col(i) = mf.normal_V.col(i).normalized();
	mf.ave_length = 0;
	for (uint32_t i = 0; i < mesh.Es.size(); i++) {
		uint32_t v0 = mesh.Es[i].vs[0];
		uint32_t v1 = mesh.Es[i].vs[1];

		mf.ave_length += (mesh.V.col(v0) - mesh.V.col(v1)).norm();
	}
	mf.ave_length /= mesh.Es.size();
	//feature edges, vs
	vector<bool> E_feature_flag(mesh.Es.size(), false);
	mf.v_types.resize(mesh.Vs.size()); std::fill(mf.v_types.begin(), mf.v_types.end(), 0);
	if (mf.read_from_file) {
		for (auto &v : mesh.Vs)sort(v.neighbor_es.begin(), v.neighbor_es.end());
		for (auto a_pair:mf.IN_v_pairs) {
			int i = a_pair[0], vid = a_pair[1];
			vector<uint32_t> sharedes;
			set_intersection(mesh.Vs[i].neighbor_es.begin(), mesh.Vs[i].neighbor_es.end(),
				mesh.Vs[vid].neighbor_es.begin(), mesh.Vs[vid].neighbor_es.end(), back_inserter(sharedes));
			if (sharedes.size()) {
				E_feature_flag[sharedes[0]] = true;

				uint32_t v0 = mesh.Es[sharedes[0]].vs[0];
				uint32_t v1 = mesh.Es[sharedes[0]].vs[1];

				mf.v_types[v0]++;
				mf.v_types[v1]++;
			}
		}
		//corners
		mf.corners.clear();
		for (uint32_t i = 0; i < mf.v_types.size(); i++) if (mf.v_types[i] == 0) mf.v_types[i] = Feature_V_Type::REGULAR;
		for (auto cid: mf.IN_corners) {
			mf.v_types[cid] = Feature_V_Type::CORNER;
			mf.corners.push_back(cid);
		}
	}
	else {
		vector<Float> Dihedral_angles(mesh.Es.size());
		for (uint32_t i = 0; i < mesh.Es.size(); i++) {
			Vector3d n0 = mf.normal_Tri.col(mesh.Es[i].neighbor_fs[0]);
			Vector3d n1 = mf.normal_Tri.col(mesh.Es[i].neighbor_fs[1]);

			Dihedral_angles[i] = PAI - acos(n0.dot(n1));

			if (Dihedral_angles[i] < mf.angle_threshold) {
				E_feature_flag[i] = true;

				uint32_t v0 = mesh.Es[i].vs[0];
				uint32_t v1 = mesh.Es[i].vs[1];

				mf.v_types[v0]++;
				mf.v_types[v1]++;
			}
		}
		mf.E_feature_flag = E_feature_flag;
		//corners
		mf.corners.clear();
		for (uint32_t i = 0; i < mf.v_types.size(); i++) {

			if (mf.v_types[i] == 0) mf.v_types[i] = Feature_V_Type::REGULAR;
			else if (mf.v_types[i] == 1 || mf.v_types[i] > 2) {
				mf.v_types[i] = Feature_V_Type::CORNER;
				mf.corners.push_back(i);
			} 
			else if (mf.v_types[i] == 2) {
				vector<Vector3d> ns; vector<int> vs;
				for (auto eid : mesh.Vs[i].neighbor_es) if (E_feature_flag[eid]) {
					uint32_t v0 = mesh.Es[eid].vs[0]; vs.push_back(v0);
					uint32_t v1 = mesh.Es[eid].vs[1]; vs.push_back(v1);
					ns.push_back((mesh.V.col(v0) - mesh.V.col(v1)).normalized());
				}

				if (vs[0] == vs[2] || vs[1] == vs[3]) ns[0] *= -1;
				double angles = PAI - acos(ns[0].dot(ns[1]));

				if (angles < mf.angle_threshold) {
					mf.v_types[i] = Feature_V_Type::CORNER;
					mf.corners.push_back(i);
				}
			}
		}
	}
	//feature curves
	mf.curve_es.clear(); mf.curve_vs.clear(); mf.circles.clear();
	uint32_t INVALID_E = mesh.Es.size();
	vector<bool> E_flag(mesh.Es.size(), false);
	for (uint32_t i = 0; i < mesh.Es.size(); i++) {

		if (!E_feature_flag[i]) continue;
		if (E_flag[i]) continue;

		std::function<bool(uint32_t, uint32_t, uint32_t &)> feature_line_proceed = [&](uint32_t vid, uint32_t eid, uint32_t &neid)->bool {

			if (mf.v_types[vid] == Feature_V_Type::REGULAR || mf.v_types[vid] == Feature_V_Type::CORNER) return false;

			for (uint32_t j = 0; j < mesh.Vs[vid].neighbor_es.size(); j++) {
				uint32_t cur_e = mesh.Vs[vid].neighbor_es[j];
				if (cur_e == eid || !E_feature_flag[cur_e] || E_flag[cur_e]) continue;
				neid = cur_e;
				return true;
			}
			return false;
		};

		uint32_t v_left = mesh.Es[i].vs[0], v_right = mesh.Es[i].vs[1];
		uint32_t sv_left, sv_right;
		std::vector<uint32_t> vs_left, vs_right, es_left, es_right;

		bool is_circle = false;
		//left 
		es_left.push_back(i); vs_left.push_back(v_left);
		uint32_t cur_e = i, next_e = INVALID_E;
		while (feature_line_proceed(v_left, cur_e, next_e)) {
			cur_e = next_e;
			E_flag[next_e] = true;
			if (cur_e == i) { is_circle = true; break; }
			es_left.push_back(next_e);
			if (mesh.Es[cur_e].vs[0] == v_left) v_left = mesh.Es[cur_e].vs[1]; else v_left = mesh.Es[cur_e].vs[0];
			vs_left.push_back(v_left);
		}

		if (is_circle) {
			mf.circles.push_back(true);
			mf.curve_es.push_back(es_left); mf.curve_vs.push_back(vs_left);
			continue;
		}
		//right
		vs_right.push_back(v_right);
		cur_e = i, next_e = INVALID_E;
		while (feature_line_proceed(v_right, cur_e, next_e)) {
			cur_e = next_e;
			E_flag[next_e] = true;
			if (mesh.Es[cur_e].vs[0] == v_right) v_right = mesh.Es[cur_e].vs[1]; else v_right = mesh.Es[cur_e].vs[0];
			vs_right.push_back(v_right);
			es_right.push_back(next_e);
		}
		std::reverse(vs_left.begin(), vs_left.end());
		vector<uint32_t> vs_link;
		vs_link = vs_left; vs_link.insert(vs_link.end(), vs_right.begin(), vs_right.end());
		std::reverse(es_left.begin(), es_left.end());
		vector<uint32_t> es_link;
		es_link = es_left; es_link.insert(es_link.end(), es_right.begin(), es_right.end());

		if (vs_link[0] == vs_link[vs_link.size() - 1]) {//one corner with a circle
			vector<uint32_t> vs_ = vs_link; vs_link.clear();
			for (uint32_t j = 0; j < vs_.size() - 1; j++)vs_link.push_back(vs_[j]);

			int vid0 = vs_link[0], vid1 = vs_link[vs_link.size() / 2];
			mf.v_types[vid1] = Feature_V_Type::CORNER;
			mf.corners.push_back(vid1);

			vector<uint32_t> curve_vs, curve_vs0, curve_vs1;
			for (auto vid : vs_link)if (vid == vid1) {
				curve_vs.push_back(vid);
				curve_vs0.swap(curve_vs);
				curve_vs.push_back(vid);
			}
			else curve_vs.push_back(vid);
			curve_vs.push_back(vid0);
			curve_vs1.swap(curve_vs);

			vector<uint32_t> es_link0, es_link1;
			for (int j = 0; j < curve_vs0.size() - 1; j++) {
				int v0 = curve_vs0[j], v1 = curve_vs0[j+1];
				vector<uint32_t> sharedes, es0 = mesh.Vs[v0].neighbor_es, es1 = mesh.Vs[v1].neighbor_es;
				std::sort(es0.begin(), es0.end()); std::sort(es1.begin(), es1.end());
				set_intersection(es0.begin(), es0.end(), es1.begin(), es1.end(), back_inserter(sharedes));
				if (sharedes.size()) es_link0.push_back(sharedes[0]); else { cout << "bug" << endl; }
			}
			for (int j = 0; j < curve_vs1.size() - 1; j++) {
				int v0 = curve_vs1[j], v1 = curve_vs1[j + 1];
				vector<uint32_t> sharedes, es0 = mesh.Vs[v0].neighbor_es, es1 = mesh.Vs[v1].neighbor_es;
				std::sort(es0.begin(), es0.end()); std::sort(es1.begin(), es1.end());
				set_intersection(es0.begin(), es0.end(), es1.begin(), es1.end(), back_inserter(sharedes));
				if (sharedes.size()) es_link1.push_back(sharedes[0]); else { cout << "bug" << endl; }
			}
			mf.curve_es.push_back(es_link0); mf.curve_vs.push_back(curve_vs0);
			mf.circles.push_back(false);
			mf.curve_es.push_back(es_link1); mf.curve_vs.push_back(curve_vs1);
			mf.circles.push_back(false);
		}
		else {
			mf.curve_es.push_back(es_link); mf.curve_vs.push_back(vs_link);
			mf.circles.push_back(false);
		}
	}
	mf.corner_curves.clear(); mf.corner_curves.resize(mf.corners.size());
	//
	for (uint32_t i = 0; i < mf.curve_vs.size(); i++) for (auto vid : mf.curve_vs[i])
		if (mf.v_types[vid] != Feature_V_Type::CORNER) mf.v_types[vid] = i;
		else {
			int pos = find(mf.corners.begin(), mf.corners.end(), vid) - mf.corners.begin();
			mf.corner_curves[pos].push_back(i);
		}

	//orphan
	if (!mf.orphan_curve || !mf.orphan_curve_single) {
		while (true) {
			bool changed = false;

			vector<int> l_count(mf.curve_es.size(), 0), c_count(mf.corners.size(), 0);
			vector<bool> l_flag(mf.curve_es.size(), false), c_flag(mf.corners.size(), false);
			vector<int> L_map(mf.curve_es.size(), -1);

			for (uint32_t i = 0; i < mf.corner_curves.size(); i++)if (mf.corner_curves[i].size() == 1) {
				l_count[mf.corner_curves[i][0]]++;
			}

			int ln = 0;
			for (uint32_t i = 0; i < mf.curve_es.size(); i++) {
				if (!mf.orphan_curve_single && l_count[i] == 1) {
					l_flag[i] = true; changed = true;
				}
				else if (!mf.orphan_curve && l_count[i] == 2) {
					l_flag[i] = true; changed = true;
				}
				else L_map[i] = ln++;
			}

			if (!changed)break;

			for (uint32_t i = 0; i < mf.corner_curves.size(); i++)if (mf.corner_curves[i].size() == 1) {
				if (l_flag[mf.corner_curves[i][0]])c_flag[i] = true;
			}else if(!mf.corner_curves[i].size()) c_flag[i] = true;

			vector<uint32_t> corner_; vector<bool> circles_;
			vector<vector<uint32_t>> corner_curves_, curve_vs_, curve_es_;
			for (uint32_t i = 0; i < mf.corners.size(); i++) {
				if (c_flag[i]) continue;
				corner_.push_back(mf.corners[i]);
				vector<uint32_t> cc;
				for (auto lid : mf.corner_curves[i])if (L_map[lid] != -1) cc.push_back(L_map[lid]);
				corner_curves_.push_back(cc);
			}
			for (uint32_t i = 0; i < mf.curve_vs.size(); i++) {
				if (l_flag[i]) {
					for (auto eid : mf.curve_es[i]) {
						mf.E_feature_flag[eid] = E_feature_flag[eid] = false;
					}
					continue;
				}
				curve_vs_.push_back(mf.curve_vs[i]);
				curve_es_.push_back(mf.curve_es[i]);
				circles_.push_back(mf.circles[i]);
			}
			mf.corners.swap(corner_);
			mf.corner_curves.swap(corner_curves_);
			mf.curve_vs.swap(curve_vs_);
			mf.curve_es.swap(curve_es_);
			mf.circles.swap(circles_);
		}

		fill(mf.v_types.begin(), mf.v_types.end(), Feature_V_Type::REGULAR);
		for (auto l : mf.curve_vs)for (auto vid : l)mf.v_types[vid] = Feature_V_Type::LINE;
		for (auto c : mf.corners)mf.v_types[c] = Feature_V_Type::CORNER;
	}


	mf.broken_curves.resize(mf.curve_es.size());
	std::fill(mf.broken_curves.begin(), mf.broken_curves.end(), false);

	return true;
}
void build_feature_graph(Mesh_Feature &mf, Feature_Graph &fg) {
	fg = Feature_Graph();
	fg.mf = mf;
	//build fg for both triangle mesh and quad mesh
	//corners
	vector<bool> TE_tag(mf.tri.Es.size(), false);
	for (uint32_t i = 0; i < mf.corners.size(); i++) {
		Vector3d v = mf.tri.V.col(mf.corners[i]);
		Feature_Corner c;
		c.id = fg.Cs.size();
		c.original = v;
		c.vs.push_back(mf.corners[i]);
		//neighbor vs
		vector<uint32_t> nts;
		for (auto vid : c.vs)nts.insert(nts.end(), mf.tri.Vs[vid].neighbor_fs.begin(), mf.tri.Vs[vid].neighbor_fs.end());
		std::sort(nts.begin(), nts.end()); nts.erase(unique(nts.begin(), nts.end()), nts.end());
		face_soup_info(mf.tri, nts, TE_tag, c.ring_vs);

		fg.Cs.push_back(c);
	}
	//corner correspondences
	fg.Ls.resize(mf.curve_vs.size());
	for (uint32_t i = 0; i < fg.Cs.size(); i++) {
		Feature_Corner &tc = fg.Cs[i];
		tc.ring_vs_tag.resize(tc.ring_vs.size(), -1);
		tc.neighbor_ls = mf.corner_curves[i];

		vector<uint32_t> vs;
		for (auto cid : mf.corner_curves[i]) {
			vector<uint32_t> &curve = mf.curve_vs[cid];
			if (curve[0] == mf.corners[i]) vs.push_back(curve[1]);
			else if (curve[curve.size() - 1] == mf.corners[i]) vs.push_back(curve[curve.size() - 2]);
			fg.Ls[cid].cs.push_back(tc.id);
		}

		for (uint32_t j = 0; j < tc.ring_vs.size(); j++) {
			if (find(vs.begin(), vs.end(), tc.ring_vs[j]) != vs.end()) {
				tc.ring_vs_tag[j] = mf.corner_curves[i][find(vs.begin(), vs.end(), tc.ring_vs[j]) - vs.begin()];
			}
		}
	}
	for (uint32_t i = 0; i < fg.Ls.size(); i++) {
		fg.Ls[i].id = i;
		//start to trace a curve
		fg.Ls[i].vs = mf.curve_vs[i];
		fg.Ls[i].es = mf.curve_es[i];
		fg.Ls[i].broken = mf.broken_curves[i];
		fg.Ls[i].circle = mf.circles[i];
	}
	//regions
	fg.Rs.clear();

	int INVALID_E = -1, INVALID_F = -1;
	std::vector<int> e_flag(mf.tri.Es.size(), INVALID_E), f_flag(mf.tri.Fs.size(), INVALID_F);
	std::vector<bool> ee_flag(fg.Ls.size(), false);

	for (uint32_t i = 0; i < fg.Ls.size(); i++) //if (fg.Ls[i].broken) continue; else 
		for (auto eid : fg.Ls[i].es) e_flag[eid] = i;
	
	while (true) {
		Feature_Region fr; fr.id = fg.Rs.size(); fr.Color_ID = -1;
		uint32_t start_f = INVALID_F;
		for (uint32_t i = 0; i < f_flag.size(); i++) if (f_flag[i] == INVALID_F) { start_f = i; break; }
		if (start_f == INVALID_F) break;

		std::queue<uint32_t> f_pool; f_pool.push(start_f);
		while (!f_pool.empty()) {
			start_f = f_pool.front(); f_pool.pop();
			if (f_flag[start_f] != INVALID_F) continue;
			f_flag[start_f] = fr.id;
			fr.tris.push_back(start_f);
			for (auto eid: mf.tri.Fs[start_f].es) {
				if (e_flag[eid] != INVALID_E) {
					if (!ee_flag[e_flag[eid]]) {
						ee_flag[e_flag[eid]] = true;
						fr.ls.push_back(e_flag[eid]);
					}
					continue;
				}
				for (auto fid: mf.tri.Es[eid].neighbor_fs) {
					if (f_flag[fid] != INVALID_F) continue;
					f_pool.push(fid);
				}
			}
		}
		sort(fr.tris.begin(), fr.tris.end());
		fg.Rs.push_back(fr);
		for (auto lid: fr.ls) ee_flag[lid] = false;
	}
	fg.F_tag = f_flag;
	//relationship, Cs, Ls, Rs
	for (uint32_t i = 0; i < fg.Rs.size(); i++) {
		for (auto lid : fg.Rs[i].ls)fg.Ls[lid].neighbor_rs.push_back(i);
		for (auto lid : fg.Rs[i].ls)for (auto cid : fg.Ls[lid].cs)fg.Rs[i].cs.push_back(cid);
		sort(fg.Rs[i].cs.begin(), fg.Rs[i].cs.end());
		fg.Rs[i].cs.erase(unique(fg.Rs[i].cs.begin(), fg.Rs[i].cs.end()), fg.Rs[i].cs.end());
	}
}
bool triangle_mesh_feature(Mesh_Feature &mf, Mesh &hmi){

	if (hmi.type != Mesh_type::Hex) return false;
	//tri-mesh
	const uint32_t INVALID_V = hmi.Vs.size();

	Mesh &mesh = mf.tri;
	mesh.Vs.clear(); mesh.Es.clear(); mesh.Fs.clear(); mesh.Hs.clear();
	mesh.type = Mesh_type::Tri;

	mf.V_map_reverse.clear();
	mf.V_map.resize(hmi.Vs.size()); std::fill(mf.V_map.begin(), mf.V_map.end(), -1);

	vector<bool> V_tag(hmi.Vs.size(), false);
	
	for (auto f : hmi.Fs) if (f.boundary) {
		for (auto vid : f.vs) V_tag[vid] = true;
		Hybrid_F t;
		t.id = mesh.Fs.size();
		t.vs.resize(3);
		t.vs[0] = f.vs[0];
		t.vs[1] = f.vs[1];
		t.vs[2] = f.vs[2];
		mesh.Fs.push_back(t);

		t.id = mesh.Fs.size();
		t.vs[0] = f.vs[2];
		t.vs[1] = f.vs[3];
		t.vs[2] = f.vs[0];
		mesh.Fs.push_back(t);
	}
	//re-indexing
	uint32_t newV_id = 0;
	for (uint32_t i = 0; i < V_tag.size(); i++) if (V_tag[i]) {
		mf.V_map[i] = newV_id++; mf.V_map_reverse.push_back(i);
	}
	mesh.V.resize(3, newV_id);
	for (uint32_t i = 0; i < V_tag.size(); i++) if (V_tag[i]) {
		Hybrid_V v;
		v.id = mesh.Vs.size(); mesh.Vs.push_back(v);
		mesh.V.col(v.id) = hmi.V.col(i);	
	}
	for (uint32_t i = 0; i < mesh.Fs.size();i++) for (uint32_t j = 0; j < 3; j++) mesh.Fs[i].vs[j] = mf.V_map[mesh.Fs[i].vs[j]];
	//orient direction
	orient_triangle_mesh(mesh);
	//connectivity
	build_connectivity(mesh);

//normals
	mf.normal_Tri.resize(3, mesh.Fs.size()); mf.normal_Tri.setZero();
	mf.normal_V.resize(3, mesh.Vs.size()); mf.normal_V.setZero();
	mf.Tcenters.clear(); mf.Tcenters.resize(mesh.Fs.size());
	for (uint32_t i = 0; i < mesh.Fs.size(); i++) {
		const auto &vs = mesh.Fs[i].vs;
		Vector3d vec0 = mesh.V.col(vs[1]) - mesh.V.col(vs[0]);
		Vector3d vec1 = mesh.V.col(vs[2]) - mesh.V.col(vs[0]);

		mf.normal_Tri.col(i) = (vec0.cross(vec1)).normalized();
		mf.normal_V.col(vs[0]) += mf.normal_Tri.col(i);
		mf.normal_V.col(vs[1]) += mf.normal_Tri.col(i);
		mf.normal_V.col(vs[2]) += mf.normal_Tri.col(i);

		mf.Tcenters[i].setZero();
		mf.Tcenters[i] += mesh.V.col(vs[0]);
		mf.Tcenters[i] += mesh.V.col(vs[1]);
		mf.Tcenters[i] += mesh.V.col(vs[2]);
		mf.Tcenters[i] /= 3;
	}
	for (uint32_t i = 0; i<mf.normal_V.cols(); ++i) 
		if (mf.normal_V.col(i) != Vector3d::Zero()) mf.normal_V.col(i) = mf.normal_V.col(i).normalized();
	mf.ave_length = 0;
	for (uint32_t i = 0; i < mesh.Es.size(); i++) {
			uint32_t v0 = mesh.Es[i].vs[0];
			uint32_t v1 = mesh.Es[i].vs[1];

			mf.ave_length += (mesh.V.col(v0) - mesh.V.col(v1)).norm();
	}
	mf.ave_length /= mesh.Es.size();
//feature edges, vs
	vector<bool> E_feature_flag(mesh.Es.size(), false);
	mf.v_types.resize(mesh.Vs.size()); fill(mf.v_types.begin(), mf.v_types.end(), 0);
	vector<Float> Dihedral_angles(mesh.Es.size());
	for (uint32_t i = 0; i < mesh.Es.size(); i++) {
		Vector3d n0 = mf.normal_Tri.col(mesh.Es[i].neighbor_fs[0]);
		Vector3d n1 = mf.normal_Tri.col(mesh.Es[i].neighbor_fs[1]);

		Dihedral_angles[i] = PAI - acos(n0.dot(n1));

		if (Dihedral_angles[i] < mf.angle_threshold) {
			E_feature_flag[i] = true;

			uint32_t v0 = mesh.Es[i].vs[0];
			uint32_t v1 = mesh.Es[i].vs[1];

			mf.v_types[v0]++;
			mf.v_types[v1]++;
		}
	}
	mf.corners.clear();
	for (uint32_t i = 0; i < mf.v_types.size(); i++) {

		if (mf.v_types[i] == 0) mf.v_types[i] = Feature_V_Type::REGULAR;
		else if (mf.v_types[i] == 1  || mf.v_types[i] > 2) {
			mf.v_types[i] = Feature_V_Type::CORNER;
			mf.corners.push_back(i);
		}
		else if (mf.v_types[i] == 2) {
			vector<Vector3d> ns; vector<int> vs;
			for (auto eid : mesh.Vs[i].neighbor_es) if (E_feature_flag[eid]) {
				uint32_t v0 = mesh.Es[eid].vs[0]; vs.push_back(v0);
				uint32_t v1 = mesh.Es[eid].vs[1]; vs.push_back(v1);
				ns.push_back((mesh.V.col(v0) - mesh.V.col(v1)).normalized());
			}

			if (vs[0] == vs[2] || vs[1] == vs[3]) ns[0] *= -1;
			//if (vs[0] == vs[2]) ns[0] *= -1;//to reproduce chord bug
			double angles = PAI - acos(ns[0].dot(ns[1]));

			if (angles < mf.angle_threshold) {
				mf.v_types[i] = Feature_V_Type::CORNER;
				mf.corners.push_back(i);
			}
		}

	}
	mf.corner_curves.clear(); mf.corner_curves.resize(mf.corners.size());
//feature curves
	mf.curve_es.clear(); mf.curve_vs.clear(); mf.circles.clear();
	uint32_t INVALID_E = mesh.Es.size();
	vector<bool> E_flag(mesh.Es.size(), false);
	for (uint32_t i = 0; i < mesh.Es.size(); i++) {

		if (!E_feature_flag[i]) continue;
		if (E_flag[i]) continue;

		std::function<bool(uint32_t, uint32_t, uint32_t &)> feature_line_proceed = [&](uint32_t vid, uint32_t eid, uint32_t &neid)->bool {

			if (mf.v_types[vid] == Feature_V_Type::REGULAR || mf.v_types[vid] == Feature_V_Type::CORNER) return false;

			for (uint32_t j = 0; j < mesh.Vs[vid].neighbor_es.size(); j++) {
				uint32_t cur_e = mesh.Vs[vid].neighbor_es[j];
				if (cur_e == eid || !E_feature_flag[cur_e] || E_flag[cur_e]) continue;
				neid = cur_e;
				return true;
			}
			return false;
		};

		uint32_t v_left = mesh.Es[i].vs[0], v_right = mesh.Es[i].vs[1];
		uint32_t sv_left, sv_right;
		std::vector<uint32_t> vs_left, vs_right, es_left, es_right;

		bool is_circle = false;
		//left 
		es_left.push_back(i); vs_left.push_back(v_left);
		uint32_t cur_e = i, next_e = INVALID_E;
		while (feature_line_proceed(v_left, cur_e, next_e)) {
			cur_e = next_e;
			E_flag[next_e] = true;
			if (cur_e == i) { is_circle = true; break; }
			es_left.push_back(next_e);
			if (mesh.Es[cur_e].vs[0] == v_left) v_left = mesh.Es[cur_e].vs[1]; else v_left = mesh.Es[cur_e].vs[0];
			vs_left.push_back(v_left);
		}

		if (is_circle) {
			mf.circles.push_back(true);
			mf.curve_es.push_back(es_left); mf.curve_vs.push_back(vs_left);
			continue;
		}
		//right
		vs_right.push_back(v_right);
		cur_e = i, next_e = INVALID_E;
		while (feature_line_proceed(v_right, cur_e, next_e)) {
			cur_e = next_e;
			E_flag[next_e] = true;
			if (mesh.Es[cur_e].vs[0] == v_right) v_right = mesh.Es[cur_e].vs[1]; else v_right = mesh.Es[cur_e].vs[0];
			vs_right.push_back(v_right);
			es_right.push_back(next_e);
		}
		std::reverse(vs_left.begin(), vs_left.end());
		vector<uint32_t> vs_link;
		vs_link = vs_left; vs_link.insert(vs_link.end(), vs_right.begin(), vs_right.end());
		std::reverse(es_left.begin(), es_left.end());
		vector<uint32_t> es_link;
		es_link = es_left; es_link.insert(es_link.end(), es_right.begin(), es_right.end());
	
		mf.curve_es.push_back(es_link); mf.curve_vs.push_back(vs_link);
		mf.circles.push_back(false);
	}
//
	for (uint32_t i = 0; i < mf.curve_vs.size(); i++) for (auto vid : mf.curve_vs[i])
		if (mf.v_types[vid] != Feature_V_Type::CORNER) mf.v_types[vid] = i;
		else {
			int pos = find(mf.corners.begin(), mf.corners.end(), vid) - mf.corners.begin();
			mf.corner_curves[pos].push_back(i);
		}
	
	return true;
}
bool initial_feature(Mesh_Feature &mf, Feature_Constraints &fc, Mesh &hmi) {

	fc.V_types.resize(hmi.Vs.size()); fill(fc.V_types.begin(), fc.V_types.end(), Feature_V_Type::INTERIOR);
	fc.V_ids.resize(hmi.Vs.size());
	fc.RV_type.resize(hmi.Vs.size());
	fill(fc.RV_type.begin(), fc.RV_type.end(), false);
	//matrices
	uint32_t num_corners = 0, num_lines = 0, num_regulars = 0;
	for (uint32_t i = 0; i < mf.v_types.size(); i++) {
		if (mf.v_types[i] == Feature_V_Type::REGULAR) {
			fc.V_types[mf.V_map_reverse[i]] = Feature_V_Type::REGULAR;
			fc.V_ids[mf.V_map_reverse[i]] = i;
			num_regulars++;
		}
		else if (mf.v_types[i] == Feature_V_Type::CORNER) {
			fc.V_types[mf.V_map_reverse[i]] = Feature_V_Type::CORNER;
			fc.V_ids[mf.V_map_reverse[i]] = i;
			num_corners++;
		}
		else if (mf.v_types[i] != Feature_V_Type::INTERIOR) {
			fc.V_types[mf.V_map_reverse[i]] = Feature_V_Type::LINE;
			fc.V_ids[mf.V_map_reverse[i]] = mf.v_types[i];
			num_lines++;
		}
	}
	fc.ids_C.resize(num_corners); fc.C.resize(num_corners, 3);
	fc.num_a = num_lines; fc.ids_L.resize(num_lines); fc.Axa_L.resize(num_lines, 3); fc.origin_L.resize(num_lines, 3);
	fc.ids_T.resize(num_regulars); fc.normal_T.resize(num_regulars, 3); fc.dis_T.resize(num_regulars); fc.V_T.resize(num_regulars, 3);
	num_corners = num_lines = num_regulars = 0;
	for (uint32_t i = 0; i < hmi.Vs.size(); i++) {
		if (fc.V_types[i] == Feature_V_Type::CORNER) {
			fc.ids_C[num_corners] = i;
			fc.C.row(num_corners++) = mf.tri.V.col(fc.V_ids[i]);
		}
		else if (fc.V_types[i] == Feature_V_Type::LINE) {
			fc.ids_L[num_lines] = i;
			fc.origin_L.row(num_lines) = hmi.V.col(i);
			uint32_t curve_id = fc.V_ids[i];
			vector<uint32_t> &curve = mf.curve_vs[curve_id];
			if (find(curve.begin(), curve.end(), mf.V_map[i]) == curve.end()) {
				cout << "ERROR in curve" << endl; system("PAUSE");
			}
			int pos = find(curve.begin(), curve.end(), mf.V_map[i]) - curve.begin();
			Vector3d tangent(0, 0, 0);
			uint32_t curve_len = curve.size();
			if (mf.circles[curve_id] || (!mf.circles[curve_id] && pos !=0 && pos != curve_len - 1)) {
				int32_t pos_0 = (pos -1 + curve_len) % curve_len, pos_1 = (pos + 1) % curve_len;
				tangent += (mf.tri.V.col(curve[pos]) - mf.tri.V.col(curve[pos_0])).normalized();
				tangent += (mf.tri.V.col(curve[pos_1]) - mf.tri.V.col(curve[pos])).normalized();
			}
			else if (!mf.circles[curve_id] && pos == 0) {
				int32_t pos_1 = (pos + 1) % curve_len;
				tangent += (mf.tri.V.col(curve[pos_1]) - mf.tri.V.col(curve[pos])).normalized();
			}
			else if (!mf.circles[curve_id] && pos == curve_len - 1) {
				int32_t pos_0 = (pos - 1 + curve_len) % curve_len;
				tangent += (mf.tri.V.col(curve[pos]) - mf.tri.V.col(curve[pos_0])).normalized();
			}
			if(tangent == Vector3d::Zero()){ 
				cout << "ERROR in curve" << endl; system("PAUSE"); 
			}
			tangent.normalize();
			fc.Axa_L.row(num_lines++) = tangent;
		}
		else if (fc.V_types[i] == Feature_V_Type::REGULAR) {
			fc.ids_T[num_regulars] = i;
			uint32_t vid = fc.V_ids[i];
			fc.normal_T.row(num_regulars) = mf.normal_V.col(vid);
			fc.V_T.row(num_regulars) = mf.tri.V.col(vid); 
			fc.dis_T[num_regulars++] = mf.normal_V.col(vid).dot(mf.tri.V.col(vid));
		}
	}

	return true;
}
bool project_surface_update_feature(Mesh_Feature &mf, Feature_Constraints &fc, MatrixXd &V, VectorXi &b, MatrixXd &bc, uint32_t Loop) {

	uint32_t bc_num = fc.ids_C.size() + fc.ids_L.size() + fc.ids_T.size();
	vector<std::tuple<Feature_V_Type, uint32_t, uint32_t, uint32_t>> CI;
	b.resize(bc_num);
	bc.resize(bc_num, 3);
	bc.setZero();
	bc_num = 0;
	uint32_t num_corners = 0, num_lines = 0, num_regulars = 0;
		
	for (uint32_t i = 0; i < fc.V_types.size(); i++) {
		if (fc.V_types[i] == Feature_V_Type::CORNER)
			CI.push_back(std::make_tuple(Feature_V_Type::CORNER, i, num_corners++, bc_num++));
		else if (fc.V_types[i] == Feature_V_Type::LINE)
			CI.push_back(std::make_tuple(Feature_V_Type::LINE, i, num_lines++, bc_num++));
		else if (fc.V_types[i] == Feature_V_Type::REGULAR)
			CI.push_back(std::make_tuple(Feature_V_Type::REGULAR, i, num_regulars++, bc_num++));
	}
	cout << "start feature projection" << endl;

	GRAIN_SIZE = 10;
#if 1
	tbb::parallel_for(
		tbb::blocked_range<uint32_t>(0u, (uint32_t)CI.size(), GRAIN_SIZE),
		[&](const tbb::blocked_range<uint32_t> &range) {
		for (uint32_t m = range.begin(); m != range.end(); m++) {
#endif
			//for (uint32_t m = 0; m < CI.size(); m++) {

			Feature_V_Type type = std::get<0>(CI[m]);
			uint32_t i = get<1>(CI[m]);
			uint32_t mi = get<2>(CI[m]);
			uint32_t bci = get<3>(CI[m]);
			Vector3d pv, v;

			if (type == Feature_V_Type::CORNER) {
				pv = mf.tri.V.col(fc.V_ids[i]);
				fc.C.row(mi) = pv;
			}
			else if (type == Feature_V_Type::LINE) {
				pv.setZero();
				v = V.row(i);
				uint32_t curve_id = fc.V_ids[i];
				vector<uint32_t> &curve = mf.curve_vs[curve_id];
				Vector3d tangent(1, 0, 0);
				uint32_t curve_len = curve.size();

				if (!mf.circles[curve_id]) curve_len--;

				vector<Vector3d> pvs, tangents;
				vector<pair<double, uint32_t>> dis_ids;

				for (uint32_t j = 0; j < curve_len; j++) {
					uint32_t pos_0 = curve[j], pos_1 = curve[(j + 1) % curve.size()];
					double t, precision_here = 1.0e1;
					point_line_projection(mf.tri.V.col(pos_0), mf.tri.V.col(pos_1), v, pv, t);
					//if ((t >= 0.0 || num_equal(t, 0.0, precision_here)) && (t <= 1.0 || num_equal(t, 1.0, precision_here))) {
					{
						//pv = mf.tri.V.col(pos_0) + t * (mf.tri.V.col(pos_1) - mf.tri.V.col(pos_0));
						tangent = (mf.tri.V.col(pos_1) - mf.tri.V.col(pos_0)).normalized();

						dis_ids.push_back(make_pair((v - pv).norm(), pvs.size()));
						pvs.push_back(pv);
						tangents.push_back(tangent);

					}
				}
				sort(dis_ids.begin(), dis_ids.end());

				if (dis_ids.size()) {
					uint32_t cloestid = dis_ids[0].second;
					pv = pvs[cloestid];
					tangent = tangents[cloestid];
				}
				else {
					//brute-force search
					//cout << "brute-force search" << endl;
					for (uint32_t j = 0; j < curve.size(); j++) {
						double dis = (mf.tri.V.col(curve[j]) - v).norm();
						dis_ids.push_back(make_pair(dis, j));
					}
					sort(dis_ids.begin(), dis_ids.end());

					int pos = dis_ids[0].second;
					pv = mf.tri.V.col(curve[pos]);

					curve_len = curve.size();
					if (mf.circles[curve_id] || (!mf.circles[curve_id] && pos != 0 && pos != curve_len - 1)) {
						uint32_t pos_0 = (pos - 1 + curve_len) % curve_len, pos_1 = (pos + 1) % curve_len;
						tangent += (mf.tri.V.col(curve[pos]) - mf.tri.V.col(curve[pos_0])).normalized();
						tangent += (mf.tri.V.col(curve[pos_1]) - mf.tri.V.col(curve[pos])).normalized();
					}
					else if (!mf.circles[curve_id] && pos == 0) {
						uint32_t pos_1 = (pos + 1) % curve_len;
						tangent += (mf.tri.V.col(curve[pos_1]) - mf.tri.V.col(curve[pos])).normalized();
					}
					else if (!mf.circles[curve_id] && pos == curve_len - 1) {
						uint32_t pos_0 = (pos - 1 + curve_len) % curve_len;
						tangent += (mf.tri.V.col(curve[pos]) - mf.tri.V.col(curve[pos_0])).normalized();
					}
					tangent.normalize();
				}
				fc.origin_L.row(mi) = pv;
				fc.Axa_L.row(mi) = tangent;
			}
			else if (type == Feature_V_Type::REGULAR) {
				//breadth-first search for the best triangle plane
				vector<uint32_t>  tids;
				uint32_t tid;
				Vector3d interpolP, interpolN;
				pv.setZero();
				v = V.row(i);

				tid = fc.V_ids[i];
				if (!fc.RV_type[i]) tids = mf.tri.Vs[tid].neighbor_fs;
				else tids.push_back(tid);

				Vector3d PreinterpolP = fc.V_T.row(mi), PreinterpolN = fc.normal_T.row(mi);
				bool found = phong_projection(tids, Loop, tid, v, interpolP, interpolN, PreinterpolP, PreinterpolN);
				
				if(!found || (v - interpolP).norm() >= mf.ave_length){
					tid = 0;
					double min_dis = (v - mf.Tcenters[0]).norm();
					for (uint32_t j = 1; j < mf.Tcenters.size(); j++) {
						if (min_dis > (v - mf.Tcenters[j]).norm()) {
							tid = j;
							min_dis = (v - mf.Tcenters[j]).norm();
						}
					}
					uint32_t tid_temp = tid;
					tids.clear();
					tids.push_back(tid_temp);
					if (phong_projection(tids, Loop, tid_temp, v, interpolP, interpolN, PreinterpolP, PreinterpolN))
						tid = tid_temp;
					else{
						//cout << "didn't find t" << endl;
						//if ((v - interpolP).norm() >= mf.ave_length) {
							interpolP = mf.Tcenters[tid_temp];
							interpolN = mf.normal_Tri.col(tid_temp);
						//}
					}
				}
				pv = interpolP;
				fc.V_ids[i] = tid;
				fc.RV_type[i] = true;
				fc.normal_T.row(mi) = interpolN;
				fc.dis_T[mi] = interpolN.dot(interpolP);
				fc.V_T.row(mi) = interpolP;
			}
			else continue;			
			b[bci] = i;
			bc.row(bci) = pv;
		}
#if 1
	}
	);
#endif

	//cout << "Finished feature projection update" << endl;
	return true;
}
bool phong_projection(vector<uint32_t> &tids, uint32_t Loop, uint32_t &tid, Vector3d &v, Vector3d &interpolP, Vector3d &interpolN, Vector3d &PreinterpolP, Vector3d &PreinterpolN) {
	vector<bool> t_flag(mf.tri.Fs.size(), false);

	vector<uint32_t> tids_;
	for (uint32_t Iter = 0; Iter < Loop; Iter++) {
		for (uint32_t j = 0; j < tids.size(); j++) {
			vector<uint32_t> &vs = mf.tri.Fs[tids[j]].vs;
			for (uint32_t k = 0; k < 3; k++) {
				for (auto ntid : mf.tri.Vs[vs[k]].neighbor_fs) {
					if (t_flag[ntid]) continue; t_flag[ntid] = true;
					tids_.push_back(ntid);
				}
			}
		}
		tids.insert(tids.end(), tids_.begin(), tids_.end()); tids_.clear();
	}
	sort(tids.begin(), tids.end()); tids.erase(unique(tids.begin(), tids.end()), tids.end());
	
	vector<uint32_t> ts;
	bool found = false;
	for (auto id : tids) { t_flag[id] = true; ts.push_back(id); }
	while (ts.size()) {
		tids.clear();
		vector<Vector3d> pvs, pns;
		vector<Vector2d> uvs; vector<pair<double, uint32_t>> dis_ids;
		for (uint32_t j = 0; j < ts.size(); j++) {
			vector<Vector3d> tri_vs(3), vs_normals(3);
			vector<uint32_t> &vs = mf.tri.Fs[ts[j]].vs;
			for (uint32_t k = 0; k < 3; k++) {
				tri_vs[k] = mf.tri.V.col(vs[k]);
				vs_normals[k] = mf.normal_V.col(vs[k]);
			}
			Vector2d uv;
			projectPointOnTriangle(tri_vs, vs_normals, v, uv, interpolP, interpolN);

			if ((uv.x() >= 0.0 || num_equal(uv.x(), 0.0, Precision_Pro)) && (uv.y() >= 0.0 || num_equal(uv.y(), 0.0, Precision_Pro)) &&
				(1 - uv.x() - uv.y() >= 0.0 || num_equal(1 - uv.x() - uv.y(), 0.0, Precision_Pro))) {

				if (PreinterpolN!=Vector3d::Zero() && interpolN.dot(PreinterpolN) <= 0)continue;

				dis_ids.push_back(make_pair((v - interpolP).norm(), tids.size()));
				pvs.push_back(interpolP); pns.push_back(interpolN); uvs.push_back(uv);
				tids.push_back(ts[j]);
			}
		}
		sort(dis_ids.begin(), dis_ids.end());

		if (dis_ids.size()) {
			found = true;
			uint32_t cloestid = dis_ids[0].second;
			interpolP = pvs[cloestid];
			tid = tids[cloestid];
			interpolN = pns[cloestid];
			return true;
		}

		vector<uint32_t> ts_;
		for (uint32_t j = 0; j < ts.size(); j++) {
			vector<uint32_t> &vs = mf.tri.Fs[ts[j]].vs;
			for (uint32_t k = 0; k < 3; k++) {
				for (auto ntid : mf.tri.Vs[vs[k]].neighbor_fs) {
					if (t_flag[ntid]) continue;
					t_flag[ntid] = true;
					ts_.push_back(ntid);
				}
			}
		}
		ts.clear();
		ts.swap(ts_);
	}
	return false;
}

void point_line_projection(const Vector3d &v1, const Vector3d &v2, const Vector3d &v, Vector3d &pv, double &t)
{
	Vector3d vv1 = v - v1, v21 = v2 - v1;
	double nv21_2 = v21.squaredNorm();
	if (nv21_2 >= Precision)
		t = vv1.dot(v21) / nv21_2;
	else
		t = 0;
	if (t >= 0.0 && t <= 1.0) pv = v1 + t * (v2 - v1);
	else if (t < 0.0)pv = v1;
	else if (t > 1.0)pv = v2;
}
void projectPointOnQuad(const vector<Vector3d>& quad_vs, vector<Vector3d> & vs_normals, const Vector3d& p, Vector2d& uv, Vector3d& interpolP, Vector3d& interpolN)
{
	Eigen::Matrix<double, 3, 2> jacobian;

	uv = Vector2d::Constant(0.5f);

	//Newton iteration for projection on quad

	//objective function:
	// F(u, v) = (p - interpolP) x interpolN = 0
	Vector3d F;

	for (int i = 0; i < 4; ++i)
	{
		interpolP = bilinear(quad_vs[0], quad_vs[1], quad_vs[2], quad_vs[3], uv);
		interpolN = bilinear(vs_normals[0], vs_normals[1], vs_normals[2], vs_normals[3], uv);

		Vector3d dPdu = (1 - uv.y()) * quad_vs[1] + uv.y() * quad_vs[2] - ((1 - uv.y()) * quad_vs[0] + uv.y() * quad_vs[3]);
		Vector3d dPdv = (1 - uv.x()) *quad_vs[3] + uv.x() * quad_vs[2] - ((1 - uv.x()) * quad_vs[0] + uv.x() * quad_vs[1]);
		Vector3d dNdu = (1 - uv.y()) * vs_normals[1] + uv.y() * vs_normals[2] - ((1 - uv.y()) * vs_normals[0] + uv.y() * vs_normals[3]);
		Vector3d dNdv = (1 - uv.x()) * vs_normals[3] + uv.x() * vs_normals[2] - ((1 - uv.x()) * vs_normals[0] + uv.x() * vs_normals[1]);

		F = (p - interpolP).cross(interpolN);
		Vector3d dFdu = (-dPdu).cross(interpolN) + (p - interpolP).cross(dNdu);
		Vector3d dFdv = (-dPdv).cross(interpolN) + (p - interpolP).cross(dNdv);

		jacobian.col(0) = dFdu;
		jacobian.col(1) = dFdv;

		//std::cout << uv.transpose() << " => " << F.transpose() << std::endl;

		Vector2d rhs = -jacobian.transpose() * F;
		auto lhs = jacobian.transpose() * jacobian;
		float norm = 1.0f / (lhs(0, 0) * lhs(1, 1) - lhs(0, 1) * lhs(1, 0));

		uv += Vector2d(lhs(1, 1) * rhs.x() - lhs(0, 1) * rhs.y(), -lhs(1, 0) * rhs.x() + lhs(0, 0) * rhs.y()) * norm;
	}

	interpolP = bilinear(quad_vs[0], quad_vs[1], quad_vs[2], quad_vs[3], uv);
	interpolN = bilinear(vs_normals[0], vs_normals[1], vs_normals[2], vs_normals[3], uv);
}
void projectPointOnTriangle(const vector<Vector3d>& tri_vs, const vector<Vector3d> & vs_normals, const Vector3d& p, Vector2d& uv, Vector3d& interpolP, Vector3d& interpolN)
{
	Eigen::Matrix<double, 3, 2> jacobian;

	uv = Vector2d::Constant(0.333f);

	//Newton iteration for projection on triangle

	//objective function:
	// F(u, v) = (p - interpolP) x interpolN = 0
	Vector3d F;

	Vector3d dPdu = tri_vs[0] - tri_vs[2];
	Vector3d dPdv = tri_vs[1] - tri_vs[2];
	Vector3d dNdu = vs_normals[0] - vs_normals[2];
	Vector3d dNdv = vs_normals[1] - vs_normals[2];

	for (int i = 0; i < 20; ++i)
	{
		interpolP = barycentric(tri_vs[0], tri_vs[1], tri_vs[2], uv);
		interpolN = barycentric(vs_normals[0], vs_normals[1], vs_normals[2], uv);

		F = (p - interpolP).cross(interpolN);
		Vector3d dFdu = (-dPdu).cross(interpolN) + (p - interpolP).cross(dNdu);
		Vector3d dFdv = (-dPdv).cross(interpolN) + (p - interpolP).cross(dNdv);

		jacobian.col(0) = dFdu;
		jacobian.col(1) = dFdv;

		//std::cout << uv.transpose() << " => " << F.transpose() << std::endl;

		Vector2d rhs = -jacobian.transpose() * F;
		auto lhs = jacobian.transpose() * jacobian;
		float norm = 1.0f / (lhs(0, 0) * lhs(1, 1) - lhs(0, 1) * lhs(1, 0));

		uv += Vector2d(lhs(1, 1) * rhs.x() - lhs(0, 1) * rhs.y(), -lhs(1, 0) * rhs.x() + lhs(0, 0) * rhs.y()) * norm;
	}

	interpolP = barycentric(tri_vs[0], tri_vs[1], tri_vs[2], uv);
	interpolN = barycentric(vs_normals[0], vs_normals[1], vs_normals[2], uv);
}

void nearest_point_loop(const MatrixXd &P, const MatrixXd &V, VectorXi &I) {
	I.resize(P.rows());
	for (int i = 0; i < P.rows(); i++) {
		double max_dis = (P.row(i)-V.row(0)).norm(); int id=0;
		for (int j = 1; j < V.rows(); j++) {
			double dis = (P.row(i) - V.row(j)).norm();
			if (max_dis > dis) {
				max_dis = dis;
				id = j;
			}
		}
		I[i] = id;
	}
}
void hausdorff_dis(Mesh &mesh0, Mesh & mesh1) {
	MatrixXd A = mesh0.V.transpose(), B = mesh1.V.transpose();
	MatrixXi FA(mesh0.Fs.size(), 3), FB(mesh1.Fs.size(), 3);
	for (uint32_t i = 0; i < mesh0.Fs.size(); i++)for (uint32_t j = 0; j < 3; j++) FA(i, j) = mesh0.Fs[i].vs[j];
	for (uint32_t i = 0; i < mesh1.Fs.size(); i++)for (uint32_t j = 0; j < 3; j++) FB(i, j) = mesh1.Fs[i].vs[j];
	double dis = 0;
	igl::hausdorff(A, FA, B, FB, dis);
	double diagonal = 0;
	diagonal = igl::bounding_box_diagonal(A);
	//cout <<dis <<" "<<diagonal<<" "<< dis / diagonal << endl;
	cout << dis / diagonal << endl;
}
bool hausdorff_dis(Mesh &mesh0, Mesh & mesh1, double &hausdorff_ratio_threshould) {
	MatrixXd A = mesh0.V.transpose(), B = mesh1.V.transpose();
	MatrixXi FA(mesh0.Fs.size(), 3), FB(mesh1.Fs.size(), 3);
	double ave_length = average_edge_length(mesh0);

	for (uint32_t i = 0; i < mesh1.Fs.size(); i++)for (uint32_t j = 0; j < 3; j++) FB(i, j) = mesh1.Fs[i].vs[j];
	double dis = 0;
	igl::hausdorff(A, FA, B, FB, dis);
	
	double ratio = dis / ave_length;
	cout << ratio << endl;

	if (ratio < hausdorff_ratio_threshould) return true;
	return false;
}
bool hausdorff_dis(Mesh &mesh0, Mesh & mesh1, vector<int> & outlierVs, double &hausdorff_dis_threshold) {
	MatrixXd A = mesh0.V.transpose(), B = mesh1.V.transpose();
	MatrixXi FA(mesh0.Fs.size(), 3), FB(mesh1.Fs.size(), 3);

	for (uint32_t i = 0; i < mesh0.Fs.size(); i++)for (uint32_t j = 0; j < 3; j++) FA(i, j) = mesh0.Fs[i].vs[j];
	for (uint32_t i = 0; i < mesh1.Fs.size(); i++)for (uint32_t j = 0; j < 3; j++) FB(i, j) = mesh1.Fs[i].vs[j];

	VectorXd sqr_DAB;
	VectorXi  I0;
	MatrixXd C0;
	igl::point_mesh_squared_distance(A, B, FB, sqr_DAB, I0, C0);

	VectorXd sqr_DBA;
	VectorXi  I1;
	MatrixXd C1;
	igl::point_mesh_squared_distance(B, A, FA, sqr_DBA, I1, C1);

	double threshold = hausdorff_dis_threshold*hausdorff_dis_threshold;
	vector<bool> flag(mesh1.Vs.size(), false);
	while (!outlierVs.size()) {
		for (uint32_t i = 0; i < sqr_DAB.size(); i++) {
			if (sqr_DAB[i] > threshold) {
				for (auto vid : mesh1.Fs[I0[i]].vs) if (!flag[vid]) {
					outlierVs.push_back(vid);
					flag[vid] = true;
				}
			}
		}
		for (uint32_t i = 0; i < sqr_DBA.size(); i++) {
			if (sqr_DBA[i] > threshold) if (!flag[i]) {
				outlierVs.push_back(i); flag[i] = true;
			}
		}

		cout << "refered total: " << outlierVs.size() << " " << mesh1.Vs.size() << endl;
		threshold *= 0.9;
	}
	return true;
}


Float rescale(Mesh &mesh, Float scaleI, bool inverse = false) {
	if (!inverse) {
		RowVector3d c; c.setZero();
		for (uint32_t i = 0; i < mesh.V.cols(); i++)
			c += mesh.V.col(i);
		c /= mesh.V.cols();
		for (uint32_t i = 0; i < mesh.V.cols(); i++)
			mesh.V.col(i) -= c;
		Vector3d min_ = mesh.V.rowwise().minCoeff();
		Vector3d max_ = mesh.V.rowwise().maxCoeff();

		double diagonal_local = (max_ - min_).norm();
		double scale = 5.0 / diagonal_local;
		mesh.V *= scale;
		return scale;
	}
	else {
		mesh.V *= scaleI;
	}
	return 1.0;
}
void translate_rescale(const Mesh &ref, Mesh &mesh) {
	Vector3d cr; cr.setZero();
	for (uint32_t i = 0; i < ref.V.cols(); i++)
		cr += ref.V.col(i);
	cr /= ref.V.cols();

	Vector3d c; c.setZero();
	for (uint32_t i = 0; i < mesh.V.cols(); i++)
		c += mesh.V.col(i);
	c /= mesh.V.cols();
	for (uint32_t i = 0; i < mesh.V.cols(); i++)
		mesh.V.col(i) = mesh.V.col(i) - c + cr;

	Vector3d min_r = ref.V.rowwise().minCoeff();
	Vector3d max_r = ref.V.rowwise().maxCoeff();

	Vector3d min_ = mesh.V.rowwise().minCoeff();
	Vector3d max_ = mesh.V.rowwise().maxCoeff();

	double diagonal_r = (max_r - min_r).norm();
	double diagonal_local = (max_ - min_).norm();
	double scale = diagonal_r / diagonal_local;
	//mesh.V *= scale;

	for (auto &v : mesh.Vs)for (int i = 0; i < v.v.size(); i++)v.v[i] = mesh.V(i, v.id);
}
void compute_referenceMesh(MatrixXd &V, vector<Hybrid_F> &F, vector<uint32_t> &Fs, vector<MatrixXd> &Vout, bool square) {
	vector<MatrixXd>().swap(Vout);

	for (auto &fid : Fs) {
		vector<MatrixXd> vout;

		quad2square(V, F[fid].vs, vout, square);
		Vout.insert(Vout.end(), vout.begin(), vout.end());
	}
}
void quad2square(MatrixXd &V, const vector<uint32_t> &vs, vector<MatrixXd> &vout, bool square) {
	Vector3d v0 = V.row(vs[0]);
	Vector3d v1 = V.row(vs[1]);
	Vector3d v2 = V.row(vs[2]);
	Vector3d v3 = V.row(vs[3]);
	Vector3d vec012 = ( v0- v1).cross(v2 - v1);
	Vector3d vec032 = (v0 - v3).cross(v2 - v3);
	double volume = 0;
	volume += std::sqrt(vec012.dot(vec012));
	volume += std::sqrt(vec032.dot(vec032));
	
	volume /= 2;
	//three types of edges
	double e0 = 0, e1 = 0;

	if (square && abs(volume)>1e-20) {
		e0 = std::sqrt(volume);
		e1 = std::sqrt(volume);
	}
	else {
		e0 += (V.row(vs[0]) - V.row(vs[1])).norm();
		e0 += (V.row(vs[3]) - V.row(vs[2])).norm();
		e0 /= 2;
		e1 += (V.row(vs[0]) - V.row(vs[3])).norm();
		e1 += (V.row(vs[1]) - V.row(vs[2])).norm();
		e1 /= 2;
		double ratio = std::sqrt(volume / (e0*e1));

		e0 *= ratio;
		e1 *= ratio;
	}

	//eight vertices
	MatrixXd v4(4, 3);
	v4 << 0, 0, 0,
		e0, 0, 0,
		e0, e1, 0,
		0, e1, 0;
	//eight tets
	for (uint32_t i = 0; i < 4; i++) {
		MatrixXd tet(3, 3);
		for (uint32_t j = 0; j < 3; j++)
			tet.row(j) = v4.row(quad_tri_table[i][j]);
		vout.push_back(tet);
	}
}

void compute_referenceMesh(MatrixXd &V, vector<Hybrid> &H, vector<bool> &H_flag, vector<uint32_t> &Hs, vector<MatrixXd> &Vout, bool cube) {
	vector<MatrixXd>().swap(Vout);

	for (int i = 0; i < Hs.size();i++) {
		const auto & hid = Hs[i];
		vector<MatrixXd> vout;
		bool h_flag = H_flag[hid];
		hex2cuboid_(V, H[hid].vs, vout, cube, h_flag);
		Vout.insert(Vout.end(), vout.begin(), vout.end());
	}
}
void hex2cuboid_(MatrixXd &V, const vector<uint32_t> &vs, vector<MatrixXd> &vout, bool &cube, bool &h_flag) {
	MatrixXd v8(8, 3);
	double e0 = 0, e1 = 0, e2 = 0;
	
	if(!h_flag)
	{
		for(int i=0;i<8;i++)
			v8.row(i) = V.row(vs[i]);
	}else{
		double volume = 0;
		hex2tet24(V, vs, volume);

		if (cube && abs(volume)>1e-20) {
				//volume = vol;
				e0 = std::cbrt(volume);
				e1 = std::cbrt(volume);
				e2 = std::cbrt(volume);
				//cout << "volume " << volume << endl;
		}
		else {
			e0 += (V.row(vs[0]) - V.row(vs[1])).norm();
			e0 += (V.row(vs[3]) - V.row(vs[2])).norm();
			e0 += (V.row(vs[4]) - V.row(vs[5])).norm();
			e0 += (V.row(vs[7]) - V.row(vs[6])).norm();
			e0 /= 4;
			e1 += (V.row(vs[0]) - V.row(vs[3])).norm();
			e1 += (V.row(vs[1]) - V.row(vs[2])).norm();
			e1 += (V.row(vs[4]) - V.row(vs[7])).norm();
			e1 += (V.row(vs[5]) - V.row(vs[6])).norm();
			e1 /= 4;
			e2 += (V.row(vs[0]) - V.row(vs[4])).norm();
			e2 += (V.row(vs[1]) - V.row(vs[5])).norm();
			e2 += (V.row(vs[2]) - V.row(vs[6])).norm();
			e2 += (V.row(vs[3]) - V.row(vs[7])).norm();
			e2 /= 4;

			double ratio = std::cbrt(volume / (e0*e1*e2));
				
			e0 *= ratio;
			e1 *= ratio;
			e2 *= ratio;
		}

		v8 << 0, 0, 0,
		e0, 0, 0,
		e0, e1, 0,
		0, e1, 0,
		0, 0, e2,
		e0, 0, e2,
		e0, e1, e2,
		0, e1, e2;
	}

	//eight tets
	for (uint32_t i = 0; i < 8; i++) {
		MatrixXd tet(4, 3);
		for (uint32_t j = 0; j < 4; j++)
			tet.row(j) = v8.row(hex_tetra_table[i][j]);
		vout.push_back(tet);
	}
}

void compute_referenceMesh(MatrixXd &V, vector<Hybrid> &H, vector<uint32_t> &Hs, vector<MatrixXd> &Vout) {
	vector<MatrixXd>().swap(Vout);

	for (auto hid : Hs) {
		vector<MatrixXd> vout;
		hex2cuboid(V, H[hid].vs, vout);
		Vout.insert(Vout.end(),vout.begin(), vout.end());
	}
}
void compute_referenceMesh(MatrixXd &V, vector<Hybrid> &H, vector<uint32_t> &Hs, vector<MatrixXd> &Vout, bool cube, vector<double> &Vols) {
	vector<MatrixXd>().swap(Vout);

	for (int i = 0; i < Hs.size();i++) {
		const auto & hid = Hs[i];
		vector<MatrixXd> vout;
		hex2cuboid(V, H[hid].vs, vout, cube, Vols[i]);
		Vout.insert(Vout.end(), vout.begin(), vout.end());
	}
}
void hex2cuboid(MatrixXd &V, const vector<uint32_t> &vs, vector<MatrixXd> &vout) {
	double volume = 0;
	hex2tet24(V, vs, volume);

	//three types of edges
	double e0 = 0, e1 = 0, e2 = 0;
	e0 += (V.row(vs[0]) - V.row(vs[1])).norm();
	e0 += (V.row(vs[3]) - V.row(vs[2])).norm();
	e0 += (V.row(vs[4]) - V.row(vs[5])).norm();
	e0 += (V.row(vs[7]) - V.row(vs[6])).norm();
	e0 /= 4;
	e1 += (V.row(vs[0]) - V.row(vs[3])).norm();
	e1 += (V.row(vs[1]) - V.row(vs[2])).norm();
	e1 += (V.row(vs[4]) - V.row(vs[7])).norm();
	e1 += (V.row(vs[5]) - V.row(vs[6])).norm();
	e1 /= 4;
	e2 += (V.row(vs[0]) - V.row(vs[4])).norm();
	e2 += (V.row(vs[1]) - V.row(vs[5])).norm();
	e2 += (V.row(vs[2]) - V.row(vs[6])).norm();
	e2 += (V.row(vs[3]) - V.row(vs[7])).norm();
	e2 /= 4;
	double ratio = std::cbrt(volume / (e0*e1*e2));
	//e0 *= ratio;
	//e1 *= ratio;
	//e2 *= ratio;
	e0 = std::cbrt(volume);
	e1 = std::cbrt(volume);
	e2 = std::cbrt(volume);

	//eight vertices
	MatrixXd v8(8, 3);
	v8 << 0, 0, 0,
		e0, 0, 0,
		e0, e1, 0,
		0, e1, 0,
		0, 0, e2,
		e0, 0, e2,
		e0, e1, e2,
		0, e1, e2;
	//eight tets
	for (uint32_t i = 0; i < 8; i++) {
		MatrixXd tet(4, 3);
		for (uint32_t j = 0; j < 4; j++)
			tet.row(j) = v8.row(hex_tetra_table[i][j]);
		vout.push_back(tet);
	}
}
void hex2cuboid(MatrixXd &V, const vector<uint32_t> &vs, vector<MatrixXd> &vout, bool cube, double vol) {
	double volume = 0;
	hex2tet24(V, vs, volume);
	//
	//three types of edges
	double e0 = 0, e1 = 0, e2 = 0;
	
	if (cube && abs(volume)>1e-20) {
		//volume = vol;
		e0 = std::cbrt(volume);
		e1 = std::cbrt(volume);
		e2 = std::cbrt(volume);
		//cout << "volume " << volume << endl;
	}
	else {
		e0 += (V.row(vs[0]) - V.row(vs[1])).norm();
		e0 += (V.row(vs[3]) - V.row(vs[2])).norm();
		e0 += (V.row(vs[4]) - V.row(vs[5])).norm();
		e0 += (V.row(vs[7]) - V.row(vs[6])).norm();
		e0 /= 4;
		e1 += (V.row(vs[0]) - V.row(vs[3])).norm();
		e1 += (V.row(vs[1]) - V.row(vs[2])).norm();
		e1 += (V.row(vs[4]) - V.row(vs[7])).norm();
		e1 += (V.row(vs[5]) - V.row(vs[6])).norm();
		e1 /= 4;
		e2 += (V.row(vs[0]) - V.row(vs[4])).norm();
		e2 += (V.row(vs[1]) - V.row(vs[5])).norm();
		e2 += (V.row(vs[2]) - V.row(vs[6])).norm();
		e2 += (V.row(vs[3]) - V.row(vs[7])).norm();
		e2 /= 4;

		double ratio = std::cbrt(volume / (e0*e1*e2));
		
		e0 *= ratio;
		e1 *= ratio;
		e2 *= ratio;
	}

	//eight vertices
	MatrixXd v8(8, 3);
	v8 << 0, 0, 0,
		e0, 0, 0,
		e0, e1, 0,
		0, e1, 0,
		0, 0, e2,
		e0, 0, e2,
		e0, e1, e2,
		0, e1, e2;
	//eight tets
	for (uint32_t i = 0; i < 8; i++) {
		MatrixXd tet(4, 3);
		for (uint32_t j = 0; j < 4; j++)
			tet.row(j) = v8.row(hex_tetra_table[i][j]);
		vout.push_back(tet);
	}
}
void hex2cuboid(MatrixXd &V, const vector<uint32_t> &vs, MatrixXd &vout, bool cube) {
	double volume = 0;
	hex2tet24(V, vs, volume);

	//three types of edges
	double e0 = 0, e1 = 0, e2 = 0;

	if (cube && abs(volume)>1e-20) {
		e0 = std::cbrt(volume);
		e1 = std::cbrt(volume);
		e2 = std::cbrt(volume);
	}
	else {
		e0 += (V.row(vs[0]) - V.row(vs[1])).norm();
		e0 += (V.row(vs[3]) - V.row(vs[2])).norm();
		e0 += (V.row(vs[4]) - V.row(vs[5])).norm();
		e0 += (V.row(vs[7]) - V.row(vs[6])).norm();
		e0 /= 4;
		e1 += (V.row(vs[0]) - V.row(vs[3])).norm();
		e1 += (V.row(vs[1]) - V.row(vs[2])).norm();
		e1 += (V.row(vs[4]) - V.row(vs[7])).norm();
		e1 += (V.row(vs[5]) - V.row(vs[6])).norm();
		e1 /= 4;
		e2 += (V.row(vs[0]) - V.row(vs[4])).norm();
		e2 += (V.row(vs[1]) - V.row(vs[5])).norm();
		e2 += (V.row(vs[2]) - V.row(vs[6])).norm();
		e2 += (V.row(vs[3]) - V.row(vs[7])).norm();
		e2 /= 4;

		double ratio = std::cbrt(volume / (e0*e1*e2));

		e0 *= ratio;
		e1 *= ratio;
		e2 *= ratio;
	}

	//eight vertices
	vout.resize(8, 3);
	vout << 0, 0, 0,
		e0, 0, 0,
		e0, e1, 0,
		0, e1, 0,
		0, 0, e2,
		e0, 0, e2,
		e0, e1, e2,
		0, e1, e2;
}
void hex2cuboid(double elen, MatrixXd &vout) {
	double e0 = 0, e1 = 0, e2 = 0;
	e0 = elen;
	e1 = elen;
	e2 = elen;

	//eight vertices
	vout.resize(8, 3);
	vout << 0, 0, 0,
		e0, 0, 0,
		e0, e1, 0,
		0, e1, 0,
		0, 0, e2,
		e0, 0, e2,
		e0, e1, e2,
		0, e1, e2;
}
void hex2tet24(MatrixXd &V, const vector<uint32_t> &vs, double & volume) {
	//6 face center
	vector<RowVector3d> fvs(6);
	for (int i = 0; i < 6; i++) {
		fvs[i].setZero();
		for (int j = 0; j < 4; j++) {
			int vid = vs[hex_face_table[i][j]];
			fvs[i] += V.row(vid);
		}
		fvs[i] /= 4;
	}
	//hex center
	RowVector3d hv; hv.setZero();
	for (int i = 0; i < 8; i++) hv += V.row(vs[i]);
	hv /= 8;
	//tet volume
	volume = 0;
	for (int i = 0; i < 6; i++)
		for (int j = 0; j < 4; j++) {
			int vid0 = vs[hex_face_table[i][j]], vid1 = vs[hex_face_table[i][(j + 1) % 4]];

			Matrix3d Jacobian;
			Jacobian.col(0) = V.row(vid0) - hv;
			Jacobian.col(1) = V.row(vid1) - hv;
			Jacobian.col(2) = fvs[i] - hv;
			volume += std::abs(Jacobian.determinant()/6);
		}

}
//===================================mesh inside/outside judgement==========================================
void points_inside_mesh(MatrixXd &Ps, Mesh &tmi, VectorXd &signed_dis) {
	Eigen::MatrixXd V = tmi.V.transpose();
	Eigen::MatrixXi F(tmi.Fs.size(),3);
	igl::AABB<Eigen::MatrixXd, 3> tree;
	Eigen::MatrixXd FN, VN, EN;
	Eigen::MatrixXi E;
	Eigen::VectorXi EMAP;

	for (uint32_t i = 0; i < tmi.Fs.size(); i++) {
		F(i, 0) = tmi.Fs[i].vs[0];
		F(i, 1) = tmi.Fs[i].vs[1];
		F(i, 2) = tmi.Fs[i].vs[2];
	}
	// Precompute signed distance AABB tree
	tree.init(V, F);
	// Precompute vertex,edge and face normals
	igl::per_face_normals(V, F, FN);
	igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, FN, VN);
	igl::per_edge_normals(V, F, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, FN, EN, E, EMAP);

	VectorXi I;
	MatrixXd N, C;
	// Bunny is a watertight mesh so use pseudonormal for signing
	signed_distance_pseudonormal(Ps, V, F, tree, FN, VN, EN, EMAP, signed_dis, I, C, N);
}
//===================================PCA Bounding Box==========================================
bool PCA_BBOX(MatrixXd &Ps, MatrixXd &T, VectorXd &S)
{
	if(!Ps.cols()) return false;

	Eigen::VectorXd center;
	center.resize(Ps.rows());
	center.setZero();
	for(int i=0;i<Ps.cols();i++)
		center += Ps.col(i);
	center /= Ps.cols();
	//difference matrix
	MatrixXd D(Ps.rows(),Ps.cols());
	for(int i=0;i<Ps.cols();i++)
		D.col(i) = Ps.col(i) - center;	
	//covariance matrix
	MatrixXd C(Ps.rows(),Ps.rows());
	C.setZero();
	for(int i=0;i<Ps.rows(); i++)
	{
		for(int j=0;j<Ps.rows(); j++)
		{
			for(int k=0;k<Ps.cols();k++)
			{
				C(i,j) += D(i, k) * D(j, k);
			}
			C(i,j)/=Ps.cols();
		}
	}
	SelfAdjointEigenSolver<Matrix3d> eigensolver(C);
   	if (eigensolver.info() != Success) return false;
   	
   	Eigen::MatrixXd dirs= eigensolver.eigenvectors();
	S.resize(3);
	S.setZero();
	for(int i=0;i<3; i++)
	{	
		double max_dis=0, min_dis = 0;
		for(int j=0;j<D.cols();j++)
		{
			double dis = D.col(j).dot(dirs.col(i));
			if(dis > max_dis)
				max_dis = dis;
			if(dis < min_dis)
				min_dis = dis;
		}
		S[i] = max_dis - min_dis;
	}
	Eigen::Vector3d dir0 = dirs.col(0), dir1 = dirs.col(1);
	dirs.col(2) = dir0.cross(dir1);
	T.setIdentity(4,4);
	T.block(0, 0, 3,3) = dirs.inverse();
	T.col(3).segment(0,3) = -center;
	return true;
}