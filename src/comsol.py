"""Read saved mesh data or prepare optional COMSOL 6.2 batch calculations."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _values(rows):
    return np.array([float(row["E_kV_per_mm"]) for row in rows])


def run_mesh_analysis(root, out):
    """Recompute peak shifts, mesh sizes, profile differences and missing probes."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "config/finite_element.json").read_text(encoding="utf-8"))
    old = _read_csv(root / "data/mesh/reference_presets/mesh_convergence.csv")
    _write_csv(out / "reference_preset_convergence.csv", old)
    preset_locations = []
    preset_peaks = _read_csv(root / "data/mesh/reference_presets/peak_sampling.csv")
    for row in old:
        records = [r for r in preset_peaks if r["mesh"] == row["mesh"]]
        chosen = next(r for r in records if int(r["lagrange_refine"]) == 5)
        preset_locations.append(dict(mesh=row["mesh"],
            E_kV_per_mm=float(chosen["peak_E_kV_per_mm"]),
            r_mm=float(chosen["r_mm_from_expression"]),
            z_mm=float(chosen["z_mm_from_expression"])))
    _write_csv(out / "reference_preset_peak_locations.csv", preset_locations)
    summary, finite_counts, changes = [], [], []
    grouped_tables = {}
    def local_rows(case_key, filename):
        # One CSV per measurement; group rows by (case, level).
        if filename not in grouped_tables:
            groups = {}
            for record in _read_csv(root / "data/mesh/local" / filename):
                groups.setdefault((record["case"], record["level"]), []).append(record)
            grouped_tables[filename] = groups
        return grouped_tables[filename][case_key]

    def local_profile(case_key, filename):
        rows = local_rows(case_key, filename)
        return rows, _values(rows)
    profile_names = ["hv_profile.csv", "hv_offset_0p05mm.csv",
                     "end_gas_offset_1e-5mm.csv", "end_gas_offset_0p05mm.csv",
                     "shared_corner_rays.csv"]
    for case in metadata["local_cases"]:
        previous = None
        for level in metadata["local_levels"]:
            folder = (case, level["name"])
            stats = local_rows(folder, "mesh_statistics.csv")[0]
            peaks = {r["scope"]: r for r in local_rows(folder, "peaks.csv")}
            floating = local_rows(folder, "floating_potential.csv")[0]
            end_edges = [float(r["length_mm"]) for r in local_rows(folder, "mesh_edges.csv")
                         if r["end_cap"].lower() == "true"]
            vertex_rows = local_rows(folder, "mesh_vertices.csv")
            vertices = {int(r["index"]): np.array([float(r["r_geometry_mm"]), float(r["z_geometry_mm"])])
                        for r in vertex_rows}
            triangles = local_rows(folder, "mesh_triangles.csv")
            # Gas triangles sharing at least one vertex with an end-cap boundary edge.
            boundary_vertices = {int(r[k]) for r in local_rows(folder, "mesh_edges.csv")
                                 if r["end_cap"].lower() == "true" for k in ["v0", "v1"]}
            incident_edges, incident_count = [], 0
            for triangle in triangles:
                ids = [int(triangle[k]) for k in ["v0", "v1", "v2"]]
                if boundary_vertices.intersection(ids):
                    incident_count += 1
                    incident_edges.extend(float(np.linalg.norm(vertices[ids[a]] - vertices[ids[b]]))
                                          for a, b in [(0, 1), (1, 2), (2, 0)])
            hv_rows, hv = local_profile(folder, "hv_profile.csv")
            in_window = np.array([250 <= float(r["requested_z_mm"]) <= 270 for r in hv_rows])
            row = dict(case=case, level=level["name"],
                end_cap_hmax_mm=float(stats["end_cap_hmax_mm"]), triangles=int(stats["triangles"]),
                vertices=int(stats["vertices"]), min_quality=float(stats["min_quality"]),
                mean_quality=float(stats["mean_quality"]), Vfloat_kV=float(floating["floating_V_kV"]),
                dofs=int(floating["solved_dofs"]), actual_cap_edge_min_mm=min(end_edges),
                actual_cap_edge_max_mm=max(end_edges),
                vertex_incident_triangle_count=incident_count,
                vertex_incident_edge_min_mm=min(incident_edges),
                vertex_incident_edge_max_mm=max(incident_edges),
                hv_window_ptp_kV_per_mm=float(np.ptp(hv[in_window])),
                hv_window_ptp_percent_mean=float(np.ptp(hv[in_window])/np.mean(hv[in_window])*100))
            for scope in ["gas", "hv", "floating"]:
                for field in ["E_kV_per_mm", "r_mm", "z_mm"]:
                    row[f"{scope}_{field}"] = float(peaks[scope][field])
            row["floating_vs_hv_percent"] = (row["floating_E_kV_per_mm"] / row["hv_E_kV_per_mm"] - 1)*100
            summary.append(row)
            for filename in profile_names:
                rows, values = local_profile(folder, filename)
                finite_counts.append(dict(case=case, level=level["name"], profile=filename,
                    rows=len(rows), finite=int(np.isfinite(values).sum()),
                    nonfinite=int((~np.isfinite(values)).sum())))
            if previous is not None:
                before = summary[-2]
                for scope in ["gas", "hv", "floating"]:
                    change = dict(case=case, previous_level=previous[1], level=level["name"],
                        quantity=scope + "_peak", maximum_absolute_profile_difference_kV_per_mm="",
                        relative_to_previous_maximum_percent="",
                        relative_to_current_maximum_percent="",
                        peak_scalar_change_percent=(row[scope+"_E_kV_per_mm"] / before[scope+"_E_kV_per_mm"]-1)*100,
                        peak_location_displacement_mm=math.hypot(
                            row[scope+"_r_mm"]-before[scope+"_r_mm"], row[scope+"_z_mm"]-before[scope+"_z_mm"]),
                        common_finite_points="")
                    changes.append(change)
                for filename in profile_names:
                    old_rows, old_values = local_profile(previous, filename)
                    new_rows, new_values = local_profile(folder, filename)
                    requested = ["requested_r_mm", "requested_z_mm"]
                    if len(old_rows) != len(new_rows) or any(
                        abs(float(a[k])-float(b[k])) > 1e-9
                        for a,b in zip(old_rows,new_rows) for k in requested):
                        raise ValueError(f"Inconsistent common probe grid: {case}/{filename}")
                    finite = np.isfinite(old_values) & np.isfinite(new_values)
                    difference = float(np.max(np.abs(new_values[finite]-old_values[finite])))
                    changes.append(dict(case=case, previous_level=previous[1], level=level["name"],
                        quantity=filename, maximum_absolute_profile_difference_kV_per_mm=difference,
                        relative_to_previous_maximum_percent=difference/float(np.max(np.abs(old_values[finite])))*100,
                        relative_to_current_maximum_percent=difference/float(np.max(np.abs(new_values[finite])))*100,
                        peak_scalar_change_percent="", peak_location_displacement_mm="",
                        common_finite_points=int(finite.sum())))
            previous = folder
    _write_csv(out / "local_mesh_peak_summary.csv", summary)
    _write_csv(out / "successive_mesh_comparisons.csv", changes)
    _write_csv(out / "profile_finite_counts.csv", finite_counts)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for case, label in [("sharp_rect", "Sharp floating end"), ("rounded_R2", "Rounded floating end, R=2 mm")]:
        rows = [r for r in summary if r["case"] == case]
        axis.plot([r["end_cap_hmax_mm"] for r in rows], [r["floating_E_kV_per_mm"] for r in rows], "o-", label=label)
    axis.set_xlabel("End-cap maximum element size (mm)")
    axis.set_ylabel("Floating-electrode peak field (kV/mm)")
    axis.invert_xaxis()
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "local_peak_refinement.png", dpi=300)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for level in metadata["local_levels"]:
        folder = ("rounded_R2", level["name"])
        rows, values = local_profile(folder, "hv_profile.csv")
        axes[0].plot([float(r["requested_z_mm"]) for r in rows], values, label=level["name"])
        rows, values = local_profile(folder, "end_gas_offset_1e-5mm.csv")
        # The first 401 points describe the LOWER rounded end, at a 0.00001 mm gas offset.
        axes[1].plot(np.linspace(180, 360, 401), values[:401], label=level["name"])
    axes[0].set(xlabel="Axial coordinate z (mm)", ylabel="Field (kV/mm)", title="High-voltage boundary")
    axes[1].set(xlabel="Lower-cap angle (degrees)", ylabel="Field (kV/mm)", title="Gas offset: 0.00001 mm")
    for axis in axes:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "common_grid_profiles.png", dpi=300)
    plt.close(figure)
    result = dict(reference_preset_count=len(old), locally_refined_case_count=len(summary),
        new_finite_element_solves=0, retained_nonfinite_probes=sum(r["nonfinite"] for r in finite_counts),
        geometry_scope=metadata["scope"], profile_offsets_mm=[0.00001, 0.05])
    (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


def _java_string(value):
    return json.dumps(str(value).replace("\\", "/"), ensure_ascii=True)


def extract_sample_records(model_path, output_path, kind):
    """Export the scalar records embedded in a completed sampling MPH file."""
    headers = {
        "morris": ["trajectory", "point", "changed_variable_index", "normalized_step",
                   "r_hv_proxy_mm", "d_mm", "r_g_proxy_mm", "l_m_mm", "h_m",
                   "l_s_proxy_mm", "theta_s_deg", "Emax_kV_per_mm"],
        "training": ["split", "index", "r_hv_eq_mm", "d_mm", "h_m", "l_m_mm",
                     "Emax_kV_per_mm", "Vfloat_kV"]}
    prefix = "MORRIS," if kind == "morris" else "DATA,"
    with zipfile.ZipFile(model_path) as archive:
        description = ET.fromstring(archive.read("dmodel.xml")).findtext("modelDescription") or ""
    records = [row.split(",")[1:] for row in description.splitlines() if row.startswith(prefix)]
    expected = 80 if kind == "morris" else 130
    if len(records) != expected:
        raise ValueError(f"Expected {expected} completed scalar records in the saved model")
    _write_csv(output_path, [dict(zip(headers[kind], row)) for row in records])
    return len(records)


def prepare_comsol(root, out, task, comsol_root=None, execute=False,
                   seeds=None, levels=None, cases=None):
    """Generate portable Java jobs; optionally compile and run with licensed COMSOL.

    ``export-fields`` reads the six supplied solved models and exports full fields
    without a study run. ``local-study`` rebuilds the reference end shapes and
    solves them. ``morris`` and ``training`` generate the stored sampling schemes.
    Generated sources and batch products are written only to ``out``.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or root in out.parents and str(out.relative_to(root)).startswith(("data", "models", "src", "config")):
        raise ValueError("Choose an output outside the supplied input directories")
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "config/finite_element.json").read_text(encoding="utf-8"))
    jobs = []
    if task in ["local-study", "export-fields"]:
        for case in (cases or metadata["local_cases"]):
            if case not in metadata["local_cases"]:
                raise ValueError("Unknown end geometry")
            for level in metadata["local_levels"]:
                if levels and level["name"] not in levels:
                    continue
                name = "GasCT" + ("Solve" if task == "local-study" else "Export") + ("Sharp" if case == "sharp_rect" else "Rounded") + level["name"]
                job_out = out / case / level["name"]
                model = root / ("models/comsol/reference_medium.mph" if task == "local-study"
                                else f"models/comsol/{case}_{level['name']}.mph")
                template = (root / "src/comsol_java/ReferenceMesh.java.template").read_text(encoding="utf-8")
                values = {"CLASS":name, "CASE":case, "LEVEL":level["name"], "HLOCAL":str(level["end_cap_hmax_mm"]),
                          "INPUT":_java_string(model), "OUTPUT":_java_string(job_out),
                          "MODE":"solve" if task == "local-study" else "export"}
                for key, value in values.items():
                    template = template.replace("__"+key+"__", value)
                jobs.append((name, template, job_out))
    elif task == "morris":
        morris = json.loads((root / "data/morris/metadata.json").read_text(encoding="utf-8"))
        for seed in (seeds or morris["design_seeds"]):
            name = "GasCTMorris"+str(seed)
            template = (root / "src/comsol_java/MorrisSamples.java.template").read_text(encoding="utf-8")
            jobs.append((name, template.replace("__CLASS__",name).replace("__SEED__",str(seed)), out))
    elif task == "training":
        name = "GasCTTrainingSamples"
        template = (root / "src/comsol_java/TrainingSamples.java.template").read_text(encoding="utf-8")
        jobs.append((name, template.replace("__CLASS__",name), out))
    else:
        raise ValueError("task must be local-study, export-fields, morris or training")
    commands = []
    compile_exe = batch_exe = None
    preferences = None
    if execute:
        if comsol_root is None:
            raise ValueError("Specify the COMSOL installation directory")
        installation = Path(comsol_root)
        compile_exe = next((p for p in [installation/"Multiphysics/bin/win64/comsolcompile.exe",
                           installation/"bin/win64/comsolcompile.exe", installation/"bin/comsolcompile"] if p.is_file()), None)
        batch_exe = next((p for p in [installation/"Multiphysics/bin/win64/comsolbatch.exe",
                         installation/"bin/win64/comsolbatch.exe", installation/"bin/comsolbatch"] if p.is_file()), None)
        if compile_exe is None or batch_exe is None:
            raise FileNotFoundError("Cannot find COMSOL compiler and batch executables in this installation")
        preferences = out / '.comsol_preferences'
        preferences.mkdir(parents=True, exist_ok=True)
        # Batch Java jobs read models and write data using this run's profile.
        # Restrict unrelated external capabilities in the same profile.
        (preferences/'comsol.prefs').write_text('\n'.join([
            'security.external.enable=on',
            'security.external.filepermission=full',
            'security.external.netpermission=off',
            'security.external.propertypermission=off',
            'security.external.reflectpermission=off',
            'security.external.runtimepermission=off',
            'security.external.securitypermission=off',
            'security.external.socketpermission=off',
            'security.comsol.allowexternalprocess=off',
            'security.comsol.allowexternallibraries=off',
        ])+'\n',encoding='utf-8')
    for name, source, job_out in jobs:
        build = out / "build" / name
        build.mkdir(parents=True, exist_ok=True)
        job_out.mkdir(parents=True, exist_ok=True)
        java = build / (name+".java")
        if "__" in source:
            raise ValueError("Unexpanded Java template token")
        java.write_text(source, encoding="utf-8")
        commands.append(dict(class_name=name, source=str(java), outputs=str(job_out)))
        if execute:
            class_file = build / (name+".class")
            class_file.unlink(missing_ok=True)
            compile_command = [str(compile_exe)]
            bundled_jdk = Path(comsol_root) / "java/win64/jre"
            if not bundled_jdk.is_dir():
                bundled_jdk = Path(comsol_root) / "Multiphysics/java/win64/jre"
            if bundled_jdk.is_dir():
                compile_command.extend(["-jdkroot", str(bundled_jdk)])
            subprocess.run(compile_command + [str(java)], cwd=build, check=True)
            if not class_file.is_file():
                raise RuntimeError("COMSOL compiler produced no class file")
            model_output=job_out/(name+'.mph')
            for previous_model in job_out.glob(name+'*.mph'):
                previous_model.unlink()
            required_outputs=[]
            if task in ['local-study','export-fields']:
                required_outputs=[job_out/n for n in ['mesh_statistics.csv','peaks.csv','gas_all_points.csv',
                    'floating_potential.csv','hv_profile.csv','end_gas_offset_0p05mm.csv']]
                for required in required_outputs:required.unlink(missing_ok=True)
            subprocess.run([str(batch_exe), '-prefsdir', str(preferences), "-inputfile", str(class_file),
                            "-outputfile", str(model_output), "-batchlog", str(build/"solver.log")],
                            cwd=build, check=True)
            # COMSOL may append the returned model tag to the requested basename.
            generated=sorted(job_out.glob(name+'*.mph'))
            if not generated or not all(zipfile.is_zipfile(p) for p in generated):
                raise RuntimeError(f'COMSOL did not create a saved model. See {build/"solver.log"}')
            missing=[p.name for p in required_outputs if not p.is_file() or p.stat().st_size==0]
            if missing:
                raise RuntimeError(f'COMSOL outputs missing: {missing}. See {build/"solver.log"}')
            commands[-1]['saved_models']=[str(p) for p in generated]
            if task in ["morris", "training"]:
                generated = sorted(job_out.glob(name+"*.mph"))
                if not generated:
                    raise RuntimeError("COMSOL batch produced no saved model")
                for candidate in generated:
                    try:
                        extract_sample_records(candidate, job_out/(name+"_samples.csv"), task)
                        break
                    except ValueError:
                        continue
                else:
                    raise RuntimeError("No saved model contains the complete sampling records")
    return dict(task=task, job_count=len(commands), executed=bool(execute), jobs=commands)
