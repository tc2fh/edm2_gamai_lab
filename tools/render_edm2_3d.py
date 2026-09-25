"""Render rotating 3D forecast GIFs for an export_flow_analysis.py output
directory, using the flow repo's OWN isolated PyVista renderer, unmodified
(docs/vivit_conditioning_plan.md Phase 5; team-lead request, 2026-09-13).

This does NOT import pyvista in-process: pyvista/vtk are not installed in
either this repo's or the flow repo's normal interpreter, by design --
tumor_flow.analysis.pyvista_renderer's own module docstring says it
"intentionally has no imports from tumor_flow so it can run in an isolated
uv environment with pinned PyVista/VTK wheels". Instead this subprocess-
invokes tumor_flow/analysis/pyvista_renderer.py the EXACT same way
tumor_flow.analysis.render_scan.render_3d_gifs() does (same uv isolation
flags, same pinned package versions, same CLI), so EDM2's GIFs are produced
by literally the same code as the flow repo's: same camera path, colours,
iso level, header/caption layout, frame count, and GIF durations. This
script itself needs no third-party imports at all (pure stdlib) -- only a
`uv` executable on PATH, so it can be run under any interpreter.

Reads an analysis directory's viz_index.json (written by
tools/export_flow_analysis.py) to find every viz/patient<pid>/scan<sid>/
folder, and renders render_3d.gif / render_3d_slow.gif into each from its
arrays.npz + metrics.json -- both already in exactly the shape
pyvista_renderer.py expects (density, target, spacing_xyz_mm; a metrics dict
with sample_dice_mean/sample_dice_sd/consensus_dice), since
tools/export_flow_analysis.py's write_viz_scan() writes the same schema the
flow repo's own Part B writer does.

Usage:
    python tools/render_edm2_3d.py --analysis-dir <export_flow_analysis.py --out dir> \\
        [--flow-repo D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet] \\
        [--only patient_id:target_scan_id[,patient_id:target_scan_id...]]
"""

import argparse
import json
import os
import shutil
import subprocess

DEFAULT_FLOW_REPO = 'D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet'

# Exactly tumor_flow.analysis.render_scan._PYVISTA_UV_PACKAGES -- kept in sync
# by hand since this script must not import anything from the flow repo (its
# own interpreter may not even have `tumor_flow` installed).
PYVISTA_UV_PACKAGES = (
    'pyvista==0.44.1',
    'vtk==9.3.1',
    'numpy==1.26.4',
    'imageio==2.34.1',
    'pillow==10.4.0',
)
PYVISTA_TIMEOUT_SECONDS = 600

#----------------------------------------------------------------------------

def render_3d_gifs(scan_dir, title, flow_repo, uv_executable=None, timeout=PYVISTA_TIMEOUT_SECONDS):
    """Mirrors tumor_flow.analysis.render_scan.render_3d_gifs() exactly (same
    isolation flags, same package pins, same pyvista_renderer.py CLI args),
    pointed at --flow-repo's copy of the script instead of importing it."""
    executable = uv_executable or shutil.which('uv')
    if executable is None:
        raise RuntimeError('uv executable is unavailable for PyVista 3D rendering')
    renderer_path = os.path.join(flow_repo, 'src', 'tumor_flow', 'analysis', 'pyvista_renderer.py')
    if not os.path.isfile(renderer_path):
        raise RuntimeError(f'pyvista_renderer.py not found at {renderer_path!r}; is --flow-repo correct?')

    command = [executable, 'run', '--isolated', '--no-project', '--python', '3.11']
    for package in PYVISTA_UV_PACKAGES:
        command.extend(('--with', package))
    command.extend((
        'python', renderer_path,
        '--data', os.path.join(scan_dir, 'arrays.npz'),
        '--metrics', os.path.join(scan_dir, 'metrics.json'),
        '--output-dir', scan_dir,
        '--title', title,
    ))
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f'PyVista renderer exited {result.returncode} for {scan_dir!r}: {detail}')

#----------------------------------------------------------------------------

def resolve_title(scan_dir, patient_id, scan_id, index_entry):
    metrics_path = os.path.join(scan_dir, 'metrics.json')
    with open(metrics_path) as f:
        doc = json.load(f)
    title = doc.get('title')
    if title:
        return title
    # Fallback for an analysis dir predating the 'title' field (contract
    # matches tumor_flow.analysis.visualizations._build_scan's convention).
    baseline_scan_id = index_entry.get('baseline_scan_id', doc.get('baseline_scan_id', '?'))
    delta_days = index_entry.get('delta_days', doc.get('delta_days', '?'))
    return f'Patient {patient_id}: {baseline_scan_id} -> {scan_id} ({delta_days} days)'

#----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir', required=True,
                    help="export_flow_analysis.py --out directory (has viz_index.json and viz/).")
    p.add_argument('--flow-repo', default=DEFAULT_FLOW_REPO)
    p.add_argument('--uv-executable', default=None)
    p.add_argument('--only', default=None,
                    help='Comma-separated patient_id:target_scan_id pairs to restrict to (default: all).')
    p.add_argument('--timeout', type=int, default=PYVISTA_TIMEOUT_SECONDS)
    args = p.parse_args()

    with open(os.path.join(args.analysis_dir, 'viz_index.json')) as f:
        viz_index = json.load(f)

    only = None
    if args.only:
        only = set()
        for spec in args.only.split(','):
            pid, _, tsid = spec.partition(':')
            if not pid or not tsid:
                raise SystemExit(f'--only expects PATIENT:TARGET_SCAN_ID, got {spec!r}')
            only.add((pid, tsid))

    n_rendered = 0
    n_skipped = 0
    for pid, block in viz_index.get('patients', {}).items():
        for sid, entry in block.get('scans', {}).items():
            if only is not None and (pid, sid) not in only:
                n_skipped += 1
                continue
            scan_dir = os.path.join(args.analysis_dir, 'viz', f'patient{pid}', f'scan{sid}')
            if not os.path.isfile(os.path.join(scan_dir, 'arrays.npz')):
                print(f'[warn] no arrays.npz for patient={pid} scan={sid} in {scan_dir!r}, skipping')
                n_skipped += 1
                continue
            title = resolve_title(scan_dir, pid, sid, entry)
            print(f'Rendering 3D GIFs: patient={pid} scan={sid} title={title!r} ...')
            render_3d_gifs(scan_dir, title, args.flow_repo, uv_executable=args.uv_executable, timeout=args.timeout)
            n_rendered += 1

    print(f'Rendered 3D GIFs for {n_rendered} scan(s) ({n_skipped} skipped) under {args.analysis_dir!r}')


if __name__ == '__main__':
    main()
