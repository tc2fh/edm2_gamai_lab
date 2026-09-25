"""Tests for tools/render_edm2_3d.py's pure-Python logic (title resolution,
--only filtering, CLI wiring). Does NOT invoke the real PyVista subprocess
(requires network/uv-cached wheels and takes real wall-clock time) --
render_3d_gifs is monkeypatched to record calls instead."""

import json
import os
import sys

import pytest

import tools.render_edm2_3d as red

#----------------------------------------------------------------------------

def _write_viz_fixture(tmp_path, patients_scans):
    """patients_scans: {patient_id: {scan_id: {'title': ..., 'baseline_scan_id':..., 'delta_days':...}}}"""
    viz_index = dict(patients={})
    for pid, scans in patients_scans.items():
        viz_index['patients'][pid] = dict(scans={})
        for sid, meta in scans.items():
            scan_dir = tmp_path / 'viz' / f'patient{pid}' / f'scan{sid}'
            scan_dir.mkdir(parents=True)
            (scan_dir / 'arrays.npz').write_bytes(b'fake')  # existence check only
            metrics_doc = {'baseline_scan_id': meta.get('baseline_scan_id'), 'delta_days': meta.get('delta_days')}
            if meta.get('title') is not None:
                metrics_doc['title'] = meta['title']
            (scan_dir / 'metrics.json').write_text(json.dumps(metrics_doc))
            viz_index['patients'][pid]['scans'][sid] = dict(
                baseline_scan_id=meta.get('baseline_scan_id'), delta_days=meta.get('delta_days'))
    (tmp_path / 'viz_index.json').write_text(json.dumps(viz_index))
    return tmp_path

#----------------------------------------------------------------------------

def test_resolve_title_uses_metrics_json_title(tmp_path):
    fx = _write_viz_fixture(tmp_path, {'100': {'100_1_10': {'title': 'Custom Title', 'baseline_scan_id': '100_0_0', 'delta_days': 10}}})
    scan_dir = str(fx / 'viz' / 'patient100' / 'scan100_1_10')
    title = red.resolve_title(scan_dir, '100', '100_1_10', index_entry={})
    assert title == 'Custom Title'


def test_resolve_title_falls_back_to_convention(tmp_path):
    fx = _write_viz_fixture(tmp_path, {'100': {'100_1_10': {'title': None, 'baseline_scan_id': '100_0_0', 'delta_days': 10}}})
    scan_dir = str(fx / 'viz' / 'patient100' / 'scan100_1_10')
    index_entry = dict(baseline_scan_id='100_0_0', delta_days=10)
    title = red.resolve_title(scan_dir, '100', '100_1_10', index_entry)
    assert title == 'Patient 100: 100_0_0 -> 100_1_10 (10 days)'

#----------------------------------------------------------------------------

def test_main_renders_every_scan_and_respects_only(tmp_path, monkeypatch):
    fx = _write_viz_fixture(tmp_path, {
        '100': {'100_1_10': {'title': 'T1', 'baseline_scan_id': '100_0_0', 'delta_days': 10}},
        '200': {'200_1_20': {'title': 'T2', 'baseline_scan_id': '200_0_0', 'delta_days': 20}},
    })

    calls = []

    def fake_render_3d_gifs(scan_dir, title, flow_repo, uv_executable=None, timeout=600):
        calls.append((scan_dir, title, flow_repo))

    monkeypatch.setattr(red, 'render_3d_gifs', fake_render_3d_gifs)

    argv = ['render_edm2_3d.py', '--analysis-dir', str(fx), '--flow-repo', 'FAKE_FLOW_REPO']
    monkeypatch.setattr(sys, 'argv', argv)
    red.main()
    assert len(calls) == 2
    assert {c[1] for c in calls} == {'T1', 'T2'}
    assert all(c[2] == 'FAKE_FLOW_REPO' for c in calls)


def test_main_only_filters_to_requested_scan(tmp_path, monkeypatch):
    fx = _write_viz_fixture(tmp_path, {
        '100': {'100_1_10': {'title': 'T1', 'baseline_scan_id': '100_0_0', 'delta_days': 10}},
        '200': {'200_1_20': {'title': 'T2', 'baseline_scan_id': '200_0_0', 'delta_days': 20}},
    })
    calls = []
    monkeypatch.setattr(red, 'render_3d_gifs', lambda *a, **k: calls.append(a))
    argv = ['render_edm2_3d.py', '--analysis-dir', str(fx), '--only', '100:100_1_10']
    monkeypatch.setattr(sys, 'argv', argv)
    red.main()
    assert len(calls) == 1


def test_main_skips_scan_without_arrays_npz(tmp_path, monkeypatch, capsys):
    viz_index = dict(patients={'100': {'scans': {'100_1_10': dict(baseline_scan_id='100_0_0', delta_days=10)}}})
    (tmp_path / 'viz_index.json').write_text(json.dumps(viz_index))
    # No viz/patient100/scan100_1_10/arrays.npz created at all.
    calls = []
    monkeypatch.setattr(red, 'render_3d_gifs', lambda *a, **k: calls.append(a))
    argv = ['render_edm2_3d.py', '--analysis-dir', str(tmp_path)]
    monkeypatch.setattr(sys, 'argv', argv)
    red.main()
    assert len(calls) == 0
    assert 'skipping' in capsys.readouterr().out


def test_render_3d_gifs_raises_on_bad_flow_repo(tmp_path):
    with pytest.raises(RuntimeError, match='pyvista_renderer.py not found'):
        red.render_3d_gifs(str(tmp_path), 'title', flow_repo=str(tmp_path / 'nonexistent'))
