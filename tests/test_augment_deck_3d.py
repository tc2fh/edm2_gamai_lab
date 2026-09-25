"""Tests for tools/augment_deck_3d.py.

MUST run with the FLOW repo's interpreter plus python-pptx (imports
tumor_flow.analysis.build_comparison_deck.shared_scans and pptx):

    uv run --project D:/Work/GrowthNet_gamailab/FlowMatchingGrowthNet --with python-pptx python \
        -m pytest D:/Work/GrowthNet_gamailab/edm2_gamai_lab/tests/test_augment_deck_3d.py -v

Skipped entirely if pptx/tumor_flow are not importable.
"""

import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip('pptx')
pytest.importorskip('tumor_flow')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

import tools.augment_deck_3d as adk  # noqa: E402

#----------------------------------------------------------------------------
# GIF downsampling.

def _make_gif(path, n_frames, size):
    frames = [Image.new('RGB', size, color=(i * 10 % 255, 50, 100)) for i in range(n_frames)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0)


def test_downsample_gif_under_small_file_copied_unchanged(tmp_path):
    src = tmp_path / 'small.gif'
    _make_gif(src, n_frames=3, size=(20, 20))
    dst = tmp_path / 'out.gif'
    result = adk.downsample_gif_under(str(src), str(dst), max_bytes=10 * 1024 * 1024)
    assert result == str(dst)
    assert dst.read_bytes() == src.read_bytes()  # untouched copy, not re-encoded


def test_downsample_gif_under_shrinks_large_file(tmp_path):
    src = tmp_path / 'big.gif'
    _make_gif(src, n_frames=40, size=(300, 300))
    original_size = os.path.getsize(src)
    dst = tmp_path / 'out.gif'
    max_bytes = original_size // 4
    result = adk.downsample_gif_under(str(src), str(dst), max_bytes=max_bytes)
    assert os.path.isfile(result)
    assert os.path.getsize(result) < original_size
    # Still a valid, openable animated GIF.
    img = Image.open(result)
    assert img.n_frames >= 1

#----------------------------------------------------------------------------
# Eyebrow format must match build_comparison_deck.py's own f-string exactly.

class _FakeScan:
    def __init__(self, patient_id, scan_id, category, delta_days, old_scan=None, new_scan=None):
        self.patient_id = patient_id
        self.scan_id = scan_id
        self.category = category
        self.delta_days = delta_days
        self.old_scan = old_scan or {}
        self.new_scan = new_scan or {}


def test_eyebrow_for_matches_flow_format():
    scan = _FakeScan('289', '289_1_404', 'grew', 404)
    assert adk.eyebrow_for(scan) == 'PATIENT 289 - GREW - +404 d'

#----------------------------------------------------------------------------
# move_slide reordering.

def _blank_prs_with_labeled_slides(labels):
    prs = Presentation()
    layout = next(l for l in prs.slide_layouts if l.name == 'Blank')
    for label in labels:
        slide = prs.slides.add_slide(layout)
        box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(5), Inches(0.5))
        box.text_frame.text = label
    return prs


def _labels_in_order(prs):
    return [adk.slide_title_text(s) for s in prs.slides]


def test_move_slide_reorders_correctly():
    prs = _blank_prs_with_labeled_slides(['A', 'B', 'C', 'D'])
    adk.move_slide(prs, 3, 1)  # move 'D' to index 1
    assert _labels_in_order(prs) == ['A', 'D', 'B', 'C']

#----------------------------------------------------------------------------
# Full end-to-end: a starter deck with 2 per-scan slides + a fabricated pair
# of analysis dirs (viz_index.json + tiny render_3d.gif each), run main() via
# sys.argv, and check the companion slides land right after their originals
# with the right consensus-Dice captions.

def _write_analysis_dir(root, patient_id, scan_id, consensus_dice, category, delta_days):
    scan_dir = root / 'viz' / f'patient{patient_id}' / f'scan{scan_id}'
    scan_dir.mkdir(parents=True)
    _make_gif(scan_dir / 'render_3d.gif', n_frames=2, size=(16, 16))

    viz_index_path = root / 'viz_index.json'
    viz_index = json.loads(viz_index_path.read_text()) if viz_index_path.is_file() else dict(patients={})
    viz_index['patients'].setdefault(patient_id, dict(scans={}))['scans'][scan_id] = dict(
        category=category, delta_days=delta_days, metrics=dict(consensus_dice=consensus_dice))
    viz_index_path.write_text(json.dumps(viz_index))


def _build_starter_deck(path, scan_specs):
    """scan_specs: list of (patient_id, category, delta_days) for per-scan
    slides, in deck order, each surrounded by an unrelated filler slide."""
    prs = Presentation()
    layout = next(l for l in prs.slide_layouts if l.name == 'Blank')

    def add_text_slide(text):
        slide = prs.slides.add_slide(layout)
        box = slide.shapes.add_textbox(Inches(0.6), Inches(0.3), Inches(12), Inches(0.4))
        box.text_frame.text = text
        return slide

    add_text_slide('00 - TITLE')
    for patient_id, scan_id, category, delta_days in scan_specs:
        add_text_slide(f'PATIENT {patient_id} - {category.upper()} - +{delta_days} d')
    add_text_slide('08 - TAKEAWAYS')
    prs.save(str(path))


def test_augment_deck_end_to_end(tmp_path):
    old_dir = tmp_path / 'old'
    new_dir = tmp_path / 'new'
    old_dir.mkdir()
    new_dir.mkdir()

    scans = [('100', '100_1_10', 'grew', 10), ('200', '200_1_20', 'stable', 20)]
    for patient_id, scan_id, category, delta_days in scans:
        _write_analysis_dir(old_dir, patient_id, scan_id, consensus_dice=0.70, category=category, delta_days=delta_days)
        _write_analysis_dir(new_dir, patient_id, scan_id, consensus_dice=0.55, category=category, delta_days=delta_days)

    pptx_path = tmp_path / 'deck.pptx'
    _build_starter_deck(pptx_path, scans)
    out_path = tmp_path / 'deck_3d.pptx'

    argv = [
        'augment_deck_3d.py',
        '--pptx', str(pptx_path),
        '--old-analysis-dir', str(old_dir),
        '--new-analysis-dir', str(new_dir),
        '--old-label', 'Flow', '--new-label', 'EDM2',
        '--out', str(out_path),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        adk.main()
    finally:
        sys.argv = old_argv

    assert out_path.is_file()
    prs = Presentation(str(out_path))
    titles = [adk.slide_title_text(s) for s in prs.slides]

    # Original 4 slides + 2 companions = 6, each companion right after its scan slide.
    assert len(titles) == 6
    assert titles[0] == '00 - TITLE'
    assert titles[1] == 'PATIENT 100 - GREW - +10 d'
    assert titles[2] == 'PATIENT 100 - GREW - +10 d - 3D ROTATION'
    assert titles[3] == 'PATIENT 200 - STABLE - +20 d'
    assert titles[4] == 'PATIENT 200 - STABLE - +20 d - 3D ROTATION'
    assert titles[5] == '08 - TAKEAWAYS'

    # Companion slide text mentions both labels and their consensus Dice values.
    companion_slide = prs.slides[2]
    all_text = '\n'.join(
        shape.text_frame.text for shape in companion_slide.shapes if shape.has_text_frame
    )
    assert 'Flow' in all_text and '0.700' in all_text
    assert 'EDM2' in all_text and '0.550' in all_text
    # Two pictures embedded (the two downsampled GIFs).
    from pptx.shapes.picture import Picture
    pictures = [s for s in companion_slide.shapes if isinstance(s, Picture)]
    assert len(pictures) == 2


def test_augment_deck_missing_scan_slide_raises(tmp_path):
    old_dir = tmp_path / 'old'
    new_dir = tmp_path / 'new'
    old_dir.mkdir()
    new_dir.mkdir()
    _write_analysis_dir(old_dir, '999', '999_1_1', 0.5, 'stable', 1)
    _write_analysis_dir(new_dir, '999', '999_1_1', 0.5, 'stable', 1)

    pptx_path = tmp_path / 'deck.pptx'
    _build_starter_deck(pptx_path, [])  # no matching scan slide in the deck at all
    argv = [
        'augment_deck_3d.py',
        '--pptx', str(pptx_path),
        '--old-analysis-dir', str(old_dir),
        '--new-analysis-dir', str(new_dir),
        '--old-label', 'Flow', '--new-label', 'EDM2',
        '--out', str(tmp_path / 'deck_3d.pptx'),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit):
            adk.main()
    finally:
        sys.argv = old_argv
