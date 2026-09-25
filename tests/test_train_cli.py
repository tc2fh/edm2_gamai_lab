"""End-to-end test of the real train_edm2.py CLI path (as opposed to
test_training_loop.py, which drives training.training_loop.training_loop
directly with hand-built kwargs dicts).

Regression test for a real launch-time crash found by the gen-eval agent:
train_edm2.py's network_kwargs and training_loop.py's interface_kwargs both
defined `sigma_data`, so `construct_class_by_name(**network_kwargs,
**interface_kwargs)` raised "got multiple values for keyword argument
'sigma_data'" on every real invocation. test_training_loop.py's mini run
missed this because it builds network_kwargs by hand without a sigma_data
key at all.

Runs train_edm2.py in a SUBPROCESS (the current interpreter, via
sys.executable -- already the pixi interpreter under pytest), not via
click.testing.CliRunner in-process. Found in review: CliRunner calls
train_edm2.cmdline() in-process, which calls torch_utils.distributed.init()
-> torch_utils.training_stats.init_multiprocessing(), which asserts
`not _sync_called`; `_sync_called` is a module-level global that latches
True the first time any real training run reports status (e.g.
test_training_loop.py's mini run, or an earlier --status!=0 CLI test), and
is never reset. Two dist.init() calls in one process are then only safe by
accident of file/test ordering never putting a status-reporting run before
a second CliRunner invocation. A subprocess per launch sidesteps this (and
any other shared-global-state risk between launches) entirely, matching how
a real user actually invokes this CLI."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.pair_fixture import build_pair_fixture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / 'train_edm2.py'
RES = (128, 128, 64)

BASE_ARGS = [
    '--channels=16', '--channel-mult=1,2,2,4', '--num-blocks=2', '--channels-per-head=32',
    '--attn-resolutions=16', '--cross-attn-resolutions=16,32', '--dropout=0.1',
    '--duration=8', '--batch=2', '--batch-gpu=1',
    '--lr=0.01', '--decay=100', '--P_mean=-0.4', '--P_std=1.0',
    '--status=0', '--snapshot=1024', '--checkpoint=0',
]


# torch_utils/distributed.py's init() only fills these in `if key not in
# os.environ`, so it only picks a fresh free port / rank 0 / world size 1
# for a genuinely clean environment. If this pytest PROCESS has already
# called dist.init() itself (e.g. test_training_loop.py, which runs
# in-process, not via subprocess), these are already set in os.environ and
# a child subprocess inherits them by default -- including a MASTER_PORT
# that init() already bound a TCPStore server on, so the child's own
# init_process_group() fails to bind the SAME port ("Only one usage of
# each socket address..."). Strip them so every launch gets a fresh pick,
# exactly like a real, separately-invoked `python train_edm2.py ...` would.
_DIST_ENV_KEYS = ('MASTER_ADDR', 'MASTER_PORT', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE')


def _launch(args, log_path):
    # Redirect to a real FILE, not subprocess.PIPE (i.e. not capture_output=True).
    # Found in review: the default DataLoader (num_workers=2, spawned via
    # 'spawn' on Windows) can leave a worker process alive fractionally
    # longer than its parent train_edm2.py process, and that worker
    # inherits the pipe's write end too -- subprocess.run(capture_output=
    # True)'s communicate() blocks until EOF, i.e. until EVERY process
    # holding the pipe open has exited, so a slightly-lagging worker can
    # hang the test indefinitely even though the actual training run (and
    # its exit code) is long done. Waiting on a real file only waits for
    # the direct child (train_edm2.py itself) to exit.
    env = {k: v for k, v in os.environ.items() if k not in _DIST_ENV_KEYS}
    with open(log_path, 'w') as log_file:
        proc = subprocess.run(
            [sys.executable, str(TRAIN_SCRIPT)] + args,
            cwd=str(REPO_ROOT), stdout=log_file, stderr=subprocess.STDOUT, timeout=300, env=env,
        )
    return proc.returncode, log_path.read_text()


def _run(tmp_path, extra_args, run_name):
    pair_dir, tokens_dir, _samples = build_pair_fixture(tmp_path / 'data', shape=RES)
    outdir = tmp_path / run_name
    args = [
        f'--outdir={outdir}', f'--data={pair_dir}', f'--tokens-dir={tokens_dir}',
    ] + BASE_ARGS + extra_args
    returncode, output = _launch(args, tmp_path / f'{run_name}.log')
    assert returncode == 0, output
    return outdir, output


def test_cli_runs_to_completion_and_writes_snapshot(tmp_path):
    outdir, output = _run(tmp_path, [], 'run_default')
    snapshots = sorted(outdir.glob('network-snapshot-*.pkl'))
    assert len(snapshots) >= 1, output


def test_cli_with_sigma_data_override(tmp_path):
    outdir, output = _run(tmp_path, ['--sigma-data=0.75'], 'run_sigma_override')
    snapshots = sorted(outdir.glob('network-snapshot-*.pkl'))
    assert len(snapshots) >= 1, output

    import pickle
    with open(snapshots[0], 'rb') as f:
        data = pickle.load(f)
    assert data['dataset_kwargs']['sigma_data'] == pytest.approx(0.75)
    assert data['ema'].sigma_data == pytest.approx(0.75)


def test_cli_with_bf16_dtype(tmp_path):
    outdir, output = _run(tmp_path, ['--dtype=bf16'], 'run_bf16')
    snapshots = sorted(outdir.glob('network-snapshot-*.pkl'))
    assert len(snapshots) >= 1, output

    import pickle
    with open(snapshots[0], 'rb') as f:
        data = pickle.load(f)
    assert data['ema'].dtype == 'bf16'


def test_cli_with_no_context(tmp_path):
    # --no-context: token-free comparison arm (Phase 0's classifier probe
    # came back at chance for every ViT tap, per team-lead). Must build with
    # context_dim=0 / no cross-attention or FiLM params, skip token I/O
    # entirely, run to completion, and record use_context=False in the
    # checkpoint's C7 metadata so generate_forecasts.py can detect it.
    outdir, output = _run(tmp_path, ['--no-context'], 'run_no_context')
    snapshots = sorted(outdir.glob('network-snapshot-*.pkl'))
    assert len(snapshots) >= 1, output

    import pickle
    with open(snapshots[0], 'rb') as f:
        data = pickle.load(f)
    assert data['dataset_kwargs']['use_context'] is False

    net = data['ema']
    assert net.context_dim == 0
    assert net.unet.ctx_proj is None
    assert net.unet.emb_context is None
    assert net.unet.context_balance is None
    state_dict_keys = net.state_dict().keys()
    assert not any('cross_attn' in k for k in state_dict_keys)
    assert not any(k.startswith('unet.ctx_') or k.startswith('unet.emb_context') for k in state_dict_keys)

    # Forward still runs with context=None (the only way this net accepts it).
    x = torch.zeros(1, 1, *RES)
    sigma = torch.ones(1)
    cond_image = torch.zeros(1, net.cond_channels, *RES)
    delta_days = torch.zeros(1)
    with torch.no_grad():
        out = net(x, sigma, cond_image=cond_image, context=None, delta_days=delta_days, force_fp32=True)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_no_context_dataset_skips_token_loading(tmp_path):
    # Unit-level companion to test_cli_with_no_context: PairDataset itself
    # must never touch the token store when use_context=False, and must
    # omit context/context_mask/context_ages from the batch dict entirely
    # (rather than e.g. returning empty (0,256,768) tensors).
    from training.dataset import PairDataset
    pair_dir, tokens_dir, _samples = build_pair_fixture(tmp_path, shape=RES)
    ds = PairDataset(pair_dir, tokens_dir='/nonexistent/path/should/never/be/opened',
        split='train', target_encoding='binary', use_context=False)
    assert ds.use_context is False
    item = ds[0]
    assert 'context' not in item
    assert 'context_mask' not in item
    assert 'context_ages' not in item
    assert item['image'].shape == (1,) + RES
    assert ds.dataset_kwargs['use_context'] is False


def test_cli_resume_continues_from_saved_checkpoint(tmp_path):
    # Regression test for a real resume-blocking bug: torch_utils/
    # distributed.py's CheckpointIO.load() called torch.load() without
    # weights_only=False. On torch >= 2.6 (default flipped to True) this
    # raised "Unsupported global: GLOBAL dnnlib.util.EasyDict was not an
    # allowed global by default" on every resume, since the training-state
    # .pt holds EasyDict objects (state, dataset_kwargs). Drives two real
    # CLI launches into the same --outdir (as a user actually resumes) and
    # checks the SECOND launch's own log output, not just that it didn't
    # crash, since a silent restart from nimg 0 would also exit 0.
    #
    # Uses a tiny architecture and small resolution (unlike the other tests
    # in this file) purely for speed: reaching a real checkpoint boundary
    # needs checkpoint_nimg to be a multiple of 1024 (training_loop.py's
    # own granularity assert), which is far more forward/backward passes
    # than the other tests' --duration=8 smoke checks.
    small_res = (32, 32, 16)
    pair_dir, tokens_dir, _samples = build_pair_fixture(tmp_path / 'data', shape=small_res)
    outdir = tmp_path / 'run_resume'

    small_arch_args = [
        '--channels=8', '--channel-mult=1,2', '--num-blocks=1', '--channels-per-head=8',
        '--attn-resolutions=', '--cross-attn-resolutions=', '--dropout=0.0',
        '--lr=0.01', '--decay=100', '--P_mean=-0.4', '--P_std=1.0',
        '--status=0', '--snapshot=0',
    ]

    def launch(duration, checkpoint, log_name):
        args = [
            f'--outdir={outdir}', f'--data={pair_dir}', f'--tokens-dir={tokens_dir}',
        ] + small_arch_args + [
            f'--duration={duration}', '--batch=32', '--batch-gpu=32', f'--checkpoint={checkpoint}',
        ]
        return _launch(args, tmp_path / log_name)

    # Run 1: train to exactly one checkpoint boundary (1024 samples).
    returncode1, output1 = launch(duration=1024, checkpoint=1024, log_name='run1.log')
    assert returncode1 == 0, output1
    state_files = sorted(outdir.glob('training-state-*.pt'))
    assert len(state_files) >= 1, output1

    # Run 2: same --outdir, larger --duration. Must resume from the saved
    # state (1 kimg), not restart from 0.
    returncode2, output2 = launch(duration=2048, checkpoint=1024, log_name='run2.log')
    assert returncode2 == 0, output2
    assert 'Training from 1 kimg to 2 kimg' in output2, output2
    assert 'Training from 0 kimg' not in output2, output2
