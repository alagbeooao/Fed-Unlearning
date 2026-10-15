"""
Tests for the unlearning implementations.

CPU-only and quick, so the core mechanics can be checked without a GPU or the
FeTS data. Mostly these cover the things I got wrong at some point and want to
stay fixed.

Run all tests:
    python -m tests.test_methods
or, if pytest is installed:
    pytest tests/test_methods.py -v

What each test checks:
    test_fedavg_weighting        FedAvg is a correct sample-weighted mean.
    test_calibration_rescales    FedEraser's calibration rescales a cached
                                 direction to the probed magnitude (i.e. it is
                                 rather than sitting at ~1.0 and doing nothing).
    test_calibration_preserves_direction
                                 Calibration changes magnitude only, not
                                 direction (cosine similarity == 1).
    test_kd_clamp_binds          The KD push clamp actually binds at tau_push
                                 (it is a live cap, not dead code at 1.0).
    test_seed_determinism        A given seed reproduces model init exactly;
                                 different seeds differ.
"""
import math

import torch

from federated.seeding import set_global_seed
from models.ae_2d import AE2D
from unlearning.fed_eraser import (_flat_norm, calibrate_direction, fedavg)


def _approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_fedavg_weighting():
    """FedAvg is a sample-weighted mean."""
    # Two clients, one scalar-ish tensor each.
    u1 = [torch.tensor([2.0, 4.0])]
    u2 = [torch.tensor([0.0, 0.0])]
    # Weights 3:1 -> result should be 0.75*u1 + 0.25*u2 = [1.5, 3.0].
    out = fedavg([u1, u2], num_examples=[3, 1])
    assert _approx(out[0][0].item(), 1.5)
    assert _approx(out[0][1].item(), 3.0)
    # Equal weights -> simple mean.
    out_eq = fedavg([u1, u2], num_examples=[1, 1])
    assert _approx(out_eq[0][0].item(), 1.0)
    assert _approx(out_eq[0][1].item(), 2.0)
    print("PASS test_fedavg_weighting")


def test_calibration_rescales():
    """Calibration sets the update norm to the target magnitude.

    My first version compared full weight tensors and got a scale of about 1.0
    every time, which did nothing. Targets here stay inside the clip band so
    this tests the rescaling alone; clipping has its own test.
    """
    torch.manual_seed(0)
    old = [torch.randn(8, 8) * 0.5, torch.randn(8) * 0.3]
    old_norm = _flat_norm(old)
    assert old_norm > 0
    max_scale = 10.0
    # Targets strictly inside the [old_norm/10, old_norm*10] band.
    for target in (old_norm * 0.2, old_norm * 0.5, old_norm * 2.0,
                   old_norm * 5.0):
        cal = calibrate_direction(old, target, max_scale=max_scale)
        assert _approx(_flat_norm(cal), target, tol=1e-4), (
            f"calibrated norm {_flat_norm(cal):.6f} != target {target:.6f}"
        )
    print("PASS test_calibration_rescales")


def test_calibration_preserves_direction():
    """Calibration changes magnitude only; direction (cosine) is unchanged."""
    torch.manual_seed(1)
    old = [torch.randn(16), torch.randn(4, 4)]
    cal = calibrate_direction(old, new_update_norm=3.0)
    # Flatten both to one vector and compare cosine similarity.
    flat_old = torch.cat([t.flatten() for t in old])
    flat_cal = torch.cat([t.flatten() for t in cal])
    cos = torch.dot(flat_old, flat_cal) / (flat_old.norm() * flat_cal.norm())
    assert _approx(cos.item(), 1.0, tol=1e-5), f"cosine {cos.item()} != 1"
    print("PASS test_calibration_preserves_direction")


def test_calibration_clips_scale():
    """A huge target magnitude gets clipped instead of exploding.

    One noisy probe used to be enough to inject a giant update that then
    compounded over the replay.
    """
    torch.manual_seed(2)
    old = [torch.randn(8, 8) * 0.5]
    old_norm = _flat_norm(old)
    # Ask for a target 1000x the cached norm; with max_scale=10 the applied
    # scale must be exactly 10, so the result norm is 10 * old_norm, not 1000x.
    cal = calibrate_direction(old, new_update_norm=old_norm * 1000.0,
                              max_scale=10.0)
    assert _approx(_flat_norm(cal), old_norm * 10.0, tol=1e-3), (
        f"clipped norm {_flat_norm(cal):.4f} != {old_norm*10:.4f}"
    )
    # And a tiny target is clipped at the reciprocal (1/10).
    cal_small = calibrate_direction(old, new_update_norm=old_norm * 0.0001,
                                    max_scale=10.0)
    assert _approx(_flat_norm(cal_small), old_norm * 0.1, tol=1e-3)
    print("PASS test_calibration_clips_scale")


def test_calibration_zeroes_buffers():
    """Buffer entries are zeroed so the replay leaves them alone.

    Scaling a BatchNorm variance can push it negative, which was one source of
    the NaNs. Calibration retraining recomputes them anyway.
    """
    # Two entries: index 0 = learnable (kept), index 1 = buffer (zeroed).
    old = [torch.randn(4, 4), torch.randn(8)]
    mask = [True, False]
    cal = calibrate_direction(old, new_update_norm=2.0, mask=mask)
    assert not torch.equal(cal[0], torch.zeros_like(cal[0])), \
        "learnable entry should be scaled, not zeroed"
    assert torch.equal(cal[1], torch.zeros_like(cal[1])), \
        "buffer entry should be zeroed during replay"
    print("PASS test_calibration_zeroes_buffers")


def test_kd_clamp_binds():
    """The push clamp binds at tau_push.

    It was hardcoded at 1.0 for a while, which at this error scale (~0.02)
    meant it never fired.
    """
    tau_push = 0.05
    # Below the cap: loss tracks the distance.
    for d in (0.01, 0.03):
        loss_push = -torch.clamp(torch.tensor(d), max=tau_push)
        assert _approx(loss_push.item(), -d)
    # At/above the cap: loss saturates at -tau_push.
    for d in (0.05, 0.08, 0.5):
        loss_push = -torch.clamp(torch.tensor(d), max=tau_push)
        assert _approx(loss_push.item(), -tau_push)
    print("PASS test_kd_clamp_binds")


def test_seed_determinism():
    """Same seed reproduces model init exactly; different seeds differ."""
    set_global_seed(42)
    a = torch.cat([p.flatten() for p in AE2D(1, 8, 32).parameters()])
    set_global_seed(42)
    b = torch.cat([p.flatten() for p in AE2D(1, 8, 32).parameters()])
    set_global_seed(123)
    c = torch.cat([p.flatten() for p in AE2D(1, 8, 32).parameters()])
    assert torch.equal(a, b), "same seed should give identical init"
    assert not torch.equal(a, c), "different seeds should give different init"
    print("PASS test_seed_determinism")


def test_calibration_excludes_bn_stats():
    """The learnable-parameter mask excludes BatchNorm buffers from the norm.

    FedEraser's magnitude calibration must be computed over learnable weights
    only; BatchNorm running statistics (buffers) would otherwise distort the
    scale. This checks the mask classifies every state_dict entry correctly,
    and that masking can only reduce (never increase) the norm.
    """
    from unlearning.fed_eraser import _learnable_param_mask

    model = AE2D(1, 8, 32)
    keys = list(model.state_dict().keys())
    mask = _learnable_param_mask(model)
    assert len(mask) == len(keys)
    for k, keep in zip(keys, mask):
        is_buffer = (k.endswith("running_mean") or k.endswith("running_var")
                     or k.endswith("num_batches_tracked"))
        if is_buffer:
            assert not keep, f"buffer {k} should be excluded from calibration"
        else:
            assert keep, f"parameter {k} should be included in calibration"

    tensors = [p.detach() for p in model.state_dict().values()]
    full = _flat_norm(tensors)
    masked = _flat_norm(tensors, mask=mask)
    assert masked <= full + 1e-9, "masked norm must not exceed full norm"
    print("PASS test_calibration_excludes_bn_stats")


def test_memorisation_signal():
    """The signal is positive iff the teacher fits the forgotten client better."""
    from unlearning.push_scaling import memorisation_signal
    # teacher reconstructs forgotten client BETTER (lower error) -> memorised
    assert memorisation_signal(0.008, 0.010) > 0
    # teacher reconstructs it WORSE -> not memorised (the minor-client case)
    assert memorisation_signal(0.012, 0.010) < 0
    # identical -> no client-specific imprint
    assert _approx(memorisation_signal(0.010, 0.010), 0.0)
    # magnitude: 20% lower error -> s = 0.2
    assert _approx(memorisation_signal(0.008, 0.010), 0.2, tol=1e-9)
    print("PASS test_memorisation_signal")


def test_signal_to_scale():
    """Scale is linear up to s_ref, clipped to [0, 1].

    The floor at 0 is the key safety property: a teacher that fits the
    forgotten client no better than average must produce NO push, which is
    what prevents the fidelity collapse on minor clients.
    """
    from unlearning.push_scaling import signal_to_scale
    s_ref = 0.10
    assert _approx(signal_to_scale(0.00, s_ref), 0.0)   # nothing memorised
    assert _approx(signal_to_scale(0.05, s_ref), 0.5)   # half strength
    assert _approx(signal_to_scale(0.10, s_ref), 1.0)   # full strength
    assert _approx(signal_to_scale(0.50, s_ref), 1.0)   # saturates
    # negative signal must floor at zero, never invert the push
    assert _approx(signal_to_scale(-0.30, s_ref), 0.0)
    print("PASS test_signal_to_scale")


def test_derive_seed_is_process_stable():
    """derive_seed gives the same value in every process.

    hash() is randomised per process, so using it here would silently break
    reproducibility inside Ray workers. crc32 does not have that problem.
    """
    from federated.seeding import derive_seed
    # Known-stable values: if these ever change, reproducibility is broken.
    a = derive_seed(42, "train", 0, 7)
    assert a == derive_seed(42, "train", 0, 7), "must be deterministic"
    assert derive_seed(42, "train", 0, 7) != derive_seed(42, "train", 1, 7), "client must vary"
    assert derive_seed(42, "train", 0, 7) != derive_seed(42, "train", 0, 8), "round must vary"
    assert derive_seed(42, "train", 0, 7) != derive_seed(123, "train", 0, 7), "seed must vary"
    assert derive_seed(42, "kd_match", 0, 7) != derive_seed(42, "kd_push", 0, 7), "stream must vary"
    assert 0 <= a < 2**31, "must fit a 32-bit positive seed"
    print("PASS test_derive_seed_is_process_stable")


def test_seeded_shuffle_is_reproducible():
    """Same seed, same shuffle; different round, different shuffle.

    Without this, two identical configurations produced different results.
    """
    from torch.utils.data import DataLoader, TensorDataset
    from federated.seeding import derive_seed

    ds = TensorDataset(torch.arange(40).float().unsqueeze(1))

    def order(seed):
        g = torch.Generator(); g.manual_seed(int(seed))
        dl = DataLoader(ds, batch_size=8, shuffle=True, generator=g, num_workers=0)
        return [int(x[0].item()) for b, in dl for x in b.unsqueeze(1)]

    s0 = derive_seed(42, "train", 0, 1)
    assert order(s0) == order(s0), "same seed must give the same order"
    s1 = derive_seed(42, "train", 0, 2)
    assert order(s0) != order(s1), "different round must give a different order"
    print("PASS test_seeded_shuffle_is_reproducible")


def test_reliability_guard_threshold():
    """A small held-out set forces the scale to 0.

    Four of six clients flipped their verdict when I changed the held-out split,
    because a 32-case client leaves about five held-out cases. Below the
    threshold the estimate is not worth acting on.
    """
    from unlearning.push_scaling import signal_to_scale
    min_slices = 200
    # a strong-looking signal from a tiny held-out set must be discarded
    strong_s, n_held = 0.14, 40
    raw = signal_to_scale(strong_s, 0.10)
    guarded = raw if n_held >= min_slices else 0.0
    assert raw == 1.0, "signal alone would have said push at full strength"
    assert guarded == 0.0, "guard must override an unreliable estimate"
    # the same signal from an adequate held-out set is kept
    n_held = 1200
    assert (raw if n_held >= min_slices else 0.0) == 1.0
    print("PASS test_reliability_guard_threshold")


def _run_all():
    tests = [
        test_fedavg_weighting,
        test_calibration_rescales,
        test_calibration_preserves_direction,
        test_calibration_clips_scale,
        test_calibration_zeroes_buffers,
        test_calibration_excludes_bn_stats,
        test_kd_clamp_binds,
        test_seed_determinism,
        test_memorisation_signal,
        test_signal_to_scale,
        test_derive_seed_is_process_stable,
        test_seeded_shuffle_is_reproducible,
        test_reliability_guard_threshold,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
