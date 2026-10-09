"""Attack tests for the sealed-noise component (fedgt/sealed_noise.py).

Each test is one thing a cheating client might try. The client controls
everything outside the box: its input, the ciphertext it relays, the nonces it
passes on, and whether it runs the box at all.
"""
import copy
import pickle

import numpy as np
import pytest
from scipy.stats import norm

from fedgt.sealed_noise import (BoxConfig, SealedNoiseBox, Server, Vendor, Rejected, Refused,
                                measurement_of)

SHAPES = (("q0", (64, 48)), ("v0", (16, 48)))
D = sum(a * b for _, (a, b) in SHAPES)


def make(clip=1.0, sigma=8.0, budget=1e9, box_cls=SealedNoiseBox, box_cfg=None, client=1):
    vendor = Vendor()
    cfg = BoxConfig(clip=clip, sigma=sigma, eps_budget=budget, shapes=SHAPES)
    server = Server(vendor.public_key, measurement_of(SealedNoiseBox), cfg)
    box = box_cls(box_cfg or cfg, client, vendor, server.sign_pk, server.enc_pk)
    return vendor, server, box


def update(rng, scale=1.0):
    return {n: rng.normal(0, scale, s) for n, s in SHAPES}


def flat(u):
    return np.concatenate([u[n].reshape(-1) for n, _ in SHAPES]).astype(np.float64)


def clipped(u, C):
    x = flat(u)
    return x * min(1.0, C / np.linalg.norm(x))


# ---------------------------------------------------------------- correct use
def test_honest_release_has_the_contracted_noise():
    rng = np.random.default_rng(0)
    _, server, box = make()
    server.enrol(1, box.certificate)
    u = update(rng, 0.1)
    out = server.receive(1, box.release(server.nonce(1, 1), u))
    resid = flat(out) - clipped(u, 1.0)
    assert abs(resid.std() / 8.0 - 1) < 0.03
    assert abs(resid.mean()) < 4 * 8.0 / np.sqrt(D)


# ---------------------------------------------------------------- under-noising
def test_box_with_smaller_sigma_is_rejected_at_enrolment():
    vendor = Vendor()
    approved = BoxConfig(clip=1.0, sigma=8.0, eps_budget=1e9, shapes=SHAPES)
    weak = BoxConfig(clip=1.0, sigma=4.0, eps_budget=1e9, shapes=SHAPES)
    server = Server(vendor.public_key, measurement_of(SealedNoiseBox), approved)
    box = SealedNoiseBox(weak, 1, vendor, server.sign_pk, server.enc_pk)
    with pytest.raises(Rejected, match="configuration"):
        server.enrol(1, box.certificate)


def test_modified_box_code_is_rejected():
    class HalfNoiseBox(SealedNoiseBox):
        def release(self, nonce, update):          # a box that would add half the noise
            return super().release(nonce, update)
    _, server, box = make(box_cls=HalfNoiseBox)
    with pytest.raises(Rejected, match="measurement"):
        server.enrol(1, box.certificate)


def test_certificate_from_a_fake_vendor_is_rejected():
    vendor, server, box = make()
    fake = Vendor()
    cert = fake.attest(measurement_of(SealedNoiseBox), box.certificate["box_pk"],
                       box.certificate["config"])
    with pytest.raises(Rejected, match="vendor"):
        server.enrol(1, cert)


# ---------------------------------------------------------------- noise selection, retries, replay
def test_second_release_for_the_same_nonce_is_refused():
    rng = np.random.default_rng(1)
    _, server, box = make()
    server.enrol(1, box.certificate)
    n = server.nonce(1, 1)
    box.release(n, update(rng))
    with pytest.raises(Refused):
        box.release(n, update(rng))


def test_replayed_release_is_rejected():
    rng = np.random.default_rng(2)
    _, server, box = make()
    server.enrol(1, box.certificate)
    ct = box.release(server.nonce(1, 1), update(rng))
    server.receive(1, ct)
    with pytest.raises(Rejected):
        server.receive(1, ct)                        # same round
    server.nonce(1, 2)
    with pytest.raises(Rejected):
        server.receive(1, ct)                        # offered as the next round


def test_old_or_forged_nonces_are_refused():
    rng = np.random.default_rng(3)
    _, server, box = make()
    server.enrol(1, box.certificate)
    box.release(server.nonce(1, 5), update(rng))
    with pytest.raises(Refused, match="not after"):
        box.release(server.nonce(1, 4), update(rng))     # rewinding the round
    other = Server(Vendor().public_key, b"", server.config)
    with pytest.raises(Refused, match="signed"):
        box.release(other.nonce(1, 9), update(rng))       # nonce not from the pinned server
    with pytest.raises(Refused, match="another client"):
        box.release(server.nonce(2, 9), update(rng))


def test_box_cannot_be_cloned_or_rolled_back():
    _, server, box = make()
    with pytest.raises(Refused):
        copy.deepcopy(box)
    with pytest.raises(Refused):
        pickle.dumps(box)
    server.enrol(1, box.certificate)
    vendor2, _, box2 = make()                           # a fresh box for the same client
    with pytest.raises(Rejected):
        server.enrol(1, box2.certificate)


def test_client_cannot_read_or_alter_the_release():
    rng = np.random.default_rng(4)
    vendor, server, box = make()
    server.enrol(1, box.certificate)
    ct = box.release(server.nonce(1, 1), update(rng))
    tampered = list(ct)
    b = bytearray(tampered[3]); b[10] ^= 1; tampered[3] = bytes(b)
    with pytest.raises(Rejected, match="tampered"):
        server.receive(1, tampered)
    eavesdropper = Server(vendor.public_key, measurement_of(SealedNoiseBox), server.config)
    eavesdropper.enrol(1, box.certificate)
    eavesdropper.nonce(1, 1)
    with pytest.raises(Rejected):
        eavesdropper.receive(1, ct)                      # without the server's key it cannot decrypt


def test_release_not_from_the_enrolled_box_is_rejected():
    rng = np.random.default_rng(5)
    vendor = Vendor()
    cfg = BoxConfig(clip=1.0, sigma=8.0, eps_budget=1e9, shapes=SHAPES)
    server = Server(vendor.public_key, measurement_of(SealedNoiseBox), cfg)
    box_a = SealedNoiseBox(cfg, 1, vendor, server.sign_pk, server.enc_pk)
    box_b = SealedNoiseBox(cfg, 1, vendor, server.sign_pk, server.enc_pk)
    server.enrol(1, box_a.certificate)
    with pytest.raises(Rejected, match="signature"):
        server.receive(1, box_b.release(server.nonce(1, 1), update(rng)))


# ---------------------------------------------------------------- noise cancellation and bad inputs
def test_noise_is_fresh_and_cannot_be_cancelled():
    rng = np.random.default_rng(6)
    _, server, box = make(sigma=1.0, budget=1e9)
    server.enrol(1, box.certificate)
    u = update(rng, 1e-3)
    r1 = flat(server.receive(1, box.release(server.nonce(1, 1), u))) - clipped(u, 1.0)
    # worst case: the client somehow learned last round's noise and feeds its negative
    v = {n: -r1[:a * b].reshape(a, b) if i == 0 else -r1[64 * 48:].reshape(a, b)
         for i, (n, (a, b)) in enumerate(SHAPES)}
    r2 = flat(server.receive(1, box.release(server.nonce(1, 2), v))) - clipped(v, 1.0)
    assert abs(np.corrcoef(r1, r2)[0, 1]) < 0.05
    assert abs(r2.std() - 1.0) < 0.03


@pytest.mark.parametrize("bad", ["nan", "inf", "shape", "missing", "huge"])
def test_malformed_or_huge_inputs_stay_bounded(bad):
    rng = np.random.default_rng(7)
    _, server, box = make(sigma=1e-6, budget=1e300)
    server.enrol(1, box.certificate)
    u = update(rng)
    if bad == "nan":
        u["q0"][0, 0] = np.nan
    elif bad == "inf":
        u["v0"][1, 1] = np.inf
    elif bad == "shape":
        u["q0"] = u["q0"][:10]
    elif bad == "missing":
        u = None
    else:
        u = {n: a * 1e30 for n, a in u.items()}
    out = flat(server.receive(1, box.release(server.nonce(1, 1), u)))
    assert np.linalg.norm(out) <= 1.0 + 1e-3              # never more than the clip norm (+ tiny noise)
    if bad != "huge":
        assert np.linalg.norm(out) < 1e-3                 # malformed input releases noise only


# ---------------------------------------------------------------- privacy accounting
def test_budget_is_enforced_across_rounds():
    rng = np.random.default_rng(8)
    _, server, box = make(sigma=8.0, budget=2.0)
    server.enrol(1, box.certificate)
    k = server.config.max_releases()
    assert k >= 1 and server.config.eps_after(k) <= 2.0 < server.config.eps_after(k + 1)
    for t in range(1, k + 1):
        server.receive(1, box.release(server.nonce(1, t), update(rng)))
    with pytest.raises(Refused, match="budget"):
        box.release(server.nonce(1, k + 1), update(rng))


def test_missing_release_excludes_the_client():
    _, server, box = make()
    server.enrol(1, box.certificate)
    server.nonce(1, 1)
    server.missing(1)
    with pytest.raises(Rejected, match="excluded"):
        server.nonce(1, 2)


# ---------------------------------------------------------------- empirical privacy (covers relabelling)
def test_worst_case_inputs_are_no_more_distinguishable_than_the_gaussian_mechanism():
    """Whatever the client claims is learning or noise, the server sees only the
    release of an input clipped to norm C. Two worst-case inputs (opposite, huge)
    should be separable no better than a Gaussian mechanism with sensitivity 2C."""
    rng = np.random.default_rng(9)
    C, sigma, M = 1.0, 4.0, 300
    u = {n: rng.normal(0, 1, s) for n, s in SHAPES}
    big = {n: a * 1e6 for n, a in u.items()}
    neg = {n: -a for n, a in big.items()}
    d = clipped(big, C) - clipped(neg, C)
    dirn = d / np.linalg.norm(d)
    stats = {0: [], 1: []}
    for lab, x in ((0, neg), (1, big)):
        for i in range(M):
            _, server, box = make(clip=C, sigma=sigma, client=i)
            server.enrol(i, box.certificate)
            stats[lab].append(flat(server.receive(i, box.release(server.nonce(i, 1), x))) @ dirn)
    thr = np.quantile(stats[0], 0.95)
    power = np.mean(np.array(stats[1]) > thr)
    mu = 2 * C / sigma
    theory = norm.cdf(mu - norm.ppf(0.95))
    assert power < theory + 4 * np.sqrt(theory * (1 - theory) / M) + 0.03
