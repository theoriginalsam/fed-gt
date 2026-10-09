"""Sealed noise: a minimal component that clips and noises a client's update.

The audits in this project show that a server cannot certify self-reported
noise from the update it receives. This module takes the noise step away from
the client instead. It is a software reference implementation of the protocol;
in deployment the SealedNoiseBox would run inside a trusted execution
environment (TEE), and the parts marked SIMULATED would be provided by it.

Protocol (one release per client per round):

  1. Enrolment. The vendor (SIMULATED: a root signing key) attests the box's
     code measurement, its public signing key and its configuration (clip
     norm C, noise sigma, delta, privacy budget). The server pins that key to
     the client and checks measurement and configuration.
  2. Nonce. For each round the server issues a signed nonce bound to
     (client, round). The box accepts only nonces signed by the pinned server
     key, with a round strictly larger than any round it has released.
  3. Release. The box advances its monotonic counter (SIMULATED hardware
     counter) before it draws noise, clips the whole update jointly to norm C,
     adds fresh Gaussian noise from its own randomness, signs the result with
     the nonce, and encrypts it to the server's key. The client relays the
     ciphertext but cannot read it, so it never sees the noise it would want
     to cancel or select on. A missing or invalid input releases clip(0)+N.
  4. Accounting. The box refuses a release that would exceed its (eps, delta)
     budget, composing Gaussian releases exactly (mu-GDP: mu adds in quadrature).
  5. Server. Accepts a release only if the certificate chain, measurement,
     configuration, nonce, counter order and signature all verify, and the
     nonce has not been used. A client whose release goes missing is excluded
     from later rounds, so withholding can happen at most once per client.

Privacy (per release): any two inputs clip to points at distance <= 2C, so the
release is a Gaussian mechanism with sensitivity 2C, whatever computation the
client ran. The guarantee covers the content of the releases; the time at
which a client stops participating can carry at most log2(R + 1) bits over R
rounds and is not covered by the DP statement.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import struct
from dataclasses import dataclass, asdict

import numpy as np
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .privacy_calibration import delta_of_eps, eps_of_sigma


class Rejected(Exception):
    """The server refused a release."""


class Refused(Exception):
    """The box refused to release."""


def _raw(pk) -> bytes:
    return pk.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class BoxConfig:
    clip: float                 # C: joint Frobenius clip norm over all modules
    sigma: float                # Gaussian noise standard deviation per entry
    delta: float = 1e-5
    eps_budget: float = 8.0     # total (eps, delta) budget over all releases
    shapes: tuple = ()          # ((name, (d1, d2)), ...) fixed module layout

    def digest(self) -> bytes:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).digest()

    @property
    def mu(self) -> float:
        """GDP parameter of one release: sensitivity 2C over sigma."""
        return 2.0 * self.clip / self.sigma

    def eps_after(self, k: int) -> float:
        """Exact eps after k releases (mu-GDP composes as sqrt(k) mu)."""
        return 0.0 if k == 0 else _eps_of_mu(np.sqrt(k) * self.mu, self.delta)

    def max_releases(self) -> int:
        """Largest k with eps_after(k) <= eps_budget (eps grows with k)."""
        if self.eps_after(1) > self.eps_budget:
            return 0
        lo, hi = 1, 2
        while self.eps_after(hi) <= self.eps_budget and hi < 1 << 30:
            lo, hi = hi, hi * 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if self.eps_after(mid) <= self.eps_budget else (lo, mid)
        return lo


@functools.lru_cache(maxsize=64)
def _max_releases(config: "BoxConfig") -> int:
    return config.max_releases()


def _eps_of_mu(mu: float, delta: float) -> float:
    # eps_of_sigma takes (sigma, sensitivity): mu = sensitivity / sigma
    return eps_of_sigma(1.0 / mu, 1.0, delta)


# --------------------------------------------------------------------------- SIMULATED hardware
class _HardwareCounters:
    """SIMULATED monotonic counters (in a TEE: a hardware or TPM counter).

    Keyed by box identity; a copy or snapshot of a box shares the same counter,
    so restoring an old state cannot rewind it.
    """
    _values: dict = {}

    @classmethod
    def read(cls, box_id: bytes) -> tuple:
        return cls._values.get(box_id, (0, 0))          # (last round, releases)

    @classmethod
    def advance(cls, box_id: bytes, rnd: int) -> int:
        last, n = cls.read(box_id)
        if rnd <= last:
            raise Refused(f"round {rnd} not after last released round {last}")
        cls._values[box_id] = (rnd, n + 1)
        return n + 1


class Vendor:
    """SIMULATED attestation root (in a TEE: the hardware vendor's key)."""

    def __init__(self):
        self._sk = Ed25519PrivateKey.generate()
        self.public_key = self._sk.public_key()

    def attest(self, measurement: bytes, box_pk: bytes, config_digest: bytes) -> dict:
        body = measurement + box_pk + config_digest
        return {"measurement": measurement, "box_pk": box_pk, "config": config_digest,
                "sig": self._sk.sign(body)}


def measurement_of(cls) -> bytes:
    """Code measurement: hash of the box class source (in a TEE: MRENCLAVE or similar)."""
    return hashlib.sha256(inspect.getsource(cls).encode()).digest()


# --------------------------------------------------------------------------- the box
class SealedNoiseBox:
    """Clip, noise, sign and encrypt one update per server nonce."""

    CHUNK = 1 << 22             # entries per encrypted chunk

    def __init__(self, config: BoxConfig, client_id: int, vendor: Vendor,
                 server_sign_pk: Ed25519PublicKey, server_enc_pk: X25519PublicKey):
        self.config = config
        self.client_id = int(client_id)
        self._sk = Ed25519PrivateKey.generate()
        self._id = _raw(self._sk.public_key())
        self._server_sign_pk = server_sign_pk
        self._server_enc_pk = server_enc_pk
        self.certificate = vendor.attest(measurement_of(type(self)), self._id, config.digest())
        self._max = _max_releases(config)

    def __deepcopy__(self, memo):
        raise Refused("the box cannot be cloned")

    def __reduce__(self):
        raise Refused("the box cannot be serialised")

    # -- release ---------------------------------------------------------------
    def release(self, nonce: dict, update: dict | None):
        """Return the encrypted release for this nonce as a list of byte chunks."""
        cfg = self.config
        body = struct.pack(">QQ", nonce["client"], nonce["round"]) + nonce["rand"]
        try:
            self._server_sign_pk.verify(nonce["sig"], body)
        except InvalidSignature:
            raise Refused("nonce not signed by the server")
        if nonce["client"] != self.client_id:
            raise Refused("nonce issued to another client")
        if _HardwareCounters.read(self._id)[1] >= self._max:
            raise Refused("privacy budget exhausted")
        n_rel = _HardwareCounters.advance(self._id, nonce["round"])   # before any noise is drawn

        # validate without copying: every module present, right shape, finite
        x = None
        if isinstance(update, dict) and set(update) == {n for n, _ in cfg.shapes}:
            x, norm2 = {}, 0.0
            for name, shape in cfg.shapes:
                a = np.asarray(update[name])
                if a.shape != tuple(shape) or not np.all(np.isfinite(a)):
                    x = None
                    break
                x[name] = a
                norm2 += float(np.sum(np.square(a, dtype=np.float64)))
        if x is None:                                   # missing or malformed input
            x, norm2 = {name: None for name, _ in cfg.shapes}, 0.0
        scale = min(1.0, cfg.clip / np.sqrt(norm2)) if norm2 > 0 else 1.0

        rng = np.random.Generator(np.random.Philox(key=int.from_bytes(os.urandom(16), "big")))
        eph = X25519PrivateKey.generate()
        shared = eph.exchange(self._server_enc_pk)
        key = HKDF(hashes.SHA256(), 32, salt=nonce["rand"], info=b"sealed-noise").derive(shared)
        aes = AESGCM(key)
        header = json.dumps({"client": self.client_id, "round": nonce["round"],
                             "rand": nonce["rand"].hex(), "release": n_rel,
                             "config": cfg.digest().hex(), "shapes": [[n, list(s)] for n, s in cfg.shapes]}
                            ).encode()
        h = hashlib.sha256(header)
        chunks = [_raw(eph.public_key()), self._seal(aes, 0, header)]
        idx = 1
        for name, shape in cfg.shapes:
            size = int(np.prod(shape))
            a = x[name]
            flat = a.reshape(-1) if a is not None else None
            for start in range(0, size, self.CHUNK):
                stop = min(size, start + self.CHUNK)
                part = (flat[start:stop].astype(np.float64) * scale) if flat is not None else np.zeros(stop - start)
                part = (part + rng.normal(0.0, cfg.sigma, stop - start)).astype(np.float32)
                b = part.tobytes()
                h.update(b)
                chunks.append(self._seal(aes, idx, b))
                idx += 1
        sig = self._sk.sign(h.digest())
        chunks.append(self._seal(aes, idx, sig))
        return chunks

    @staticmethod
    def _seal(aes, idx, data):
        return aes.encrypt(struct.pack(">IQ", 0, idx), data, struct.pack(">Q", idx))


# --------------------------------------------------------------------------- server
class Server:
    def __init__(self, vendor_pk: Ed25519PublicKey, measurement: bytes, config: BoxConfig):
        self._vendor_pk = vendor_pk
        self._measurement = measurement
        self.config = config
        self._sign = Ed25519PrivateKey.generate()
        self._enc = X25519PrivateKey.generate()
        self.sign_pk = self._sign.public_key()
        self.enc_pk = self._enc.public_key()
        self._pinned: dict[int, Ed25519PublicKey] = {}
        self._issued: dict[bytes, tuple] = {}
        self._used: set = set()
        self._last_release: dict[int, int] = {}
        self.excluded: set = set()

    def enrol(self, client_id: int, cert: dict):
        try:
            self._vendor_pk.verify(cert["sig"], cert["measurement"] + cert["box_pk"] + cert["config"])
        except InvalidSignature:
            raise Rejected("certificate not signed by the vendor")
        if cert["measurement"] != self._measurement:
            raise Rejected("box code measurement does not match the approved box")
        if cert["config"] != self.config.digest():
            raise Rejected("box configuration (clip, sigma, budget) is not the approved one")
        if client_id in self._pinned:
            raise Rejected("client already enrolled with a box")
        self._pinned[client_id] = Ed25519PublicKey.from_public_bytes(cert["box_pk"])

    def nonce(self, client_id: int, rnd: int) -> dict:
        if client_id in self.excluded:
            raise Rejected("client excluded after a missing release")
        rand = os.urandom(16)
        body = struct.pack(">QQ", client_id, rnd) + rand
        self._issued[rand] = (client_id, rnd)
        return {"client": client_id, "round": rnd, "rand": rand, "sig": self._sign.sign(body)}

    def missing(self, client_id: int):
        """A release did not arrive by the deadline: exclude the client."""
        self.excluded.add(client_id)

    def receive(self, client_id: int, chunks: list) -> dict:
        if client_id not in self._pinned:
            raise Rejected("client not enrolled")
        if client_id in self.excluded:
            raise Rejected("client excluded")
        try:
            eph = X25519PublicKey.from_public_bytes(chunks[0])
            shared = self._enc.exchange(eph)
            aes_input = chunks[1:]
            header_ct = aes_input[0]
            # the salt is the nonce's random part, read from the header after decryption
            for rand, (cid, rnd) in list(self._issued.items()):
                if cid != client_id:
                    continue
                key = HKDF(hashes.SHA256(), 32, salt=rand, info=b"sealed-noise").derive(shared)
                aes = AESGCM(key)
                try:
                    header = aes.decrypt(struct.pack(">IQ", 0, 0), header_ct, struct.pack(">Q", 0))
                    break
                except InvalidTag:
                    continue
            else:
                raise Rejected("no outstanding nonce decrypts this release")
            meta = json.loads(header)
            if bytes.fromhex(meta["rand"]) != rand or meta["client"] != client_id or meta["round"] != rnd:
                raise Rejected("release not bound to the nonce")
            if rand in self._used:
                raise Rejected("nonce already used (replay)")
            if meta["config"] != self.config.digest().hex():
                raise Rejected("wrong configuration")
            if meta["release"] <= self._last_release.get(client_id, 0):
                raise Rejected("release counter did not advance")
            h = hashlib.sha256(header)
            out, idx = {}, 1
            for name, shape in self.config.shapes:
                size = int(np.prod(shape))
                parts = []
                for start in range(0, size, SealedNoiseBox.CHUNK):
                    b = aes.decrypt(struct.pack(">IQ", 0, idx), aes_input[idx], struct.pack(">Q", idx))
                    h.update(b)
                    parts.append(np.frombuffer(b, dtype=np.float32))
                    idx += 1
                out[name] = np.concatenate(parts).reshape(shape)
            sig = aes.decrypt(struct.pack(">IQ", 0, idx), aes_input[idx], struct.pack(">Q", idx))
            if idx + 1 != len(aes_input):
                raise Rejected("unexpected trailing data")
        except (InvalidTag, ValueError, IndexError, KeyError) as e:
            raise Rejected(f"malformed or tampered release: {type(e).__name__}")
        try:
            self._pinned[client_id].verify(sig, h.digest())
        except InvalidSignature:
            raise Rejected("signature does not match the client's enrolled box")
        self._used.add(rand)
        del self._issued[rand]
        self._last_release[client_id] = meta["release"]
        return out
