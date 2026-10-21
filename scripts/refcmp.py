#!/usr/bin/env python3
"""Measure reference captures (bevy_aurora::util::capture).

    scripts/refcmp.py DIR            one capture: the reference's convergence (its checkpoints), the noisy input's
                                     noise and clumping, RR's output against the reference
    scripts/refcmp.py DIR_A DIR_B    and B's reference against A's: the tracer's bias, no denoiser
    scripts/refcmp.py --png DIR      reference.png, noisy-0.png, output-0.png beside the PFMs

All values are linear radiance (nits). Errors are relative: |x - ref| / (ref + floor) over
16x16-pixel block means (where light lands), floor 5 % of the mean.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

LUMA = np.array([0.2126, 0.7152, 0.0722])


def load_pfm(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"PF"
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4")
    return data.reshape(h, w, 3)[::-1].astype(np.float64)


def luma(a):
    return a @ LUMA


def blocks(a, n=16):
    h, w = a.shape[:2]
    h, w = h // n * n, w // n * n
    return a[:h, :w].reshape(h // n, n, w // n, n, *a.shape[2:]).mean(axis=(1, 3))


def block_error(x, ref):
    bx, br = blocks(luma(x)), blocks(luma(ref))
    floor = max(br.mean() * 0.05, 1e-6)
    return float((np.abs(bx - br) / (br + floor)).mean())


def resize(a, w, h):
    return np.stack(
        [np.asarray(Image.fromarray(a[..., c].astype(np.float32), "F").resize((w, h), Image.BILINEAR)) for c in range(3)],
        axis=-1,
    ).astype(np.float64)


def frames(d, name):
    out = []
    k = 0
    while (d / f"{name}-{k}.pfm").exists():
        out.append(load_pfm(d / f"{name}-{k}.pfm"))
        k += 1
    return out


def autocorr(res, offsets=(1, 2, 4, 8, 16)):
    """Correlation of the residual with itself shifted by each offset (x and y averaged): 0 for
    independent per-pixel noise, toward 1 where it comes in clumps that size."""
    r = luma(res)
    r = r - r.mean()
    var = (r * r).mean()
    out = []
    for o in offsets:
        cx = (r[:, o:] * r[:, :-o]).mean()
        cy = (r[o:, :] * r[:-o, :]).mean()
        out.append(0.5 * (cx + cy) / max(var, 1e-30))
    return out


def convergence(d, ref, m):
    """From the checkpoints (reference-<n>.pfm at N/16 and N/4): how fast the reference's noise
    falls. Each step is 4x the frames; with independent frames the change between successive
    checkpoints halves (N^-0.5). Median and 99.9 %-clipped rms of the per-pixel change."""
    cps = sorted((int(p.stem.split("-")[1]), p) for p in d.glob("reference-*.pfm"))
    if len(cps) < 2:
        return
    imgs = [luma(load_pfm(p)) for _, p in cps] + [luma(ref)]
    def change(a, b):
        x = np.abs(a - b) / max(m, 1e-12)
        return np.median(x), np.sqrt((np.minimum(x, np.percentile(x, 99.9)) ** 2).mean())
    c1, c2 = change(imgs[-3], imgs[-2]), change(imgs[-2], imgs[-1])
    exps = [np.log(c1[i] / max(c2[i], 1e-30)) / np.log(4.0) for i in range(2)]
    print(f"  convergence ({cps[-2][0]} -> {cps[-1][0]} -> N frames): noise falls as N^-{exps[0]:.2f} (median), "
          f"N^-{exps[1]:.2f} (rms); independent frames N^-0.50")


def one(d):
    d = Path(d)
    ref = load_pfm(d / "reference.pfm")
    noisy = frames(d, "noisy")
    out = frames(d, "output")
    m = luma(ref).mean()
    print(f"{d}: reference mean {m:.4g} nits, {ref.shape[1]}x{ref.shape[0]}")
    if noisy:
        n = np.stack([luma(x) for x in noisy])
        # Per-pixel spread of single frames around the reference, and the reference's own standard
        # error from it (frames in capture.json).
        import json

        count = json.loads((d / "capture.json").read_text()).get("frames", 1)
        sigma = np.sqrt(((n - luma(ref)[None]) ** 2).mean(axis=0))
        stderr = float((sigma / np.sqrt(max(count, 1))).mean() / max(m, 1e-12))
        print(f"  reference: {count} frames, relative standard error ~{stderr:.3f} (if frames are independent)")
        rel = float(np.sqrt(((n - luma(ref)[None]) ** 2).mean()) / max(m, 1e-12))
        zero = float((n <= 0).mean())
        ac = autocorr(noisy[0] - ref)
        print(f"  noisy input: relative rms {rel:.2f}, {zero * 100:.0f} % black pixels, "
              f"clumping (corr at 1/2/4/8/16 px) {' '.join(f'{c:.2f}' for c in ac)}")
    convergence(d, ref, m)
    if out:
        h, w = ref.shape[:2]
        o = [resize(x, w, h) for x in out]
        ratio = luma(o[0]).mean() / max(m, 1e-12)
        err = block_error(o[0], ref)
        change = np.mean([np.abs(luma(o[i + 1]) - luma(o[i])).mean() for i in range(len(o) - 1)]) / max(m, 1e-12) if len(o) > 1 else float("nan")
        print(f"  RR output: ratio {ratio:.3f}, block error {err:.3f}, frame-to-frame change {change:.3f}")
    return ref


def png(d):
    d = Path(d)
    ref = load_pfm(d / "reference.pfm")
    scale = 0.18 / max(np.median(luma(ref)), 1e-12)
    for name in ["reference", "noisy-0", "output-0"]:
        p = d / f"{name}.pfm"
        if not p.exists():
            continue
        a = load_pfm(p) * scale
        a = a / (1.0 + a)
        a = np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(np.clip(a, 0, None), 1 / 2.4) - 0.055)
        Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(d / f"{name}.png")
        print(d / f"{name}.png")


def main(args):
    if not args:
        sys.exit(__doc__)
    if args[0] == "--png":
        for d in args[1:]:
            png(d)
        return
    a = one(args[0])
    if len(args) > 1:
        b = one(args[1])
        if a.shape != b.shape:
            b = resize(b, a.shape[1], a.shape[0])
        print(f"bias, {args[1]} against {args[0]} (references, no denoiser): "
              f"ratio {luma(b).mean() / max(luma(a).mean(), 1e-12):.3f}, block error {block_error(b, a):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
