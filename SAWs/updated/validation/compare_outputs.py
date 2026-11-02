#!/usr/bin/env python
"""Bit-level comparison of two output trees (golden vs candidate).

Every regular file under DIR_A is compared with the file at the same relative
path under DIR_B.  The standard is *bit identity* of the scientific content:

* ``.npz``  every key; numeric/bool arrays must match dtype, shape and raw bytes
            (stricter than ``==``: NaN payloads and signed zeros count); object /
            string arrays are compared element-wise after path normalization.
* ``.json`` recursive equality after path normalization.
* ``.csv`` / ``.tsv`` / ``.txt``  text equality after path normalization; on a
            mismatch the first differing CSV cells are reported.
* ``.h5`` / ``.hdf5``  every dataset (dtype, shape, raw bytes) and attribute.
* Plots (``.png``, ``.pdf``, ``.svg``) and logs are skipped and listed.

Some content legitimately differs between two runs of identical science:
wall-clock timings, timestamps, host names, absolute output paths, and the
worker count when the comparison is deliberately across worker counts.  Such
fields are excluded by *name* (``--ignore-key`` regexes; defaults below) and
paths are normalized with ``--normalize OLD=NEW``.  Every excluded difference is
still reported under "ignored" so nothing is silently hidden.

Exit status: 0 when no non-ignored difference exists, 1 otherwise, 2 on usage
error.  Requires numpy; h5py only when HDF5 files are present.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Field names that carry run-environment information rather than science.  The
# match is a full, case-insensitive regex match against the *last* key component
# (JSON key, NPZ key, CSV column header, HDF5 attribute/dataset name).
DEFAULT_IGNORE_KEYS = [
    r"timestamp", r"(start|end|finish)_?time", r"date(time)?", r"created(_at)?",
    r"wall_?time(_seconds|_s|_hours)?", r"elapsed(_seconds|_s|_time)?",
    r".*_elapsed(_s|_seconds)?", r"timing", r"timings", r"t_(sweep|swap|struct|total)(_total)?",
    r"(sweep|swap|struct|total)_(seconds|time|time_s|wall)", r"runtime(_seconds|_s)?",
    r".*_seconds", r".*_secs", r".*overhead.*",
    r"host(name)?", r"node(name)?", r"pid", r"cwd", r"user(name)?",
    r"python(_executable|_version)?", r"executable", r"sys_version", r"platform",
    r"slurm_.*", r"git(_.*)?",
    # Implementation provenance added by the Numba stage (which kernel ran, its
    # start-up self-check, the RNG scheme); the sampled data do not depend on it.
    r"sampler_implementation",
]
SKIP_SUFFIXES = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".log", ".out", ".err", ".pyc"}
SKIP_NAME_RE = re.compile(r"(^|_)(stdout|stderr)(\.|_|$)|^\.|^_", re.IGNORECASE)
TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".dat", ".md"}
HDF5_SUFFIXES = {".h5", ".hdf5"}


class Report:
    def __init__(self, max_per_file: int) -> None:
        self.max_per_file = max_per_file
        self.diffs: Dict[str, List[str]] = {}
        self.ignored: Dict[str, List[str]] = {}
        self.skipped: List[str] = []
        self.compared: List[str] = []

    def diff(self, rel: str, msg: str) -> None:
        lst = self.diffs.setdefault(rel, [])
        if len(lst) < self.max_per_file:
            lst.append(msg)
        elif len(lst) == self.max_per_file:
            lst.append("... (further differences in this file suppressed)")

    def ignore(self, rel: str, msg: str) -> None:
        lst = self.ignored.setdefault(rel, [])
        if len(lst) < self.max_per_file:
            lst.append(msg)


class Comparator:
    def __init__(self, ignore_keys: Sequence[str], normalize: Sequence[Tuple[str, str]],
                 report: Report) -> None:
        self.ignore_re = [re.compile(p, re.IGNORECASE) for p in ignore_keys]
        # Longest first so nested roots normalize before their parents.
        self.normalize = sorted(normalize, key=lambda kv: -len(kv[0]))
        self.r = report

    # -- helpers ---------------------------------------------------------
    def ignored_name(self, name: str, full: bool = False) -> bool:
        """Match the leaf name (default) or, with full=True, the dotted JSON path."""
        leaf = str(name) if full else str(name).split("/")[-1].split(".")[-1]
        return any(p.fullmatch(leaf) for p in self.ignore_re)

    def norm_str(self, s: str) -> str:
        for old, new in self.normalize:
            if old:
                s = s.replace(old, new)
        return s

    def norm_obj(self, o: Any) -> Any:
        if isinstance(o, str):
            return self.norm_str(o)
        if isinstance(o, bytes):
            try:
                return self.norm_str(o.decode("utf-8")).encode("utf-8")
            except UnicodeDecodeError:
                return o
        if isinstance(o, dict):
            return {k: self.norm_obj(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(self.norm_obj(v) for v in o)
        return o

    # -- JSON-like trees -------------------------------------------------
    def cmp_tree(self, rel: str, a: Any, b: Any, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b), key=str):
                sub = f"{path}.{k}" if path else str(k)
                if self.ignored_name(str(k)) or self.ignored_name(sub, full=True):
                    if a.get(k, "<missing>") != b.get(k, "<missing>"):
                        self.r.ignore(rel, f"{sub}: {_short(a.get(k))} != {_short(b.get(k))}")
                    continue
                if k not in a:
                    self.r.diff(rel, f"{sub}: missing in A")
                elif k not in b:
                    self.r.diff(rel, f"{sub}: missing in B")
                else:
                    self.cmp_tree(rel, a[k], b[k], sub)
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                self.r.diff(rel, f"{path}: list length {len(a)} != {len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                self.cmp_tree(rel, x, y, f"{path}[{i}]")
            return
        if isinstance(a, float) and isinstance(b, float):
            if not (a == b or (a != a and b != b)) or (a == 0.0 and np.signbit(a) != np.signbit(b)):
                self.r.diff(rel, f"{path}: {a!r} != {b!r}")
            return
        if type(a) is not type(b) or a != b:
            self.r.diff(rel, f"{path}: {_short(a)} != {_short(b)}")

    # -- arrays ------------------------------------------------------------
    def cmp_array(self, rel: str, name: str, a: np.ndarray, b: np.ndarray) -> None:
        # Fixed-width strings: the itemsize follows the (path-dependent) longest
        # element, so compare kind + normalized content rather than dtype width.
        strings = a.dtype.kind in "US" and b.dtype.kind == a.dtype.kind
        if a.dtype != b.dtype and not strings:
            self.r.diff(rel, f"{name}: dtype {a.dtype} != {b.dtype}")
            return
        if a.shape != b.shape:
            self.r.diff(rel, f"{name}: shape {a.shape} != {b.shape}")
            return
        if a.dtype.kind in "biufcmM?":
            if np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
                msg = f"{name}: values differ"
                if a.dtype.kind in "fc" and a.size:
                    with np.errstate(invalid="ignore"):
                        d = np.abs(a.astype(complex) - b.astype(complex))
                    finite = np.isfinite(d)
                    n_bad = int(np.count_nonzero(~((a == b) | (np.isnan(a) & np.isnan(b)))))
                    msg += (f" ({n_bad}/{a.size} elements; max |diff| "
                            f"{float(d[finite].max()) if finite.any() else float('nan'):.3e})")
                elif a.size:
                    n_bad = int(np.count_nonzero(a != b))
                    msg += f" ({n_bad}/{a.size} elements)"
                self.r.diff(rel, msg)
            return
        # strings / objects: element-wise after normalization
        fa = [self.norm_obj(x) for x in a.reshape(-1).tolist()]
        fb = [self.norm_obj(x) for x in b.reshape(-1).tolist()]
        for i, (x, y) in enumerate(zip(fa, fb)):
            if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
                    self.r.diff(rel, f"{name}[{i}]: type differs")
                else:
                    self.cmp_array(rel, f"{name}[{i}]", x, y)
            elif isinstance(x, (dict, list)) or isinstance(y, (dict, list)):
                self.cmp_tree(rel, x, y, f"{name}[{i}]")
            elif x != y and not (isinstance(x, float) and x != x and isinstance(y, float) and y != y):
                self.r.diff(rel, f"{name}[{i}]: {_short(x)} != {_short(y)}")
                return

    # -- file types --------------------------------------------------------
    def cmp_npz(self, rel: str, pa: Path, pb: Path) -> None:
        with np.load(pa, allow_pickle=True) as A, np.load(pb, allow_pickle=True) as B:
            ka, kb = set(A.files), set(B.files)
            for k in sorted(ka ^ kb):
                (self.r.ignore if self.ignored_name(k) else self.r.diff)(
                    rel, f"key {k!r} only in {'A' if k in ka else 'B'}")
            for k in sorted(ka & kb):
                a, b = A[k], B[k]
                if self.ignored_name(k):
                    if a.shape != b.shape or a.dtype != b.dtype or (
                            a.dtype.kind != "O" and a.tobytes() != b.tobytes()):
                        self.r.ignore(rel, f"key {k!r} differs")
                    continue
                self.cmp_array(rel, k, a, b)

    def cmp_json(self, rel: str, pa: Path, pb: Path) -> None:
        try:
            a = json.loads(pa.read_text(encoding="utf-8"))
            b = json.loads(pb.read_text(encoding="utf-8"))
        except ValueError as exc:
            self.r.diff(rel, f"unparseable JSON: {exc}")
            return
        self.cmp_tree(rel, self.norm_obj(a), self.norm_obj(b), "")

    def cmp_text(self, rel: str, pa: Path, pb: Path) -> None:
        ta = self.norm_str(pa.read_text(encoding="utf-8", errors="replace"))
        tb = self.norm_str(pb.read_text(encoding="utf-8", errors="replace"))
        if ta == tb:
            return
        if pa.suffix.lower() in {".csv", ".tsv"}:
            delim = "\t" if pa.suffix.lower() == ".tsv" else ","
            ra = list(csv.reader(io.StringIO(ta), delimiter=delim))
            rb = list(csv.reader(io.StringIO(tb), delimiter=delim))
            header = ra[0] if ra else []
            if not ra or not rb or ra[0] != rb[0]:
                self.r.diff(rel, f"CSV header differs: {ra[:1]} vs {rb[:1]}")
                return
            if len(ra) != len(rb):
                self.r.diff(rel, f"CSV row count {len(ra)} != {len(rb)}")
            ignored_cols = {j for j, h in enumerate(header) if self.ignored_name(h)}
            for i, (x, y) in enumerate(zip(ra, rb)):
                if len(x) != len(y):
                    self.r.diff(rel, f"row {i}: {len(x)} != {len(y)} cells")
                    continue
                for j, (u, v) in enumerate(zip(x, y)):
                    if u != v:
                        col = header[j] if j < len(header) else str(j)
                        (self.r.ignore if j in ignored_cols else self.r.diff)(
                            rel, f"row {i} col {col!r}: {u!r} != {v!r}")
            return
        la, lb = ta.splitlines(), tb.splitlines()
        for i, (x, y) in enumerate(zip(la, lb)):
            if x != y:
                self.r.diff(rel, f"line {i + 1}: {x[:120]!r} != {y[:120]!r}")
                return
        self.r.diff(rel, f"line count {len(la)} != {len(lb)}")

    def cmp_h5(self, rel: str, pa: Path, pb: Path) -> None:
        import h5py  # local import: only needed when HDF5 outputs exist

        def walk(g, prefix=""):
            out = {}
            for k in g:
                obj = g[k]
                name = f"{prefix}/{k}"
                out[name] = obj
                if isinstance(obj, h5py.Group):
                    out.update(walk(obj, name))
            return out

        with h5py.File(pa, "r") as A, h5py.File(pb, "r") as B:
            self.cmp_attrs(rel, "/", A.attrs, B.attrs)
            wa, wb = walk(A), walk(B)
            for k in sorted(set(wa) ^ set(wb)):
                self.r.diff(rel, f"{k}: only in {'A' if k in wa else 'B'}")
            for k in sorted(set(wa) & set(wb)):
                oa, ob = wa[k], wb[k]
                self.cmp_attrs(rel, k, oa.attrs, ob.attrs)
                if isinstance(oa, h5py.Dataset) and isinstance(ob, h5py.Dataset):
                    if self.ignored_name(k.split("/")[-1]):
                        continue
                    self.cmp_array(rel, k, np.asarray(oa[()]), np.asarray(ob[()]))
                elif type(oa) is not type(ob):
                    self.r.diff(rel, f"{k}: group/dataset type differs")

    def cmp_attrs(self, rel: str, where: str, aa, ab) -> None:
        for k in sorted(set(aa.keys()) | set(ab.keys())):
            sub = f"{where}@{k}"
            if k not in aa or k not in ab:
                (self.r.ignore if self.ignored_name(k) else self.r.diff)(
                    rel, f"{sub}: only in {'A' if k in aa else 'B'}")
                continue
            va, vb = self.norm_obj(_attr(aa[k])), self.norm_obj(_attr(ab[k]))
            if self.ignored_name(k):
                if _neq(va, vb):
                    self.r.ignore(rel, f"{sub}: {_short(va)} != {_short(vb)}")
                continue
            if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
                self.cmp_array(rel, sub, va, vb)
            elif isinstance(va, str) and isinstance(vb, str) and va.lstrip().startswith(("{", "[")):
                try:
                    self.cmp_tree(rel, json.loads(va), json.loads(vb), sub)
                except ValueError:
                    if va != vb:
                        self.r.diff(rel, f"{sub}: {_short(va)} != {_short(vb)}")
            elif _neq(va, vb):
                self.r.diff(rel, f"{sub}: {_short(va)} != {_short(vb)}")

    def cmp_file(self, rel: str, pa: Path, pb: Path) -> None:
        suf = pa.suffix.lower()
        if suf == ".npz":
            self.cmp_npz(rel, pa, pb)
        elif suf == ".json":
            self.cmp_json(rel, pa, pb)
        elif suf in HDF5_SUFFIXES:
            self.cmp_h5(rel, pa, pb)
        elif suf in TEXT_SUFFIXES:
            self.cmp_text(rel, pa, pb)
        elif suf == ".npy":
            self.cmp_array(rel, "array", np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True))
        else:
            if pa.read_bytes() != pb.read_bytes():
                self.r.diff(rel, "binary content differs")


def _attr(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray) and v.dtype.kind in "SO" and v.size == 1:
        x = v.reshape(-1)[0]
        return x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x
    if isinstance(v, np.generic):
        return v.item()
    return v


def _neq(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes()
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return False
    return type(a) is not type(b) or a != b


def _short(x: Any, n: int = 80) -> str:
    s = repr(x)
    return s if len(s) <= n else s[: n - 3] + "..."


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not SKIP_NAME_RE.search(d) and d != "__pycache__")
        for f in sorted(filenames):
            p = Path(dirpath) / f
            yield p.relative_to(root).as_posix(), p


def compare_trees(dir_a: Path, dir_b: Path, ignore_keys: Sequence[str],
                  normalize: Sequence[Tuple[str, str]], max_per_file: int = 8) -> Report:
    rep = Report(max_per_file)
    comp = Comparator(ignore_keys, normalize, rep)
    fa = dict(iter_files(dir_a))
    fb = dict(iter_files(dir_b))
    for rel in sorted(set(fa) | set(fb)):
        name = rel.split("/")[-1]
        p = fa.get(rel) or fb[rel]
        if p.suffix.lower() in SKIP_SUFFIXES or SKIP_NAME_RE.search(name):
            rep.skipped.append(rel)
            continue
        if rel not in fa:
            rep.diff(rel, "file only in B")
            continue
        if rel not in fb:
            rep.diff(rel, "file only in A")
            continue
        rep.compared.append(rel)
        try:
            comp.cmp_file(rel, fa[rel], fb[rel])
        except Exception as exc:  # noqa: BLE001 - any read failure is a difference
            rep.diff(rel, f"comparison failed: {type(exc).__name__}: {exc}")
    return rep


def print_report(rep: Report, out=sys.stdout, show_ignored: bool = True) -> None:
    print(f"compared {len(rep.compared)} files, skipped {len(rep.skipped)} "
          f"(plots/logs/underscore-prefixed)", file=out)
    if show_ignored and rep.ignored:
        n = sum(len(v) for v in rep.ignored.values())
        print(f"ignored by name rule: {n} field difference(s) in {len(rep.ignored)} file(s)", file=out)
        for rel, msgs in sorted(rep.ignored.items()):
            for m in msgs[:3]:
                print(f"  [ignored] {rel}: {m}", file=out)
    if rep.diffs:
        print(f"DIFFERENCES in {len(rep.diffs)} file(s):", file=out)
        for rel, msgs in sorted(rep.diffs.items()):
            for m in msgs:
                print(f"  {rel}: {m}", file=out)
    else:
        print("IDENTICAL (no non-ignored differences)", file=out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--ignore-key", action="append", default=[],
                    help="extra regex (full match, case-insensitive) of field names to ignore")
    ap.add_argument("--no-default-ignores", action="store_true")
    ap.add_argument("--normalize", action="append", default=[], metavar="OLD=NEW",
                    help="replace OLD with NEW in every string before comparing")
    ap.add_argument("--max-per-file", type=int, default=8)
    ap.add_argument("--quiet-ignored", action="store_true")
    args = ap.parse_args(argv)
    a, b = Path(args.dir_a), Path(args.dir_b)
    if not a.is_dir() or not b.is_dir():
        print("both arguments must be directories", file=sys.stderr)
        return 2
    pairs = []
    for item in args.normalize:
        if "=" not in item:
            print(f"--normalize expects OLD=NEW, got {item!r}", file=sys.stderr)
            return 2
        old, new = item.split("=", 1)
        pairs.append((old, new))
    # The two roots themselves always normalize to a common token.
    pairs += [(str(a.resolve()), "<ROOT>"), (str(b.resolve()), "<ROOT>"),
              (str(a), "<ROOT>"), (str(b), "<ROOT>")]
    keys = ([] if args.no_default_ignores else list(DEFAULT_IGNORE_KEYS)) + list(args.ignore_key)
    rep = compare_trees(a, b, keys, pairs, args.max_per_file)
    print_report(rep, show_ignored=not args.quiet_ignored)
    return 1 if rep.diffs else 0


if __name__ == "__main__":
    sys.exit(main())
