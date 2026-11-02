#!/usr/bin/env python
"""Compare a baseline and a candidate run_cases.py output tree, case by case.

    python golden_report.py --baseline B --candidate C \
        --baseline-code <baseline SAWs/updated> --candidate-code <candidate SAWs/updated>

Two checks, both required to pass:

1. golden:   baseline/<case>_w<W>  ==  candidate/<case>_w<W>   for every case
             directory present in the baseline tree;
2. workers:  candidate/<case>_w<a> == candidate/<case>_w<b>    for the family
             pair in run_cases.CROSS_WORKER_PAIRS, ignoring only fields that
             record the worker count itself.

Writes a plain-text report next to the trees and exits 0/1.
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_outputs as co  # noqa: E402
import run_cases as rc  # noqa: E402

CROSS_WORKER_IGNORES = [r"execution", r"n_workers", r"workers", r"worker_source",
                        r"command_line", r"command"]


def _cmp(a: Path, b: Path, normalize, extra_ignores=()):
    rep = co.compare_trees(a, b, list(co.DEFAULT_IGNORE_KEYS) + list(extra_ignores), normalize)
    buf = io.StringIO()
    co.print_report(rep, out=buf, show_ignored=False)
    return (not rep.diffs), buf.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline-code", required=True)
    ap.add_argument("--candidate-code", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--cases", default="", help="comma-separated fnmatch patterns of case names (default: all)")
    args = ap.parse_args(argv)
    B, C = Path(args.baseline).resolve(), Path(args.candidate).resolve()
    norm = [(str(Path(args.baseline_code).resolve()), "<CODE>"),
            (str(Path(args.candidate_code).resolve()), "<CODE>"),
            (str(B), "<OUT>"), (str(C), "<OUT>")]
    lines, ok_all = [], True
    pats = [p for p in args.cases.split(",") if p]

    def wanted(case_id: str) -> bool:
        name = case_id.rsplit("_w", 1)[0]
        return not pats or any(fnmatch.fnmatchcase(name, p) for p in pats)

    cases = sorted(p.name for p in B.iterdir()
                   if p.is_dir() and not p.name.startswith("_") and wanted(p.name))
    for cid in cases:
        if not (C / cid).is_dir():
            lines.append(f"FAIL golden  {cid}: missing in candidate")
            ok_all = False
            continue
        ok, txt = _cmp(B / cid, C / cid, norm + [(str(B / cid), "<CASE>"), (str(C / cid), "<CASE>")])
        ok_all &= ok
        lines.append(f"{'ok  ' if ok else 'FAIL'} golden  {cid}")
        if not ok:
            lines += ["      " + l for l in txt.splitlines()]
    names = sorted({c.rsplit("_w", 1)[0] for c in (p.name for p in C.iterdir() if p.is_dir())
                    if "_w" in c and not c.startswith("_")})
    for name in names:
        if not wanted(name + "_w0"):
            continue
        fam = rc.family(name)
        if fam not in rc.CROSS_WORKER_PAIRS:
            continue
        wa, wb = rc.CROSS_WORKER_PAIRS[fam]
        da, db = C / f"{name}_w{wa}", C / f"{name}_w{wb}"
        if not (da.is_dir() and db.is_dir()):
            continue
        ok, txt = _cmp(da, db, norm + [(str(da), "<CASE>"), (str(db), "<CASE>")], CROSS_WORKER_IGNORES)
        ok_all &= ok
        lines.append(f"{'ok  ' if ok else 'FAIL'} workers {name} (w{wa} vs w{wb})")
        if not ok:
            lines += ["      " + l for l in txt.splitlines()]
    for name in names:
        fam = rc.family(name)
        if fam not in rc.CROSS_SCHEME_PAIRS or not wanted(name + "_w0"):
            continue
        target, wa, wb = rc.CROSS_SCHEME_PAIRS[fam]
        other = target + name[len(fam):]
        da, db = C / f"{name}_w{wa}", C / f"{other}_w{wb}"
        if not (da.is_dir() and db.is_dir()):
            lines.append(f"FAIL scheme  {name}: missing {da.name} or {db.name}")
            ok_all = False
            continue
        ok, txt = _cmp(da, db, norm + [(str(da), "<CASE>"), (str(db), "<CASE>")], CROSS_WORKER_IGNORES)
        ok_all &= ok
        lines.append(f"{'ok  ' if ok else 'FAIL'} scheme  {name}_w{wa} == {other}_w{wb}")
        if not ok:
            lines += ["      " + l for l in txt.splitlines()]
    lines.append("RESULT: " + ("ALL IDENTICAL" if ok_all else "DIFFERENCES FOUND"))
    text = "\n".join(lines) + "\n"
    print(text, end="")
    report = Path(args.report) if args.report else C.parent / "golden_report.txt"
    report.write_text(text)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
