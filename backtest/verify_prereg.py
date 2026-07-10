# backtest/verify_prereg.py — machine-check the two-commit pre-registration proofs.
#
# The experiment protocol claims: the header (hypothesis + bar) is committed BEFORE the
# run, results appended in a LATER commit, so the git hash chain proves the bar predated
# the outcome. This script makes that claim checkable instead of narrative (seventh
# external review): for every experiments/*.py whose RESULTS block names its
# pre-registration commit, verify (1) the named commit exists, (2) it is the FIRST
# commit that touched the file (the header), (3) at least one later commit touched it
# (the results), and (4) — eighth review — HEADER CONTENT INTEGRITY: the text before
# the "# RESULTS" marker at the results commit is byte-identical to the pre-registration
# blob, proving the bar was not edited between registration and outcome. (Cosmetic edits
# AFTER the results commit — e.g. path portability — are visible in git and don't taint
# the run.) Exit 1 on any violation.
#
# KNOWN LIMITATION (tenth review, stated rather than papered over): this proves the BAR
# predated the outcome and wasn't edited; it does NOT re-execute experiments to confirm
# the transcribed RESULTS numbers match a fresh run. Byte-exact re-verification is
# structurally impossible on this data: yfinance re-scales the entire adjusted history
# whenever a dividend posts, so every re-run drifts slightly. The mitigation is that
# each experiment file IS the runnable procedure — anyone (including a future Erik who
# stops trusting his past self) can re-run it today and compare within tolerance.
#
#   .venv/bin/python -m backtest.verify_prereg

import re
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent / "experiments"
REPO = EXP_DIR.parents[1]
PATTERN = re.compile(r"unmodified from (?:the )?pre-registration (?:commit )?`?([0-9a-f]{7,10})`?")
RESULTS_MARKER = "\n# RESULTS"


def file_commits(path):
    out = subprocess.run(["git", "log", "--follow", "--format=%h", "--", str(path)],
                         capture_output=True, text=True, cwd=path.parent).stdout.split()
    return out                                             # newest first


def blob_at(commit, relpath):
    r = subprocess.run(["git", "show", f"{commit}:{relpath}"],
                       capture_output=True, text=True, cwd=REPO)
    return r.stdout if r.returncode == 0 else None


def main():
    failures, verified, unclaimed = [], [], []
    for f in sorted(EXP_DIR.glob("*.py")):
        m = PATTERN.search(f.read_text())
        if not m:
            unclaimed.append(f.name)
            continue
        claimed = m.group(1)
        commits = file_commits(f)
        if not commits:
            failures.append(f"{f.name}: not in git")
            continue
        oldest = commits[-1]
        relpath = str(f.relative_to(REPO))
        if not (oldest.startswith(claimed) or claimed.startswith(oldest)):
            failures.append(f"{f.name}: claims pre-reg {claimed} but file's first commit is {oldest}")
            continue
        if len(commits) < 2:
            failures.append(f"{f.name}: results claim a later commit but file has only one")
            continue
        # (4) content integrity: pre-reg blob == pre-RESULTS portion at the results commit
        prereg_blob = blob_at(oldest, relpath)
        results_commit, results_blob = None, None
        for c in reversed(commits[:-1]):                   # oldest-after-prereg first
            b = blob_at(c, relpath)
            if b and RESULTS_MARKER in b:
                results_commit, results_blob = c, b
                break
        if results_blob is None:
            failures.append(f"{f.name}: no commit containing a RESULTS block found")
            continue
        header_at_results = results_blob.split(RESULTS_MARKER)[0].rstrip()
        if prereg_blob is None or prereg_blob.rstrip() != header_at_results:
            failures.append(f"{f.name}: HEADER EDITED between pre-reg {oldest} and results "
                            f"{results_commit} — the bar text changed after registration")
        else:
            verified.append(f"{f.name}: header {oldest} -> results {results_commit} "
                            f"({len(commits)} commits, header content INTACT)")
    for v in verified:
        print(f"  VERIFIED  {v}")
    for u in unclaimed:
        print(f"  no claim  {u} (no 'unmodified from pre-registration commit' line)")
    for x in failures:
        print(f"  FAILED    {x}")
    print(f"\nverify_prereg: {len(verified)} verified, {len(unclaimed)} without claims, "
          f"{len(failures)} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
