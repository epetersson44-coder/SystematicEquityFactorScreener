# backtest/verify_prereg.py — machine-check the two-commit pre-registration proofs.
#
# The experiment protocol claims: the header (hypothesis + bar) is committed BEFORE the
# run, results appended in a LATER commit, so the git hash chain proves the bar predated
# the outcome. This script makes that claim checkable instead of narrative (seventh
# external review): for every experiments/*.py whose RESULTS block names its
# pre-registration commit, verify that (1) the named commit exists, (2) it is the
# FIRST commit that touched the file (the header), and (3) at least one later commit
# touched it (the results). Exit 1 on any violation.
#
#   .venv/bin/python -m backtest.verify_prereg

import re
import subprocess
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent / "experiments"
PATTERN = re.compile(r"unmodified from (?:the )?pre-registration (?:commit )?`?([0-9a-f]{7,10})`?")


def file_commits(path):
    out = subprocess.run(["git", "log", "--follow", "--format=%h", "--", str(path)],
                         capture_output=True, text=True, cwd=path.parent).stdout.split()
    return out                                             # newest first


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
        if not (oldest.startswith(claimed) or claimed.startswith(oldest)):
            failures.append(f"{f.name}: claims pre-reg {claimed} but file's first commit is {oldest}")
        elif len(commits) < 2:
            failures.append(f"{f.name}: results claim a later commit but file has only one")
        else:
            verified.append(f"{f.name}: header {oldest} -> results {commits[0]} ({len(commits)} commits)")
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
