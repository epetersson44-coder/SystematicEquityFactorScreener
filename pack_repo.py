# pack_repo.py — flatten the lab into ONE markdown file for external LLM review
# (a local code2prompt: no install, no repo access needed by the reviewer).
#
#   .venv/bin/python pack_repo.py          -> writes repo_pack.md (gitignored)
#
# Ordered for a cold reader: CURRENT_STATE.md first (the status map — prevents the
# review-the-retired-screener-as-the-product misread), then code in dependency order,
# tests last. Data/caches/artifacts excluded. Regenerate fresh before each share.
#
# PRIVACY NOTE (shown at run time too): the pack contains portfolio.py (real holdings —
# they ARE the review subject, so they stay). Name/email/home-path strings are REDACTED
# from the pack text below (ninth review housekeeping): the reviewer doesn't need them,
# and every paste into an external LLM is a copy you can't recall.

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "repo_pack.md"

# The review prompt embedded at the top of the pack, so pasting the single file into any
# LLM is a complete request — role, rules, and code in one shot. Derived from the
# research-methodology audit prompt (the one that produced useful reviews), tightened.
PROMPT = """\
# Quantitative research audit — full repository below

You are acting as a Principal Quantitative Researcher, Senior Quant Developer, and
Systematic Portfolio Manager joining a small research lab. The ENTIRE repository is
included below this prompt, one file per section, in dependency order.

## What this is (read before judging)

A personal quantitative research lab. It is NOT a commercial product and NOT one
strategy: it contains LIVE strategies, RETIRED negative results kept deliberately as
findings, and BANKED tested-but-not-adopted options. The first file, CURRENT_STATE.md,
is the authoritative map of which is which — read it first and score each module against
its actual status. Critiquing a retired module as if it were the product is the #1
misread to avoid.

The lab exists to answer one question: can a systematic strategy beat SPY over long
periods, net of realistic frictions — and can its operator hold it through drawdowns?
Real money runs on the sso_stack construction; treat anything touching the live pipeline
(tracker, preflight, shopping list) as production-critical.

## Your job

Determine whether this research process can be trusted — and find what it still gets
wrong. Challenge every assumption; do not assume a methodology is correct because it is
implemented, or because the comments argue for it confidently. Assume false positives
are more likely than true edge.

Evaluate, in order of importance:
1. VALIDITY — hidden look-ahead, survivorship, or data-leakage paths the existing
   guards miss; unrealistic execution/cost/financing assumptions; errors in the LETF,
   financing, or vol-targeting math.
2. STATISTICS — is the significance framework (trial ledger, deflated Sharpe, block
   bootstrap, timing-luck sweeps, synthetic falsification) sound and honestly applied?
   Where could it still fool the operator?
3. EXPERIMENT DISCIPLINE — pre-registration, the ledger, banked/closed verdicts:
   real safeguards or theater? What would make them airtight?
4. CODE-VS-CLAIM DRIFT — places where comments/docs assert something the code does
   not do (this repo treats that as a serious bug).
5. NEW TESTS worth running — concrete, pre-registrable experiments that could change
   a live decision, with the adoption bar you would set. Respect the constraints: one
   operator, free/cheap data, a small cash (no-margin) brokerage account, monthly cadence.

Skip entirely: SaaS/auth/API/database/UI concerns, deployment, multi-user anything —
unless it directly corrupts research conclusions.

## Output format

1. Your understanding of the system (brief — prove you read it, flag anything unclear).
2. Findings ranked by severity, each with: file/function, why it matters, and the
   concrete failure scenario. Cite code you can point to, not vibes.
3. The 5 highest-impact improvements, effort-rated.
4. What evidence would convince you the live edge is real, and what would falsify it.
5. Scores /10: data validity, backtest realism, statistical rigor, experiment
   discipline, code quality — each with the single change that would most raise it.

Be blunt. A confirmed weakness is worth more than a compliment. If something is good,
say so once and move on.
"""

# Explicit front matter + core order; everything else picked up alphabetically after.
FIRST = [
    "CURRENT_STATE.md",
    "requirements.txt",
    "config.py",
    "fundamentals.py",
    "edgar.py",
    "factors.py",
    "score.py",
    "screen.py",
    "export.py",
    "portfolio.py",
    "backtest/constants.py",
    "backtest/costs.py",
    "backtest/metrics.py",
    "backtest/engine.py",
    "backtest/engine_xs.py",
    "backtest/trend_sleeve.py",
    "backtest/timing_luck.py",
    "backtest/significance.py",
    "backtest/leverage_study.py",
    "backtest/synthetic.py",
    "backtest/volatility.py",
    "backtest/tracker.py",
    "backtest/dashboard.py",
    "backtest/preflight.py",
]
SKIP_DIRS = {".git", ".venv", "__pycache__", "output", "cache", "data"}
SKIP_FILES = {OUT.name, "desk_data.json", "desk_history.json", "desk_daily.csv"}
KEEP_EXT = {".py", ".md", ".txt"}

# PII scrubbed from the PACK only (source files untouched — SEC requires the real
# contact in edgar.py's User-Agent; the reviewer doesn't need it).
REDACT = [
    (re.compile(r"epetersson44@gmail\.com"), "<email-redacted>"),
    (re.compile(r"Erik Petersson"), "<name-redacted>"),
    (re.compile(r"/Users/erik\.petersson"), "/Users/<user>"),
]


def wanted(p: Path) -> bool:
    rel = p.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in rel.parts) or p.name in SKIP_FILES:
        return False
    if rel.parts[0] == "backtest" and rel.parts[1:2] == ("picks",):
        return False                       # lock JSONs are data, not code
    return p.suffix in KEEP_EXT


def main():
    files = [ROOT / f for f in FIRST if (ROOT / f).exists()]
    rest = sorted(p for p in ROOT.rglob("*") if p.is_file() and wanted(p)
                  and str(p.relative_to(ROOT)) not in FIRST)
    # experiments + tests after core, tests very last
    rest.sort(key=lambda p: (p.parts.__contains__("tests"), str(p)))
    files += rest

    parts = [PROMPT]
    total_lines = 0
    for p in files:
        rel = p.relative_to(ROOT)
        text = p.read_text(errors="replace").rstrip("\n")
        for pat, sub in REDACT:
            text = pat.sub(sub, text)
        total_lines += text.count("\n") + 1
        fence = "```" if p.suffix != ".md" else "````"
        lang = {".py": "python", ".txt": "text", ".md": "markdown"}[p.suffix]
        parts.append(f"\n---\n## FILE: {rel}\n{fence}{lang}\n{text}\n{fence}\n")
    out = "\n".join(parts)
    OUT.write_text(out)
    print(f"wrote {OUT.name}: {len(files)} files, {total_lines:,} lines, "
          f"{len(out):,} chars (~{len(out) // 4:,} tokens)")
    print("PRIVACY: includes portfolio.py (real holdings); name/email/home-path REDACTED from the pack.")


if __name__ == "__main__":
    main()
