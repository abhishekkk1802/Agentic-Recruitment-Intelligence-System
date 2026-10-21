"""
Explainability Layer for the LangGraph Matching Agent
=====================================================

Turns raw match scores into decisions a hiring manager can defend.

The screening pipeline (``screening_pipeline.py``) already produces scores,
hire/no-hire verdicts and key strengths. What it does not produce is the other
half of an honest assessment: what a candidate is *missing*, whether they sit
close enough to the bar to be worth a conversation, and what would have to be
true for a borderline candidate to clear it. This module supplies that, plus
the ranking-delta machinery needed to explain why a leaderboard moved after the
user refines their criteria mid-conversation.

Responsibilities:
- Detailed per-candidate match reports with strengths *and* gaps (Part C.2)
- Improvement suggestions for borderline candidates (Part C.2)
- Side-by-side comparison tables (Part B.1)
- Head-to-head "why did X rank above Y" explanations (Part B.1)
- Ranking deltas so re-ranking is explained, not just performed (Part B.2)

Everything here is deterministic and free of LLM calls. Narratives are
templated from the underlying numbers, so a report can never claim a strength
the data does not support, and the module stays testable and cheap to call.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent_state import (
    CandidateMatchDict,
    JobRequirementDict,
    MatchReportData,
    RankingChangeData,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Scoring bands
# =============================================================================
# These bands are the single source of truth for how a score becomes a verdict.
# `screening_pipeline` uses its own thresholds for round *filtering*; these are
# for round-independent *interpretation* of a candidate's standing.

STRONG_THRESHOLD = 85.0
SOLID_THRESHOLD = 70.0
BORDERLINE_THRESHOLD = 55.0

VERDICT_LABELS = {
    "strong": "Strong match",
    "solid": "Solid match",
    "borderline": "Borderline",
    "weak": "Weak match",
}


def candidate_score(candidate: CandidateMatchDict) -> float:
    """
    Resolve a candidate's effective score.

    Round 2 of screening adds an ``enhanced_score``; when present it supersedes
    the original ``match_score`` because it reflects deeper analysis.

    Args:
        candidate: Candidate match record

    Returns:
        Effective score on a 0-100 scale
    """
    score = candidate.get("enhanced_score")
    if score is None:
        score = candidate.get("match_score", 0)
    try:
        return float(score or 0)
    except (TypeError, ValueError):
        return 0.0


def classify_verdict(score: float) -> str:
    """Map a 0-100 score onto a verdict band."""
    if score >= STRONG_THRESHOLD:
        return "strong"
    if score >= SOLID_THRESHOLD:
        return "solid"
    if score >= BORDERLINE_THRESHOLD:
        return "borderline"
    return "weak"


def is_borderline(score: float) -> bool:
    """
    Whether a candidate sits in the band worth a second look.

    Borderline means "did not clear the bar, but close enough that a targeted
    conversation could change the answer" - precisely the population the brief
    asks for improvement suggestions on.
    """
    return BORDERLINE_THRESHOLD <= score < SOLID_THRESHOLD


# =============================================================================
# Requirement helpers
# =============================================================================

def _normalise(values: Optional[Sequence[str]]) -> List[str]:
    """Lower-case, strip and de-duplicate a list of strings, preserving order."""
    out: List[str] = []
    seen = set()
    for value in values or []:
        if not value:
            continue
        cleaned = str(value).strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def _titlecase(values: Sequence[str]) -> List[str]:
    """Present skill tokens back to the user in a readable form."""
    out = []
    for value in values:
        # Preserve tokens that are already deliberately cased (AWS, C++, Node.js)
        if any(ch.isupper() for ch in value) or not value.isalpha():
            out.append(value)
        else:
            out.append(value.title())
    return out


def split_requirement_coverage(
    candidate: CandidateMatchDict,
    requirements: Optional[JobRequirementDict]
) -> Dict[str, List[str]]:
    """
    Work out which requirements a candidate actually covers.

    Compares the candidate's matched skills against both the must-have and
    nice-to-have sides of the requirements, so a gap in a "preferred" skill is
    never reported with the same weight as a missing hard requirement.

    Args:
        candidate: Candidate match record
        requirements: Active job requirements (may be None)

    Returns:
        Dict with keys must_met, must_missing, nice_met, nice_missing
    """
    matched = set(_normalise(candidate.get("matched_skills")))

    # A candidate's own missing_skills list is a useful signal but not
    # authoritative, so requirement coverage is recomputed from scratch.
    must_have: List[str] = []
    nice_have: List[str] = []

    if requirements:
        must_have = _normalise(
            list(requirements.get("skills_required") or [])
            + list(requirements.get("must_have_keywords") or [])
        )
        nice_have = _normalise(
            list(requirements.get("skills_preferred") or [])
            + list(requirements.get("nice_to_have_keywords") or [])
        )

    # Anything listed as must-have is excluded from nice-to-have to avoid
    # double-counting a skill that appears on both lists.
    nice_have = [s for s in nice_have if s not in set(must_have)]

    def covered(requirement: str) -> bool:
        """Substring-tolerant match: 'react' should satisfy 'react.js'."""
        if requirement in matched:
            return True
        return any(requirement in skill or skill in requirement for skill in matched)

    must_met = [r for r in must_have if covered(r)]
    must_missing = [r for r in must_have if not covered(r)]
    nice_met = [r for r in nice_have if covered(r)]
    nice_missing = [r for r in nice_have if not covered(r)]

    return {
        "must_met": must_met,
        "must_missing": must_missing,
        "nice_met": nice_met,
        "nice_missing": nice_missing,
    }


def assess_experience(
    candidate: CandidateMatchDict,
    requirements: Optional[JobRequirementDict]
) -> Tuple[str, Optional[int]]:
    """
    Describe how the candidate's experience compares to the requirement.

    Returns:
        (human-readable assessment, shortfall in years or None)
    """
    required = None
    if requirements:
        required = requirements.get("experience_years")

    profile = candidate.get("profile_analysis") or {}
    actual = profile.get("experience_years")
    if actual is None:
        actual = candidate.get("experience_years")

    if required in (None, 0):
        if actual is None:
            return "No explicit experience requirement stated.", None
        return f"{actual} years of experience on file; no explicit requirement to compare against.", None

    if actual is None:
        return (
            f"Requirement is {required}+ years, but years of experience could not be "
            f"determined from the resume.",
            None,
        )

    try:
        actual_i = int(actual)
        required_i = int(required)
    except (TypeError, ValueError):
        return f"Requirement is {required}+ years; resume reports {actual}.", None

    if actual_i >= required_i:
        surplus = actual_i - required_i
        if surplus == 0:
            return f"Meets the {required_i}-year requirement exactly.", None
        return f"Exceeds the {required_i}-year requirement by {surplus} year(s).", None

    shortfall = required_i - actual_i
    return (
        f"Falls {shortfall} year(s) short of the {required_i}-year requirement "
        f"({actual_i} years on file).",
        shortfall,
    )


# =============================================================================
# Match reports (Part C.2)
# =============================================================================

def build_improvement_suggestions(
    candidate: CandidateMatchDict,
    coverage: Dict[str, List[str]],
    experience_shortfall: Optional[int],
    score: float
) -> List[str]:
    """
    Concrete, candidate-specific suggestions for closing the gap.

    Only generated where they are actionable. A candidate who already clears
    the bar does not need advice, and a candidate far below it needs a
    different role rather than a checklist.

    Args:
        candidate: Candidate match record
        coverage: Output of split_requirement_coverage
        experience_shortfall: Years short, if any
        score: Effective score

    Returns:
        Ordered list of suggestions, most impactful first
    """
    suggestions: List[str] = []
    name = candidate.get("candidate_name", "This candidate")

    must_missing = coverage["must_missing"]
    nice_missing = coverage["nice_missing"]

    # Hard requirements first - these are what actually block a hire.
    if must_missing:
        top = _titlecase(must_missing[:3])
        suggestions.append(
            f"Verify {', '.join(top)} during screening. These are listed as required "
            f"but do not appear in the resume, and a false negative here is the most "
            f"likely reason {name} is under-scored."
        )
        if len(must_missing) > 3:
            suggestions.append(
                f"{len(must_missing) - 3} further required skill(s) are also unevidenced: "
                f"{', '.join(_titlecase(must_missing[3:6]))}."
            )

    # Experience shortfall is often negotiable, so frame it as a probe.
    if experience_shortfall:
        suggestions.append(
            f"Probe depth over duration: {name} is {experience_shortfall} year(s) short on "
            f"paper, so establish whether recent scope and ownership offset the gap."
        )

    # Preferred skills are upside, not blockers.
    if nice_missing:
        suggestions.append(
            f"Treat {', '.join(_titlecase(nice_missing[:3]))} as ramp-up areas rather than "
            f"blockers, since they are preferred rather than required."
        )

    # If the resume itself is thin, the score may be an artefact of the document.
    excerpts = candidate.get("relevant_excerpts") or []
    if len(excerpts) <= 1:
        suggestions.append(
            "Resume yielded little matchable detail. Request a fuller work history or a "
            "portfolio link before scoring this candidate out."
        )

    if not suggestions:
        suggestions.append(
            f"No specific gaps identified. {name} scored {score:.1f}% and can move forward "
            f"on the current evidence."
        )

    return suggestions


def build_match_report(
    candidate: CandidateMatchDict,
    requirements: Optional[JobRequirementDict],
    rank: int,
    total_candidates: int
) -> MatchReportData:
    """
    Build a full explainability report for a single candidate.

    Args:
        candidate: Candidate match record
        requirements: Active job requirements
        rank: 1-based position in the current ranking
        total_candidates: Size of the ranking, used for relative framing

    Returns:
        Populated MatchReportData
    """
    name = candidate.get("candidate_name", "Unknown")
    score = candidate_score(candidate)
    verdict = classify_verdict(score)
    borderline = is_borderline(score)

    coverage = split_requirement_coverage(candidate, requirements)
    experience_assessment, shortfall = assess_experience(candidate, requirements)

    # ---- Strengths -------------------------------------------------------
    strengths: List[str] = []
    matched_skills = candidate.get("matched_skills") or []
    if matched_skills:
        strengths.append(
            f"Matches {len(matched_skills)} required/relevant skill(s): "
            f"{', '.join(_titlecase(_normalise(matched_skills))[:6])}"
        )
    if coverage["must_met"]:
        strengths.append(
            f"Covers {len(coverage['must_met'])} of "
            f"{len(coverage['must_met']) + len(coverage['must_missing'])} hard requirement(s)"
        )
    if coverage["nice_met"]:
        strengths.append(
            f"Brings preferred extras: {', '.join(_titlecase(coverage['nice_met'])[:4])}"
        )
    if shortfall is None and requirements and requirements.get("experience_years"):
        strengths.append(experience_assessment)
    if candidate.get("education_match"):
        strengths.append("Education requirement satisfied")
    if rank == 1:
        strengths.append("Highest-scoring candidate in the current ranking")
    elif total_candidates and rank <= max(1, total_candidates // 3):
        strengths.append(f"Top-third candidate (rank {rank} of {total_candidates})")

    deep = candidate.get("deep_analysis") or {}
    if deep.get("skill_diversity", 0) >= 8:
        strengths.append(f"Broad technical surface ({deep['skill_diversity']} distinct skills)")

    if not strengths:
        strengths.append("No distinguishing strengths surfaced from the available resume text")

    # ---- Gaps ------------------------------------------------------------
    gaps: List[str] = []
    if coverage["must_missing"]:
        gaps.append(
            f"Missing {len(coverage['must_missing'])} hard requirement(s): "
            f"{', '.join(_titlecase(coverage['must_missing'])[:6])}"
        )
    if shortfall:
        gaps.append(experience_assessment)
    if coverage["nice_missing"]:
        gaps.append(
            f"Lacks preferred skill(s): {', '.join(_titlecase(coverage['nice_missing'])[:4])}"
        )
    declared_missing = _normalise(candidate.get("missing_skills"))
    already_reported = set(coverage["must_missing"]) | set(coverage["nice_missing"])
    residual = [s for s in declared_missing if s not in already_reported]
    if residual:
        gaps.append(f"Other unevidenced skills: {', '.join(_titlecase(residual)[:4])}")
    if candidate.get("education_match") is False:
        gaps.append("Education requirement not evidenced")

    if not gaps:
        gaps.append("No material gaps identified against the stated requirements")

    suggestions = build_improvement_suggestions(candidate, coverage, shortfall, score)

    # ---- Narrative -------------------------------------------------------
    must_total = len(coverage["must_met"]) + len(coverage["must_missing"])
    coverage_clause = (
        f"covering {len(coverage['must_met'])}/{must_total} hard requirements"
        if must_total
        else "with no hard requirements specified to check against"
    )
    narrative = (
        f"{name} ranks {rank} of {total_candidates} at {score:.1f}% "
        f"({VERDICT_LABELS[verdict]}), {coverage_clause}. {experience_assessment}"
    )
    if borderline:
        narrative += (
            " Sits in the borderline band, so the decision hinges on verifying the gaps "
            "below rather than on the score itself."
        )

    return MatchReportData(
        candidate_name=name,
        rank=rank,
        match_score=round(score, 1),
        verdict=verdict,
        strengths=strengths,
        gaps=gaps,
        must_have_met=_titlecase(coverage["must_met"]),
        must_have_missing=_titlecase(coverage["must_missing"]),
        nice_to_have_met=_titlecase(coverage["nice_met"]),
        experience_assessment=experience_assessment,
        improvement_suggestions=suggestions,
        is_borderline=borderline,
        narrative=narrative,
    )


def build_match_reports(
    candidates: Sequence[CandidateMatchDict],
    requirements: Optional[JobRequirementDict]
) -> List[MatchReportData]:
    """Build reports for a whole ranking, in rank order."""
    total = len(candidates)
    reports: List[MatchReportData] = []
    for index, candidate in enumerate(candidates, start=1):
        try:
            reports.append(build_match_report(candidate, requirements, index, total))
        except Exception as exc:  # a bad record must not sink the whole report
            logger.error(
                "Failed to build match report for %s: %s",
                candidate.get("candidate_name", "unknown"), exc
            )
    return reports


def collect_borderline(reports: Sequence[MatchReportData]) -> List[str]:
    """Names of candidates in the borderline band."""
    return [r["candidate_name"] for r in reports if r.get("is_borderline")]


def format_match_report(report: MatchReportData) -> str:
    """Render a single match report as readable text."""
    lines = [
        f"{report['rank']}. {report['candidate_name']} - {report['match_score']:.1f}% "
        f"({VERDICT_LABELS.get(report.get('verdict', ''), 'Unrated')})",
        f"   {report.get('narrative', '')}",
        "   Strengths:",
    ]
    for item in report.get("strengths", []):
        lines.append(f"     + {item}")

    lines.append("   Gaps:")
    for item in report.get("gaps", []):
        lines.append(f"     - {item}")

    if report.get("is_borderline"):
        lines.append("   Suggestions to resolve borderline status:")
        for item in report.get("improvement_suggestions", []):
            lines.append(f"     > {item}")

    return "\n".join(lines)


def format_match_reports(
    reports: Sequence[MatchReportData],
    job_title: str = "the role"
) -> str:
    """Render a full set of match reports, with a borderline callout."""
    if not reports:
        return "No candidates available to report on."

    header = f"Detailed match reports for {job_title} ({len(reports)} candidate(s))"
    blocks = [header, "=" * len(header), ""]
    blocks.extend(format_match_report(r) + "\n" for r in reports)

    borderline = collect_borderline(reports)
    if borderline:
        blocks.append(
            f"Borderline candidates needing a decision: {', '.join(borderline)}.\n"
            f"Each has targeted suggestions listed above."
        )
    return "\n".join(blocks)


# =============================================================================
# Side-by-side comparison (Part B.1)
# =============================================================================

def format_comparison_table(
    candidates: Sequence[CandidateMatchDict],
    requirements: Optional[JobRequirementDict] = None
) -> str:
    """
    Render a fixed-width side-by-side comparison table.

    Answers queries of the form "compare the top 3 matches side by side"
    without needing an LLM round-trip.

    Args:
        candidates: Candidates to compare, in rank order
        requirements: Active requirements, used for coverage rows

    Returns:
        Plain-text table
    """
    if not candidates:
        return "No candidates available to compare."
    if len(candidates) == 1:
        return (
            f"Only one candidate available ({candidates[0].get('candidate_name')}). "
            f"At least two are needed for a side-by-side comparison."
        )

    names = [c.get("candidate_name", "Unknown") for c in candidates]
    col_width = max(16, max(len(n) for n in names) + 2)
    label_width = 22

    def row(label: str, values: Sequence[str]) -> str:
        cells = "".join(str(v)[: col_width - 1].ljust(col_width) for v in values)
        return f"{label.ljust(label_width)}{cells}"

    lines = [row("Attribute", names), "-" * (label_width + col_width * len(names))]

    lines.append(row("Score", [f"{candidate_score(c):.1f}%" for c in candidates]))
    lines.append(row("Verdict", [
        VERDICT_LABELS[classify_verdict(candidate_score(c))] for c in candidates
    ]))
    lines.append(row("Rank", [str(i) for i in range(1, len(candidates) + 1)]))

    coverages = [split_requirement_coverage(c, requirements) for c in candidates]
    lines.append(row("Hard reqs met", [
        f"{len(cov['must_met'])}/{len(cov['must_met']) + len(cov['must_missing'])}"
        if (cov["must_met"] or cov["must_missing"]) else "n/a"
        for cov in coverages
    ]))
    lines.append(row("Preferred extras", [str(len(cov["nice_met"])) for cov in coverages]))
    lines.append(row("Skills matched", [
        str(len(c.get("matched_skills") or [])) for c in candidates
    ]))
    lines.append(row("Experience OK", [
        "yes" if c.get("experience_match") else "no" for c in candidates
    ]))
    lines.append(row("Education OK", [
        "yes" if c.get("education_match") else "no" for c in candidates
    ]))

    # Per-candidate detail below the grid, where full skill lists fit.
    lines.append("")
    for candidate, cov in zip(candidates, coverages):
        name = candidate.get("candidate_name", "Unknown")
        lines.append(f"{name}:")
        matched = _titlecase(_normalise(candidate.get("matched_skills")))
        lines.append(f"  Matched:  {', '.join(matched[:8]) if matched else 'none recorded'}")
        if cov["must_missing"]:
            lines.append(f"  Missing:  {', '.join(_titlecase(cov['must_missing'])[:8])}")
        else:
            lines.append("  Missing:  no hard requirements outstanding")

    # A comparison that does not conclude anything is not much use.
    best = max(candidates, key=candidate_score)
    spread = candidate_score(best) - min(candidate_score(c) for c in candidates)
    lines.append("")
    if spread < 5:
        lines.append(
            f"Verdict: {best.get('candidate_name')} leads, but the spread is only "
            f"{spread:.1f} points. Treat these as effectively tied and decide on "
            f"interview signal rather than score."
        )
    else:
        lines.append(
            f"Verdict: {best.get('candidate_name')} leads by {spread:.1f} points over the "
            f"lowest-scoring candidate shown."
        )
    return "\n".join(lines)


# =============================================================================
# Head-to-head explanation (Part B.1)
# =============================================================================

def _find_candidate(
    ranking: Sequence[CandidateMatchDict],
    name_fragment: str
) -> Optional[Tuple[int, CandidateMatchDict]]:
    """Locate a candidate by partial, case-insensitive name match."""
    needle = name_fragment.strip().lower()
    if not needle:
        return None
    for index, candidate in enumerate(ranking, start=1):
        if needle in str(candidate.get("candidate_name", "")).lower():
            return index, candidate
    # Fall back to matching any single name token (handles "John" vs "John Doe")
    for index, candidate in enumerate(ranking, start=1):
        tokens = str(candidate.get("candidate_name", "")).lower().split()
        if any(needle == token for token in tokens):
            return index, candidate
    return None


def explain_head_to_head(
    ranking: Sequence[CandidateMatchDict],
    name_a: str,
    name_b: str,
    requirements: Optional[JobRequirementDict] = None
) -> str:
    """
    Explain why one candidate out-ranked another.

    Handles the "Why did John rank higher than Jane?" class of question by
    attributing the score difference to concrete, checkable factors rather than
    restating the scores.

    Args:
        ranking: Current ranking, best first
        name_a: First candidate name or fragment
        name_b: Second candidate name or fragment
        requirements: Active requirements

    Returns:
        Explanation text
    """
    if not ranking:
        return "There is no active ranking to explain. Run a candidate search first."

    found_a = _find_candidate(ranking, name_a)
    found_b = _find_candidate(ranking, name_b)

    missing = []
    if not found_a:
        missing.append(name_a)
    if not found_b:
        missing.append(name_b)
    if missing:
        available = ", ".join(c.get("candidate_name", "?") for c in ranking[:10])
        return (
            f"Could not find {' or '.join(missing)} in the current ranking. "
            f"Candidates currently ranked: {available}."
        )

    rank_a, cand_a = found_a
    rank_b, cand_b = found_b

    # Normalise so `hi` is always the better-ranked candidate.
    if rank_a <= rank_b:
        (rank_hi, cand_hi), (rank_lo, cand_lo) = (rank_a, cand_a), (rank_b, cand_b)
    else:
        (rank_hi, cand_hi), (rank_lo, cand_lo) = (rank_b, cand_b), (rank_a, cand_a)

    name_hi = cand_hi.get("candidate_name", "Candidate A")
    name_lo = cand_lo.get("candidate_name", "Candidate B")
    score_hi, score_lo = candidate_score(cand_hi), candidate_score(cand_lo)
    gap = score_hi - score_lo

    cov_hi = split_requirement_coverage(cand_hi, requirements)
    cov_lo = split_requirement_coverage(cand_lo, requirements)

    lines = [
        f"{name_hi} (rank {rank_hi}, {score_hi:.1f}%) ranked above "
        f"{name_lo} (rank {rank_lo}, {score_lo:.1f}%) by {gap:.1f} points.",
        "",
        "Drivers of the difference:",
    ]

    # Factor 1: hard requirement coverage - the heaviest driver.
    if len(cov_hi["must_met"]) != len(cov_lo["must_met"]):
        better, worse = (
            (name_hi, name_lo) if len(cov_hi["must_met"]) > len(cov_lo["must_met"])
            else (name_lo, name_hi)
        )
        lines.append(
            f"  - Hard requirements: {name_hi} covers {len(cov_hi['must_met'])}, "
            f"{name_lo} covers {len(cov_lo['must_met'])}. Advantage {better}."
        )
        exclusive = set(cov_lo["must_missing"]) - set(cov_hi["must_missing"])
        if exclusive:
            lines.append(
                f"    {name_lo} is missing {', '.join(_titlecase(sorted(exclusive))[:4])}, "
                f"which {name_hi} evidences."
            )
    else:
        lines.append(
            f"  - Hard requirements: both cover {len(cov_hi['must_met'])}, so this was not "
            f"the differentiator."
        )

    # Factor 2: raw skill overlap.
    skills_hi = len(cand_hi.get("matched_skills") or [])
    skills_lo = len(cand_lo.get("matched_skills") or [])
    if skills_hi != skills_lo:
        lines.append(
            f"  - Skill overlap: {skills_hi} matched skills vs {skills_lo}."
        )

    # Factor 3: experience.
    exp_hi, short_hi = assess_experience(cand_hi, requirements)
    exp_lo, short_lo = assess_experience(cand_lo, requirements)
    if (short_hi or 0) != (short_lo or 0):
        lines.append(f"  - Experience: {name_hi} - {exp_hi}")
        lines.append(f"                {name_lo} - {exp_lo}")

    # Factor 4: preferred extras act as tie-breakers.
    if len(cov_hi["nice_met"]) != len(cov_lo["nice_met"]):
        lines.append(
            f"  - Preferred extras: {len(cov_hi['nice_met'])} vs {len(cov_lo['nice_met'])} "
            f"(tie-breaker weight only)."
        )

    # Factor 5: education.
    if bool(cand_hi.get("education_match")) != bool(cand_lo.get("education_match")):
        holder = name_hi if cand_hi.get("education_match") else name_lo
        lines.append(f"  - Education: requirement evidenced only by {holder}.")

    lines.append("")
    if gap < 5:
        lines.append(
            f"Caveat: a {gap:.1f}-point gap is inside the noise of this scoring model. "
            f"{name_hi} and {name_lo} should be treated as comparable, and the ordering "
            f"here should not by itself decide who advances."
        )
    else:
        lines.append(
            f"The {gap:.1f}-point gap is material and driven mainly by the factors above."
        )
    return "\n".join(lines)


# =============================================================================
# Ranking deltas (Part B.2)
# =============================================================================

def compute_ranking_delta(
    previous: Sequence[CandidateMatchDict],
    current: Sequence[CandidateMatchDict],
    trigger: str = "criteria refinement",
    criteria_before: Optional[Dict[str, Any]] = None,
    criteria_after: Optional[Dict[str, Any]] = None
) -> RankingChangeData:
    """
    Diff two rankings so the agent can explain what a refinement changed.

    Args:
        previous: Ranking before the change, best first
        current: Ranking after the change, best first
        trigger: What caused the re-rank, quoted back to the user
        criteria_before: Criteria snapshot before
        criteria_after: Criteria snapshot after

    Returns:
        Populated RankingChangeData including a ready-to-print narrative
    """
    prev_pos = {
        c.get("candidate_name"): i
        for i, c in enumerate(previous, start=1) if c.get("candidate_name")
    }
    curr_pos = {
        c.get("candidate_name"): i
        for i, c in enumerate(current, start=1) if c.get("candidate_name")
    }
    prev_scores = {
        c.get("candidate_name"): candidate_score(c)
        for c in previous if c.get("candidate_name")
    }
    curr_scores = {
        c.get("candidate_name"): candidate_score(c)
        for c in current if c.get("candidate_name")
    }

    entered = [n for n in curr_pos if n not in prev_pos]
    dropped = [n for n in prev_pos if n not in curr_pos]

    moved_up: List[Dict[str, Any]] = []
    moved_down: List[Dict[str, Any]] = []
    unchanged: List[str] = []

    for name, new_rank in curr_pos.items():
        old_rank = prev_pos.get(name)
        if old_rank is None:
            continue
        movement = {
            "candidate_name": name,
            "from_rank": old_rank,
            "to_rank": new_rank,
            "positions": abs(old_rank - new_rank),
            "score_before": round(prev_scores.get(name, 0), 1),
            "score_after": round(curr_scores.get(name, 0), 1),
        }
        if new_rank < old_rank:
            moved_up.append(movement)
        elif new_rank > old_rank:
            moved_down.append(movement)
        else:
            unchanged.append(name)

    moved_up.sort(key=lambda m: -m["positions"])
    moved_down.sort(key=lambda m: -m["positions"])

    score_deltas = {
        name: round(curr_scores[name] - prev_scores.get(name, 0.0), 1)
        for name in curr_scores if name in prev_scores
    }

    narrative = _narrate_ranking_change(
        trigger, entered, dropped, moved_up, moved_down, unchanged,
        previous, current, criteria_before, criteria_after
    )

    return RankingChangeData(
        trigger=trigger,
        criteria_before=criteria_before or {},
        criteria_after=criteria_after or {},
        entered=entered,
        dropped=dropped,
        moved_up=moved_up,
        moved_down=moved_down,
        unchanged=unchanged,
        score_deltas=score_deltas,
        narrative=narrative,
    )


def _narrate_ranking_change(
    trigger: str,
    entered: List[str],
    dropped: List[str],
    moved_up: List[Dict[str, Any]],
    moved_down: List[Dict[str, Any]],
    unchanged: List[str],
    previous: Sequence[CandidateMatchDict],
    current: Sequence[CandidateMatchDict],
    criteria_before: Optional[Dict[str, Any]],
    criteria_after: Optional[Dict[str, Any]]
) -> str:
    """Turn a ranking diff into prose."""
    if not previous:
        return (
            f"Initial ranking produced {len(current)} candidate(s). "
            f"No prior ranking to compare against."
        )

    lines = [f"Ranking updated after {trigger}."]

    # Surface exactly which criteria moved, when we know.
    if criteria_before and criteria_after:
        changes = []
        for key in sorted(set(criteria_before) | set(criteria_after)):
            before, after = criteria_before.get(key), criteria_after.get(key)
            if before != after:
                changes.append(f"{key}: {before!r} -> {after!r}")
        if changes:
            lines.append("Criteria changed: " + "; ".join(changes))

    old_top = previous[0].get("candidate_name") if previous else None
    new_top = current[0].get("candidate_name") if current else None
    if new_top and old_top and new_top != old_top:
        lines.append(f"Top candidate changed: {old_top} -> {new_top}.")
    elif new_top:
        lines.append(f"Top candidate unchanged: {new_top}.")

    if entered:
        lines.append(f"Entered the ranking: {', '.join(entered[:6])}.")
    if dropped:
        lines.append(
            f"Dropped out (no longer meets criteria): {', '.join(dropped[:6])}."
        )
    for movement in moved_up[:4]:
        lines.append(
            f"  Up {movement['positions']}: {movement['candidate_name']} "
            f"rank {movement['from_rank']} -> {movement['to_rank']} "
            f"({movement['score_before']}% -> {movement['score_after']}%)"
        )
    for movement in moved_down[:4]:
        lines.append(
            f"  Down {movement['positions']}: {movement['candidate_name']} "
            f"rank {movement['from_rank']} -> {movement['to_rank']} "
            f"({movement['score_before']}% -> {movement['score_after']}%)"
        )

    if not entered and not dropped and not moved_up and not moved_down:
        lines.append(
            "No candidate changed position. The refinement narrowed the criteria without "
            "altering relative standing."
        )
    elif unchanged:
        lines.append(f"Held position: {len(unchanged)} candidate(s).")

    return "\n".join(lines)


def format_ranking_change(change: Optional[RankingChangeData]) -> str:
    """Render a ranking delta for display."""
    if not change:
        return "No ranking change recorded."
    return change.get("narrative", "Ranking changed.")


def build_refinement_record(
    turn: int,
    user_request: str,
    parsed_adjustments: Dict[str, Any],
    criteria_before: Dict[str, Any],
    criteria_after: Dict[str, Any]
) -> Dict[str, Any]:
    """Build an audit record for one mid-conversation criteria adjustment."""
    return {
        "turn": turn,
        "timestamp": time.time(),
        "user_request": user_request,
        "parsed_adjustments": parsed_adjustments,
        "criteria_before": criteria_before,
        "criteria_after": criteria_after,
    }
