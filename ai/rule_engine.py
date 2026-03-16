"""
Rule-Based Expert System
Forward chaining: derives conclusions from student facts.
Backward chaining: traces which facts caused a given conclusion.
~20 rules covering stress, workload, confidence, deadline proximity.
"""


class Rule:
    def __init__(self, rule_id, description, conditions, conclusions):
        """
        conditions: dict of fact_name -> callable(value) -> bool
        conclusions: list of (fact_name, value) to assert
        """
        self.rule_id = rule_id
        self.description = description
        self.conditions = conditions
        self.conclusions = conclusions

    def matches(self, facts):
        for fact_name, test_fn in self.conditions.items():
            if fact_name not in facts:
                return False
            if not test_fn(facts[fact_name]):
                return False
        return True

    def __repr__(self):
        return f"Rule({self.rule_id}: {self.description})"


# ── Define Rules ──────────────────────────────────────────────

RULES = [
    # Stress rules
    Rule("R1", "Very high stress indicates burnout risk",
         {"stress": lambda v: v >= 8},
         [("stress_level", "very_high"), ("burnout_risk", True)]),
    Rule("R2", "High stress detected",
         {"stress": lambda v: 6 <= v < 8},
         [("stress_level", "high")]),
    Rule("R3", "Moderate stress detected",
         {"stress": lambda v: 4 <= v < 6},
         [("stress_level", "moderate")]),
    Rule("R4", "Low stress detected",
         {"stress": lambda v: v < 4},
         [("stress_level", "low")]),

    # Confidence rules
    Rule("R5", "Very low confidence — needs encouragement",
         {"confidence": lambda v: v <= 3},
         [("confidence_level", "very_low"), ("needs_encouragement", True)]),
    Rule("R6", "Low confidence detected",
         {"confidence": lambda v: 3 < v <= 5},
         [("confidence_level", "low")]),
    Rule("R7", "Good confidence detected",
         {"confidence": lambda v: 5 < v <= 7.5},
         [("confidence_level", "good")]),
    Rule("R8", "High confidence detected",
         {"confidence": lambda v: v > 7.5},
         [("confidence_level", "high")]),

    # Deadline rules
    Rule("R9", "Urgent deadline — within 3 days",
         {"days_to_deadline": lambda v: v <= 3},
         [("deadline_urgency", "critical"), ("needs_immediate_action", True)]),
    Rule("R10", "Close deadline — within 7 days",
         {"days_to_deadline": lambda v: 3 < v <= 7},
         [("deadline_urgency", "high")]),
    Rule("R11", "Approaching deadline — within 14 days",
         {"days_to_deadline": lambda v: 7 < v <= 14},
         [("deadline_urgency", "moderate")]),
    Rule("R12", "Deadline is far away",
         {"days_to_deadline": lambda v: v > 14},
         [("deadline_urgency", "low")]),

    # Workload rules
    Rule("R13", "Heavy workload — 5+ modules",
         {"workload": lambda v: v >= 5},
         [("workload_level", "heavy"), ("overloaded", True)]),
    Rule("R14", "Moderate workload — 3-4 modules",
         {"workload": lambda v: 3 <= v < 5},
         [("workload_level", "moderate")]),
    Rule("R15", "Light workload — 1-2 modules",
         {"workload": lambda v: v < 3},
         [("workload_level", "light")]),

    # Study hours relative to workload
    Rule("R16", "Insufficient study time for workload",
         {"study_hours": lambda v: v is not None, "workload": lambda v: v is not None},
         []),  # conclusions added dynamically below

    # Combined risk rules
    Rule("R17", "High risk: high stress + low confidence",
         {"stress_level": lambda v: v in ("very_high", "high"),
          "confidence_level": lambda v: v in ("very_low", "low")},
         [("risk_flag", "high"), ("recommendation", "Seek academic support and reduce workload if possible")]),

    Rule("R18", "High risk: critical deadline + heavy workload",
         {"deadline_urgency": lambda v: v == "critical",
          "workload_level": lambda v: v == "heavy"},
         [("risk_flag", "high"), ("recommendation", "Prioritise the nearest deadline and defer non-urgent tasks")]),

    Rule("R19", "Medium risk: high stress or low confidence alone",
         {"stress_level": lambda v: v in ("very_high", "high")},
         [("risk_contributing_factor", "stress")]),

    Rule("R20", "Low risk: good balance across factors",
         {"stress_level": lambda v: v in ("low", "moderate"),
          "confidence_level": lambda v: v in ("good", "high"),
          "deadline_urgency": lambda v: v in ("low", "moderate")},
         [("risk_flag", "low"), ("recommendation", "You are on track — maintain your current routine")]),

    Rule("R21", "Burnout risk requires rest recommendation",
         {"burnout_risk": lambda v: v is True},
         [("recommendation", "Schedule regular breaks to avoid burnout")]),

    Rule("R22", "Encouragement needed for low confidence",
         {"needs_encouragement": lambda v: v is True},
         [("recommendation", "Start with easier tasks to build momentum and confidence")]),
]


def _check_study_ratio(facts):
    """Special logic for R16 — study hours vs workload ratio."""
    if "study_hours" in facts and "workload" in facts:
        ratio = facts["study_hours"] / max(facts["workload"], 1)
        if ratio < 3:
            return [("study_adequacy", "insufficient"),
                    ("recommendation", "Increase weekly study hours — aim for at least 3 hours per module")]
        elif ratio < 5:
            return [("study_adequacy", "borderline")]
        else:
            return [("study_adequacy", "adequate")]
    return []


# ── Forward Chaining ──────────────────────────────────────────

def forward_chain(facts):
    """
    Iteratively fire rules until no new facts are derived.
    Returns: (derived_facts, fired_rules) where fired_rules is a list of (Rule, new conclusions).
    """
    working = dict(facts)
    fired = []
    fired_ids = set()
    changed = True

    while changed:
        changed = False
        for rule in RULES:
            if rule.rule_id in fired_ids:
                continue

            if rule.rule_id == "R16":
                # Special rule with dynamic conclusions
                extra = _check_study_ratio(working)
                if extra:
                    new_facts = []
                    for fname, fval in extra:
                        if fname not in working or (fname == "recommendation" and fval != working.get(fname)):
                            working.setdefault(fname, fval)
                            # For recommendations, collect them as a list
                            new_facts.append((fname, fval))
                    if new_facts:
                        fired.append((rule, new_facts))
                        fired_ids.add(rule.rule_id)
                        changed = True
                continue

            if rule.matches(working):
                new_facts = []
                for fname, fval in rule.conclusions:
                    if fname not in working:
                        working[fname] = fval
                        new_facts.append((fname, fval))
                if new_facts:
                    fired.append((rule, new_facts))
                    fired_ids.add(rule.rule_id)
                    changed = True

    return working, fired


# ── Backward Chaining ─────────────────────────────────────────

def backward_chain(facts, goal_fact, goal_value=None):
    """
    Given a goal (e.g., risk_flag=high), trace back which rules and base facts
    would lead to that conclusion.
    Returns a list of explanation strings.
    """
    explanations = []
    _backward_recurse(facts, goal_fact, goal_value, explanations, visited=set())
    return explanations


def _backward_recurse(facts, goal_fact, goal_value, explanations, visited):
    for rule in RULES:
        if rule.rule_id in visited:
            continue

        # Check if this rule concludes the goal
        concludes_goal = False
        for fname, fval in rule.conclusions:
            if fname == goal_fact:
                if goal_value is None or fval == goal_value:
                    concludes_goal = True
                    break

        if not concludes_goal:
            continue

        visited.add(rule.rule_id)

        # Check which conditions are met
        if rule.matches(facts):
            condition_strs = []
            for fact_name in rule.conditions:
                if fact_name in facts:
                    condition_strs.append(f"{fact_name} = {facts[fact_name]}")
            explanations.append(
                f"[{rule.rule_id}] {rule.description} "
                f"(because {', '.join(condition_strs)})"
            )
            # Recurse: check if any condition fact was itself derived by another rule
            for fact_name in rule.conditions:
                _backward_recurse(facts, fact_name, facts.get(fact_name), explanations, visited)


# ── Convenience Functions ─────────────────────────────────────

def assess_student(student_data):
    """
    Main entry point: take raw student data, run forward chaining,
    collect risk level and recommendations.
    Returns dict with risk_level, recommendations, fired_rules, all_facts.
    """
    facts = dict(student_data)  # copy
    all_facts, fired = forward_chain(facts)

    # Determine overall risk
    risk = all_facts.get("risk_flag", "medium")  # default to medium if no definitive rule fired

    # Collect all recommendations
    recommendations = []
    for rule, new_facts in fired:
        for fname, fval in new_facts:
            if fname == "recommendation":
                recommendations.append(fval)

    # If no recommendations were generated, add a default
    if not recommendations:
        if risk == "medium":
            recommendations.append("Consider balancing your workload and managing stress proactively.")

    return {
        "risk_level": risk,
        "recommendations": recommendations,
        "fired_rules": [(r.rule_id, r.description) for r, _ in fired],
        "all_facts": all_facts,
    }


def explain_risk(all_facts, risk_level):
    """Use backward chaining to explain why a risk level was assigned."""
    explanations = backward_chain(all_facts, "risk_flag", risk_level)
    if not explanations:
        # Try explaining contributing factors
        for goal in ["stress_level", "confidence_level", "deadline_urgency", "workload_level", "study_adequacy"]:
            if goal in all_facts:
                more = backward_chain(all_facts, goal, all_facts[goal])
                explanations.extend(more)
    return explanations


# ── Main (demo) ───────────────────────────────────────────────

def main():
    student = {
        "stress": 8.5,
        "confidence": 3.0,
        "days_to_deadline": 4,
        "workload": 5,
        "study_hours": 8,
        "gender": "female",
    }

    print("=== Forward Chaining ===")
    result = assess_student(student)
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommendations:")
    for r in result["recommendations"]:
        print(f"  - {r}")
    print(f"\nFired Rules:")
    for rid, desc in result["fired_rules"]:
        print(f"  [{rid}] {desc}")

    print("\n=== Backward Chaining ===")
    explanations = explain_risk(result["all_facts"], result["risk_level"])
    for exp in explanations:
        print(f"  {exp}")


if __name__ == "__main__":
    main()
