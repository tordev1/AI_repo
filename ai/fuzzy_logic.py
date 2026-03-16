"""
Fuzzy Logic Module (Optional Enhancement)
Provides fuzzy membership functions for stress and confidence,
yielding more nuanced categorisation than crisp thresholds.
"""

import numpy as np

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False


def create_fuzzy_system():
    """Build the fuzzy inference system for risk assessment."""
    if not FUZZY_AVAILABLE:
        return None

    # Define universe of discourse
    stress = ctrl.Antecedent(np.arange(0, 11, 0.1), "stress")
    confidence = ctrl.Antecedent(np.arange(0, 11, 0.1), "confidence")
    risk = ctrl.Consequent(np.arange(0, 11, 0.1), "risk")

    # Stress membership functions
    stress["low"] = fuzz.trimf(stress.universe, [0, 0, 4])
    stress["moderate"] = fuzz.trimf(stress.universe, [3, 5, 7])
    stress["high"] = fuzz.trimf(stress.universe, [6, 8, 10])
    stress["very_high"] = fuzz.trimf(stress.universe, [8, 10, 10])

    # Confidence membership functions
    confidence["very_low"] = fuzz.trimf(confidence.universe, [0, 0, 3])
    confidence["low"] = fuzz.trimf(confidence.universe, [2, 4, 6])
    confidence["good"] = fuzz.trimf(confidence.universe, [5, 7, 9])
    confidence["high"] = fuzz.trimf(confidence.universe, [7, 10, 10])

    # Risk output membership functions
    risk["low"] = fuzz.trimf(risk.universe, [0, 0, 4])
    risk["medium"] = fuzz.trimf(risk.universe, [3, 5, 7])
    risk["high"] = fuzz.trimf(risk.universe, [6, 10, 10])

    # Fuzzy rules
    rules = [
        ctrl.Rule(stress["very_high"] & confidence["very_low"], risk["high"]),
        ctrl.Rule(stress["high"] & confidence["low"], risk["high"]),
        ctrl.Rule(stress["high"] & confidence["very_low"], risk["high"]),
        ctrl.Rule(stress["moderate"] & confidence["low"], risk["medium"]),
        ctrl.Rule(stress["moderate"] & confidence["good"], risk["medium"]),
        ctrl.Rule(stress["high"] & confidence["good"], risk["medium"]),
        ctrl.Rule(stress["low"] & confidence["good"], risk["low"]),
        ctrl.Rule(stress["low"] & confidence["high"], risk["low"]),
        ctrl.Rule(stress["moderate"] & confidence["high"], risk["low"]),
        ctrl.Rule(stress["low"] & confidence["low"], risk["medium"]),
        ctrl.Rule(stress["low"] & confidence["very_low"], risk["medium"]),
        ctrl.Rule(stress["very_high"] & confidence["high"], risk["medium"]),
        ctrl.Rule(stress["very_high"] & confidence["good"], risk["high"]),
        ctrl.Rule(stress["moderate"] & confidence["very_low"], risk["high"]),
        ctrl.Rule(stress["high"] & confidence["high"], risk["medium"]),
        ctrl.Rule(stress["very_high"] & confidence["low"], risk["high"]),
    ]

    system = ctrl.ControlSystem(rules)
    simulator = ctrl.ControlSystemSimulation(system)
    return simulator


def fuzzy_risk_assessment(stress_val, confidence_val, simulator=None):
    """
    Run fuzzy inference to get a risk score (0-10) and category.
    Returns (risk_score, risk_category, membership_info) or None if fuzzy not available.
    """
    if not FUZZY_AVAILABLE or simulator is None:
        return None

    # Clamp inputs
    stress_val = max(0.1, min(9.9, stress_val))
    confidence_val = max(0.1, min(9.9, confidence_val))

    simulator.input["stress"] = stress_val
    simulator.input["confidence"] = confidence_val

    try:
        simulator.compute()
        risk_score = simulator.output["risk"]
    except Exception:
        return None

    # Categorise
    if risk_score >= 6.5:
        category = "high"
    elif risk_score >= 3.5:
        category = "medium"
    else:
        category = "low"

    # Get membership degrees for explainability
    membership_info = {
        "stress_input": stress_val,
        "confidence_input": confidence_val,
        "risk_score": round(risk_score, 2),
        "risk_category": category,
    }

    return membership_info


# Singleton simulator
_simulator = None


def get_simulator():
    global _simulator
    if _simulator is None:
        _simulator = create_fuzzy_system()
    return _simulator


def main():
    if not FUZZY_AVAILABLE:
        print("scikit-fuzzy not installed. Install with: pip install scikit-fuzzy")
        return

    sim = get_simulator()

    test_cases = [
        (9, 2),   # very high stress, very low confidence
        (5, 5),   # moderate stress, moderate confidence
        (2, 8),   # low stress, high confidence
        (7, 4),   # high stress, low confidence
    ]

    for stress_val, conf_val in test_cases:
        result = fuzzy_risk_assessment(stress_val, conf_val, sim)
        if result:
            print(f"Stress={stress_val}, Confidence={conf_val} -> "
                  f"Risk Score={result['risk_score']}, Category={result['risk_category']}")
        else:
            print(f"Stress={stress_val}, Confidence={conf_val} -> Could not compute")


if __name__ == "__main__":
    main()
