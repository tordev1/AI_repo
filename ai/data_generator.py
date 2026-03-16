"""
Synthetic Student Dataset Generator
Generates 1000+ realistic student records for training the ML risk model.
"""

import pandas as pd
import numpy as np
import os

SEED = 42
NUM_RECORDS = 1200


def generate_dataset(n=NUM_RECORDS, seed=SEED):
    rng = np.random.RandomState(seed)

    genders = rng.choice(["male", "female", "non-binary"], size=n, p=[0.45, 0.45, 0.10])

    # Workload: number of active modules/subjects (1-6)
    workload = rng.randint(1, 7, size=n)

    # Study hours per week (2-40, skewed toward lower values)
    study_hours = np.clip(rng.gamma(4, 3, size=n), 2, 40).round(1)

    # Confidence (1-10 scale)
    confidence = np.clip(rng.normal(6, 2, size=n), 1, 10).round(1)

    # Stress (1-10 scale)
    stress = np.clip(rng.normal(5.5, 2.5, size=n), 1, 10).round(1)

    # Days to nearest deadline (1-60)
    days_to_deadline = rng.randint(1, 61, size=n)

    # --- Derive risk label from features using logical rules ---
    risk = []
    for i in range(n):
        score = 0.0

        # High stress increases risk
        if stress[i] >= 8:
            score += 3
        elif stress[i] >= 6:
            score += 1.5

        # Low confidence increases risk
        if confidence[i] <= 3:
            score += 3
        elif confidence[i] <= 5:
            score += 1.5

        # Low study hours relative to workload
        ratio = study_hours[i] / max(workload[i], 1)
        if ratio < 3:
            score += 2.5
        elif ratio < 5:
            score += 1

        # Close deadline
        if days_to_deadline[i] <= 3:
            score += 3
        elif days_to_deadline[i] <= 7:
            score += 1.5
        elif days_to_deadline[i] <= 14:
            score += 0.5

        # High workload
        if workload[i] >= 5:
            score += 1.5
        elif workload[i] >= 4:
            score += 0.5

        # Add noise
        score += rng.normal(0, 0.8)

        if score >= 6:
            risk.append("high")
        elif score >= 3.5:
            risk.append("medium")
        else:
            risk.append("low")

    df = pd.DataFrame({
        "gender": genders,
        "workload": workload,
        "study_hours": study_hours,
        "confidence": confidence,
        "stress": stress,
        "days_to_deadline": days_to_deadline,
        "risk": risk,
    })

    return df


def main():
    df = generate_dataset()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_students.csv")
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} records -> {out_path}")
    print(f"\nRisk distribution:\n{df['risk'].value_counts()}")
    print(f"\nSample rows:\n{df.head()}")


if __name__ == "__main__":
    main()
