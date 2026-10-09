"""
Student Performance Explorer
IEEE LGU AI/ML Cohort One - Week 3: Data Analysis with NumPy & Pandas
AI/ML Leads: Abdullah Faisal & Alina Irshad
"""

import numpy as np
import pandas as pd

DATA_PATH = "data/students.csv"
SCORE_COLS = ["quiz_score", "assignment_score", "exam_score"]

# Weights used for the final mark (must add up to 1.0)
WEIGHTS = {"quiz_score": 0.20, "assignment_score": 0.30, "exam_score": 0.50}


def load_data(path):
    """Load the raw student CSV into a DataFrame."""
    return pd.read_csv(path)


def clean_data(df):
    """Clean the raw data and print what was changed at every step."""
    print("=" * 60)
    print("DATA CLEANING REPORT")
    print("=" * 60)
    print(f"Raw rows loaded          : {len(df)}")

    # 1. Remove duplicate records. The same student can be entered twice
    #    with a different student_id, so compare every column EXCEPT student_id.
    before = len(df)
    content_cols = [c for c in df.columns if c != "student_id"]
    df = df.drop_duplicates(subset=content_cols, keep="first")
    print(f"Duplicate rows removed   : {before - len(df)}")

    # 2. Convert score columns to numbers; text like 'abc' becomes NaN
    for col in SCORE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Scores must be between 0 and 100, otherwise invalid
    for col in SCORE_COLS:
        df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan

    print("Missing/invalid values before cleaning:")
    for col in SCORE_COLS:
        print(f"   {col:<18}: {df[col].isna().sum()}")

    # 4. Fill missing scores with the MEDIAN of the same subject.
    #    Median is used because it is not pulled by very high/low marks.
    for col in SCORE_COLS:
        df[col] = df.groupby("subject")[col].transform(lambda s: s.fillna(s.median()))

    print(f"Missing values after fill: {int(df[SCORE_COLS].isna().sum().sum())}")
    print(f"Clean rows remaining     : {len(df)}")
    return df.reset_index(drop=True)


def add_final_mark(df):
    """Final mark = weighted average of the three scores."""
    scores = df[SCORE_COLS].to_numpy()
    weights = np.array([WEIGHTS[c] for c in SCORE_COLS])
    df["final_mark"] = np.round(scores @ weights, 2)
    return df


def class_statistics(df):
    """Class-wide descriptive statistics using NumPy arrays."""
    marks = df["final_mark"].to_numpy()
    return {
        "mean": np.mean(marks),
        "std": np.std(marks),
        "median": np.median(marks),
        "min": np.min(marks),
        "max": np.max(marks),
    }


def print_report(df, stats):
    print("\n" + "=" * 60)
    print("STUDENT PERFORMANCE SUMMARY REPORT")
    print("=" * 60)
    print(f"Total Students     : {len(df)}")
    print(f"Mean Final Mark    : {stats['mean']:.2f}")
    print(f"Standard Deviation : {stats['std']:.2f}")
    print(f"Median Final Mark  : {stats['median']:.2f}")
    print(f"Minimum Final Mark : {stats['min']:.2f}")
    print(f"Maximum Final Mark : {stats['max']:.2f}")

    # Subject ranking
    subject_avg = (
        df.groupby("subject")["final_mark"].mean().sort_values(ascending=False)
    )
    print("\n--- Subject Ranking (average final mark) ---")
    for rank, (name, val) in enumerate(subject_avg.items(), 1):
        print(f"{rank}. {name:<18} {val:6.2f}")
    print(f"Strongest subject: {subject_avg.index[0]}")
    print(f"Weakest subject  : {subject_avg.index[-1]}")

    # Section comparison
    section_avg = df.groupby("section")["final_mark"].mean().sort_values(ascending=False)
    print("\n--- Section Comparison (average final mark) ---")
    for name, val in section_avg.items():
        print(f"{name:<10} {val:6.2f}")

    # Filtering by threshold
    high_exam = df[df["exam_score"] >= 80]
    print(f"\n--- Students with exam score >= 80: {len(high_exam)} ---")
    print(high_exam[["student_id", "name", "subject", "exam_score"]]
          .sort_values("exam_score", ascending=False)
          .to_string(index=False))

    # Top performers
    top5 = df.sort_values("final_mark", ascending=False).head(5)
    print("\n--- Top 5 Students ---")
    print(top5[["student_id", "name", "subject", "final_mark"]].to_string(index=False))

    # Students needing support
    low = df[df["final_mark"] < 60]
    print(f"\n--- Students below 60 final mark: {len(low)} ---")
    if len(low):
        print(low[["student_id", "name", "subject", "final_mark"]].to_string(index=False))


def main():
    raw = load_data(DATA_PATH)
    clean = clean_data(raw)
    clean = add_final_mark(clean)
    stats = class_statistics(clean)
    print_report(clean, stats)


if __name__ == "__main__":
    main()
