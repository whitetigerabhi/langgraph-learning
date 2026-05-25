import argparse
import random
import numpy as np
import pandas as pd

def generate(n=4000, seed=7):
    random.seed(seed)
    np.random.seed(seed)

    n_both = int(0.08 * n)
    n_diab_only = int(0.22 * n)
    n_asth_only = int(0.17 * n)
    n_none = n - (n_both + n_diab_only + n_asth_only)

    labels = [3]*n_both + [1]*n_diab_only + [2]*n_asth_only + [0]*n_none
    random.shuffle(labels)

    rows = []
    for member_id, g in enumerate(labels, start=1):
        diabetes = 1 if g in (1, 3) else 0
        asthma = 1 if g in (2, 3) else 0

        base = 0.05
        if g == 1: base = 0.10
        if g == 2: base = 0.08
        if g == 3: base = 0.12

        cortico = (np.random.rand() < (0.78 if g==3 else 0.50 if g==2 else 0.10 if g==1 else 0.04))
        cortico = int(cortico)

        if cortico:
            if g==3:
                fill_6 = np.random.rand() < 0.14
                fill_4 = (not fill_6) and (np.random.rand() < 0.32)
                fill_2 = (not fill_6 and not fill_4) and (np.random.rand() < 0.62)
            elif g==2:
                fill_6 = np.random.rand() < 0.07
                fill_4 = (not fill_6) and (np.random.rand() < 0.16)
                fill_2 = (not fill_6 and not fill_4) and (np.random.rand() < 0.60)
            elif g==1:
                fill_6 = np.random.rand() < 0.04
                fill_4 = (not fill_6) and (np.random.rand() < 0.09)
                fill_2 = (not fill_6 and not fill_4) and (np.random.rand() < 0.45)
            else:
                fill_6 = np.random.rand() < 0.02
                fill_4 = (not fill_6) and (np.random.rand() < 0.05)
                fill_2 = (not fill_6 and not fill_4) and (np.random.rand() < 0.35)
        else:
            fill_2 = fill_4 = fill_6 = False

        fill_2 = int(fill_2); fill_4 = int(fill_4); fill_6 = int(fill_6)

        no_eye_exam = (np.random.rand() < (0.44 if g==1 else 0.24 if g==3 else 0.18 if g==2 else 0.08))
        no_eye_exam = int(no_eye_exam)

        retinopathy = (np.random.rand() < (0.36 if g==1 else 0.18 if g==3 else 0.05 if g==2 else 0.02))
        retinopathy = int(retinopathy)

        age = int(np.clip(np.random.normal(24, 7), 10, 85)) if g in (2,3) else int(np.clip(np.random.normal(40, 11), 10, 85))

        prob = base
        prob += 0.05*diabetes
        prob += 0.03*asthma
        prob += 0.04*no_eye_exam
        prob += 0.10*retinopathy
        prob += 0.03*cortico
        prob += 0.06*fill_2
        prob += 0.11*fill_4
        prob += 0.16*fill_6

        # interaction to make realized != unrealized in combined cohort
        if diabetes and asthma:
            prob += 0.06*fill_4 + 0.08*fill_6
            prob -= 0.03*retinopathy

        prob = float(np.clip(prob + np.random.normal(0, 0.02), 0, 0.98))

        rows.append({
            "member_id": member_id,
            "asthma": asthma,
            "diabetes": diabetes,
            "cortico": cortico,
            "fill_2": fill_2,
            "fill_4": fill_4,
            "fill_6": fill_6,
            "no_eye_exam": no_eye_exam,
            "retinopathy": retinopathy,
            "age": age,
            "future_avoidable_ed_prob": round(prob, 6),
        })

    return pd.DataFrame(rows)

def write_sql(df: pd.DataFrame, out_path: str):
    create_sql = """
DROP TABLE IF EXISTS members;
CREATE TABLE cohort_data.members (
  member_id INT PRIMARY KEY,
  asthma INT,
  diabetes INT,
  cortico INT,
  fill_2 INT,
  fill_4 INT,
  fill_6 INT,
  no_eye_exam INT,
  retinopathy INT,
  age INT,
  future_avoidable_ed_prob DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_members_diabetes ON cohort_data.members(diabetes);
CREATE INDEX IF NOT EXISTS idx_members_asthma ON cohort_data.members(asthma);
CREATE INDEX IF NOT EXISTS idx_members_cortico ON cohort_data.members(cortico);
CREATE INDEX IF NOT EXISTS idx_members_fill4 ON cohort_data.members(fill_4);
CREATE INDEX IF NOT EXISTS idx_members_fill6 ON cohort_data.members(fill_6);
CREATE INDEX IF NOT EXISTS idx_members_no_eye_exam ON cohort_data.members(no_eye_exam);
CREATE INDEX IF NOT EXISTS idx_members_retinopathy ON cohort_data.members(retinopathy);
""".strip()

    cols = list(df.columns)
    values = []
    for row in df.itertuples(index=False, name=None):
        values.append("(" + ",".join(str(x) for x in row) + ")")

    chunk_size = 800
    insert_prefix = "INSERT INTO members (" + ", ".join(cols) + ") VALUES\n"
    chunks = []
    for i in range(0, len(values), chunk_size):
        chunks.append(insert_prefix + ",\n".join(values[i:i+chunk_size]) + ";")

    sql = create_sql + "\n\n" + "\n\n".join(chunks)

    with open(out_path, "w") as f:
        f.write(sql)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sql/seed_members.sql")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = generate(n=args.n, seed=args.seed)
    write_sql(df, args.out)
    print(f"Wrote seed SQL to: {args.out}  (rows={len(df)})")

if __name__ == "__main__":
    main()
