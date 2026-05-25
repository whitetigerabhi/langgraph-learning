import os
import psycopg2

def get_conn():
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        port=int(os.environ.get("PGPORT", "5432")),
        sslmode=os.environ.get("PGSSLMODE", "require"),
    )

def cohort_metrics(cur, sql_list: list[str]) -> tuple[int, float]:
    """
    Compute:
      - member_count
      - avg(future_avoidable_ed_prob)
    for the INTERSECT of member_id subqueries.
    """
    intersect = " INTERSECT ".join([f"({s})" for s in sql_list])

    q = f"""
    WITH cohort AS (
      {intersect}
    )
    SELECT
      COUNT(*)::INT AS member_count,
      COALESCE(AVG(m.future_avoidable_ed_prob), 0)::DOUBLE PRECISION AS avg_prob
    FROM cohort_data.members m
    JOIN cohort c ON m.member_id = c.member_id;
    """

    cur.execute(q)
    row = cur.fetchone()
    return int(row[0]), float(row[1])