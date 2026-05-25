from graph_def import NODES
from nx_engine import build_graph, enumerate_candidates, union_dedupe
from pg_executor import get_conn, cohort_metrics

def explain(baseline_count, baseline_avg, cand_count, cand_avg):
    delta_n = baseline_count - cand_count
    delta_avg = cand_avg - baseline_avg
    lift_cr = (cand_avg / baseline_avg) if baseline_avg > 1e-9 else None
    return {
        "baseline_count": baseline_count,
        "baseline_avg_prob": baseline_avg,
        "candidate_count": cand_count,
        "candidate_avg_prob": cand_avg,
        "delta_n": delta_n,
        "delta_avg_prob": delta_avg,
        "lift_cr": lift_cr,
    }

def run_suggest_flow(anchor_concepts: list[str],
                     topk_probe: int = 6,
                     topk_return: int = 3,
                     min_support: int = 25,
                     min_coverage: float = 0.01):
    """
    Workflow:
    1) Baseline metrics for anchor intersection
    2) NetworkX candidate paths (1-hop & 2-hop) across anchors
    3) Union+dedupe candidates across anchors
    4) Rank by Unrealized Gain and select topk_probe
    5) For each candidate, compute Realized metrics in Postgres:
       - count, avg future_avoidable_ed_prob
       - compute ΔN, Δprob, CR lift
    6) Filter by min_support and min_coverage
    7) Rank by Realized lift then delta
    8) Return topk_return suggestions
    """

    if not anchor_concepts:
        return {"anchor_concepts": [], "baseline": {"count": 0, "avg_prob": 0.0}, "suggestions": []}

    G = build_graph()
    anchor_sqls = [NODES[c]["cached_sql"] for c in anchor_concepts]

    with get_conn() as conn:
        with conn.cursor() as cur:
            base_count, base_avg = cohort_metrics(cur, anchor_sqls)

            cand = enumerate_candidates(G, anchor_concepts)
            cand = union_dedupe(cand)

            cand.sort(key=lambda x: x["unrealized_gain"], reverse=True)
            cand = cand[:topk_probe]

            evaluated = []
            for c in cand:
                apply_nodes = c["apply_nodes"]
                sqls = anchor_sqls + [NODES[n]["cached_sql"] for n in apply_nodes]
                cnt, avg = cohort_metrics(cur, sqls)

                # Guardrails
                if cnt < min_support:
                    continue
                if base_count > 0 and (cnt / base_count) < min_coverage:
                    continue

                realized = explain(base_count, base_avg, cnt, avg)
                evaluated.append({**c, "realized": realized})

    # Rank by Realized lift then delta_avg_prob
    def sort_key(x):
        lift = x["realized"]["lift_cr"]
        lift = lift if lift is not None else -1
        return (lift, x["realized"]["delta_avg_prob"])

    evaluated.sort(key=sort_key, reverse=True)

    suggestions = []
    for e in evaluated[:topk_return]:
        r = e["realized"]
        suggestions.append({
            "path": e["path"],
            "apply_nodes": e["apply_nodes"],
            "unrealized_gain": e["unrealized_gain"],
            **r
        })

    return {
        "anchor_concepts": anchor_concepts,
        "baseline": {"count": base_count, "avg_prob": base_avg},
        "suggestions": suggestions,
    }

def run_finalize_flow(anchor_concepts: list[str], accepted_nodes: list[str]):
    """
    After user accepts one/more suggestions, compute final cohort metrics:
      (anchor intersection) AND (accepted criteria)
    """
    anchor_sqls = [NODES[c]["cached_sql"] for c in anchor_concepts]
    accepted_sqls = [NODES[n]["cached_sql"] for n in accepted_nodes]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cnt, avg = cohort_metrics(cur, anchor_sqls + accepted_sqls)

    return {
        "final_conditions": anchor_concepts + accepted_nodes,
        "final_count": cnt,
        "final_avg_avoidable_ed_prob": avg,
    }