from concept_resolver import resolve
from runtime_engine import run_suggest_flow, run_finalize_flow

def simple_split(query: str):
    q = query.lower()
    terms = []
    if "diabetes" in q:
        terms.append("diabetes")
    if "asthma" in q:
        terms.append("asthma")
    return terms

def main():
    query = "Show members with diabetes and asthma"
    terms = simple_split(query)
    anchors = [resolve(t) for t in terms]
    anchors = [a for a in anchors if a]

    print("Query:", query)
    print("Resolved anchor concepts:", anchors)

    out = run_suggest_flow(anchors)

    print("\nBaseline cohort:", out["baseline"])

    print("\nTop suggestions:")
    for s in out["suggestions"]:
        print("- Path:", " -> ".join(s["path"]))
        print(f"  Unrealized Gain: {s['unrealized_gain']:.2f}x")
        print(f"  Narrow population by ΔN={s['delta_n']} "
              f"(from {s['baseline_count']} to {s['candidate_count']})")
        print(f"  Avoidable ED prob: {s['baseline_avg_prob']:.4f} -> {s['candidate_avg_prob']:.4f} "
              f"(Δ={s['delta_avg_prob']:.4f}, CR={s['lift_cr']:.2f}x)\n")

    # Simulated user acceptance
    if out["suggestions"]:
        accepted = out["suggestions"][0]["apply_nodes"]
        final = run_finalize_flow(anchors, accepted)
        print("Simulated acceptance:", accepted)
        print("Final cohort:", final)

if __name__ == "__main__":
    main()