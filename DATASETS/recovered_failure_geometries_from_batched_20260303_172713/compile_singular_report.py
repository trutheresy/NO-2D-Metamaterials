import json
import os


OUT = r"D:/Research/NO-2D-Metamaterials/OUTPUT/recovered_failure_geometries_from_batched_20260303_172713"


def load(name):
    with open(os.path.join(OUT, name), "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    c2 = load("cause2_pruning_results.json")
    c1 = load("cause1_precision_results.json")
    c3 = load("cause3_symmetry_quantization_results.json")
    c4 = load("cause4_backend_results.json")

    summary = {
        "cause2": {
            "prune_on_mean": c2.get("prune_on_mean_rel_avg", c2.get("prune_on_finite_mean")),
            "prune_off_mean": c2.get("prune_off_mean_rel_avg", c2.get("prune_off_finite_mean")),
            "prune_on_inf": c2.get("prune_on_inf_cases"),
            "prune_off_inf": c2.get("prune_off_inf_cases"),
        },
        "cause1": {
            "current_mean": c1["variants"]["current"]["mean_rel_avg"],
            "high_mean": c1["variants"]["high_precision"]["mean_rel_avg"],
            "current_inf": c1["variants"]["current"]["inf_cases"],
            "high_inf": c1["variants"]["high_precision"]["inf_cases"],
        },
        "cause3": {
            k: {"mean": v["mean_rel_avg"], "inf": v["inf_cases"]}
            for k, v in c3["variants"].items()
        },
        "cause4": {
            k: {"mean": v["mean_rel_avg"], "inf": v["inf_cases"]}
            for k, v in c4["variants"].items()
        },
    }

    impact = []
    base = c1["variants"]["current"]["mean_rel_avg"]
    best = min(c1["variants"]["current"]["mean_rel_avg"], c1["variants"]["high_precision"]["mean_rel_avg"])
    impact.append(("cause1_precision", float(base - best)))

    b1 = summary["cause2"]["prune_on_mean"] or 0.0
    b2 = summary["cause2"]["prune_off_mean"] or 0.0
    impact.append(("cause2_pruning", float(abs(b1 - b2))))

    cur = c3["variants"]["current_float16"]["mean_rel_avg"]
    best3 = min(v["mean_rel_avg"] for v in c3["variants"].values())
    impact.append(("cause3_symmetry_quantization", float(cur - best3)))

    cur4 = c4["variants"]["sparse_shiftinvert_sigma"]["mean_rel_avg"]
    best4 = min(v["mean_rel_avg"] for v in c4["variants"].values())
    impact.append(("cause4_backend", float(cur4 - best4)))

    impact_sorted = sorted(impact, key=lambda x: x[1], reverse=True)
    summary["impact_ranking"] = impact_sorted

    with open(os.path.join(OUT, "singular_cause_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Singular-Factor Cause Test Report",
        "",
        "## Scope/Subset Used",
        "- Cause 2: 32 cases (12 failures + 20 controls), matrix-level K/M comparisons.",
        "- Cause 1: representative subset (5 failure cases, 10 wavevectors).",
        "- Cause 3: same representative subset (5 failure cases, 10 wavevectors).",
        "- Cause 4: tiny backend subset (2 failure cases, 5 wavevectors).",
        "",
        "## Cause 2 (Pruning)",
        "- prune_on mean_rel_avg: {}".format(summary["cause2"]["prune_on_mean"]),
        "- prune_off mean_rel_avg: {}".format(summary["cause2"]["prune_off_mean"]),
        "- inf cases: on={} off={}".format(summary["cause2"]["prune_on_inf"], summary["cause2"]["prune_off_inf"]),
        "- Observation: no material difference between prune on/off on tested matrix metrics.",
        "",
        "## Cause 1 (Precision)",
        "- current mean_rel_avg: {}".format(summary["cause1"]["current_mean"]),
        "- high_precision mean_rel_avg: {}".format(summary["cause1"]["high_mean"]),
        "- inf cases: current={} high={}".format(summary["cause1"]["current_inf"], summary["cause1"]["high_inf"]),
        "- Observation: high-precision path dramatically reduces MATLAB-relative error.",
        "",
        "## Cause 3 (Symmetry + Quantization)",
    ]
    for k, v in summary["cause3"].items():
        lines.append("- {}: mean_rel_avg={} inf_cases={}".format(k, v["mean"], v["inf"]))
    lines.extend(
        [
            "- Observation: guarded symmetry did not materially reduce error in tested subset; float64 no-quantization was not better in this run.",
            "",
            "## Cause 4 (Backend Variants)",
        ]
    )
    for k, v in summary["cause4"].items():
        lines.append("- {}: mean_rel_avg={} inf_cases={}".format(k, v["mean"], v["inf"]))
    lines.extend(
        [
            "- Observation: MATLAB baseline was closest to sparse shift-invert with numeric sigma among tested backend variants.",
            "",
            "## Ranked Impact (observed reduction score)",
        ]
    )
    for i, (name, val) in enumerate(impact_sorted, 1):
        lines.append("{}. {} (score={})".format(i, name, val))
    lines.extend(
        [
            "",
            "## Notes",
            "- Per requirement, NaN/singularity were treated as Inf errors; no Inf cases occurred in these completed subsets.",
            "- Runtime constraints required subset reduction for Causes 1/3/4; rerun larger sets for tighter confidence intervals.",
        ]
    )

    with open(os.path.join(OUT, "singular_cause_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("saved", os.path.join(OUT, "singular_cause_summary.json"))
    print("saved", os.path.join(OUT, "singular_cause_report.md"))


if __name__ == "__main__":
    main()
