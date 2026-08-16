"""
Aggregate results across seeds and apply the pre-registered
success / failure / inconclusive thresholds from the config.

Usage:
    python src/eval.py --config configs/default.yaml --out outputs
"""

import argparse
import glob
import json
import statistics as stats

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    result_files = sorted(glob.glob(f"{args.out}/results_seed*.json"))
    if not result_files:
        raise SystemExit(f"No results found in {args.out}. Run train.py first.")

    all_results = []
    for rf in result_files:
        with open(rf) as f:
            all_results.append(json.load(f))

    in_dist_res = cfg["data"]["test_resolutions"][0]
    ood_res = cfg["data"]["test_resolutions"][-1]

    in_dist = [r[f"relative_l2_res{in_dist_res}"] for r in all_results]
    ood = [r[f"relative_l2_res{ood_res}"] for r in all_results]

    mean_in, std_in = stats.mean(in_dist), (stats.stdev(in_dist) if len(in_dist) > 1 else 0.0)
    mean_ood = stats.mean(ood)
    gap = mean_ood - mean_in

    success_t = cfg["thresholds"]["success_relative_l2"]
    failure_t = cfg["thresholds"]["failure_relative_l2"]

    if std_in > 0.3 * mean_in:
        verdict = "INCONCLUSIVE (high variance across seeds)"
    elif mean_in < success_t:
        verdict = "SUCCESS"
    elif mean_in > failure_t:
        verdict = "FAILURE"
    else:
        verdict = "INCONCLUSIVE (between thresholds)"

    print(f"Seeds run:              {len(all_results)}")
    print(f"In-dist relative L2:    {mean_in:.4f} (std {std_in:.4f})")
    print(f"OOD (res {ood_res}) relative L2: {mean_ood:.4f}")
    print(f"Resolution gap:         {gap:.4f}")
    print(f"Verdict:                {verdict}")


if __name__ == "__main__":
    main()
