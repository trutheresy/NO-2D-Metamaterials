"""Extract band-by-group table from wave_stats cache (read-only)."""

from __future__ import annotations

from common import CACHE, load_json, save_json


def main() -> None:
    out = {}
    for tag in ("c_test", "b_test"):
        stats = load_json(CACHE / f"wave_stats_{tag}.json")
        out[tag] = stats.get("by_group_band", {})

    save_json(CACHE / "h7_band_by_group.json", out)
    print("Wrote", CACHE / "h7_band_by_group.json")
    for tag in out:
        if "ky0" in out[tag]:
            ky = out[tag]["ky0"]
            print(f"  {tag} ky0:", [f"b{b}={ky[str(b)]['nmae_mean']:.3f}" for b in range(6)])


if __name__ == "__main__":
    main()
