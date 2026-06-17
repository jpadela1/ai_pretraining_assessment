"""Look at the per-sub-dimension rights breakdown for Folktables."""
import json

with open("results/rubric_scores/folktables_acsincome_rubric_scores.json") as f:
    d = json.load(f)

print(f"{'state':<30s} {'R':>6s} {'dem_gap':>9s} {'consent':>9s} {'reident':>9s} {'inferent':>10s} {'contest':>9s}")
print("-" * 90)
for sl in d["slices"]:
    name = sl["name"]
    R = sl["R"]
    rights = sl["per_sub_dimension"]["rights"]
    def fmt(v):
        return f"{v:.3f}" if v is not None else "  N/A "
    print(f"{name:<30s} {R:.3f}  "
          f"{fmt(rights['demographic_representation_gap']):>9s}  "
          f"{fmt(rights['consent_provenance']):>9s}  "
          f"{fmt(rights['reidentification_risk']):>9s}  "
          f"{fmt(rights['inferential_harm_potential']):>10s}  "
          f"{fmt(rights['contestability']):>9s}")