import json

with open("results/rubric_scores/civilcomments_rubric_scores.json") as f:
    d = json.load(f)

print(f"Dataset: {d['dataset']}")
print(f"Number of slices: {len(d['slices'])}\n")
print("Slice details:")
for s in d["slices"]:
    name = s["name"]
    safety = s["per_sub_dimension"]["safety"]
    h = safety.get("harm_content_density")
    n = s.get("n_rows", "?")
    S = s.get("S", "?")
    print(f"  {name:40s} n={n:>6}  S={S}  harm_content_density={h}")