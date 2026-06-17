import json

with open("results/rubric_scores/civilcomments_rubric_scores.json") as f:
    d = json.load(f)

print("S composite across slices:")
for sl in d["slices"]:
    s = sl["S"]
    hcd = sl["per_sub_dimension"]["safety"]["harm_content_density"]
    print(f"  {sl['name']}:  S={s:.3f}  hcd={hcd:.3f}")