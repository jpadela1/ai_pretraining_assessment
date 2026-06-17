from pathlib import Path
for p in sorted(Path("data_cache/civilcomments").rglob("*")):
    print(p)