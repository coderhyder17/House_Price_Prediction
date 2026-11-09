import json
from pathlib import Path

nb = json.loads(Path("house-price-prediction.ipynb").read_text(encoding="utf-8"))
print("n_cells", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    print(f"===== CELL {i} [{c['cell_type']}] exec={c.get('execution_count')} =====")
    print(src)
    print()