"""포켓몬 캡션 데이터셋에서 시험용 그림을 받아 data/pokemon/에 저장한다.

python prepare_data.py
일반 사진 실험(notebooks/01)은 data/photos/에 아무 jpg나 넣으면 된다.
"""
import json
import os

from datasets import load_dataset

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pokemon")
os.makedirs(OUT, exist_ok=True)

ds = load_dataset("reach-vb/pokemon-blip-captions")["train"]
test = ds.train_test_split(test_size=0.1, seed=42)["test"]
meta = []
for i, row in enumerate(test):
    name = f"test_{i:03d}.jpg"
    row["image"].convert("RGB").save(os.path.join(OUT, name), quality=92)
    meta.append({"file": name, "text": row["text"]})
with open(os.path.join(OUT, "captions.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print(f"saved {len(meta)} images to {OUT}")
