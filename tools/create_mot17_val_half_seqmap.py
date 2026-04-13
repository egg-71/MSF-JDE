import configparser
from pathlib import Path

GT_ROOT = Path("/home/success/YOLO11-JDE-main/tracker/evaluation/TrackEval/data/gt/mot_challenge/MOT17/val_half")
SEQMAP_OUT = GT_ROOT.parent / "seqmaps" / "MOT17-val_half.txt"

lines = []
for seq_dir in sorted(GT_ROOT.iterdir()):
    if not (seq_dir / "seqinfo.ini").exists():
        continue
    cfg = configparser.ConfigParser()
    cfg.read(seq_dir / "seqinfo.ini")
    L = int(cfg["Sequence"]["seqLength"])
    start = (L // 2) + 1
    end = L
    lines.append(f"{seq_dir.name} {start:06d} {end:06d}")

SEQMAP_OUT.parent.mkdir(parents=True, exist_ok=True)
SEQMAP_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[OK] wrote {SEQMAP_OUT}")
