from pathlib import Path

SEQMAP = Path("/home/success/YOLO11-JDE-main/tracker/evaluation/TrackEval/data/gt/mot_challenge/seqmaps/MOT17-val_half.txt")

def parse_line(line: str):
    parts = line.strip().split()
    if len(parts) == 3:
        name, s, e = parts
        return name, int(s), int(e)
    elif len(parts) == 1:
        # 兼容只有序列名（全帧）的情况；这里不建议用
        return parts[0], None, None
    else:
        raise ValueError(f"Bad seqmap line: {line}")

txt = SEQMAP.read_text().strip().splitlines()
print("[DRYRUN] SeqMap ranges:")
for ln in txt:
    name, s, e = parse_line(ln)
    if s is None:
        print(f"  {name}: [FULL]")
    else:
        print(f"  {name}: {s:06d}..{e:06d}")
