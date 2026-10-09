import os
from pathlib import Path
import cv2
import numpy as np

ROOT = Path.home() / "automotive-ai" / "data" / "culane"
OUT = Path.home() / "automotive-ai" / "data" / "culane_prepared"
LINE_WIDTH = 16

def parse_lines(txt_path):
    lanes = []
    for line in txt_path.read_text().strip().splitlines():
        nums = list(map(float, line.split()))
        points = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
        if len(points) >= 2:
            lanes.append(points)
    return lanes

def process_split(split_name, driver_folder):
    img_root = ROOT / split_name / driver_folder
    out_img_root = OUT / "images" / driver_folder
    out_mask_root = OUT / "masks" / driver_folder
    list_lines = []

    for txt_path in img_root.rglob("*.lines.txt"):
        img_path = txt_path.with_name(txt_path.name.replace(".lines.txt", ".jpg"))
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lanes = parse_lines(txt_path)
        # نرتب الحارات من الشمال لليمين حسب أقرب نقطة لأسفل الصورة
        lanes.sort(key=lambda pts: pts[-1][0])

        mask = np.zeros((h, w), dtype=np.uint8)
        exist = [0, 0, 0, 0]
        num_lanes = min(len(lanes), 4)
        # لو أقل من 4 حارات، نحطهم في النص (حوالين حارة السواق)
        start_slot = (4 - num_lanes) // 2

        for i in range(num_lanes):
            slot = start_slot + i
            pts = np.array(lanes[i], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], False, slot + 1, LINE_WIDTH)
            exist[slot] = 1

        rel_path = img_path.relative_to(img_root)
        out_img_path = out_img_root / rel_path
        out_mask_path = out_mask_root / rel_path.with_suffix(".png")
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_img_path), img)
        cv2.imwrite(str(out_mask_path), mask)

        img_rel = f"/{driver_folder}/{rel_path.as_posix()}"
        mask_rel = f"/{driver_folder}/{rel_path.with_suffix('.png').as_posix()}"
        list_lines.append(f"{img_rel} {mask_rel} {' '.join(map(str, exist))}")

    return list_lines


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_lines = process_split("train", "driver_182_30frame")
    test_lines = process_split("test", "driver_100_30frame")

    (OUT / "train_gt.txt").write_text("\n".join(train_lines))
    (OUT / "test_gt.txt").write_text("\n".join(test_lines))

    print(f"Train samples: {len(train_lines)}")
    print(f"Test samples: {len(test_lines)}")


if __name__ == "__main__":
    main()
