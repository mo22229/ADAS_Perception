from pathlib import Path
import shutil

project_root = Path.home() / "automotive-ai"
kitti_root = project_root / "data" / "kitti"

image_dir = kitti_root / "training" / "image_2"
label_dir = kitti_root / "training" / "labels_yolo"

image_sets_dir = kitti_root / "ImageSets"


yolo_root = kitti_root / "yolo"

train_image_dir = yolo_root / "images" / "train"
val_image_dir = yolo_root / "images" / "val"

train_label_dir = yolo_root / "labels" / "train"
val_label_dir = yolo_root / "labels" / "val"

train_image_dir.mkdir(parents=True, exist_ok=True)
val_image_dir.mkdir(parents=True, exist_ok=True)

train_label_dir.mkdir(parents=True, exist_ok=True)
val_label_dir.mkdir(parents=True, exist_ok=True)

# split_name train OR val
def split_dataset(split_name):

    split_file = image_sets_dir / f"{split_name}.txt"

    if split_name == "train":
        output_image_dir = train_image_dir
        output_label_dir = train_label_dir
    else:
        output_image_dir = val_image_dir
        output_label_dir = val_label_dir

    with open(split_file, "r") as file:

        for line in file:

            image_id = line.strip()

            image_path = image_dir / f"{image_id}.png"
            label_path = label_dir / f"{image_id}.txt"

            shutil.copy2(
                image_path,
                output_image_dir / image_path.name
            )

            shutil.copy2(
                label_path,
                output_label_dir / label_path.name
            )


split_dataset("train")
split_dataset("val")