from pathlib import Path
from PIL import Image

project_root = Path.home()/"automotive-ai"
kitti_root = project_root / "data" / "kitti"
training_dir = kitti_root / "training"
image_dir = training_dir / "image_2"
label_dir = training_dir / "label_2"

# YOLO labels will be saved here
yolo_label_dir = training_dir / "labels_yolo"
yolo_label_dir.mkdir(parents=True, exist_ok=True)

############################################
# CONVERT KITTI LABELS ----> YOLO LABELS

CLASS_MAPPING = {"Car": 0,"Cyclist": 1,"Misc": 2,"Pedestrian": 3,"Person_sitting": 4,
                 "Tram": 5,"Truck": 6,"Van": 7,}


def convert_kitti_label(label_path,image_width,image_height):
    yolo_lines=[]
    with open(label_path,"r") as file:
        for line in file:
            parts=line.strip().split()
            class_name = parts[0]
            if class_name == "DontCare":
                continue
            class_id = CLASS_MAPPING[class_name]

            x_min = float(parts[4])
            y_min = float(parts[5])
            x_max = float(parts[6])
            y_max = float(parts[7])

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            width = x_max - x_min
            height = y_max - y_min

            x_center /= image_width
            y_center /= image_height
            width /= image_width
            height /= image_height

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{width:.6f} {height:.6f}"
            )
    return yolo_lines


for img in sorted(image_dir.glob("*.png")):
    label_path = label_dir/ f"{img.stem}.txt"
    with Image.open(img)as image:
        image_width,image_hight = image.size

    yolo_lines = convert_kitti_label(label_path,image_width,image_hight)
    print(img.name, len(yolo_lines))

    output_label_path = yolo_label_dir / f"{img.stem}.txt"

    with open(output_label_path, "w") as file:
        file.write("\n".join(yolo_lines))