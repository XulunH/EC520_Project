import json

with open('jsonfiles/Edge_Cases.json', 'r') as file:
    data = json.load(file)
    annotations = data['annotations']

bboxes_by_image = {}

for item in annotations:
    image_id = item["image_id"]
    bbox = item["bbox"]
    if image_id not in bboxes_by_image:
        bboxes_by_image[image_id] = []
    bboxes_by_image[image_id].append(bbox)

for image_id, bboxes in bboxes_by_image.items():
    with open(f"annotations/Edge_Cases/{image_id}.txt", "w") as file:
        for bbox in bboxes:
            bbox_string = "0 " + " ".join(map(str, bbox))
            file.write(f"{bbox_string}\n")

