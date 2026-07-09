# Detection Utilities

TensorStudio provides compact NumPy-backed detection utilities for local object
detection experiments and dataset preparation.

## Box Operations

Boxes use `xyxy` coordinates:

```python
import tensorstudio as ts

boxes = ts.tensor([[0.0, 0.0, 16.0, 16.0], [4.0, 4.0, 20.0, 20.0]])
scores = ts.tensor([0.9, 0.7])

print(ts.vision.box_area(boxes))
print(ts.vision.box_iou(boxes, boxes))
print(ts.vision.generalized_box_iou(boxes, boxes))
print(ts.vision.distance_box_iou(boxes, boxes))
print(ts.vision.nms(boxes, scores, iou_threshold=0.5))
```

Supported helpers:

- `box_area`
- `box_iou`
- `generalized_box_iou`
- `distance_box_iou`
- `nms`
- `box_xyxy_to_cxcywh`
- `box_cxcywh_to_xyxy`

## Anchors

```python
anchors = ts.vision.generate_anchors((8, 8), stride=16, sizes=(32.0, 64.0))
deltas = ts.vision.encode_boxes(target_boxes, anchors[: len(target_boxes)])
decoded = ts.vision.decode_boxes(deltas, anchors[: len(target_boxes)])
```

Anchor helpers are intentionally simple and inspectable. They do not include a
full detector training stack yet.

## DetectionFolder

`DetectionFolder` expects:

```text
dataset/
  images/
    image-001.png
  annotations/
    image-001.json
```

Each JSON annotation contains:

```json
{
  "boxes": [[0, 0, 16, 16]],
  "labels": [1]
}
```

```python
dataset = ts.vision.DetectionFolder("dataset")
image, target = dataset[0]
print(target["boxes"], target["labels"])
```
