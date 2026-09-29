---
license: cc-by-4.0
tags:
- football
- ball
- image
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/images/*.jpg
  - split: validation
    path: data/valid/images/*.jpg
  - split: test
    path: data/test/images/*.jpg
task_categories:
- object-detection
---

# Football Ball Detection Dataset

## Dataset Description

This dataset consists of images captured from a football match in a stadium, taken from a camera positioned near the middle of the pitch. Each image contains annotations for **football balls**.

For every ball in an image, a **bounding box** is provided, represented in the format:

- `x_center`: X-coordinate of the bounding box center
- `y_center`: Y-coordinate of the bounding box center
- `width`: Width of the bounding box
- `height`: Height of the bounding box

This dataset is suitable for tasks such as **object detection**, and can be used to detect balls in football game images or videos.

Detecting football balls is a challenging task, as the ball is small, can be occluded by players, may appear in the air, or can be confused with other objects on the pitch.


## Usage

Download dataset in your current working directory ``cwd``:
```
hf download martinjolif/football-ball-detection --repo-type dataset --local-dir cwd
```

Finetune on this dataset:
```
yolo detect train data=data/data.yaml model=yolo11n.pt epochs=1 batch=32 imgsz=640 device=mps
```

Validate custom-trained model:
```
yolo detect val model=path/to/best.pt
```


## References

The dataset comes from [Roboflow Universe](https://universe.roboflow.com/football-project-pifbc/football-ball-detection-rejhg-tl9ep), where you can also visualize the bounding boxes drawn around each object in the images.

Cite the dataset:

```bibtex
@misc{
  football-ball-detection-rejhg-tl9ep_dataset,
  title = { football-ball-detection Dataset },
  type = { Open Source Dataset },
  author = { Football project },
  howpublished = { \url{ https://universe.roboflow.com/football-project-pifbc/football-ball-detection-rejhg-tl9ep } },
  url = { https://universe.roboflow.com/football-project-pifbc/football-ball-detection-rejhg-tl9ep },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2025 },
  month = { oct },
  note = { visited on 2025-12-12 },
  }
