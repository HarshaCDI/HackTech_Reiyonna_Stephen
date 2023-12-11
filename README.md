# Real-time multi-object tracking using Yolov8 with DeepOCSORT and StrongSORT


## Introduction

This repo contains a collections of pluggable state-of-the-art multi-object trackers on the UA-DETRAC dataset. We provide examples on how to use this package together with popular object detection model: [Yolov8](https://github.com/ultralytics).

<details>
<summary>Supported tracking methods</summary>

[DeepOCSORT](https://arxiv.org/abs/2302.11813) , [BoTSORT](https://arxiv.org/abs/2206.14651) , [StrongSORT](https://github.com/dyhBUPT/StrongSORT)[](https://arxiv.org/abs/2202.13514), [OCSORT](https://github.com/noahcao/OC_SORT)[](https://arxiv.org/abs/2203.14360) and [ByteTrack](https://github.com/ifzhang/ByteTrack)[](https://arxiv.org/abs/2110.06864). DeepOCSORT, BoTSORT and StrongSORT are based on motion + appearance description; OCSORT and ByteTrack are based on motion only. 

</details>

<details>
<summary>Tutorials</summary>

* [Yolov8 training (link to external repository)](https://docs.ultralytics.com/modes/train/)&nbsp;
* [Deep appearance descriptor training (link to external repository)](https://kaiyangzhou.github.io/deep-person-reid/user_guide.html)&nbsp;
* [Evaluation on custom tracking dataset](https://github.com/mikel-brostrom/yolov8_tracking/wiki/How-to-evaluate-on-custom-tracking-dataset)&nbsp;

</details>

## Why using this tracking toolbox?

Everything is designed with simplicity and flexibility in mind. We don't hyperfocus on results on a single dataset, we prioritize real-world results. 

## Installation

If you only want to import the tracking modules you can simply:

```
pip install boxmot
```


## YOLOv8

<details>
<summary>Tracking</summary>

<details>
<summary>Yolo models</summary>



```bash
$ python examples/track.py --yolo-model yolov8n       # bboxes only
  python examples/track.py --yolo-model yolo_nas_s    # bboxes only
  python examples/track.py --yolo-model yolox_n       # bboxes only
                                        yolov8n-seg   # bboxes + segmentation masks
                                        yolov8n-pose  # bboxes + pose estimation

```

  </details>

<details>
<summary>Tracking methods</summary>

```bash
$ python examples/track.py --tracking-method deepocsort
                                             strongsort
                                             ocsort
                                             bytetrack
                                             botsort
```
Tracking can be run on most video formats
</details>

<details>
<summary>Select ReID model</summary>

Some tracking methods combine appearance description and motion in the process of tracking. For those which use appearance, you can choose a ReID model based on your needs from this [ReID model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO). These model can be further optimized for you needs by the [reid_export.py](https://github.com/mikel-brostrom/yolo_tracking/blob/master/boxmot/deep/reid_export.py) script

```bash
$ python examples/track.py --source 0 --reid-model lmbn_n_cuhk03_d.pt
                                                   osnet_x0_25_market1501.pt
                                                   mobilenetv2_x1_4_msmt17.engine
                                                   resnet50_msmt17.onnx
                                                   osnet_x1_0_msmt17.pt
                                                   ...
```

</details>

<summary>Evaluation</summary>

Evaluate a combination of detector, tracking method and ReID model on standard MOT dataset or you custom one by

```bash
$ python3 examples/val.py --yolo-model yolo_nas_s.pt --reid-model osnetx1_0_dukemtcereid.pt --tracking-method deepocsort --benchmark MOT16
                          --yolo-model yolox_n.pt    --reid-model osnet_ain_x1_0_msmt17.pt  --tracking-method ocsort     --benchmark MOT17
                          --yolo-model yolov8s.pt    --reid-model lmbn_n_market.pt          --tracking-method strongsort --benchmark <your-custom-dataset>
```

</details>



## Custom object detection model example

<details>
<summary>Minimalistic</summary>

```python
from boxmot import DeepOCSORT
from pathlib import Path


tracker = DeepOCSORT(
  model_weights=Path('osnet_x0_25_msmt17.pt'),  # which ReID model to use
  device='cuda:0',  # 'cpu', 'cuda:0', 'cuda:1', ... 'cuda:N'
  fp16=True,  # wether to run the ReID model with half precision or not
)

cap = cv.VideoCapture(0)
while True:
    ret, im = cap.read()
    ...
    # dets (numpy.ndarray):
    #  - your model's nms:ed outputs of shape Nx6 (x, y, x, y, conf, cls)
    # im   (numpy.ndarray):
    #  - the original hxwx3 image (for better ReID results)
    #  - the downscaled hxwx3 image fed to you model (faster)
    tracker_outputs = tracker.update(dets, im)  # --> (x, y, x, y, id, conf, cls)
    ...
```

</details>


<details>
<summary>Complete</summary>

```python
from boxmot import DeepOCSORT
from pathlib import Path
import cv2
import numpy as np

tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='cuda:0',
    fp16=True,
)

vid = cv2.VideoCapture(0)
color = (0, 0, 255)  # BGR
thickness = 2
fontscale = 0.5

while True:
    ret, im = vid.read()

    # substitute by your object detector, output has to be N X (x, y, x, y, conf, cls)
    dets = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])

    ts = tracker.update(dets, im) # --> (x, y, x, y, id, conf, cls)

    xyxys = ts[:, 0:4].astype('int') # float64 to int
    ids = ts[:, 4].astype('int') # float64 to int
    confs = ts[:, 5]
    clss = ts[:, 6]

    # print bboxes with their associated id, cls and conf
    if ts.shape[0] != 0:
        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
            im = cv2.rectangle(
                im,
                (xyxy[0], xyxy[1]),
                (xyxy[2], xyxy[3]),
                color,
                thickness
            )
            cv2.putText(
                im,
                f'id: {id}, conf: {conf}, c: {cls}',
                (xyxy[0], xyxy[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness
            )

    # show image with bboxes, ids, classes and confidences
    cv2.imshow('frame', im)

    # break on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
```

</details>

