import keras_cv
import keras_core as keras
import numpy as np
import tensorflow as tf

filepath = keras.utils.get_file(origin="https://i.imgur.com/zwKEcAQ.jpeg")
image = np.array(keras.utils.load_img(filepath))
image_resized = keras.ops.image.resize(image,(640,640))[None, ...]

# Pretrained backbone
# model = keras_cv.models.YOLOV8Backbone.from_preset(
#     "yolo_v8_xs_backbone_coco"
# )
# output = model(input_data)

# Randomly initialized backbone with a custom config
# model = keras_cv.models.YOLOV8Backbone(
#     stackwise_channels=[128, 256, 512, 1024],
#     stackwise_depth=[3, 9, 9, 3],
#     include_rescaling=False,
# )
# output = model(input_data)
# print(output)

model=keras_cv.models.YOLOV8Detector.from_preset(
    "yolo_v8_m_pascalvoc",
    bounding_box_format="xywh",
)
# model = keras_cv.models.RetinaNet.from_preset(
#     "resnet50_v2_imagenet",
#     num_classes=20,
#     bounding_box_format="xywh",
# )
predictions = model.predict(image_resized)
# backbone = keras_cv.models.ResNetBackbone.from_preset("resnet50_imagenet")

class_ids = {
    1: "Person",
    2: "Car",
    3: "Bicycle",
    4: "Dog",
    5: "Cat",
    6: "Bird",
    7: "Tree",
    8: "Building",
    9: "House",
    10: "Street",
    11: "Sidewalk",
    12: "Traffic Light",
    13: "Traffic Sign",
    14: "Bus",
    15: "Bag",
    16: "Motorcycle",
    17: "Boat",
    18: "Bridge",
}

keras_cv.visualization.plot_bounding_box_gallery(
    image_resized,
    value_range=(0,255),
    rows=1,
    cols=1,
    y_pred=predictions,
    scale=5,
    font_scale=0.7,
    bounding_box_format="xywh",
    class_mapping=class_ids,
)

l = []
for prediction in predictions:
    if prediction == 'classes':
      arr = predictions['classes']
      for i in arr[0]:
        if i!= -1:
          if i not in l:
            l.append(class_ids[i])
print(list(set(l)))