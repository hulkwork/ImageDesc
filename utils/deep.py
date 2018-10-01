#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################
# Code from https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolo_opencv.py
import cv2
import numpy as np
import os.path

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):
    label = "%s %.2f" % (str(classes[class_id]), confidence)

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_object_id(img, object_id):
    for id in object_id:
        x = object_id[id][0]
        y = object_id[id][1]
        label = 'bject_id_%sd' % id
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def yolo_detection(image, class_filename=os.path.join(dir_path, "../data/yolo/yolov3.txt"),
                   weights=os.path.join(dir_path, "../data/yolo/yolov3.weights"),
                   config=os.path.join(dir_path, "../data/yolo/yolov3.cfg")):
    object_id = {}
    classes = None
    details = []

    with open(class_filename, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    # loop over the image paths
    net = cv2.dnn.readNet(weights, config)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    center_boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                center_boxes.append([center_x, center_y])
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    tmp_boxes = {}
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        tmp_details = {"y": y, "x": x, "w": w, "h": h, "class": classes[class_ids[i]], "confidences": confidences[i]}
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h),
                        classes=classes, COLORS=COLORS)
        tmp_boxes[i] = center_boxes[i]
        details.append(tmp_details)

    draw_object_id(image, object_id)
    return image, details
