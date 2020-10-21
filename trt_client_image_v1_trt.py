import requests
import base64
import json
import cv2
import time
import numpy as np
import colorsys
import random

# image_path = "/home/dcs/work/PythonProject/triton_client/samples/bus.jpg"
image_path = "/data/home/fjc/workspace/yolov5/inference/images/zidane.jpg"
label_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone',
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
              'hair drier', 'toothbrush']


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    cv2.imwrite('show.jpg', resized)
    # return the resized image
    return resized


def draw_bbox(image, image_size, bboxes, labels, scores, thres=0.4):
    num_classes = 80
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    fontScale = 0.5

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for k, score in enumerate(scores):
        if score > thres:
            box = bboxes[k]
            class_ind = int(labels[k])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1 = (int(box[0] / 640 * image_w), int(box[1] / 640 * image_h))
            c2 = (int(box[2] / 640 * image_w), int(box[3] / 640 * image_h))

            cv2.rectangle(image, c1, c2, bbox_color, 2)

            if 1:
                bbox_mess = '%s: %.2f' % (label_list[class_ind], score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1]
                                          - t_size[1] - 3), bbox_color, -1)  # filled

                cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
                cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image


def main():
    image0 = image = cv2.imread(image_path)
    # image = image_resize(image, 640, 640)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))
    image_size = image.shape[:2]
    image = np.asarray(image, dtype=np.float32)
    # image = np.transpose(image, (2, 0, 1))

    image /= 255.0
    print(image.shape)
    image = np.transpose(image, (2, 0, 1))
    data = image.tobytes()
    # # data = image_content
    # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # data = image.tobytes()

    input_names = ['images']
    output_names = ["nmsed_boxes", "nmsed_classes", "nmsed_scores", "num_detections_reshaped"]
    batch_size = 1
    dims = [1]
    server = '192.168.60.73'
    port = 8000
    model = "yolov5"
    version = 1

    nv_inferreq = (
            ['batch_size: 1 '] +
            [' input { name: "%s" }' % input_name for input_name in input_names] +
            [' output { name: "%s"}' % output_name for output_name in output_names])
    nv_inferreq = ''.join(nv_inferreq)
    # print(nv_inferreq)

    headers = {
        'Content-Type': 'application/octet-stream',
        # 'content_length': str(640 * 640 * 3 * 4),
        'NV-InferRequest': nv_inferreq}
    # print(headers)
    url = 'http://{}:{}/api/infer/{}/{}'.format(server, port, model, version)

    r = requests.post(url, headers=headers, data=data)
    result = {}
    trt_model = True
    if trt_model:
        t = np.frombuffer(r.content[:1600], dtype=np.float32)
        print(r.content)
        result['detection_boxes'] = np.reshape(t, [100, 4])

        t = np.frombuffer(r.content[1600: 1600 + 400], dtype=np.float32)
        result['detection_scores'] = np.reshape(t, [100])

        t = np.frombuffer(r.content[1600 + 400: 1600 + 400 + 400], dtype=np.float32)
        result['detection_classes'] = np.reshape(t, [100])

        t = np.frombuffer(r.content[1600 + 400 + 400: 1600 + 400 + 400 + 4], dtype=np.int32)
        result['num_detections'] = np.reshape(t, [1])
    else:
        t = np.frombuffer(r.content[:1600], dtype=np.float32)
        result['detection_boxes'] = np.reshape(t, [100, 4])

        t = np.frombuffer(r.content[1600: 1600 + 400], dtype=np.float32)
        result['detection_classes'] = np.reshape(t, [100])

        t = np.frombuffer(r.content[1600 + 400: 1600 + 400 + 400], dtype=np.float32)
        result['detection_scores'] = np.reshape(t, [100])

        t = np.frombuffer(r.content[1600 + 400 + 400: 1600 + 400 + 400 + 4], dtype=np.int32)
        result['num_detections'] = np.reshape(t, [1])

    detection_boxes = result['detection_boxes']
    detection_classes = result['detection_classes']
    detection_scores = result['detection_scores']

    print("result['num_detections']", result['num_detections'])
    print("result['detection_boxes']", result['detection_boxes'])
    print("result['detection_scores']", result['detection_scores'])
    print("result['detection_classes']", result['detection_classes'])

    result = draw_bbox(image0, image0.shape[:2], detection_boxes, detection_classes, detection_scores)

    cv2.imwrite('drawn_image.jpg', result)


if __name__ == '__main__':
    main()
