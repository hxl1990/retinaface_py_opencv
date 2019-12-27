import cv2
import time
import numpy as np
from anchor import AnchorCfg
from anchor import AnchorGenerator

cls_threshold = 0.8
nms_threshold = 0.4
mean = (0, 0, 0)
blob_size = 640

feat_stride_fpn = [32, 16, 8]
anchor_cfg = {
    32: AnchorCfg([32, 16], [1], 16),
    16: AnchorCfg([8, 4,], [1], 16),
    8: AnchorCfg([2, 1], [1], 16)
}

ac = [AnchorGenerator().Init(s, anchor_cfg[s]) for s in feat_stride_fpn]

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = np.array([d[0] for d in dets])
    y1 = np.array([d[1] for d in dets])
    x2 = np.array([d[2] for d in dets])
    y2 = np.array([d[3] for d in dets])
    scores = np.array([d.score for d in dets])

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def main():
    net = cv2.dnn.readNetFromCaffe("caffemodel/mnet.prototxt", "caffemodel/mnet.caffemodel")
    cap = cv2.VideoCapture(0)

    while(True):
        ret, img = cap.read()
        if not ret:
            continue

        height, width, _ = img.shape
        h_f = height / blob_size
        w_f = width / blob_size

        anchor_list = []
        blob = cv2.dnn.blobFromImage(img, 1, (blob_size, blob_size), mean, False, False)
        s_time = time.time()

        output_name = []
        for i, s in enumerate(feat_stride_fpn):
            s_time = time.time()
            output_name.append("face_rpn_cls_prob_reshape_stride{}".format(s))
            output_name.append("face_rpn_bbox_pred_stride{}".format(s))
            output_name.append("face_rpn_landmark_pred_stride{}".format(s))
        
        s_time = time.time()
        net.setInput(blob)
        forward_out = net.forward(output_name)
        print("cost: {}".format(time.time() - s_time))

        for i, s in enumerate(feat_stride_fpn):
            c_out = forward_out[i*3+0]
            b_out = forward_out[i*3+1]
            l_out = forward_out[i*3+2]
            det_i = ac[i].FilterAnchor(c_out, b_out, l_out)
            if det_i is not None and len(det_i) > 0:
                anchor_list.extend(det_i)

        k_index = py_cpu_nms(anchor_list, nms_threshold)
        for i in k_index:
            anchor = anchor_list[i]
            p0 = (int(anchor[0]* w_f), int(anchor[1] * h_f))
            p1 = (int(anchor[2]* w_f), int(anchor[3] * h_f))
            cv2.rectangle(img, p0, p1, (0, 255, 255), 2 )

        cv2.imshow("test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("done")




if __name__ == "__main__":
    main()