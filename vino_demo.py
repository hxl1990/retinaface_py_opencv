import os
import sys
import cv2
import time
import logging as log
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin
from anchor import AnchorCfg, AnchorGenerator, Anchor



device = "CPU"
config = ""
model_xml = "model/mnet.xml"
model_bin = os.path.splitext(model_xml)[0] + ".bin"
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

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)




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
    plugin = IEPlugin(device=device, plugin_dirs="")
    net = IENetwork(model=model_xml, weights=model_bin)
    if 'CPU' in device:
        plugin.add_cpu_extension("cpu_extension_avx2")

    if "CPU" in plugin.device:
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                    format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                    "or --cpu_extension command line argument")
            sys.exit(1)


    exec_net = plugin.load(network=net, config=config)

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    n, c, h, w = net.inputs[input_blob].shape

    cap = cv2.VideoCapture(0)

    while(True):
        ret, img = cap.read()
        if not ret:
            continue

        height, width, _ = img.shape
        h_f = height / blob_size
        w_f = width / blob_size

        anchor_list = []
        input_data = cv2.resize(img, (w, h))
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        input_data = input_data.transpose((2, 0, 1))
        input_data = input_data.reshape((n, c, h, w))

        time_s = time.time()
        res = exec_net.infer(inputs={input_blob: input_data})
        print("cost: {}".format(time.time() - time_s))

        for i, s in enumerate(feat_stride_fpn):
            c_out = res["face_rpn_cls_prob_reshape_stride{}".format(s)]
            b_out = res["face_rpn_bbox_pred_stride{}".format(s)]
            l_out = res["face_rpn_landmark_pred_stride{}".format(s)]
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


if __name__ == "__main__":
    main()