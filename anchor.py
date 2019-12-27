
import math


class Point:
    x = 0 
    y = 0
    def __init__(self, x, y):
        self.x = x
        self.y = y
    

class AnchorCfg:
    SCALES = []
    RATIOS = []
    BASE_SIZE = 0

    def __init__(self, s, r, size):
        self.SCALES = s
        self.RATIOS = r
        self.BASE_SIZE = size


class CRect2f:
    val = []
    
    def __init__(self, x1, y1, x2, y2):
        
        self.val.append(x1)
        self.val.append(y1)
        self.val.append(x2)
        self.val.append(y2)

    def __getitem__(self, key):
        return self.val[key]


    def print(self):
	    print("rect: {} {} {} {}".format(self.val[0], self.val[1], self.val[2], self.val[3]))
	


class Anchor:

    anchor = []
    reg = [0, 0, 0, 0]
    center = [0, 0]
    score = 0.
    pts = []
    finalbox = [0, 0, 0, 0]

    def __init__(self):
        super().__init__()


    def __getitem__(self, key):
        return self.finalbox[key]


    def __gt__(self, other):
        return self.score > other.score

    def __lt__(self, other):
        return self.score < other.score



class AnchorGenerator:
    anchor_size = [] 
    anchor_ratio =  []
    anchor_step = 0.
    anchor_stride = 0
    feature_w = 0
    feature_h = 0
    preset_anchors = []
    anchor_num = 0

    def  __init__(self):
        super().__init__()

    def Init(self,  stride, cfg, dense_anchor=False):
        base_anchor = [0, 0, cfg.BASE_SIZE-1, cfg.BASE_SIZE-1]
        ratio_anchors = []
        ratio_anchors = self._ratio_enum(base_anchor, cfg.RATIOS)
        self.preset_anchors = self._scale_enum(ratio_anchors, cfg.SCALES)

        if dense_anchor:
            assert(stride % 2 == 0)
            for anchor in preset_anchors:
                self.preset_anchors.append(
                    [anchor[0]+int(stride/2),
                    anchor[1]+int(stride/2),
                    anchor[2]+int(stride/2),
                    anchor[3]+int(stride/2)]

                )

        self.anchor_stride = stride

        self.anchor_num = len(self.preset_anchors)
        return self


    def Generate(self, fwidth, fheight, stride, step, size, ratio, dense_anchor):
        pass

    def FilterAnchor(self, cls, reg, pts):
        assert(cls.shape[1] == self.anchor_num*2)
        assert(reg.shape[1] == self.anchor_num*4)
        pts_length = 0
        if pts is not None:
            assert(pts.shape[1] % self.anchor_num == 0)
            pts_length = int(pts.shape[1]/self.anchor_num/2)
        

        w = cls.shape[3]
        h = cls.shape[2]
        step = h*w
        cls_threshold = 0.8

        result = []
        for i in range(h):
            for j in range(w):
                for a in range(self.anchor_num):
                    cls_s = cls[0][self.anchor_num + a][i][j]
                    if  cls_s> cls_threshold:
                        # print(cls_s)
                        box = [
                            j * self.anchor_stride + self.preset_anchors[a][0],
                            i * self.anchor_stride + self.preset_anchors[a][1],
                            j * self.anchor_stride + self.preset_anchors[a][2],
                            i * self.anchor_stride + self.preset_anchors[a][3]
                        ]

                        # print("{} {} {} {}".format(box[0], box[1], box[2], box[3]))

                        delta = [
                            reg[0][a*4+0][i][j],
                            reg[0][a*4+1][i][j],
                            reg[0][a*4+2][i][j],
                            reg[0][a*4+3][i][j]
                        ]
                        res = Anchor()
                        res.anchor = box
                        res.finalbox = self.bbox_pred(box, delta)
                        res.score = cls_s
                        res.center = [i, j]
                        # print("center: {}".format(res.center))


                        if pts is not None:
                            pts_delta = []
                            for p in range(pts_length):
                                pts_delta.append(Point(
                                    pts[0][a*pts_length*2+p*2][i][j],
                                    pts[0][a*pts_length*2+p*2+1][i][j]
                                ))

                            res.pts = self.landmark_pred(box, pts_delta)
                        result.append(res)

        return result

    def _ratio_enum(self, anchor: list, ratios: list):
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)

        ratio_anchors  = []
        sz = w * h

        for r in ratios:
            size_ratios = sz / r
            ws = math.sqrt(size_ratios)
            hs = ws * r
            ratio_anchors.append(
                [x_ctr - 0.5 * (ws - 1),
                y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1),
                y_ctr + 0.5 * (hs - 1)]
            )
        
        return ratio_anchors

    def _scale_enum(self, ratio_anchor: list, scales: list):
        scale_anchors = []
        for anchor in ratio_anchor:
            w = anchor[2] - anchor[0] + 1
            h = anchor[3] - anchor[1] + 1
            x_ctr = anchor[0] + 0.5 * (w - 1)
            y_ctr = anchor[1] + 0.5 * (h - 1)

            for s in scales:
                ws = w * s
                hs = h * s
                scale_anchors.append(
                    [x_ctr - 0.5 * (ws - 1),
                    y_ctr - 0.5 * (hs - 1),
                    x_ctr + 0.5 * (ws - 1),
                    y_ctr + 0.5 * (hs - 1)]
                )

        return scale_anchors

    def bbox_pred(self, anchor: list, delta: list):
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)

        dx = delta[0]
        dy = delta[1]
        dw = delta[2]
        dh = delta[3]

        pred_ctr_x = dx * w + x_ctr
        pred_ctr_y = dy * h + y_ctr
        pred_w = math.exp(dw) * w
        pred_h = math.exp(dh) * h

        box = (pred_ctr_x - 0.5 * (pred_w - 1.0),
	    pred_ctr_y - 0.5 * (pred_h - 1.0),
	    pred_ctr_x + 0.5 * (pred_w - 1.0),
	    pred_ctr_y + 0.5 * (pred_h - 1.0))

        return box

    def landmark_pred(self, anchor: CRect2f, delta: list):
        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)

        pts = []
        for d  in delta:
            p = Point(d.x*w + x_ctr,  d.y*h + y_ctr)
            pts.append(p)

        return pts


