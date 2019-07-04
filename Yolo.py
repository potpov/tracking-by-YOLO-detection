import torch
import conf
from content.darknet import *

class Yolo:
    def __init__(self, device="cuda"):
        self.model = Darknet(conf.YOLO_CONF_PATH)
        self.model.load_weights(conf.YOLO_WEIGHT_PATH)
        self.model.net_info["height"] = conf.RESOLUTION
        inp_dim = int(self.model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32

        self.device = device
        self.model.to(self.device)
        # Set the model in evaluation mode
        self.model.eval()

    def predict(self, frame):
        net_input = prep_image(frame, int(self.model.net_info["height"]))
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
        im_dim = im_dim.to(self.device)
        net_input = net_input.to(self.device)

        with torch.no_grad():
            output = self.model(Variable(net_input), torch.cuda.is_available())

        output = write_results(
            output,
            conf.MIN_CONFIDENCE,
            conf.CLASS_NUMBER,
            nms_conf=conf.NMS_TH
        )

        if type(output) == int:
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                exit(0)  # EOF
            return  # just skip this prediction

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (self.model.net_info["height"] - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.model.net_info["height"] - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        return output
