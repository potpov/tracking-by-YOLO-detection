import numpy as np
import conf


class Trajectory:

    def __init__(self):
        pass

    def point_predict(self, m, v, x, y):
        return np.cos(np.deg2rad(m)) * v + x, np.sin(np.deg2rad(m)) * v + y

    def coeff_predict(self, bucket):
        """
        predict angular coefficient and module according to previous model observation 
        """
        
        m = 0
        v = 0

        min_len = min(len(bucket), conf.MAX_LEN_BUCKET)

        bucket = bucket[-min_len:]

        for i in range(min_len - 1):
            x0, y0 = bucket[i]
            x1, y1 = bucket[i + 1]

            m += np.rad2deg(np.arctan2(y0 - y0, x1 - x0))
            v += np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        m = m / (min_len - 1)
        v = v / (min_len - 1)
        return m, v

    def next_point_prediction(self, target_bucket, point_x, point_y):
        """
        predict point position
        """
        m, v = self.coeff_predict(target_bucket)
        pred_x, pred_y = self.point_predict(m, v, point_x, point_y)
        return pred_x, pred_y

    def score(self, pred_x, pred_y, x, y):
        """
        Compare point prediction with observation
        """
        
        return np.sqrt((x - pred_x) ** 2 + (y - pred_y) ** 2)
