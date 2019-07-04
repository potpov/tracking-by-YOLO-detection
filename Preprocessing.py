import conf
import cv2


class Distortion:
    def __init__(self):
        pass

    def correct(self, frame):
        camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            conf.CAMERA_MATRIX,
            conf.DIST_K,
            (frame.shape[1], frame.shape[0]),
            alpha=1,
        )

        map1, map2 = cv2.initUndistortRectifyMap(
            conf.CAMERA_MATRIX,
            conf.DIST_K,
            None,
            camera_matrix,
            (frame.shape[1], frame.shape[0]),
            cv2.CV_16SC2
        )
        return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


class Prospective:
    def __init__(self):
        pass

    def frame_transform(self, frame):
        return cv2.warpPerspective(frame, conf.UP_TRANSF_MATRIX, dsize=(774, 555))

    def point_transform(self, point):
        point = cv2.perspectiveTransform(point, conf.UP_TRANSF_MATRIX)
        return [point[0][0][0], point[0][0][1]]
