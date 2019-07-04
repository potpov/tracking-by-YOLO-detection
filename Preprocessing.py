import conf
import cv2


class Distortion:
    """
    correct image distortion using camera matrix and distortion coefficient k1
    """
    def __init__(self):
        pass

    def correct(self, frame):
        """
        :param frame: numpy matrix
        :return: corrected numpy matrix
        """
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
    """
    homographic transformation from the starting position to the upside prospective
    """
    def __init__(self):
        pass

    def frame_transform(self, frame):
        """
        :param frame: input numpy matrix to be transformed
        :return: transformed numpy matrix
        """
        return cv2.warpPerspective(frame, conf.UP_TRANSF_MATRIX, dsize=(774, 555))

    def point_transform(self, point):
        """
        :param point: tuple of coordinates to be transformed
        :return: tuple of transformed coord
        """
        point = cv2.perspectiveTransform(point, conf.UP_TRANSF_MATRIX)
        return [point[0][0][0], point[0][0][1]]
