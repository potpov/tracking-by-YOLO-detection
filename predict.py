import conf
import torch
import Preprocessing
import Trajectory
import Yolo
import cv2
import pickle as pkl
import numpy as np
import random


def progress_bar(current_value, total):
    increments = 50
    percentual = int((current_value/ total) * 100)
    i = int(percentual // (100 / increments ))
    text = "\r[{0: <{1}}] {2}%".format('=' * i, increments, percentual)
    print(text, end="\n" if percentual == 100 else "")


# loading available classes (only person will be used)
fp = open(conf.CLASS_NAME_PATH, 'r')
classes = fp.read().split("\n")[:-1]  # discard the last

# loading colors from palette
colors = pkl.load(open(conf.PALETTE_PATH, "rb"))

# checking if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading network...")
yolo = Yolo.Yolo(device)
print("Network successfully loaded")

cap = cv2.VideoCapture(conf.VIDEO_PATH)
assert cap.isOpened(), 'Cannot capture source, bad video path?'
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
loading = 0  # for the status bar

distortion = Preprocessing.Distortion()

prospective = Preprocessing.Prospective()
trajectory = Trajectory.Trajectory()

# GLOBAL TRACKING VARS
buckets_colors = []  # id -> (R, G, B)
buckets_cords = []  # id -> [(x,y), (x2, y2)]
buckets_cords_orig = []

# saving result as a new video?
if conf.SAVE_RESULT:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vout = cv2.VideoWriter(conf.VIDEO_OUT_PATH, fourcc, 20.0, (640, 480))
    vout1 = cv2.VideoWriter(conf.VIDEO_UP_OUT_PATH, fourcc, 20.0, (640, 480))

print("analyzing video...")
while cap.isOpened():
    loading = loading + 1  # status bar progressing
    progress_bar(loading, video_length)
    ret, frame = cap.read()

    if ret:

        # ----------------------
        # START preprocessing |
        # ----------------------

        # camera distortion correction
        frame = distortion.correct(frame)

        # calculating upper visual
        frame_up = prospective.frame_transform(frame)

        # noise filtering
        frame = cv2.bilateralFilter(
            frame,
            conf.BILATERAL_D,
            conf.BILATERAL_SIGMA,
            conf.BILATERAL_SIGMA
        )

        # -------------------------------------------
        # END pre-processing | START yolo prediction|
        # -------------------------------------------

        output = yolo.predict(frame)

        if type(output) != torch.Tensor:
            continue

        # --------------------------------------
        # END yolo prediction | START tracking|
        # --------------------------------------

        # buckets scores is a ndarray which has people on rows and buckets on columns:
        #
        # ----------------------------------------------------
        #    -    | bucket1  |  bucket2 |   ...  |  bucketN  |
        # --------|----------|----------|--------|-----------|
        # person1 |          |          |        |           |
        # --------|----------|----------|--------|-----------|
        # person2 |          |          |        |           |
        # --------|----------|----------|--------|-----------|
        #   ...   |          |          |        |           |
        # --------|----------|----------|--------|-----------|
        # personM |          |          |        |           |
        # ----------------------------------------------------

        counter = len(['' for ot in output if classes[int(ot[-1])] == 'person'])
        if conf.TESTING:
            print("number of people detected", counter)

        buckets_score = np.ndarray(shape=(counter, len(buckets_cords)))  # distance matrix
        new_coords = []  # using this to recover coords from the distance matrix
        new_coords_orig = []  # old reference for new coords

        old_bc = buckets_cords

        for ip, person in enumerate(output):

            # check if class is person
            if classes[int(person[-1])] == 'person':
                
                # find detection center
                c1 = tuple(person[1:3].int().cpu())
                c2 = tuple(person[3:5].int().cpu())
                center = (np.asarray((c1[0] + c2[0]) // 2), np.asarray(c2[1]))  # cx, cy

                # get the coords of those points from the other perspective
                pts = np.array([[center[0], center[1]]], dtype="float32")
                pts = np.array([pts])
                center_upp = prospective.point_transform(pts)

                # SPECIAL CASE: avoid deadlock at the first iteration (no buckets)
                # adding the first person met
                if len(old_bc) == 0:
                    buckets_cords.append([center_upp])  # retrieve coords from the new person vector
                    buckets_cords_orig.append([center])
                    buckets_colors.append(random.choice(colors))  # link a color to this new index
                    if conf.TESTING:
                        print("new element in bucket list (INIT). buckets size: ", len(buckets_cords))
                    continue

                new_coords.append(center_upp)  # saving this for later
                new_coords_orig.append(center)

                # let's create a score person->bucket foreach bucket.
                for k, bucket in enumerate(old_bc):
                    cx1, cy1 = bucket[-1]  # compare last position
                    pos_dist = abs(center_upp[0] - cx1) + abs(center_upp[1] - cy1)

                    if conf.TESTING:
                        print("pos dist: ", pos_dist)

                    if len(bucket) > 1:  # if we have enough points, compare also trajectory for the bucket
                        next_x, next_y = trajectory.next_point_prediction(bucket, cx1, cy1)
                        trj_score = trajectory.score(next_x, next_y, center_upp[0], center_upp[1])

                        pos_dist = (pos_dist + trj_score) / 2

                        if conf.TESTING:
                            print("trj_score: ", trj_score)

                    buckets_score[ip][k] = pos_dist

        # find best people score for each buckets and return indexes
        if buckets_score.shape[0] > 0 and buckets_score.shape[1] > 0:
            # execute this cycle as many times as number of people
            for personBucket in buckets_score:
                if conf.TESTING:
                    print("buckets_score: ", buckets_score)

                # find min val in matrix
                ind = np.unravel_index(np.argmin(buckets_score, axis=None), buckets_score.shape)

                if conf.TESTING:
                    print("best index: ", ind)
                    print("new chords: ", new_coords)
                # check if min score is under a give threshold
                if buckets_score[ind] <= conf.MIN_TRACKING_TH:

                    if buckets_score.shape[1] > 1:
                        bucket_score_temp = buckets_score[:, ind[1]].copy()

                        for cp, per in enumerate(buckets_score):
                            idx = np.argmin(per)
                            if per[idx] < conf.MIN_TRACKING_TH:
                                if idx == ind[1]:
                                    A, B = np.partition(per, 1)[0:2]
                                    bucket_score_temp[cp] = A / B
                        ind = list(ind)
                        ind[0] = np.argmin(bucket_score_temp)
                        ind = tuple(ind)

                    # append person coords to bucktes
                    buckets_cords[ind[1]].append(new_coords[ind[0]])
                    buckets_cords_orig[ind[1]].append(new_coords_orig[ind[0]])
                    # setting score out of image view
                    buckets_score[ind[0], :] = 9998
                    buckets_score[:, ind[1]] = 9998
                    buckets_score[ind[0], ind[1]] = 9999

                    if conf.TESTING:
                        print("find position in bucket: ", ind[1])

                    col_idx = ind[1]

                # if it is not under a given threshold add buckets
                else:
                    buckets_cords.append([new_coords[ind[0]]])
                    buckets_cords_orig.append([new_coords_orig[ind[0]]])
                    buckets_colors.append(random.choice(colors))
                    if conf.TESTING:
                        print("new element in bucket list. buckets: ", buckets_cords)
                    col_idx = buckets_score.shape[1]

                cv2.circle(
                    frame_up, (
                        new_coords[ind[0]][0],
                        new_coords[ind[0]][1]
                    ),
                    7,
                    buckets_colors[col_idx],
                    -1
                )

                cv2.circle(
                    frame, (
                        new_coords_orig[ind[0]][0],
                        new_coords_orig[ind[0]][1]
                    ),
                    7,
                    buckets_colors[col_idx],
                    -1
                )

            for bkt_idx, bkt in enumerate(buckets_cords):
                bkt = np.asarray(bkt).reshape((-1, 1, 2))
                cv2.polylines(frame_up, np.int32([bkt]), isClosed=False, color=buckets_colors[bkt_idx], thickness=3,
                              lineType=10)
                bkt_orig = np.asarray(buckets_cords_orig[bkt_idx]).reshape((-1, 1, 2))
                cv2.polylines(frame, np.int32([bkt_orig]), isClosed=False, color=buckets_colors[bkt_idx], thickness=3,
                              lineType=10)

        frame_up = cv2.resize(frame_up, (640, 480))

        if conf.SAVE_RESULT:
            vout1.write(frame_up)
            vout.write(frame)

        if conf.LIVE_RESULTS:
            cv2.imshow("frame", frame)
            cv2.imshow("frame_up", frame_up)
            key = cv2.waitKey(1)

    else:  # exit if video if over
        break

# saving results and exiting
if conf.SAVE_RESULT:
    cap.release()
    vout.release()
    vout1.release()
    print("Your video is ready!")
    cv2.destroyAllWindows()







