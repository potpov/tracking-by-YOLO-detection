# main idea

detect people for each frame using yolo v3.
compare each person coordinates (center of the binding box) with a bucket 
where each bucket contains the history of each person tracked during the previous frames.
a score is evaluated by considering the distance from the last coord of the 
bucket and the next coord expected from that bucket according to its trajectory in time.

full idea can be found here: https://drive.google.com/open?id=1cwVWWydWiCwkSIiFqZewdeJUecBgM4L9

# installation

download YOLO weights
```
cd content
wget https://pjreddie.com/media/files/yolov3.weights
cd ..
```

install requirements:
```
python3.7 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

run the network
``` python predict.py ```

more sample videos available at: https://www.dropbox.com/s/6egotfb5n5a1lkp/Archivio_video.zip

if you want to calibrate your camera, you first need to estimate the camera matrix
(use https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html as 
reference). for a quick estimator of the distortion coefficient (k1) we implemented a nice algorithm, 
check it out https://github.com/potpov/camera-correction

![alt text](https://i.ibb.co/zFJxDQx/example.png)

