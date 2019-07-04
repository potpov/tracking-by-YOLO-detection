# main idea

# installation

download YOLO weights
```
cd content
wget https://pjreddie.com/media/files/yolov3.weights
```

install requirements:
```
pip install -r requirements.txt
```

more videos at: https://www.dropbox.com/s/6egotfb5n5a1lkp/Archivio_video.zip

if you want to calibrate your camera, we implemented a coefficient distortion estimator 
based on the paper [PUT LINK].
as it is kinda heavy we wrote a different file.
in order to calibrate your camera ensure you set up the right video path in 
the configuration file, then run:
```python calibration```
we'll try to estimate the best coefficient, initial and best final result will be shown for 
the first frame of the input video.


