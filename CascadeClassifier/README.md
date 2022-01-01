# Cascascade Classification

## Precodition
### Uses the following [OpenCV 3.4](https://github.com/opencv/opencv/tree/3.4) applications:

* opencv_createsamples
* opencv_traincascade

### Data
* Images are in folder `Images`.
* Annotations are in folder `Airplanes_Annotations`

## Steps

### 1. Images preparation
* Run `python3 categorical_image_generator.py`
* Use data from folder `0` as negative images.
* (Optional) Data from folder `1` can be used as positive images.

### 2. Negative images list preparation
* Run `find ./Images/0/ -name *.png > bg.txt`

### 3. Image list preparation
* Run `python3 image_list_generator.py`

### 4. Create samples
* Run `opencv_createsamples -info airplanes.info -num 355 -w 32 -h 32 -vec airplanes.vec`
* Check `opencv_createsamples -w 32 -h 32 -vec airplanes.vec`

### 5. Train cascade classifier
* Create `data` folder 
* Run `opencv_traincascade -data data -vec airplanes.vec -bg bg.txt -numPos 330 -numNeg 1200 -numStages 10 -w 32 -h 32 -featureType HAAR -maxFalseAlarmRate 0.1` for HAAR classifiers
* Replace `HAAR` with `LBP` in the above command to get the LBP classifiers.
* Trained classifier is in `data/cascade.xml` file
* Run `python3 detect.py` to test.

## Links
* [Cascade Classifier Training](https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html)
* [Training your own Cascade/Classifier/Detector — OpenCV](https://dikshit18.medium.com/training-your-own-cascade-classifier-detector-opencv-9ea6055242c2)
* [Creating your own Haar Cascade OpenCV Python Tutorial](https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/)