# Assignment for CAMS

## Task  
Take a black and white/greyscale image of a person and match with video in real time. 
Note : Video should be in colour mode and Photograph should be of Black & White / Grey scale   

## Approach 
A Siamese network named FaceNet is used to solve this problem which does not require extensive training samples for recognition tasks. 
It uses a single shot learning approach which requires just a single black and white image. 

Initially to crop the faces from the images, a Haar cascade classifier (which is a pre-trained model to detect faces and eyes in an image) alongwith OpenCV is taken.
Then to obtain the high quality feature embeddings (128-element vectors) from the face crop, a lite model of FaceNet is used. 

Basically, the Euclidean distance between the vector embeddings of the target image and the embeddings of each frame in the video is taken.
The lesser the distance, more similar the images are. 

Once the minimum distance of the face embedding among all the embeddings in the video is calculated, the corresponding label is given.


	## Folder Structure

	├── face_recognizer.py
	├── helpers
	│   ├── faces
	│   │   ├── messi
	│   │   │   └── messi.jpg
	│   │   └── ronaldo
	│   │       └── ronaldo.jpg
	│   ├── fr_utils.py
	│   ├── haarcascades
	│   │   └── haarcascade_frontalface_default.xml
	│   ├── image_dataset_gen.py
	│   ├── nn4.small2.v1.h5
	│   └── shape_predictor_68_face_landmarks.dat
	├── README.md
	├── requirements.txt
	├── ronaldo.mp4
	├── ronaldo_result.avi
	└── target_faces
	├── messi.jpg
	└── ronaldo.jpg

- Currently the target greyscale image is saved in the 'target_faces/' folder. The label taken right now is the name of the image.
- The 'helpers/' folder consists of all the helper codes for the main python file 'face_recognizer.py'
	- The 'helpers/faces' folder consists of face crops.
- The output video is saved in the same directory as the face_recognizer.py.

## Usage:
The script can be used as given below.  
```

python face_recognizer.py -img [PATH_TO_TARGET_IMAGE(S)] -vid [PATH_TO_TARGET_VIDEO][default=webcam] 

```
The image path can be a folder or file as needed. 


Note: 
Assumption is taken as only one image is there for a class. Hence no training is done, pretrained models have been used. 
Model performance can be improved by saving more black and white images of the target person.
If it is to be tested with a video, give video path. ELse webcam is taken as default. 
