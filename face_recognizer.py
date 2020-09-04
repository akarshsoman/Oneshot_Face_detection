'''Face Recognition Main File'''
import argparse
import cv2
import glob
import os

import numpy as np
import tensorflow as tf

from imutils import face_utils
from keras.models import load_model
from helpers.fr_utils import img_path_to_encoding, img_to_encoding
from helpers.image_dataset_gen import crop_face
from scipy.spatial import distance


ap= argparse.ArgumentParser()
ap.add_argument('-img','--image_path',required=True,help='Please input the target image')
ap.add_argument('-vid','--video_path',required=False, default = 'webcam', help='Please input the video path')
args=vars(ap.parse_args())


input_video = args['video_path']
target_image = args['image_path']

targetFace_dir = crop_face(target_image)

FR_model = load_model('helpers/nn4.small2.v1.h5')
print("Total Params:", FR_model.count_params())

face_cascade = cv2.CascadeClassifier('helpers/haarcascades/haarcascade_frontalface_default.xml')

threshold = 0.25

face_database = {}

for name in os.listdir(targetFace_dir):
	for image in os.listdir(os.path.join(targetFace_dir,name)):
		identity = os.path.splitext(os.path.basename(image))[0]
		face_database[identity] = img_path_to_encoding(os.path.join('helpers/faces',name,image), FR_model)

if input_video == 'webcam':
	video_capture = cv2.VideoCapture(0)
else:
	video_capture = cv2.VideoCapture(input_video)

frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
if input_video != 'webcam':
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	outfile = os.path.splitext(os.path.basename(input_video))[0]+'_result.avi'
	out = cv2.VideoWriter(outfile, fourcc, 20.0, (frame_width,frame_height))

while True:
	ret, frame = video_capture.read()
	if ret == False:
		break

	frame_bg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	w, h = frame_bg.shape
	ret = np.empty((w, h, 3),dtype=np.uint8)
	ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = frame_bg

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)
	for(x,y,w,h) in faces:
		# cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
		roi = ret[y:y+h, x:x+w]
		encoding = img_to_encoding(roi, FR_model)
		min_dist = 100
		identity = None

		for(name, encoded_image_name) in face_database.items():
			dist = np.linalg.norm(encoding - encoded_image_name)
			if(dist < min_dist):
				min_dist = dist
				identity = name
			# print('Min dist: ',min_dist)

		if min_dist < 0.05:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
			cv2.putText(frame, "Face : " + identity, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
			# cv2.putText(frame, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
		else:
			pass
			# cv2.putText(frame, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
	if input_video != 'webcam':
		out.write(frame)
	cv2.imshow('Face Recognition System', frame)
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

video_capture.release()
if input_video != 'webcam':
	out.release()
cv2.destroyAllWindows()

print('Output video saved as:', outfile)

