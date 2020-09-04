import os
import cv2
import dlib
import argparse

from imutils import face_utils
from imutils.face_utils import FaceAligner


# ap= argparse.ArgumentParser()
# ap.add_argument('-i','--image_path',required=True,help='Please input an image')
# args=vars(ap.parse_args())

def face_align(img):
	detector = dlib.get_frontal_face_detector()
	shape_predictor = dlib.shape_predictor("helpers/shape_predictor_68_face_landmarks.dat")
	face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=200)
	
	frame = cv2.imread(img)
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = detector(frame_gray)
	if len(faces) == 1:
	    face = faces[0]
	    (x, y, w, h) = face_utils.rect_to_bb(face)
	    face_img = frame_gray[y-50:y + h+100, x-50:x + w+100]
	    face_aligned = face_aligner.align(frame, frame_gray, face)
	return face_aligned

def crop_face(image_path):
	img_folder = image_path
	target_dir = 'helpers/faces/'

	if os.path.isdir(img_folder):
		for img in os.listdir(img_folder):
			name = os.path.splitext(os.path.basename(img))[0]
			target_path = os.path.join(target_dir, name)
			if not os.path.exists(target_path):
				os.makedirs(target_path, exist_ok = 'True')
			img_path = os.path.join(img_folder, img)
			face_aligned = face_align(img_path)
			cv2.imwrite(os.path.join(target_path, img), face_aligned)

		print('Extracted faces to directory: ', target_dir)

	elif os.path.isfile(img_folder):
		name = os.path.splitext(os.path.basename(img_folder))[0]
		target_path = os.path.join(target_dir, name)
		if not os.path.exists(target_path):
			os.makedirs(target_path, exist_ok = 'True')
		face_aligned = face_align(img_folder)
		print(face_aligned)
		cv2.imwrite(os.path.join(target_path, img_folder.split('/')[-1]), face_aligned)

		print('Extracted faces to directory: ', target_dir)


	return target_dir
       
# cv2.imshow('img', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


