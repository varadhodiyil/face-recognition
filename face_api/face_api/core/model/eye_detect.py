
import argparse
import os
import time

import dlib
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import FileVideoStream, VideoStream
from scipy.spatial import distance as dist

import cv2

DEBUG = False


class EyeBlinkDetector():

	def __init__(self):
		print("[INFO] loading facial landmark predictor...")
		path = os.path.dirname(os.path.abspath(__file__))
		self.predictor = dlib.shape_predictor(
			os.path.join(path, 'models/dlib_model.dat'))

	def eye_aspect_ratio(self, eye):

		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])

		C = dist.euclidean(eye[0], eye[3])
		ear = (A + B) / (2.0 * C)

		return ear

	def get_num_blinks(self, media_file):

		EYE_AR_THRESH = 0.3
		EYE_AR_CONSEC_FRAMES = 3

		COUNTER = 0
		TOTAL = 0

		detector = dlib.get_frontal_face_detector()

		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

		print("[INFO] starting video stream thread for %s  ..."% media_file)
		vs = cv2.VideoCapture(media_file)
		fileStream = True
		# vs = VideoStream(src=0).start()
		# vs = VideoStream().start()
		# vs = cv2.VideoCapture(0)
		# fileStream = False
		# time.sleep(1.0)

		# loop over frames from the video stream
		while True:
			k = cv2.waitKey(30) & 0xff
			# print vs.more()
			if k == 27:
				break
			# if fileStream and not vs.more():
			# 	break
			ret ,frame = vs.read()
			if not ret:
				break
			rects = detector(frame, 0)
			# cv2.imshow('image',frame)
			for rect in rects:

				shape = self.predictor(frame, rect)
				shape = face_utils.shape_to_np(shape)
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = self.eye_aspect_ratio(leftEye)
				rightEAR = self.eye_aspect_ratio(rightEye)

				ear = (leftEAR + rightEAR) / 2.0
				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

				if ear < EYE_AR_THRESH:
					COUNTER += 1
				else:
					if COUNTER >= EYE_AR_CONSEC_FRAMES:
						TOTAL += 1
					COUNTER = 0

				if DEBUG:
					cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			if DEBUG:
				cv2.imshow("Frame", frame)
		# vs.destroy()
		cv2.destroyAllWindows()
		return TOTAL


if __name__ == '__main__':
	ey = EyeBlinkDetector()
	ap = argparse.ArgumentParser()

	ap.add_argument("-v", "--video", type=str,
                 help="path to input video file")
	args = vars(ap.parse_args())
	print ey.get_num_blinks(args['video'])
