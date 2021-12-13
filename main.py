import cv2
import numpy

# pre-trained data on face frontals from opencv using haar cascade algorithm
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# captures video from webcam, or just a video
# webcam = cv2.VideoCapture(0)
webcam = cv2.VideoCapture('shmurda.mp4')

# loops over frames until we end the webcam/video
while True: 
	# read current frame of video
	is_frame_read, frame = webcam.read()
	
	# converts image to grayscale
	grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces
	face_coordinates = trained_face_data.detectMultiScale(grayscaled_img, 1.3, 5)

	# draws rectangles around faces
	for (x, y, w, h) in face_coordinates:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)

	# shows frame
	cv2.imshow('Face Detection', frame)

	# wait in milliseconds before updating webcam/video
	key = cv2.waitKey(1)

	# quit if q key pressed
	if key == 81 or key ==  113:
		break

# release video capture object
webcam.release()


