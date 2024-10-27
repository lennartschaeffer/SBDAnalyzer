import cv2
import numpy as np
import time
from Pose import PoseDetector


jamalVid = 'exercises/jamal browner squat.mov'
myVid = '/Users/lennartschaeffer/PycharmProjects/PoseEstimation/squat-vid.MOV'
badDepth = '/Users/lennartschaeffer/PycharmProjects/PoseEstimation/bad depth.mov'
deadlift = '/Users/lennartschaeffer/PycharmProjects/PoseEstimation/deadlift.MOV'
bench = '/Users/lennartschaeffer/Projects/SBDAnalyzer/SBDDetection/exercises/bench.mov'
bench2 = '/Users/lennartschaeffer/PycharmProjects/PoseEstimation/exercises/bench2.mov'

cap = cv2.VideoCapture(bench)

prev_time = 0
detector = PoseDetector()
count = 0
direction = 0

while True:
    success, img = cap.read()
    img = detector.findPose(img)


    # curr_time = time.time()
    # fps = 1 / (curr_time - prev_time)
    # prev_time = curr_time
    #
    # cv2.imshow("image", img)
    # cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # img = cv2.imread("/Users/lennartschaeffer/PycharmProjects/PoseEstimation/PersonalTrainer/angle.png")
    img = detector.findPose(img,False)
    landmark_list = detector.findPosition(img)

    maxAngle = 0
    minAngle = 0

    if len(landmark_list) != 0:

        angle = detector.findAngle(img,23,25,27)
        detector.findAngle(img, 24, 26, 28)
        detector.analyzeDepth(img)
        detector.determineExercise(img)
        #get a percentage score on our angle based on a range
        percentage = np.interp(angle,(180,300),(0,100))

        #check for completion of repetition
        if percentage == 100: #if we reached 100% of our range, we completed half the rep
            if direction == 0:
                count += 0.5
                direction = 1  #change the direction since we've reached the top or bottom of the rep(depending on exercise)
        if percentage == 0:
            if direction == 1:
                count += 0.5
                direction = 0


        cv2.putText(img, f'Reps: {int(count)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.putText(img, f'{int(percentage)}%', (600, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.rectangle(img, (600, 100), (700, (3 * int(percentage) + 100)), (255, 0, 0), -1)

    cv2.imshow("image",img)
    cv2.waitKey(1)