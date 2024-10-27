import cv2
import mediapipe as mp
import time
import math

class PoseDetector:

    def __init__(self, mode=False, upBody=False,smooth=True,detectionCon=0.5,trackCon=0.5):

        self.angle_list = []
        self.landmark_list = None
        self.results = None
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self,img,draw=True):
        self.landmark_list = []
        if self.results.pose_landmarks:
            for index, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                realX = int(landmark.x * width)
                realY = int(landmark.y * height)
                self.landmark_list.append([index,realX,realY])
                if draw:
                    cv2.circle(img,(realX,realY),5,(255,0,0),cv2.FILLED)
        return self.landmark_list

    def analyzeDepth(self,img,draw=True):
        leg_landmarks = {}
        if self.results.pose_landmarks:
            for index, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                realX = int(landmark.x * width)
                realY = int(landmark.y * height)
                self.landmark_list.append([index, realX, realY])
                if draw:
                    #get the y value of the knee joint and hip joint
                    if index == 23 or index == 24 or index == 25 or index == 26:
                        leg_landmarks[index] = realY
                        cv2.circle(img, (realX, realY), 5, (255, 0, 0), cv2.FILLED)

        #check if the y value of the hip joint is greater than the y value of the knee joint at any time
        #this means the hip crease is below the knee bend, which means depth

        if leg_landmarks[24] > leg_landmarks[26] and leg_landmarks[23] > leg_landmarks[25]:
            print(str(leg_landmarks[24])+" is greater than "+str(leg_landmarks[26]))
            cv2.putText(img, 'DEPTH', (300, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)


    def findAngle(self, img, point1, point2, point3,draw=True):

        #get landmarks
        x1,y1 = self.landmark_list[point1][1:] #ignore the first index since we need x and y which are stored in index 1 and 2
        x2,y2 = self.landmark_list[point2][1:]
        x3,y3 = self.landmark_list[point3][1:]

        #find angle

        angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2, x1-x2))

        if angle < 0:
            angle += 360

        #draw the points and connecting lines
        if draw:
            cv2.line(img, (x1,y1),(x2,y2),(255,255,255),3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
            cv2.circle(img, (x1,y1), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (255, 0, 255), 2)
            cv2.putText(img,str(int(angle)),(x2-20,y2+50),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),3)

        return angle

    # TODO
    # classify if its a squat,bench, or deadlift so we can determine what to analyze
    def determineExercise(self, img, draw=True):
        # for bench, we look at the shoulder and hip joints if theyre at a reasonably flat angle
        angle = self.findAngle(img,20,14,12)
        #if the angle between the hand, elbow, and shoulder joint is relatively large we can assume we're starting
        #with either a bench or a deadlift
        # Keep the last 10 angle values for averaging
        self.angle_list.append(angle)
        if len(self.angle_list) > 10:  # change 10 to a suitable window size
            self.angle_list.pop(0)

        # Calculate the average of the recent angles
        avg_angle = sum(self.angle_list) / len(self.angle_list)

        # Check if the average angle indicates a squat
        if avg_angle < 100:
            cv2.putText(img, 'SQUAT', (300, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        else:
            cv2.putText(img, 'BENCH', (300, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
def main():
    cap = cv2.VideoCapture('exercises/jamal browner squat.mov')
    prev_time = 0
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        landmark_list = detector.findPosition(img)
        print(landmark_list)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.imshow("image", img)
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.waitKey(20)

if __name__ == "__main__":
    main()