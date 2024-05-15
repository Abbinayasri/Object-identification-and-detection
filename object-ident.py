import cv2
from espeak import espeak
import time
import VL53L0X

# Create a VL53L0X object
tof = VL53L0X.VL53L0X()
# Start ranging
tof.start_ranging(VL53L0X.VL53L0X_BEST_ACCURACY_MODE)

#thres = 0.45 # Threshold to detect object

classNames = []
classFile = "/home/pi/myworkspace/objectidentification/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/myworkspace/objectidentification/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/myworkspace/objectidentification/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(200,200)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    className = ""
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo,className


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    espeak.synth("Hello")
    #cap.set(10,70)
    objectcounter = 0
    previousclassName = ""

    espeak.rate = 50
    time.sleep(1)
    espeak.synth("Object deteting glass is initialized")

    while True:
        success, img = cap.read()
        result, objectInfo, className = getObjects(img,0.45,0.2)
        distance = tof.get_distance()
        if className != "":
            if previousclassName == className and objectcounter >= 2:
                #print(objectInfo)
                if distance > 0 and distance < 1000:
                    outputtext = "object" + className + "is detected at " + str(distance/10) + "centimeters"
                else:
                    outputtext = "object" + className + "is detected"
                espeak.synth(outputtext)
                print(className)
                #cv2.imshow("Output",img)
                #cv2.imshow("Preview",img)
                cv2.waitKey(1)
                previousclassName = ""
                objectcounter = 0
            else:
                previousclassName = className
                objectcounter += 1
