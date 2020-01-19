# coding:utf-8
import cv2
import numpy as np
import os
capture = cv2.VideoCapture(0)


face_cascade = cv2.CascadeClassifier('model.xml')


def loadImageSet(add):

    images = []
    for i in os.listdir(add):

        image = cv2.imread(os.path.join(add, i))
        print(os.path.join(add, i), 'has loaded')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image, 1.1, 5)

        for box in faces:
            lx, ly, rx, ry = [int(box[0]), int(box[1]), int(box[0]) + int(box[2]), int(box[1]) + int(box[3])]
            img_aoi = image[ly:ry, lx:rx]
            img = cv2.resize(img_aoi, (24, 24))
            images.append(img.flatten().tolist())


    return np.array(images, dtype=np.int).T


def ReconginitionVector(FaceMat, selecthr=0.8):

    avgImg = np.mean(FaceMat, 1, keepdims=True)
    diffTrain = FaceMat - avgImg
    eigvals, eigVects = np.linalg.eig(np.dot(diffTrain, diffTrain.T))
    eigSortIndex = np.argsort(-eigvals)
    for i in range(np.shape(FaceMat)[0]):
        if (eigvals[eigSortIndex[:i]] / eigvals.sum()).sum() >= selecthr:
            eigSortIndex = eigSortIndex[:i]
            break
    covVects = np.dot(diffTrain.T, eigVects[:, eigSortIndex])
    return avgImg, covVects, eigVects[:, eigSortIndex]


def judgeFace(judgeImg, covVects, avgImg, lib_face):
    # lib_face fd*n
    diff = judgeImg - avgImg
    weiVec = np.dot(covVects.T, diff)
    res = 0
    dist = np.power((weiVec.T - lib_face), 2)
    dist = np.sqrt(np.sum(dist, 1))
    return dist

images = loadImageSet('images')
avgImg, lib_feat, eigVects = ReconginitionVector(images)
print('begin analysis, press Q to terminate')
cv2.namedWindow('Beauty rating program, press Q to terminate')
while True:
    ret, image = capture.read()
    if not ret:
        break
    image_bck = image[:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(image, 1.1, 5)

    for box in faces:
        lx, ly, rx, ry = [int(box[0]), int(box[1]), int(box[0])+int(box[2]), int(box[1])+int(box[3])]
        cv2.rectangle(image, (lx, ly), (rx, ry), (0, 255, 0), 4)
        image_test = image[ly:ry, lx:rx]
        image_test = cv2.resize(image_test, (24, 24))
        image_test = np.expand_dims(image_test.flatten(), 1)
        dist = judgeFace(image_test, eigVects, avgImg, lib_feat)

        cv2.rectangle(image_bck, (lx, ly), (rx, ry), (0, 255, 0), 2)

        dist = np.hstack((dist, np.array([600])))
        score = np.real(1 - (dist - np.min(dist))/(np.max(dist) - np.min(dist)))
        t = 'Score: ' + '%d' % float(score[:][0]*100)
        cv2.putText(image_bck, t, (lx, ly), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))

    cv2.imshow('Beauty rating program, press Q to terminate', image_bck)
    if cv2.waitKey(1) == ord('q'):
        break




