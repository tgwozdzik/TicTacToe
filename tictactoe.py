import cv2
import numpy as np
from skimage import data, io, filters, morphology, feature, measure
from matplotlib import pyplot as plt
from math import hypot

def getEdges(image):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200,apertureSize = 3)
    edges = morphology.dilation(edges,morphology.disk(4))
    return edges

def getLineIntersectionPoint(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def getEndPointsOfLines(lines):
    points = []
    minDistance = 50
    if (lines is not None):
        print(len(lines))
        if (len(lines)>=4)and(len(lines)<30):
            for line in lines:
                rho,theta = line[0],line[1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1280*(-b))
                y1 = int(y0 + 1280*(a))
                x2 = int(x0 - 1280*(-b))
                y2 = int(y0 - 1280*(a))

                # Add point if is unique (not so close to others)
                if (len(points)>0):
                    bool = True
                    for i in points:
                        if (hypot(i[0] - x1, i[1] - y1)<minDistance):
                            bool = False
                            break
                        if (hypot(i[2] - x1, i[3] - y1)<minDistance):
                            bool = False
                            break
                    if (bool):
                        points.append([x1,y1,x2,y2])
                else:
                    points.append([x1,y1,x2,y2])
    return points

def getIntersectionPoints(points):
    intersectionPoints = []
    for i in range(0, len(points)-1):
        for j in range(i+1 ,len(points)):
            line1 = [[points[i][0],points[i][1]],[points[i][2],points[i][3]]]
            line2 = [[points[j][0],points[j][1]],[points[j][2],points[j][3]]]
            point = getLineIntersectionPoint(line1,line2)
            if (point):
                intersectionPoints.append(point)
    return intersectionPoints

def getContours(blackAndWhiteImage,limitContourLength):
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if len(contour)>limitContourLength]
    return contours

def getCentroid(points):
    x = [] ; y = []
    for p in points:
        x.append(p[1])
        y.append(p[0])
    centroid = (sum(x) / len(points), sum(y) / len(points))
    return centroid

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    edges = getEdges(frame)
    cv2.imshow('Edges', edges)

    contours = getContours(edges,100)

    if (contours):
        centroids = [getCentroid(contour[0]) for contour in contours]

    lines = cv2.HoughLines(edges,1,10*np.pi/180,250)
    if (lines is not None):
        lines = lines[0]
        points = getEndPointsOfLines(lines)

        # draw contours
        cv2.drawContours(frame, contours, -1, (0,255,0), 2)

        # draw lines
        for i in points:
            cv2.line(frame,(i[0], i[1]),(i[2],i[3]),(0,255,0),2)

        intersectionPoints = getIntersectionPoints(points)

        # draw intersectionPoints
        for i in intersectionPoints:
            cv2.circle(frame,(int(i[0]),int(i[1])), 5, (0,0,255), -1)

        cv2.imshow('Video',frame)
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()