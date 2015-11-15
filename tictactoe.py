import cv2
import cv2.cv as cv
import numpy as np
import random
from skimage import data, io, filters, morphology, feature, measure
from matplotlib import pyplot as plt
from math import hypot

def getEdges(image):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    edges = morphology.dilation(edges,morphology.disk(4))
    return gray, edges

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
        #print("Number of lines: ", len(lines))
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

def checkY(point1, point3, i):
    position = -1
    if(i[1] < point3[1]):
        position=0
        return position
    if(point1[1] > i[1] > point3[1]):
        position=1
        return position
    if(i[1] > point1[1]):
        position=2
        return position

    return position

def checkX(point1, point2, i):
    position = -1
    if(i[0] < point1[0]):
        position=0
        return position
    if(point1[0] < i[0] < point2[0]):
        position=1
        return position
    if(i[0] > point2[0]):
        position=2
        return position
    
    return position


def checkGameState(intersectionPoints, circles, gameState):
    game = [[0,0,0],[0,0,0],[0,0,0]]
    
    for i in [0,1,2]:
        for j in [0,1,2]:
            game[i][j] = gameState[i][j]

    point1 = intersectionPoints[0]
    point2 = intersectionPoints[1]
    point3 = intersectionPoints[2]
    point4 = intersectionPoints[3]
    
    max_x = max([point1[0], point2[0],point3[0], point4[0]])
    min_x = min([point1[0], point2[0],point3[0], point4[0]])
        
    max_y = max([point1[1], point2[1],point3[1], point4[1]])
    min_y = min([point1[1], point2[1],point3[1], point4[1]])
                
    point1 = [min_x, max_y]
    point2 = [max_x, max_y]
    point3 = [min_x, min_y]
    point4 = [max_x, min_y]
    #print(point1, point2, point3, point4)
    
    positionX=-1
    positionY=-1
    
    for i in circles:
        positionX = checkX(point1, point2, i)
        positionY = checkY(point1, point3, i)
        #print("position", positionX, positionY, i)
        if(positionX != -1 and positionY != -1):
            game[positionY][positionX] = 1;
    return game

def aiMakeDecision(gameState):
    possibilities = [];
    for x in [0,1,2]:
        for y in [0,1,2]:
            if gameState[x][y] == 0:
                possibilities.append([x,y])
    chosen = possibilities[random.randint(0,len(possibilities)-1)]
    gameState[chosen[0]][chosen[1]] = 2

    return gameState

def checkState(gameState, tempGameState):
    isCorrect = False
    numberOfPlayerNewMoves = 0

    for i in [0,1,2]:
        for j in [0,1,2]:
            if(gameState[i][j] != tempGameState[i][j]):
                if(gameState[i][j] == 2):
                    print("Incorrect move! Computer already placed here his 'X'. Please change your move!")
                    return False
                numberOfPlayerNewMoves += 1

    if(numberOfPlayerNewMoves == 0):
        return -1

    if(numberOfPlayerNewMoves == 1):
        isCorrect = True
    else:
        print("Incorrect move! You placed 2 or more 'O's. Please change your move!")

    return isCorrect

def drawCurrentSituation(intersectionPoints, gameState):
    
    point1 = intersectionPoints[0]
    point2 = intersectionPoints[1]
    point3 = intersectionPoints[2]
    point4 = intersectionPoints[3]
                
    max_x = max([point1[0], point2[0],point3[0], point4[0]])
    min_x = min([point1[0], point2[0],point3[0], point4[0]])
                
    max_y = max([point1[1], point2[1],point3[1], point4[1]])
    min_y = min([point1[1], point2[1],point3[1], point4[1]])

    point1 = [min_x, max_y]
    point2 = [max_x, max_y]
    point3 = [min_x, min_y]
    point4 = [max_x, min_y]
                
    cv2.line(frame, (point1[0], point1[1]), (point2[0],point2[1]), (0,255,0), 2)
    cv2.line(frame, (point1[0], point1[1]), (point3[0],point3[1]), (0,255,0), 2)
    cv2.line(frame, (point3[0], point3[1]), (point4[0],point4[1]), (0,255,0), 2)
    cv2.line(frame, (point2[0], point2[1]), (point4[0],point4[1]), (0,255,0), 2)
                
    cv2.line(frame, (point1[0], point1[1]), (point1[0],point1[1]+(point1[1]-point3[1])), (0,255,0), 2)
    cv2.line(frame, (point1[0], point1[1]), (point1[0]-(point2[0]-point1[0]),point1[1]), (0,255,0), 2)
                
    cv2.line(frame, (point2[0], point2[1]), (point2[0],point2[1]+(point2[1]-point4[1])), (0,255,0), 2)
    cv2.line(frame, (point2[0], point2[1]), (point2[0]+(point2[0]-point1[0]),point2[1]), (0,255,0), 2)
                
    cv2.line(frame, (point3[0], point3[1]), (point3[0],point3[1]-(point1[1]-point3[1])), (0,255,0), 2)
    cv2.line(frame, (point3[0], point3[1]), (point3[0]-(point4[0]-point3[0]),point3[1]), (0,255,0), 2)
                
    cv2.line(frame, (point4[0], point4[1]), (point4[0],point4[1]-(point2[1]-point4[1])), (0,255,0), 2)
    cv2.line(frame, (point4[0], point4[1]), (point4[0]+(point4[0]-point3[0]),point4[1]), (0,255,0), 2)



    #bottom left
    if(gameState[2][0] ==2):
        cv2.line(frame, (point1[0] - 20, point1[1] + 20), (point1[0]-(point2[0]-point1[0]) + 20,point1[1]+(point1[1]-point3[1]) - 20), (0,255,0), 2)
        cv2.line(frame, (point1[0] - (point2[0]-point1[0]) + 20, point1[1] + 20), (point1[0] - 20,point1[1]+(point1[1]-point3[1]) - 20), (0,255,0), 2)
            
    #bottom middle
    if(gameState[2][1] ==2):
        cv2.line(frame, (point1[0] + 20, point1[1] + 20), (point1[0] + (point2[0]-point1[0]) - 20, point1[1]+(point1[1]-point3[1]) - 20), (0,255,0), 2)
        cv2.line(frame, (point1[0] + 20, point1[1] + (point1[1]-point3[1]) - 20), (point1[0] + (point2[0]-point1[0]) - 20, point1[1] + 20), (0,255,0), 2)
                
    #bottom right
    if(gameState[2][2] ==2):
        cv2.line(frame, (point2[0] + 20, point2[1] + 20), (point2[0]+(point2[0]-point1[0]) - 20,point1[1]+(point2[1]-point4[1]) - 20), (0,255,0), 2)
        cv2.line(frame, (point2[0] + (point2[0]-point1[0]) - 20, point2[1] + 20), (point2[0] + 20,point2[1]+(point2[1]-point4[1]) - 20), (0,255,0), 2)
                
    #middle left
    if(gameState[1][0] ==2):
        cv2.line(frame, (point1[0] - 20, point1[1] - 20), (point1[0]-(point2[0]-point1[0]) + 20,point1[1]-(point1[1]-point3[1]) + 20), (0,255,0), 2)
        cv2.line(frame, (point1[0] - (point2[0]-point1[0]) + 20, point1[1] - 20), (point1[0] - 20,point1[1]-(point1[1]-point3[1]) + 20), (0,255,0), 2)
                
    #middle
    if(gameState[1][1] ==2):
        cv2.line(frame, (point1[0] + 20, point1[1] - 20), (point1[0]+(point2[0]-point1[0]) - 20,point1[1]-(point1[1]-point3[1]) + 20), (0,255,0), 2)
        cv2.line(frame, (point1[0] + (point2[0]-point1[0]) - 20, point1[1] - 20), (point1[0] + 20,point1[1]-(point1[1]-point3[1]) + 20), (0,255,0), 2)
                
    #middle right
    if(gameState[1][2] ==2):
        cv2.line(frame, (point2[0] + 20, point2[1] - (point2[1] - point4[1]) + 20), (point2[0] + (point2[0]-point1[0]) - 20,point1[1] - 20), (0,255,0), 2)
        cv2.line(frame, (point2[0] + 20, point2[1] - 20), (point2[0] + (point2[0] - point1[0]) - 20, point2[1] - (point2[1] - point4[1]) + 20), (0,255,0), 2)
                
    #top left
    if(gameState[0][0] ==2):
        cv2.line(frame, (point3[0] - 20, point3[1] - 20), (point3[0]-(point4[0]-point3[0]) + 20,point3[1]-(point1[1]-point3[1]) + 20), (0,255,0), 2)
        cv2.line(frame, (point3[0] - (point4[0]-point3[0]) + 20, point3[1] - 20), (point3[0] - (point4[1]-point3[1]) - 20, point3[1]-(point1[1]-point3[1]) + 20), (0,255,0), 2)
                
    #top middle
    if(gameState[0][1] ==2):
        cv2.line(frame, (point3[0] + 20, point3[1] - 20), (point3[0] + (point4[0] - point3[0]) - 20, point3[1] + (point3[1] - point1[1]) + 20), (0,255,0), 2)
        cv2.line(frame, (point3[0] + 20, point3[1] - (point1[1] - point3[1]) + 20), (point3[0] + (point4[0] - point3[0]) - 20, point3[1] - 20), (0,255,0), 2)
                
    #top right
    if(gameState[0][2] ==2):
        cv2.line(frame, (point4[0] + 20, point4[1] - 20), (point4[0] + (point4[0] - point3[0]) - 20, point4[1] + (point3[1] - point1[1]) + 20), (0,255,0), 2)
        cv2.line(frame, (point4[0] + 20, point4[1] - (point1[1] - point3[1]) + 20), (point4[0] + (point4[0] - point3[0]) - 20, point4[1] - 20), (0,255,0), 2)

def checkWinner(gameState):
    winner = -1
    if(gameState[0][0] == gameState[0][1] == gameState[0][2]):
        winner = gameState[0][0]
    if(gameState[1][0] == gameState[1][1] == gameState[1][2]):
        winnder = gameState[1][0]
    if(gameState[2][0] == gameState[2][1] == gameState[2][2]):
        winnder = gameState[2][0]

    if(gameState[0][0] == gameState[1][0] == gameState[2][0]):
        winner = gameState[0][0]
    if(gameState[0][1] == gameState[1][1] == gameState[2][1]):
        winnder = gameState[0][1]
    if(gameState[0][2] == gameState[1][2] == gameState[2][2]):
        winnder = gameState[0][2]

    if(gameState[0][0] == gameState[1][1] == gameState[2][2]):
        winner = gameState[0][0]
    if(gameState[2][0] == gameState[1][1] == gameState[0][2]):
        winner = gameState[2][0]

    if(winner == 1):
        print("CONGRATULATION! YOU WIN!")
    if(winner == 2):
        print("COMPUTER WIN THIS GAME! TRY AGAIN!")

    return winner

video_capture = cv2.VideoCapture(0)
gameState=[[0,0,0],[0,0,0],[0,0,0]]

while True:
    ret, frame = video_capture.read()
    gray, edges = getEdges(frame)
    #cv2.imshow('Edges', edges)

    contours = getContours(edges,100)

    if (contours):
        centroids = [getCentroid(contour[0]) for contour in contours]

    lines = cv2.HoughLines(edges,1,10*np.pi/180,250)
    if (lines is not None):
        lines = lines[0]
    points = getEndPointsOfLines(lines)

    # draw contours
    #cv2.drawContours(frame, contours, -1, (0,255,0), 2)

    # draw lines
    #for i in points:
    #    cv2.line(frame,(i[0], i[1]),(i[2],i[3]),(0,255,0),1)

    intersectionPoints = getIntersectionPoints(points)

    # draw intersectionPoints
    #print("Number of intersections: ", len(intersectionPoints))


    for i in intersectionPoints:
        cv2.circle(frame,(int(i[0]),int(i[1])), 5, (0,0,255), -1)
        
    #draw detected circles
    circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
    if (circles is not None):
        circles = circles[0]
        #print("Number of circles: ", len(circles))
        circles = np.uint16(np.around(circles))
        for i in circles:
            # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)

    if(len(intersectionPoints) == 4):
        if(circles is not None):
            tempGameState = checkGameState(intersectionPoints, circles, gameState)
            state = checkState(gameState, tempGameState)
            if(state == -1):
                state = False
                drawCurrentSituation(intersectionPoints, gameState)
                cv2.imshow('Video',frame)
            
            if(state):
                if(checkWinner(gameState) != -1):
                    if cv2.waitKey(1) & 0xFF == 32: #'space' restart
                        gameState=[[0,0,0],[0,0,0],[0,0,0]]
                    if cv2.waitKey(1) & 0xFF == 113: #'q' quit
                        break;
            
                
                print("Player")
                print("-------------")
                print(tempGameState[0])
                print(tempGameState[1])
                print(tempGameState[2])
                print("-------------")
                
                drawCurrentSituation(intersectionPoints, gameState)
                cv2.imshow('Video',frame)
                
                while True:
                    if cv2.waitKey(1) & 0xFF == 32: #'space' accept
                        newGameState = aiMakeDecision(tempGameState)
                        gameState = newGameState
                        print("Computer")
                        print("-------------")
                        print(gameState[0])
                        print(gameState[1])
                        print(gameState[2])
                        print("-------------")
                        break
                    if cv2.waitKey(1) & 0xFF == 114: #'r' reject
                        break

    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF == 113: #'q' quit
        break

video_capture.release()
cv2.destroyAllWindows()