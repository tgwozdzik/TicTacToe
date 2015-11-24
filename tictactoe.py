#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import cv2.cv as cv
import numpy as np
import random
from skimage import data, io, filters, morphology, feature, measure
from matplotlib import pyplot as plt
from math import hypot


def getEdges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
    edges = morphology.dilation(edges, morphology.disk(4))
    
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


def computePoints(intersectionPoints):
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

    return point1, point2, point3, point4


def checkGameState(intersectionPoints, circles, gameState):
    game = [[0,0,0],[0,0,0],[0,0,0]]
    
    for i in [0,1,2]:
        for j in [0,1,2]:
            game[i][j] = gameState[i][j]
                
    point1, point2, point3, point4 = computePoints(intersectionPoints)
    
    positionX = -1
    positionY = -1
    
    for i in circles:
        positionX = checkX(point1, point2, i)
        positionY = checkY(point1, point3, i)
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


def checkPlayerState(actualGameState, gameState, circles, playerCircles):
    isCorrect = False
    numberOfPlayerNewMoves = 0

    for i in [0,1,2]:
        for j in [0,1,2]:
            if(gameState[i][j] != actualGameState[i][j]):
                if(gameState[i][j] == 2):
                    print("Incorrect move! Computer already placed here his 'X'. Please change your move!")
                    return isCorrect
                numberOfPlayerNewMoves += 1

    if(numberOfPlayerNewMoves == 0):
        return isCorrect

    if((numberOfPlayerNewMoves == 1) and (playerCircles == circles - 1)):
        isCorrect = True
    else:
        print("Incorrect move! You placed 2 or more 'O's. Please change your move!")

    return isCorrect

def drawCurrentSituation(intersectionPoints, gameState, frame):
    point1, point2, point3, point4 = computePoints(intersectionPoints)
    pointx21 = point2[0]-point1[0]
    pointy13 = point1[1]-point3[1]
    
    cv2.line(frame, (point1[0], point1[1]), (point2[0],point2[1]), (0,255,0), 2)
    cv2.line(frame, (point1[0], point1[1]), (point3[0],point3[1]), (0,255,0), 2)
    cv2.line(frame, (point3[0], point3[1]), (point4[0],point4[1]), (0,255,0), 2)
    cv2.line(frame, (point2[0], point2[1]), (point4[0],point4[1]), (0,255,0), 2)
                
    cv2.line(frame, (point1[0], point1[1]), (point1[0],point1[1]+pointy13), (0,255,0), 2)
    cv2.line(frame, (point1[0], point1[1]), (point1[0]-pointx21,point1[1]), (0,255,0), 2)
                
    cv2.line(frame, (point2[0], point2[1]), (point2[0],point2[1]+pointy13), (0,255,0), 2)
    cv2.line(frame, (point2[0], point2[1]), (point2[0]+pointx21,point2[1]), (0,255,0), 2)
                
    cv2.line(frame, (point3[0], point3[1]), (point3[0],point3[1]-pointy13), (0,255,0), 2)
    cv2.line(frame, (point3[0], point3[1]), (point3[0]-pointx21,point3[1]), (0,255,0), 2)
                
    cv2.line(frame, (point4[0], point4[1]), (point4[0],point4[1]-pointy13), (0,255,0), 2)
    cv2.line(frame, (point4[0], point4[1]), (point4[0]+pointx21,point4[1]), (0,255,0), 2)

    #bottom left
    if(gameState[2][0] ==2):
        cv2.line(frame, (point1[0] - 20, point1[1] + 20), (point1[0] - pointx21 + 20, point1[1] +pointy13- 20), (0,255,0), 2)
        cv2.line(frame, (point1[0] - pointx21 + 20, point1[1] + 20), (point1[0] - 20,point1[1]+pointy13- 20), (0,255,0), 2)
            
    #bottom middle
    if(gameState[2][1] ==2):
        cv2.line(frame, (point1[0] + 20, point1[1] + 20), (point1[0] + pointx21 - 20, point1[1]+pointy13 - 20), (0,255,0), 2)
        cv2.line(frame, (point1[0] + 20, point1[1] + pointy13 - 20), (point1[0] + pointx21 - 20, point1[1] + 20), (0,255,0), 2)
                
    #bottom right
    if(gameState[2][2] ==2):
        cv2.line(frame, (point2[0] + 20, point2[1] + 20), (point2[0]+pointx21 - 20,point1[1]+pointy13- 20), (0,255,0), 2)
        cv2.line(frame, (point2[0] + pointx21 - 20, point2[1] + 20), (point2[0] + 20,point2[1]+pointy13- 20), (0,255,0), 2)
                
    #middle left
    if(gameState[1][0] ==2):
        cv2.line(frame, (point1[0] - 20, point1[1] - 20), (point1[0]-pointx21 + 20,point1[1]-pointy13+ 20), (0,255,0), 2)
        cv2.line(frame, (point1[0] - pointx21 + 20, point1[1] - 20), (point1[0] - 20,point1[1]-pointy13+ 20), (0,255,0), 2)
                
    #middle
    if(gameState[1][1] ==2):
        cv2.line(frame, (point1[0] + 20, point1[1] - 20), (point1[0]+pointx21 - 20,point1[1]-pointy13 + 20), (0,255,0), 2)
        cv2.line(frame, (point1[0] + pointx21 - 20, point1[1] - 20), (point1[0] + 20,point1[1]-pointy13 + 20), (0,255,0), 2)
                
    #middle right
    if(gameState[1][2] ==2):
        cv2.line(frame, (point2[0] + 20, point2[1] - pointy13 + 20), (point2[0] + pointx21 - 20,point1[1] - 20), (0,255,0), 2)
        cv2.line(frame, (point2[0] + 20, point2[1] - 20), (point2[0] + pointx21 - 20, point2[1] - pointy13 + 20), (0,255,0), 2)
                
    #top left
    if(gameState[0][0] ==2):
        cv2.line(frame, (point3[0] - 20, point3[1] - 20), (point3[0]-pointx21 + 20,point3[1]-pointy13 + 20), (0,255,0), 2)
        cv2.line(frame, (point3[0] - pointx21 + 20, point3[1] - 20), (point3[0] - 20, point3[1]-pointy13 + 20), (0,255,0), 2)
                
    #top middle
    if(gameState[0][1] ==2):
        cv2.line(frame, (point3[0] + 20, point3[1] - 20), (point3[0] + pointx21 - 20, point3[1] - pointy13 + 20), (0,255,0), 2)
        cv2.line(frame, (point3[0] + 20, point3[1] - pointy13 + 20), (point3[0] + pointx21 - 20, point3[1] - 20), (0,255,0), 2)
                
    #top right
    if(gameState[0][2] ==2):
        cv2.line(frame, (point4[0] + 20, point4[1] - 20), (point4[0] + pointx21 - 20, point4[1] - pointy13 + 20), (0,255,0), 2)
        cv2.line(frame, (point4[0] + 20, point4[1] - pointy13 + 20), (point4[0] + pointx21 - 20, point4[1] - 20), (0,255,0), 2)


def checkWinner(gameState, frame, intersectionPoints, playerCircles):
    if(playerCircles == 5):
        print("DRAW! TRY AGAIN!")
        drawCurrentSituation(intersectionPoints, gameState, frame)
        cv2.putText(frame,"DRAW!", (300,500), cv2.FONT_HERSHEY_SIMPLEX, 5, 255)
        cv2.imshow('Video',frame)
    
        return 3
    
    winner = -1
    point1, point2, point3, point4 = computePoints(intersectionPoints)
    pointx21 = point2[0]-point1[0]
    pointy13 = point1[1]-point3[1]
    
    if(gameState[0][0] == gameState[0][1] == gameState[0][2] and gameState[0][0] != 0):
        winner = gameState[0][0]
        cv2.line(frame, (point1[0]-pointx21, point1[1]-int(1.5*pointy13)), (point2[0]+pointx21,point2[1]-int(1.5*pointy13)), (0,255,0), 2)

    if(gameState[1][0] == gameState[1][1] == gameState[1][2] and gameState[1][0] != 0):
        winner = gameState[1][0]
        cv2.line(frame, (point1[0]-pointx21, point1[1]-int(0.5*pointy13)), (point2[0]+pointx21,point2[1]-int(0.5*pointy13)), (0,255,0), 2)

    if(gameState[2][0] == gameState[2][1] == gameState[2][2] and gameState[2][0] != 0):
        winner = gameState[2][0]
        cv2.line(frame, (point1[0]-pointx21, point1[1]+int(0.5*pointy13)), (point2[0]+pointx21,point2[1]+int(0.5*pointy13)), (0,255,0), 2)

    if(gameState[0][0] == gameState[1][0] == gameState[2][0] and gameState[0][0] != 0):
        winner = gameState[0][0]
        cv2.line(frame, (point1[0]-int(0.5*pointx21), point1[1]+int(0.5*pointy13)), (point3[0]-int(0.5*pointx21),point3[1]-int(0.5*pointy13)), (0,255,0), 2)

    if(gameState[0][1] == gameState[1][1] == gameState[2][1] and gameState[0][1] != 0):
        winner = gameState[0][1]
        cv2.line(frame, (point1[0]+int(0.5*pointx21), point1[1]+int(0.5*pointy13)), (point3[0]+int(0.5*pointx21),point3[1]-int(0.5*pointy13)), (0,255,0), 2)

    if(gameState[0][2] == gameState[1][2] == gameState[2][2] and gameState[0][2] != 0):
        winner = gameState[0][2]
        cv2.line(frame, (point1[0]+int(1.5*pointx21), point1[1]+int(0.5*pointy13)), (point3[0]+int(1.5*pointx21),point3[1]-int(0.5*pointy13)), (0,255,0), 2)

    if(gameState[0][0] == gameState[1][1] == gameState[2][2] and gameState[0][0] != 0):
        winner = gameState[0][0]
        cv2.line(frame, (point3[0]-int(0.5*pointx21), point3[1]-int(0.5*pointy13)), (point2[0]+int(0.5*pointx21),point2[1]+int(0.5*pointy13)), (0,255,0), 2)

    if(gameState[2][0] == gameState[1][1] == gameState[0][2] and gameState[2][0] != 0):
        winner = gameState[2][0]
        cv2.line(frame, (point1[0]-int(0.5*pointx21), point1[1]+int(0.5*pointy13)), (point4[0]+int(0.5*pointx21),point4[1]-int(0.5*pointy13)), (0,255,0), 2)

    if(winner == 1):
        print("CONGRATULATION! YOU WIN!")
        drawCurrentSituation(intersectionPoints, gameState, frame)
        cv2.putText(frame,"PLAYER WIN!", (15,150), cv2.FONT_HERSHEY_SIMPLEX, 3, 100)
        cv2.imshow('Video',frame)

    if(winner == 2):
        print("COMPUTER WIN THIS GAME! TRY AGAIN!")
        drawCurrentSituation(intersectionPoints, gameState, frame)
        cv2.putText(frame,"COMPUTER WIN!", (30,150), cv2.FONT_HERSHEY_SIMPLEX, 3, 100)
        cv2.imshow('Video',frame)

    return winner


def consoleStatus(player, state):
    print(player)
    print("-------------")
    print(state[0])
    print(state[1])
    print(state[2])
    print("-------------")


def isWinner(actualGameState, frame, intersectionPoints, playerCircles):
    if(checkWinner(actualGameState, frame, intersectionPoints, playerCircles) != -1):
        return True

    return False


def detectContoursAndLines(edges, frame):
    #contours = getContours(edges,100)
    
    #if (contours):
    #    centroids = [getCentroid(contour[0]) for contour in contours]
    
    lines = cv2.HoughLines(edges,1,5*np.pi/180,250)
    if (lines is not None):
        lines = lines[0]

    points = getEndPointsOfLines(lines)
    intersectionPoints = getIntersectionPoints(points)

    for i in intersectionPoints:
        cv2.circle(frame,(int(i[0]),int(i[1])), 5, (0,0,255), -1)

    return intersectionPoints


def detectCircles(gray, frame):
    circles = cv2.HoughCircles(gray, cv.CV_HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
    if (circles is not None):
        circles = np.uint16(np.around(circles[0]))
        for i in circles:
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)

    return circles


def changeGameState(intersectionPoints, frame, circles, gameState, playerCircles):
    actualGameState = checkGameState(intersectionPoints, circles, gameState)
    
    if(not checkPlayerState(actualGameState, gameState, len(circles), playerCircles)):
        drawCurrentSituation(intersectionPoints, gameState, frame)
        cv2.imshow('Video',frame)
        return False, gameState, playerCircles
    
    else:
        if(not isWinner(actualGameState, frame, intersectionPoints, playerCircles)):
            consoleStatus("Player", actualGameState)
            drawCurrentSituation(intersectionPoints, gameState, frame)
            cv2.imshow('Video',frame)
        
            while(True):
                #if cv2.waitKey(1) & 0xFF == 32: #'space'
                playerCircles += 1
                newGameState = aiMakeDecision(actualGameState)
            	gameState = newGameState
            	consoleStatus("Computer", gameState)
            	if(not isWinner(actualGameState, frame, intersectionPoints, playerCircles)):
                	return False, newGameState, playerCircles
            	else:
                	return True, newGameState, playerCircles
                
                if cv2.waitKey(1) & 0xFF == 114: #'r'
                    return False, gameState, playerCircles
        else:
            return True, gameState, playerCircles


def main():
    gameState=[[0,0,0],[0,0,0],[0,0,0]]
    playerCircles = 0
    video_capture = cv2.VideoCapture(0)

    endGame = False
    while(not endGame):
        ret, frame = video_capture.read()

        gray, edges = getEdges(frame)
        
        intersectionPoints = detectContoursAndLines(edges, frame)
        circles = detectCircles(gray, frame)
        
        isWinner = False
        if((len(intersectionPoints) == 4) and (circles is not None)):
            isWinner, newGameState, newPlayerCircles = changeGameState(intersectionPoints, frame, circles, gameState, playerCircles)
            gameState = newGameState
            playerCircles = newPlayerCircles
            
        if(isWinner):
            while(True):
                if cv2.waitKey(1) & 0xFF == 32: #'space'
                    gameState=[[0,0,0],[0,0,0],[0,0,0]]
                    playerCircles = 0
                    break

                if cv2.waitKey(1) & 0xFF == 114: #'r'
                    endGame = True
        
        cv2.imshow('Video',frame)

        if cv2.waitKey(1) & 0xFF == 113: #'q'
            endGame = True
    
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
