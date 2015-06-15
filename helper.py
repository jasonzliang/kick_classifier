#!/usr/bin/python
'''
Created on Jul 28, 2014

@author: jason
'''
import math
import numpy as np
import glob
import os

def angle2Turn(ball, goal, oppAngle):
  ballDir = (goal[0] - ball[0], goal[1] - ball[1])
  ballAngle = math.atan2(ballDir[1], ballDir[0])
  a = normalizeAngle(ballAngle) - normalizeAngle(oppAngle)
  return abs(normalizeAngle(a))
  
def distToKickPosition(ball, goal, opp):
  ballDir = (ball[0] - goal[0], ball[1] - goal[1])
  ballAngle = normalizeAngle(math.atan2(ballDir[1], ballDir[0]))
  kickPos = (ball[0] + 0.5*math.cos(ballAngle), ball[1] + 0.5*math.sin(ballAngle))
  return getDistTo(kickPos, opp)

def closestDist2Line(start, end, pt): # x3,y3 is the point
    x1 = start[0]; y1 = start[1]
    x2 = end[0]; y2 = end[1]
    x3 = pt[0]; y3 = pt[1]
    px = x2-x1
    py = y2-y1
    something = px*px + py*py + 1e-6
    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)
    if u > 1:
      u = 1
    elif u < 0:
      u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    dist = math.sqrt(dx*dx + dy*dy)
    return dist
  
def getDist2KickLine(kickTarget, ball, opp):
  kickDir = (kickTarget[0] - ball[0], kickTarget[1] - ball[1])
  kickAngle = math.atan2(kickDir[1], kickDir[0])
  start = ball
  end = (ball[0] + math.cos(kickAngle), ball[1] + math.sin(kickAngle))
  return closestDist2Line(start, end, opp)

def getDistTo(p1, p2):
  return norm(p1[0]-p2[0], p1[1]-p2[1])

def norm(x,y):
  return math.sqrt(x*x + y*y)

def norm2(pt):
  return math.sqrt(pt[0]*pt[0] + pt[1]*pt[1])

def removeFallenOpponents(oppArray):
  newOppArray = []
  for opp in oppArray:
    if opp[3] == 0:
      newOppArray.append(opp)
  return newOppArray

def findClosestOpponentToBall(oppArray, ball):
  closestDist=1e308
  closestOpp=None
  
  for i,opp in enumerate(oppArray):
    dist = getDistTo(opp, ball)
    if dist < closestDist:
      closestDist = dist
      closestOpp = i
  return closestDist, closestOpp

def findNClosestOpponentToBall(oppArray, ball, n=3):
  if n > len(oppArray):
    n = len(oppArray)  
  distArray = np.zeros(len(oppArray))
  oppIndices = np.arange(len(oppArray))
  for i,opp in enumerate(oppArray):
    distArray[i] = getDistTo(opp, ball)
  indices = np.argsort(distArray)
  distArray = distArray[indices]; oppIndices = oppIndices[indices]
  return distArray[:n], oppIndices[:n], n

def oppCloserThanMe(oppArray, me, ball):
  counter = 0
  meDist = getDistTo(me, ball)
  for i,opp in enumerate(oppArray):
    oppDist = getDistTo(opp, ball)
    if oppDist < meDist:
      counter += 1 
  return counter

def oppCloserThanDist(oppArray, ball, dist=1.5):
  counter = 0
  for i,opp in enumerate(oppArray):
    oppDist = getDistTo(opp, ball)
    if oppDist < dist:
      counter += 1 
  return counter

def oppWithinKickDir(oppArray, kickTarget, ball, angleThres=30):
  angleThres = math.radians(angleThres)
  counter = 0
  for i,opp in enumerate(oppArray):
    angle = angBetween3Pts(ball, kickTarget, opp)
    if angle < angleThres:
      counter += 1 
  return counter

def isOpponentCloserThanMe(me, opp, ball):
  return int(getDistTo(me, ball) > getDistTo(opp, ball))

def isOpponentWithinKickDir(opp, target, ball, angleThres=0.5):
  return int(angBetween3Pts(ball, target, opp) < angleThres)

def isOpponentBehindBall(me, opp, ball):
  if angBetween3Pts(me, opp, ball) < 0.5:
    if getDistTo(opp, me) > getDistTo(opp, ball):
      return 2
    else:
      return 1
  else:
    return 0

def normalizeAngle(angle):
  while angle > math.pi:
    angle -= 2*math.pi
  while angle < -math.pi:
    angle += 2*math.pi
  return angle

def angBetween3Pts(pt1, pt2, pt3):
  v1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
  v2 = (pt1[0] - pt3[0], pt1[1] - pt3[1])
  dotProduct = v1[0]*v2[0] + v1[1]*v2[1]
  try:
    angle = abs(normalizeAngle(math.acos(dotProduct/(norm2(v1)*norm2(v2)))))
  except:
    angle = 0.0
  return angle

def findTargetAngleandDist(pose, target):
  targetDirection = (target[0] - pose[0], target[1] - pose[1])
  distance = norm2(targetDirection)
  targetDirection = (targetDirection[0]/distance, targetDirection[1]/distance)
  myDirection = (math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2])))
#   print myDirection, targetDirection
  angle = angBetween3Pts((0.,0.), targetDirection, myDirection)
  return angle, distance

def mergeFolders(f1="new_data",f2="old_data",f3="."):
  for f1_name in glob.glob(f1+"/*.txt"):
    f2_name = os.path.join(f2, f1_name.split("/")[1])
    f3_name = os.path.join(f3, f1_name.split("/")[1])
    os.system("cat " + f1_name + " " + f2_name + " > " + f3_name)

if __name__ == '__main__':
  mergeFolders()