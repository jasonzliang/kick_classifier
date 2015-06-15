#!/usr/bin/python

# Script to compute average and standard error of results data

import os
import sys
import math

def isLeft(x): return x.find("_left") != -1
def isRight(x): return x.find("_right") != -1

def stderr(x):
  if len(x) < 2:
    return float('nan')
  mean_x = float(sum(x))/len(x)
  var_x = 0.0
  for e in x:
    var_x = var_x + (e-mean_x)*(e-mean_x)
  var_x = var_x/(len(x)-1)
  stddev_x = math.sqrt(var_x)
  return stddev_x/math.sqrt(len(x))


me = []
me_left = []
me_right = []
opp = []
opp_left = []
opp_right = []
gd = []
gamed = []
gameme = []
gameopp = []
gamexball = []
xBallSum = 0.0
meDict = {}
oppDict = {}
scoresDict = {}
xBallDict = {}
win = 0
loss = 0
tie = 0
initialKickOffsScored = 0
initialKickOffsMissed = 0
otherKickOffsScored = 0
otherKickOffsMissed = 0

for arg in sys.argv[1:]:
  f = open(arg)
  lines = f.readlines()
  tokens = lines[0].split()

  me_goals = int(tokens[2])
  opp_goals = int(tokens[3])
  
  me.append(me_goals)
  opp.append(opp_goals)
  if isLeft(arg):
    me_left.append(me_goals)
    opp_left.append(opp_goals)
  if isRight(arg):
    me_right.append(me_goals)
    opp_right.append(opp_goals)

  tokens = lines[1].split()
  xBall = float(tokens[2])
  
  tokens = lines[2].split()
  initialKickOffScored = int(tokens[2])
  initialKickOffMissed = int(tokens[3])

  tokens = lines[3].split()
  otherKickOffScored = int(tokens[2])
  otherKickOffMissed = int(tokens[3])
  
  diff = me_goals - opp_goals
  gameNum = os.path.basename(arg).split("_")[1]
  if gameNum in scoresDict:
    gameDiff = diff + scoresDict[gameNum]
    if gameDiff > 0:
      win = win+1
    elif gameDiff < 0:
      loss = loss + 1
    else:
      tie = tie + 1
    gamed.append(gameDiff)
    gameme.append(me_goals + meDict[gameNum])
    gameopp.append(opp_goals + oppDict[gameNum])
    gamexball.append((xBall + xBallDict[gameNum])/2.0)
  else:
    scoresDict[gameNum] = diff
    meDict[gameNum] = me_goals
    oppDict[gameNum] = opp_goals
    xBallDict[gameNum] = xBall
  
  gd.append(diff)
  xBallSum = xBallSum + xBall
  initialKickOffsScored = initialKickOffsScored + initialKickOffScored
  initialKickOffsMissed = initialKickOffsMissed + initialKickOffMissed
  otherKickOffsScored = otherKickOffsScored + otherKickOffScored
  otherKickOffsMissed = otherKickOffsMissed + otherKickOffMissed

  f.close()


mean_me = float(sum(me))/len(me)
mean_opp = float(sum(opp))/len(opp)
mean_gd = float(sum(gd))/len(gd)

mean_gameme = float('nan')
mean_gameopp = float('nan')
mean_gamed = float('nan')
mean_gamexball = float('nan')
if len(gameme) > 0:
  mean_gameme = float(sum(gameme))/len(gameme)
if len(gameopp) > 0:
  mean_gameopp = float(sum(gameopp))/len(gameopp)
if len(gamed) > 0:
  mean_gamed = float(sum(gamed))/len(gamed)
if len(gamexball) > 0:
  mean_gamexball = float(sum(gamexball))/len(gamexball)

perc_initialKickOffs = float('nan')
perc_otherKickOffs = float('nan')
if initialKickOffsScored + initialKickOffsMissed > 0:
  perc_initialKickOffs = float(initialKickOffsScored) / float(initialKickOffsScored + initialKickOffsMissed) * 100.0
if otherKickOffsScored + otherKickOffsMissed > 0:
  perc_otherKickOffs = float(otherKickOffsScored) / float(otherKickOffsScored + otherKickOffsMissed) * 100.0

'''
var_gd = 0.0
for x in gd:
  var_gd = var_gd + (x-mean_gd)*(x-mean_gd)

var_gd = var_gd/(len(gd)-1)

stddev_gd = math.sqrt(var_gd)

stderr_gd = stddev_gd/len(gd)


me_stddev_sum = 0.0
for x in me:
  me_stddev_sum = me_stddev_sum + x
me_stddev = me_stddev_sum/len(me)


opp_stddev_sum = 0.0
for x in opp:
  opp_stddev_sum = opp_stddev_sum + x
opp_stddev = opp_stddev_sum/len(opp)
'''
mean_gd_weighted = float('nan')
if len(me_left) > 0 and len(me_right) > 0:
  mean_gd_weighted = float(sum(me_left)-sum(opp_left))/float(len(me_left)) + float(sum(me_right)-sum(opp_right))/float(len(me_right))
print "Avg goal diff = " + str(round(2.0*mean_gd, 3)) + " (+/-" + str(round(2.0*stderr(gd), 3)) + "), game = " + str(round(mean_gamed, 3)) + " (+/-" + str(round(stderr(gamed), 3)) + "), weighted = " + str(round(mean_gd_weighted, 3))
print "Avg goals me = " + str(round(2.0*mean_me, 3)) + " (+/-" + str(round(2.0*stderr(me), 3)) + "), game = " + str(round(mean_gameme, 3)) + " (+/-" + str(round(stderr(gameme), 3)) + ")"
print "Avg goals opp = " + str(round(2.0*mean_opp, 3)) + " (+/-" + str(round(2.0*stderr(opp), 3)) + "), game = " + str(round(mean_gameopp, 3)) + " (+/-" + str(round(stderr(gameopp), 3)) + ")"
print "Avg xBall = " + str(round(xBallSum/float(len(sys.argv)-1), 3)) + ", game = " + str(round(mean_gamexball, 3))
print "Goals (me / opp) = " + str(sum(me)) + " / " + str(sum(opp))
print "Kickoff me goals (me / opp) = " + str(sum(me_left)) + " / " + str(sum(opp_left))
print "Kickoff opp goals (me / opp) = " + str(sum(me_right)) + " / " + str(sum(opp_right))
print "Most goals half (me / opp) = " + str(max(me)) + " / " + str(max(opp))
print "Initial kickoffs (made / missed) = " + str(initialKickOffsScored) + " / " + str(initialKickOffsMissed) + " = " + str(round(perc_initialKickOffs,2)) + "%"
print "Other kickoffs (made / missed) = " + str(otherKickOffsScored) + " / " + str(otherKickOffsMissed) + " = " + str(round(perc_otherKickOffs,2)) + "%"
print "Record (W-L-T) = " + str(win) + "-" + str(loss) + "-" + str(tie)
