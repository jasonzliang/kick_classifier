#!/usr/bin/python
'''
Created on Sep 25, 2014

@author: jason
'''
import numpy as np
import glob
import os, sys
from scipy.stats import ttest_ind, ttest_1samp
from scipy.stats import sem
from scipy.stats import norm
from sets import Set
import matplotlib.pyplot as plt

def listofmodels(directory):
  listOfFiles = glob.glob(directory + "/*")
  models = Set()
  for fileName in listOfFiles:
    fileName = fileName.split("/")[-1]
    model = fileName[:-4].split("_")[-1]
    models.add(model)
  models = list(models)
  print "models:", models 
  return models

def summarizeGameData(directory="6_30_type4_apollo3d_test/results", enemyTeam="apollo3d"):
  if not os.path.exists(directory):
    print direction + " not found!"
    print "Are you sure directory exists?"
    sys.exit(-1)

  def getScoreInformation(scoreFiles):
    processed = {}
    goalDiff = []
    average_ball_pos = []
    kicks = []
    myGoalsTotal = 0.0
    theirGoalsTotal = 0.0
    
    for scoreFile in scoreFiles:
      if "left" in scoreFile:
        scoreFile2 = scoreFile.replace("left", "right")
      else:
        scoreFile2 = scoreFile.replace("right", "left")
      if not os.path.exists(scoreFile2) or scoreFile in processed or scoreFile2 in processed:
        continue
      
      myScore = 0.0
      theirScore = 0.0
      goalPos = 0.0
      __kicks = 0.0
      for __scoreFile in [scoreFile, scoreFile2]:
        f = open(__scoreFile)
        for line in f:
          if "score" in line:
            myScore += float(line.rstrip().split()[2])
            theirScore += float(line.rstrip().split()[3])
          elif "average_ball_posX" in line:
            goalPos += float(line.rstrip().split()[2])
          elif "kicks" in line:
            __kicks += float(line.rstrip().split()[2])
        f.close()
      
      myGoalsTotal += myScore
      theirGoalsTotal += theirScore
      average_ball_pos.append(goalPos/2.0)
      goalDiff.append(myScore - theirScore)
      kicks.append(__kicks)
      processed[scoreFile] = True
      processed[scoreFile2] = True
            
    goalDiff = np.array(goalDiff)
    average_ball_pos = np.array(average_ball_pos)
    return goalDiff, average_ball_pos, myGoalsTotal, theirGoalsTotal, kicks
  
  listofresults = []
  
  for modelName in listofmodels(directory):
#     print "parsing " + modelName
    scoreFiles = glob.glob(os.path.join(directory, "*" + modelName + ".txt"))
    goalDiff, avgballpos, myGoalsTotal, theirGoalsTotal, kicks = getScoreInformation(scoreFiles)
    avgGoalDiff = np.mean(goalDiff)
    zScore = (1.0 - avgGoalDiff)/np.std(goalDiff)
    pValue = norm.cdf(zScore)
    listofresults.append((avgGoalDiff, zScore, pValue, modelName, goalDiff, 
                          avgballpos, myGoalsTotal, theirGoalsTotal, kicks))
  
  def sort_key(t):
    return t[2]
  
  sortedResults = sorted(listofresults, reverse=False, key=sort_key)
  
  avgGoalDiffs = []; stderrs = []; pValues = []; meanKickList = []; kickStdErrList = []
  averageOppGoal = []
  modelNames = [x[3] for x in sortedResults]
  for avgGoalDiff, zScore, pValue, modelName, goalDiff, avgballpos, \
                myGoalsTotal, theirGoalsTotal, kicks in sortedResults:
    n = len(goalDiff)
    if n == 0:
      continue
    stderr = sem(goalDiff)
    avgKicks = np.mean(kicks)
    meanKickList.append(avgKicks)
    kickStdErr = sem(kicks)
    kickStdErrList.append(kickStdErr)
    avgGoalDiffs.append(avgGoalDiff)
    stderrs.append(stderr)
    pValues.append(pValue)
    averageOppGoal.append(theirGoalsTotal/n)
    
    print "model: " + modelName
    print "number games: " + str(n)
    print "wins/loss/ties: " + str(np.sum(goalDiff > 0)) + "/" + str(np.sum(goalDiff < 0)) + "/" + \
          str(np.sum(goalDiff == 0))
    print "average goal difference (us vs " + enemyTeam + "): " + str(round(avgGoalDiff,3)) + \
          " stderr: " + str(round(stderr,3))
    print "z-score/p-value (goal difference < 1): " + str(round(zScore,3)) + "/" + \
          str(round(pValue,3))
    print "total goals (us vs " + enemyTeam + "): " + str(int(myGoalsTotal)) + "/" + str(int(theirGoalsTotal))
    print "average goals (us vs " + enemyTeam + "): " + str(round(myGoalsTotal/n,3)) + "/" + \
           str(round(theirGoalsTotal/n,3))
    print "average kicks: " + str(round(avgKicks,3)) + " stderr: " + str(round(kickStdErr,3))
#     print "average ball posX (us vs " + enemyTeam + "): " + str(round(np.mean(avgballpos),3)) + \
#           " stderr: " + str(round(sem(avgballpos),3))
    print

  def autolabel(rects, ax, offset, color):
    # attach some text labels
    for rect in rects:
      height = rect.get_height()
      ax.text(rect.get_x()+rect.get_width()/2., offset + height, round(height, 3),
              ha='center', va='bottom', size=10, backgroundcolor="black", weight="bold", alpha=1.0, color=color)
  
  ind = np.arange(len(modelNames))
  width = 0.23
  fig, ax = plt.subplots()
  plt.title("Comparison of kick classifier performance against " + enemyTeam)
  fig.set_size_inches(int(len(modelNames)*1.5),8)
  ax.set_xticks(ind+width*2)
  ax.set_xticklabels(modelNames)
  plt.setp(ax.get_xticklabels(), fontsize=10)

  rects1 = ax.bar(ind, avgGoalDiffs, width, color='g', yerr=stderrs, ecolor='r')
  ax.set_ylabel('Average Goal Difference', color='g')
  for tl in ax.get_yticklabels():
    tl.set_color('g')
  ax2 = ax.twinx()
  rects2 = ax2.bar(ind+width, pValues, width, color='b')
  ax2.set_ylabel('Prob of Tie/Loss', color='b')
  for tl in ax2.get_yticklabels():
    tl.set_color('b')  
  ax3 = ax.twinx()
  rects3 = ax3.bar(ind+width*2, meanKickList, width, yerr=kickStdErrList, color='y')
  autolabel(rects3, ax3, np.mean(meanKickList)/10, 'y')
  ax3.yaxis.set_visible(False)
  ax4 = ax.twinx()
  rects4 = ax4.bar(ind+width*3, averageOppGoal, width, color='c')
  autolabel(rects4, ax4, np.mean(averageOppGoal)/20, 'c')
  ax4.set_ylim(bottom=0)
  ax4.yaxis.set_visible(False)

  ax.grid()
  ax.set_axisbelow(True)

  plt.savefig(directory.split("/")[0] + "_" + enemyTeam + "_results.png", bbox_inches='tight')
  
  
if __name__ == '__main__':
#   if len(sys.argv) != 2:
#     print "usage: ./summarizeResults.py {directory name}"
#     sys.exit(-1)
#   else:
    summarizeGameData()