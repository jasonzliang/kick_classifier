#!/usr/bin/python
'''
Created on Feb 26, 2014

@author: jason
'''
import numpy as np
np.set_printoptions(suppress=True, formatter={'all':lambda x: str(x) + ','}, linewidth=9999)
import matplotlib.pyplot as plt
import glob, time
import sys
import os
import math
import operator
from helper import *
from scipy.stats import ttest_ind, ttest_rel
from sklearn import svm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest, RFECV, RFE
from smote import *
import subprocess
import random

shouldhavefailed = 0
success = 0

def preprocessGameState(gameState, removeFallenOpp=False):
  x, initTime, x, agentInfo, x, opp, x, me, x, ball, x, kickTarget, x, directFeatures =  gameState.split(":")
  initTime = float(initTime)
  agentType = int(agentInfo.split()[1])
  ball = [float(x) for x in ball.split()]
  me = [float(x) for x in me.split()]
  kickTarget = [float(x) for x in kickTarget.split()]
  directFeatures = directFeatures.split()
  directFeatures = [float(x) for x in directFeatures]
  
  if len(opp) == 0:
    oppArray = []
  else:
    oppArray = [[float(x) for x in temp.split()] for temp in opp.split(';')[:-1]]
  if removeFallenOpp:
    oppArray = removeFallenOpponents(oppArray)  
  return initTime, oppArray, me, ball, kickTarget, agentType, directFeatures

# def parseGameState2(gameState, nFeatures, skipNoOpp=True):
#   initTime, oppArray, me, ball, kickTarget, agentType, directFeatures = preprocessGameState(gameState)
#   if len(oppArray) == 0 and skipNoOpp:
#     return np.zeros(nFeatures), False
#   
#   features = np.zeros(nFeatures)
#   oppDists, oppIndices, n = findNClosestOpponentToBall(oppArray, ball, n=2)
#   
#   t=5
#   for i in xrange(n):
#     oppI = oppIndices[i]
#     oppDist = oppDists[i]
#     opp = oppArray[oppI]
#     oppAngle, dist = findTargetAngleandDist(opp, ball)
#     angBetweenOppAndBall = angBetween3Pts(me, ball, opp)
#     oppAngleFromMe, oppDistFromMe = findTargetAngleandDist(me, opp)
#     a = isOpponentCloserThanMe(me, opp, ball)
#     b = isOpponentWithinKickDir(opp, kickTarget, ball)
#     c = isOpponentBehindBall(me, opp, ball)
#     features[t*i:t*(i+1)] = np.array([oppDist, oppAngle, a, b, c])
# 
#   ballAngle, ballDist = findTargetAngleandDist(me, ball)
#   angleofKick = angBetween3Pts(me, ball, kickTarget)
#   features[-3:] = np.array([ballAngle, ballDist, angleofKick])
#   return features, True

def isOpponent2Far(gameState, threshold=2.5, removeFallenOpp=False):
  oppArray, me, ball, kickTarget, agentType, directFeatures = gameState
  closestOppDist, closestOppIndex = findClosestOpponentToBall(oppArray, ball)
  return closestOppDist > threshold 

def parseGameState(gameState, nFeatures, skipNoOpp=True, kickType=[7], removeFallenOpp=False, useDirectFeatures=True):
  initTime, oppArray, me, ball, kickTarget, agentType, directFeatures = gameState
  
  if (len(oppArray) == 0 or kickTarget[2] not in kickType) and skipNoOpp:
    return np.zeros(nFeatures), False
  
  if useDirectFeatures:
    return np.abs(np.array(directFeatures)), True
   
  closestOppDist, closestOppIndex = findClosestOpponentToBall(oppArray, ball)
  oppDist2KickLine = getDist2KickLine(kickTarget, ball, oppArray[closestOppIndex])
  angleBetweenOppandKickDir = angBetween3Pts(ball, kickTarget, oppArray[closestOppIndex])
  numOppCloserThanDist = oppCloserThanDist(oppArray, ball, dist=2.0)
  angBetweenOppAndBall = angBetween3Pts(me, ball, oppArray[closestOppIndex])
  ang2TurnOpp = angle2Turn(ball, (-15.0,0.0), math.radians(oppArray[closestOppIndex][2]))
  angle2TurnMe = angle2Turn(ball, kickTarget, math.radians(me[2]))
  oppBallAngle, oppBallDist = findTargetAngleandDist(oppArray[closestOppIndex], ball)
  angleBetweenMeAndBall = angBetween3Pts(oppArray[closestOppIndex], me, ball)
  oppFallen = oppArray[closestOppIndex][3]
  numOppWithinKickDir = oppWithinKickDir(oppArray, kickTarget, ball, angleThres=30)
  closestOppAngleFromMe, closestOppDistFromMe = findTargetAngleandDist(me, oppArray[closestOppIndex])
  distToKickPosOpp = distToKickPosition(ball, (-15.0,0.0), oppArray[closestOppIndex])
  oppdist, indices, n = findNClosestOpponentToBall(oppArray, ball)
  meanOppDist = np.mean(oppdist)
  maxOppDist = np.max(oppdist)    
  ballAngle, ballDist = findTargetAngleandDist(me, ball)
  angleOfKick = angBetween3Pts(me, ball, kickTarget)
  ballVelocity = norm(ball[2], ball[3])
  
  featureArray = np.array([closestOppDist,oppDist2KickLine,angleBetweenOppandKickDir,numOppCloserThanDist,angBetweenOppAndBall,ang2TurnOpp,
                           angle2TurnMe,oppBallAngle,oppFallen,numOppWithinKickDir, closestOppAngleFromMe, closestOppDistFromMe,
                           distToKickPosOpp, meanOppDist, maxOppDist, ballAngle, ballDist, angleOfKick, ballVelocity, angleBetweenMeAndBall])
  return featureArray, True
 
def parseKickSuccess(gameState, ballMovement, ezKickSuccess=False, ballTravelDist=2):
  initTime, oppArray, me, ball, kickTarget, agentType, directFeatures = gameState
  x, immedAfterKick, x, afterKick, x, kickInfo = ballMovement.split(":")
  kickMode = int(kickInfo.split()[0])
  if (kickMode != 2):
    return -1
  if ezKickSuccess:
    return 1
  afterKick = [float(x) for x in afterKick.split()]
  immedAfterKick = [float(x) for x in immedAfterKick.split()]
#   print afterKick, immedAfterKick
  
  timeElapsed = afterKick[2] - immedAfterKick[4]   
  target = (immedAfterKick[2], immedAfterKick[3])
  ballBefore = (immedAfterKick[0], immedAfterKick[1])
  ballAfter = (afterKick[0], afterKick[1])
  
#   ballTravelDist = getDistTo(ballBefore, ballAfter)
#   ballVel = ballTravelDist/timeElapsed
#   angBetweenBallAndTarget = angBetween3Pts(ballBefore, ballAfter, target)

  ballBeforeDist = getDistTo(ballBefore, target)
  ballAfterDist = getDistTo(ballAfter, target)

  if ballBeforeDist - ballAfterDist > ballTravelDist:
    return 1
  else:
    return -1

def parseKickSuccess_new(gameState, ballMovement, ezKickSuccess=False):
  def getDribbleDist(timeElapsed):
    if timeElapsed < 1.5:
      dist = 0.0
    else:
      dist = 4.0/6.5 * (timeElapsed - 1.5)
      
    return dist
  
  initTime, oppArray, me, ball, kickTarget, agentType, directFeatures = gameState
  x, immedAfterKick, x, afterKick, x, kickInfo = ballMovement.split(":")
  kickMode = int(kickInfo.split()[0])
  if (kickMode != 2):
    return -1
  if ezKickSuccess:
    return 1
  afterKick = [float(x) for x in afterKick.split()]
  immedAfterKick = [float(x) for x in immedAfterKick.split()]
#   print afterKick, immedAfterKick
  
  timeElapsed = afterKick[2] - initTime  
#   target = (immedAfterKick[2],immedAfterKick[3])
  target = (kickTarget[0], kickTarget[1])
  ballBefore = (immedAfterKick[0],immedAfterKick[1])
  ballAfter = (afterKick[0], afterKick[1])

  travelDist = getDistTo(ballBefore, target) - getDistTo(ballAfter, target)
  
  if travelDist > getDribbleDist(timeElapsed):
    return 1
  else:
    return -1

def parse(filename="combined.txt", useCache=False, nfeatures=20, ezKickSuccess=False, 
          kickType=[10], removeFallenOpp=False, ignoreSelfFailure=False, useDirectFeatures=True):
  
  featuresFile = "cache/" + filename[:-4] + str(nfeatures) + "_" + \
  str(ezKickSuccess) + "_" + str(kickType) + "_features.npy"
  
  labelsFile = "cache/" + filename[:-4] + str(nfeatures) + "_" + \
  str(ezKickSuccess) + "_" + str(kickType) + "_labels.npy"
  
  if useCache and os.path.exists(featuresFile) and os.path.exists(labelsFile):
    print "cached files exists, using cache!"
    features = np.load(featuresFile)
    labels = np.load(labelsFile)
    return features, labels
    
  w = open(filename)
  lines = w.readlines()
  features = np.empty((len(lines), nfeatures))
  labels = np.empty(len(lines), dtype=np.int)
  successIndex = []
  for index, line in enumerate(lines):
    if index % 1000 == 0:
      print "parsed " + str(index) + " samples"
    rawGameState, ballMovement = line.rstrip().split("#")
    
    try:
      gameState = preprocessGameState(rawGameState, removeFallenOpp=removeFallenOpp)
      featureArray, success = parseGameState(gameState, nfeatures, kickType=kickType, 
                                             removeFallenOpp=removeFallenOpp, useDirectFeatures=useDirectFeatures)
    except:
      success = False
      
    if success:
      features[index,:] = featureArray
      labels[index] = parseKickSuccess_new(gameState, ballMovement, ezKickSuccess=ezKickSuccess)
      if ignoreSelfFailure:
        if labels[index] == 1 or not isOpponent2Far(gameState, removeFallenOpp=removeFallenOpp):
          successIndex.append(index)
      else:
        successIndex.append(index)
    else:
      pass
#       print "failed: " + line.rstrip()
  
  print "ignored " + str(len(lines) - len(successIndex)) + " lines."
  features = features[successIndex, :]
  labels = labels[successIndex]
#   features = features[labels != -2,:]
#   labels = labels[labels != -2]
  print "saving processed features and labels to file..."
  np.save(featuresFile, features)
  np.save(labelsFile, labels)
  return features, labels

def visualize(useFeatures=[0,8], frac=0.005):
  features, labels = parse(useCache=False)
  features = features[:, useFeatures]
  r = np.random.random(features.shape[0]) < frac
  features2 = features[r,:]; labels2 = labels[r]
  plt.figure(figsize=(16,16))
  plt.xlabel("feature #1")
  plt.ylabel("feature #2")
  cm = plt.cm.get_cmap('cool')
  sc = plt.scatter(x=features2[:,0],y=features2[:,1], s=60, c=labels2, cmap=cm)
  plt.draw()
  plt.savefig('scatter.png', bbox_inches='tight')
  plt.clf()
  
def batch_evalAccuracy(filenames=['type0_apollo3d.txt', 'type0_fc.txt','type1_apollo3d.txt', 
                     'type1_fc.txt','type2_apollo3d.txt', 'type2_fc.txt',
                     'type3_apollo3d.txt', 'type3_fc.txt','type4_apollo3d.txt', 'type4_fc.txt'], 
           useFeatures=[4, 6, 8, 11, 0, 15, 17,],kickType=[10]):
  accuracys = []
  for i, filename in enumerate(filenames):
    try:
      accuracyRate = classify(filename=filename, useFeatures=useFeatures, equalClassSize=False, 
                              useAll=False, batch=True, useCache=True, kickType=kickType, draw=False)
      accuracys.append(round(accuracyRate,3))
    except:
      accuracys.append(0.0)
    
  for i, filename in enumerate(filenames):
    print filename[:-4] + ":", accuracys[i]
    
def balanceClasses(newFeatures, labels):
    pos = newFeatures[labels == 1]; pos_labels = labels[labels == 1]
    neg = newFeatures[labels == -1]; neg_labels = labels[labels == -1]
    r = np.random.random(neg.shape[0]) < float(pos.shape[0])/neg.shape[0]
    neg = neg[r,:]; neg_labels = neg_labels[r]
    newFeatures = np.concatenate((neg, pos))
    labels = np.concatenate((neg_labels, pos_labels))
    return newFeatures, labels

def smote(newFeatures, labels):
  percentage = 100*(float(len(labels) - np.sum(labels))/np.sum(labels))
  percentage = int(round(percentage, -2))
  safe, synthetic, danger = borderlineSMOTE(newFeatures, labels, 1, percentage, 5)
  numNewFeatures = synthetic.shape[0]
  
  newFeatures = np.concatenate((newFeatures, synthetic))
  labels = np.concatenate((labels, np.ones(numNewFeatures)))
  return newFeatures, labels
  
#0:closestOppDist,1:oppDist2KickLine,2:angleBetweenOppandKickDir,3:numOppCloserThanDist
#4:angBetweenOppAndBall,5:ang2TurnOpp,6:angle2TurnMe,7:oppBallAngle
#8:oppFallen,9:numOppWithinKickDir,10:closestOppAngleFromMe,11:closestOppDistFromMe
#12:distToKickPosOpp,13:meanOppDist,14:maxOppDist,15:ballAngle
#16:ballDist,17:angleOfKick,18:ballVelocity,19:angleBetweenMeAndBall
def classify(filename='4_14_type4_apollo3d.txt', useFrac=1.0, trainFraction=0.5, equalClassSize=True, 
             thres=0.5, useFeatures=[0], useAll=True, batch=False, useCache=True, 
             featureSelect=False, kickType=[11], draw=False, scale=False, C=1.0, B=1.0, returnProb=False): 
  
  features, labels = parse(filename=filename, useCache=useCache, ezKickSuccess=False, 
                           kickType=kickType, ignoreSelfFailure=False, useDirectFeatures=True, 
                           nfeatures=8)
  num2Use = int(useFrac*len(features))
  features = features[:num2Use]; labels = labels[:num2Use]
  if scale:
    features = StandardScaler().fit_transform(features)
  print "features mean:", features.mean(axis=0)
  print "features std:", features.std(axis=0)
  if not useAll:
#     labels = np.random.random(features.shape[0]) < 0.5
    newFeatures = features[:, useFeatures]
#     print newFeatures[:100,:]
#     newFeatures = np.random.random((features.shape[0], 9))
  else:
    newFeatures = features
  
  if equalClassSize:
    newFeatures, labels = balanceClasses(newFeatures, labels)
  
  print "we have " + str(newFeatures.shape[0]) + " samples."
  print "we have " + str(np.sum(labels == 1)) + " positive labels"
  print "ratio: " + str(float(np.sum(labels == -1))/np.sum(labels == 1))
  print "using approximately " + str(trainFraction*100) + "% as training examples"
  
  r = np.random.random(newFeatures.shape[0]) < trainFraction; r2 = np.invert(r)
  trainingSet = newFeatures[r, :]; trainLabels = labels[r]
  testingSet = newFeatures[r2, :]; testLabels = labels[r2]
      
  if not equalClassSize:
    testingSet, testLabels = balanceClasses(testingSet, testLabels)
    clf = LogisticRegression(C=C, class_weight='auto', intercept_scaling=B, penalty='l2')
#     clf = svm.SVC(C=C, kernel='rbf', class_weight='auto', probability=returnProb)
  else:
    clf = LogisticRegression(C=C, intercept_scaling=B, penalty='l2')
#     clf = svm.SVC(C=C, kernel='rbf', class_weight='auto', probability=returnProb)
#     clf = RandomForestClassifier()
#     clf = KNeighborsClassifier(n_neighbors=15)
#   print np.arange(20)[clf2.get_support()]
#     clf = AdaBoostClassifier()
#   clf = GradientBoostingClassifier(init=LogisticRegression)
#     clf = GaussianNB()
#   clf = DecisionTreeClassifier()
  
  if featureSelect:
    rfecv = RFE(estimator=clf, step=1,  n_features_to_select=8)
#     rfecv = RFECV(estimator=clf, step=1, cv=10)
    rfecv.fit(newFeatures, labels)
    print("Optimal number of features : %d" % rfecv.n_features_)
    print rfecv.ranking_
    print np.arange(20)[rfecv.support_]
    return
  
  clf.fit(trainingSet, trainLabels)
  
  def myPredict(clf, x, thres=0.5):
    probArray = clf.predict_proba(x)[:,1]
    predictLabels = 1*(probArray > thres)
    predictLabels = 2*predictLabels - 1
    return predictLabels, probArray
  
#   d = np.reshape(np.linspace(0, 10, num=1000), (-1, 1))
# #   print d.shape
#   results = clf.predict(d)
#   for i in xrange(1000):
#     if results[i] == 1:
#       print "dist:", i*0.01
#       break
  
  if returnProb:
    predictLabels, probArray = myPredict(clf, testingSet, thres=thres)
  else:
    predictLabels = clf.predict(testingSet)
#     print "accuracy rate from classifier: " + str(clf.score(testingSet, testLabels))
    
  suffix = "" if useAll else str(features)
  
  if draw and returnProb:
    area = drawPrecisionRecallCurve(filename[:-4] + suffix, testLabels, probArray)
    roc_auc = drawROCCurve(filename[:-4] + suffix, testLabels, probArray)
  
  false_neg = false_pos = true_neg = true_pos = 0
  for i in xrange(len(predictLabels)):
    if predictLabels[i] == testLabels[i] == -1:
      true_neg += 1
    elif predictLabels[i] == testLabels[i] == 1:
      true_pos += 1
    elif predictLabels[i] == -1 and testLabels[i] == 1:
      false_neg += 1
    else:
      false_pos += 1
  good = true_neg + true_pos
  print "accuracy rate: ", good/float(len(predictLabels)), good
  print "true negative rate: ", true_neg/float(len(predictLabels)), true_neg
  print "true positive rate: ", true_pos/float(len(predictLabels)), true_pos
  print "false negative rate: ", false_neg/float(len(predictLabels)), false_neg
  print "false positive rate: ", false_pos/float(len(predictLabels)), false_pos
  precision = true_pos/float(true_pos + false_pos)
  recall = true_pos/float(true_pos + false_neg)
  print "precision: ", precision
  print "recall: ", recall
  print "f1 score: ", 2*(precision*recall)/(precision + recall)
  return good/float(len(predictLabels))
  
def drawPrecisionRecallCurve(filename, testLabels, probArray):
  precision, recall, thresholds = precision_recall_curve(testLabels, probArray)
  area = auc(recall, precision)
  plt.clf()
  plt.plot(recall, precision, label='Precision-Recall curve')
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])
  plt.grid()
  plt.title('Precision-Recall: AUC=%0.2f' % area)
  plt.legend(loc="lower left")
  plt.savefig(filename+"_precRecallCurve")
  return area
  
def drawROCCurve(filename, testLabels, probArray):
  fpr, tpr, thresholds = roc_curve(testLabels, probArray)
  roc_auc = auc(fpr, tpr)
  plt.clf()
  plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.grid()
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.savefig(filename+"_ROC")
  return roc_auc

def visualizeDribbleData(dribble="type4_dribble.txt", kick="type4_rc.txt"):
  def parseKickData(kick):
    w = open(kick)
    lines = w.readlines()
    myList = []
    for index, line in enumerate(lines):
      if index % 1000 == 0:
        print "parsed " + str(index) + " samples"
      gameState, ballMovement = line.rstrip().split("#")
      x, initTime, x, agentInfo, x, opp, x, me, x, ball, x, kickTarget, x, directFeatures =  gameState.split(":")
      initTime = float(initTime)
      x, immedAfterKick, x, afterKick, x, kickInfo = ballMovement.split(":")
      kickMode = int(kickInfo.split()[0])
      if (kickMode != 2):
        continue
      afterKick = [float(x) for x in afterKick.split()]
      immedAfterKick = [float(x) for x in immedAfterKick.split()]
  
      finalTime = afterKick[2]
      target = (immedAfterKick[2],immedAfterKick[3])
      ballBefore = (immedAfterKick[0],immedAfterKick[1])
      ballAfter = (afterKick[0], afterKick[1])  
      ballBeforeDist = getDistTo(ballBefore, target)
      ballAfterDist = getDistTo(ballAfter, target)
      
      distTraveled = ballBeforeDist - ballAfterDist 
      timeElapsed = finalTime - initTime
      myList.append([timeElapsed, distTraveled])
    return np.array(myList)
  
  plt.figure(figsize=(16,12))
  mat = np.loadtxt(dribble)
#   mat2 = parseKickData(kick)
#   print np.sum(mat2[:,1] <  0)
#   print np.sum(mat2[:,1] >  0)
  plt.scatter(mat[:,0], mat[:,1], alpha=0.3, c='g', label="dibble")
#   plt.scatter(mat2[:,0], mat2[:,1], alpha=0.3, label="kick")
  plt.legend()
  plt.xlabel("time elapsed in seconds")
  plt.ylabel("distance ball traveled in meters")
  plt.grid()
  ax = plt.gca()
  start, end = ax.get_xlim()
  ax.xaxis.set_ticks(np.arange(start, end, 0.5))
  
  r = make_pipeline(PolynomialFeatures(9), Ridge(alpha=0.0001))
#   from sklearn.svm import SVR
#   r = SVR()
   
  r.fit(np.reshape(mat[:,0], (-1, 1)), mat[:,1])
  x = np.linspace(0, 12, 100)
  x = np.reshape(x, (-1, 1))
  y = r.predict(x)
  plt.plot(x, y, c='g', linewidth=3)
  
  plt.savefig("dibble_kick_compare.png", bbox_inches="tight")
  
if __name__ == '__main__':
  if len(sys.argv) == 2:
    eval(sys.argv[1])()
  else:
    classify()