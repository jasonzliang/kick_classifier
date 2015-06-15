#!/usr/bin/python
'''
Created on Oct 15, 2014

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
from smote import *
import subprocess
import random
import cStringIO
import cPickle

from classify import parse, balanceClasses
sys.path.append("./liblinear-1.94/python/")
import liblinearutil

def formatKickType(kickType):
  if type(kickType) is list:
    s = ""
    for element in kickType:
      s += str(element) + "_"
    s = s[:-1]
    return s
  else:
    return str(kickType)
  
def tuneParamsWeights(useCache=True, trainFiles=['type2_fc_10.train', 'type2_fc_11.train'],
                      kickTypes = [10,11], modelDir="models"):
  for trainFile, kickType in zip(trainFiles, kickTypes):
    basename = modelDir + "/" + str(kickType) + "optimalParameters"
    if not os.path.exists(basename + ".pkl") or not useCache:
      d = {}
      f = open(basename + ".txt", 'a+')
      g = open(basename + ".pkl", 'wb')
      for weight in myCustomWeights:
        weightString = "-w-1 1.0 -w1 " + str(float(weight)) + " "
        a, C, B = tuneParameters(weightString, os.path.join(modelDir, trainFile))
        d[float(weight)] = (a,C,B)
        f.write("weight: " + str(weight) + " ")
        f.write(str((C,B)) + ": " + str(a) + "\n")
        f.flush()
        print "optimal params C: " + str(C) + " B: " + str(B) + " acc: " + str(a)
      cPickle.dump(d, g)
      f.close()
      g.close()
    else:
      d = cPickle.load(open(basename + ".pkl"))
      for key, value in d.items():
        print str(key) + ": " + str(value)
    
def tuneParameters(weightString="-w-1 1.0 -w1 1.0 ", trainFile='models/type2_fc_10_11.train',
                   CList=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], BList=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], v=10):
  labels = []
  features = []
  bestAccuracy = 0
  bestParams = None
  with open(trainFile) as f:
    for line in f:
      if line == '':
        continue
      data = line.rstrip().split()
      labels.append(float(data[0]))
      data = [float(x.split(':')[1]) for x in data[1:]]
      features.append(data)
  for C in CList:
    for B in BList:
      myOptions =  weightString + "-s 0 -e 0.000001 -c " + str(C) + " -B " + str(B) + " -v " + str(v)
      accuracy = liblinearutil.train(labels, features, myOptions)
      
      print "C: " + str(C) + " B: " + str(B) + " " + str(accuracy)
      if accuracy > bestAccuracy:
        bestAccuracy = accuracy
        bestParams = (C,B)
  
  return bestAccuracy, bestParams[0], bestParams[1]

myCustomWeights = [0.5, 0.75, 1.0, 1.5, 2.0]
customProb=[0.5, 0.75, 0.9]

def outputTrainingFile(filename="5_4_type4_apollo3d.txt", outputDir="models_5_4", trainFrac=1.0,
                       useCache=True, kickTypes=[[10],[11]], useFeatures=[0,1,2,3,4,5,6,7], 
                       useAll=True, scaling=False, equalClassSize=True, customWeights=myCustomWeights, 
                       customProb=customProb, C=1.0, B=1.0, paramFile=None):
  if not os.path.exists(outputDir):
    os.system("mkdir -p " + outputDir)
  
  for kickType in kickTypes:      
    features, labels = parse(filename=filename, useCache=useCache, ezKickSuccess=False, 
                             kickType=kickType, ignoreSelfFailure=False, useDirectFeatures=True,
                             nfeatures=8)
    kickType = formatKickType(kickType)
    
    if paramFile != None:
      with open(outputDir + "/" + str(kickType) + "optimalParameters.pkl") as f:
        paramDict = cPickle.load(f)
        
    if not useAll:
      newFeatures = features[:, useFeatures]
    else:
      newFeatures = features
      
    sortedIndices = np.argsort(labels)
    newFeatures = newFeatures[sortedIndices, :]
    labels = labels[sortedIndices]
      
    if equalClassSize:
      newFeatures, labels = balanceClasses(newFeatures, labels)
      weightString = "-w-1 1.0 -w1 1.0 "
    else:
      neg = float(np.sum(labels == -1))
      pos = float(np.sum(labels == 1))
      pos_weight = (neg + pos)/pos
      neg_weight = (neg + pos)/neg
      weightString = "-w-1 1.0 -w1 {0:f} ".format(pos_weight/neg_weight)
    print "we have " + str(newFeatures.shape[0]) + " samples."
    print "we have " + str(np.sum(labels == 1)) + " positive labels"
  
    if scaling:
      minimum = np.amin(newFeatures, axis=0)
      maximum = np.amax(newFeatures, axis=0)
      dataRange = maximum - minimum
      for i in xrange(newFeatures.shape[0]):
        newFeatures[i,:] = np.divide(newFeatures[i,:] - minimum, dataRange)
      assert np.max(newFeatures) <= 1.0
      assert np.min(newFeatures) >= 0.0
      
    trainFile = os.path.join(outputDir, filename[:-4]  + "_" + kickType + ".train")
    if trainFrac == 1.0:
      testFile = trainFile
    else:
      testFile = os.path.join(outputDir, filename[:-4]  + "_" + kickType + ".test")
      
    f = open(trainFile, 'wb'); g = open(testFile, 'wb')
    numFeatures = newFeatures.shape[1]
    for i in xrange(newFeatures.shape[0]):
#       if i % 10000 == 0:
#         print "writing training file, line: " + str(i)
      writeString = str(int(labels[i])) + " "
      for j in xrange(numFeatures):
        writeString += str(j+1) + ":" + '{0:f}'.format(newFeatures[i,j]) + " "
      writeString = writeString.rstrip() + "\n"
      if random.random() < trainFrac:
        f.write(writeString)
      else:
        g.write(writeString)
        
    f.close(); g.close()
    
    if scaling:
      def parse_a(array):
        l = np.round(minimum,6).tolist()
        l = [str(x) for x in l]
        return " ".join(l)
      
      f = open(trainFile[:-6] + ".scale", 'wb')
      f.write("minimum " + parse_a(minimum) + "\n")
      f.write("dataRange " + parse_a(dataRange) + "\n")
      f.close()
      
    def createModelFile(weightString, folderName, trainFile, testFile, C, B, p, alreadyTrained):
      if paramFile != None:
        weight = float(weightString.split()[-1]) 
        a,C,B = paramDict[weight]
      print "class weights: " + weightString
      print "C: " + str(C) + " B: " + str(B) + " p: " + str(p)
      weightOutputDir = os.path.join(outputDir, folderName)
      if not os.path.exists(weightOutputDir):
        os.system("mkdir -p " + weightOutputDir)
      with open(os.path.join(weightOutputDir, kickType + ".prob"), 'wb') as f:
        f.write(str(p)+'\n')
      
      if alreadyTrained != None:
        os.system("cp " + alreadyTrained + ".model " + weightOutputDir + "/")
        return alreadyTrained
      else:
        modelFile = os.path.join(weightOutputDir, kickType + ".train")
        
        print subprocess.check_output("./liblinear-1.94/train " + weightString + "-s 0 -e 0.000001 -c " + str(C) + 
                                      " -B " + str(B) + " " + trainFile + " " + modelFile + ".model", shell=True) 
        print subprocess.check_output("./liblinear-1.94/predict " +  testFile + " " + 
                                      modelFile + ".model" + " /tmp/temp_out", shell=True)
        os.system("rm /tmp/temp_out")
        return modelFile
      
    if customWeights == None:
      if equalClassSize:
        folderName = "balanced"
      else:
        folderName = "unbalanced"
      for p in customProb:
        createModelFile(weightString, (folderName + "vp" + str(p)).replace(".", "d"), trainFile, testFile, C, B, p)
    else:
      for w in customWeights:
        alreadyTrained = None
        for p in customProb:
          weightString = "-w-1 1.0 -w1 " + str(float(w)) + " "
          alreadyTrained = createModelFile(weightString, ("w" + str(w) + "vp" + str(p)).replace(".", "d"), trainFile, testFile, C, B, p, alreadyTrained)

if __name__ == '__main__':
  outputTrainingFile()