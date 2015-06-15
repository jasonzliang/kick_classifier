#!/usr/bin/python
'''
Created on Feb 26, 2014

@author: jason
'''
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import glob
import random
import networkx as nx

np.set_printoptions(suppress=True, formatter={'all':lambda x: str(x) + ','}, linewidth=150)
colors=('k','y','m','c','b','g','r','#aaaaaa')
linestyles=('-','--','-.',':')
styles=[(color,linestyle) for linestyle in linestyles for color in colors]
numPlots = 6

def parseFile(name, mode=0):
  f = open(name)
  x = np.zeros(40)
  y = np.zeros(60)
  loc = []
  std = []
  counter = 0
  for i, line in enumerate(f):
    if i == 0:
      key = line.rstrip()
      continue
    counter +=1
    rawdata = line.rstrip().split()
#     print rawdata
    locError, angleError, avgStd = float(rawdata[4]), float(rawdata[7]), float(rawdata[9])
    loc.append(locError)
    std.append(avgStd)
    if int(locError/0.25) < 40:
      x[int(locError/0.25)] += 1
    if int(angleError) < 60:
      y[int(angleError)] += 1
    
  x = x/counter
  y = y/counter
  if mode == 0:
    return x,y,key
  else:
    return np.array(loc), np.array(std)

def drawHistogram(data, keys, prefix=""):
  matplotlib.rcParams.update({'font.size': 18})
  plt.figure(figsize=(8, 7))
  plt.title('Localization Error Performance')
  plt.ylabel('CDF')
  plt.xlabel('Accuracy in Meters')
  x = np.arange(40)*0.25
  
  for num, data in enumerate(zip(data, keys)):
    datum,key = data
    if key != 'noplot':
      plt.plot(x, np.cumsum(datum), label=key, linewidth=1, color=styles[num][0],ls=styles[num][1])
    
  plt.legend(loc=4)
  plt.grid()
  plt.savefig(prefix + "locError.png", bbox_inches='tight', dpi=300)
  plt.clf()


def drawHistogram2(data, keys, prefix=""):
  matplotlib.rcParams.update({'font.size': 18})
  plt.figure(figsize=(8, 7))
  plt.title('Yaw Error Performance')
  plt.ylabel('CDF')
  plt.xlabel('Accuracy in Degrees')
  x = np.arange(60)
  
  for num,data in enumerate(zip(data, keys)):
    datum,key = data
    if key != 'noplot':
      plt.plot(x, np.cumsum(datum), label=key, linewidth=1, color=styles[num][0],ls=styles[num][1])
    
  plt.legend(loc=4)
  plt.grid()
  plt.savefig(prefix + "angleError.png", bbox_inches='tight', dpi=300)
  plt.clf()
  
def main():
  data = []; keys = []; data2 = []
  for i in xrange(numPlots):
    name = "l" + str(i+1) + '.txt'
    x,y,k = parseFile(name)
    data.append(x); keys.append(k); data2.append(y)
  drawHistogram(data, keys)
  drawHistogram2(data2, keys)
  
def parseFile2(filename, keys=None, data1=None, data2=None, counter=0):
  if keys == None:
    keys = []; data1 = []; data2 = []
    for i in xrange(numPlots):
      data1.append(np.zeros(40))
      data2.append(np.zeros(60))
  f = open(filename)
  lines = f.readlines()
  counter += len(lines) - 1
  
  locData = np.zeros((len(lines) - 1, numPlots))
  angleData = np.zeros((len(lines) - 1, numPlots))

  for i, line in enumerate(lines):
    if i == 0:
      keys = line.rstrip()[:-1].split(';')
      continue
    
    rawdata = line.rstrip()[:-1].split(';')
    
    for j in xrange(0, numPlots*2, 2):
      locError, angError = float(rawdata[j]), float(rawdata[j+1])
      if np.isnan(locError) or np.isnan(angError):
        continue
      locData[i-1, j/2] = locError
      angleData[i-1, j/2] = angError
      if int(locError/0.25) < 40:
        x = data1[j/2]
        x[int(locError/0.25)] += 1.0
        
      if int(angError) < 60:
        y = data2[j/2]
        y[int(angError)] += 1.0
     
  return keys, data1, data2, counter, np.mean(locData,axis=0), np.mean(angleData,axis=0)
    
def multFiles(basedir="old/results_4_5-3"):
  listoffiles = glob.glob(basedir+"/*.txt")
  locAvgErrors = np.zeros((len(listoffiles), numPlots))
  t_test = np.zeros((numPlots, numPlots))
  ks_test = np.zeros((numPlots, numPlots))
  
  keys = []; data1 = []; data2 = []; counter = 0
  for i in xrange(numPlots):
    data1.append(np.zeros(40))
    data2.append(np.zeros(60))
  
    
  for i, name in enumerate(listoffiles):
    print "parsing file #" + str(i) + ":" + name
    keys, data1, data2, counter, avgError, avgAngleError = parseFile2(name, keys, data1, data2, counter)
    locAvgErrors[i, :] = avgError
    
  for i in xrange(numPlots):
    data1[i] = data1[i]/counter
    data2[i] = data2[i]/counter
    
  for i in xrange(numPlots):
    for j in xrange(numPlots):
      t_test[i,j] = stats.ttest_rel(locAvgErrors[:,i], locAvgErrors[:,j])[1]
      ks_test[i,j] = stats.ks_2samp(data1[i], data1[j])[1]
  
  #hack override
#   keys = ["p-1000,l-11","p-500,l-11","p-200,l-11","p-1000,l-3","p-500,l-3","p-200,l-3"]
#   keys =['p1000,l3', 'p1000,l2', 'p1000,l1', 'p500,l3', 'p500,l2', 'p500,l1']
#   keys = ['p500,l3,ns', 'p500,l2,ns', 'p500,l1,ns','p500,l3,s2', 'p500,l2,s2', 'p500,l1,s2']
  keys = ['noplot', 'noplot', 'no line info', '3 lines', '2 lines', '1 line']
#   keys = ['original', 'p1000,l1,s1', 'p500,l11,s2', 'p500,l3,s2', 'p500,l2,s2', 'p500,l1,s2', 'p500,l11,s1', 'p1000,l11,s2']
    
  compare = [(0,3), (1,2), (4,5), (0,1)]
  for a,b in compare:
    print keys[a] + " <-> " + keys[b], t_test[a,b], ks_test[a,b]

  drawHistogram(data1, keys, prefix=basedir)
  drawHistogram2(data2, keys, prefix=basedir)
  
def singleFile():
  keys, data1, data2, counter, avgError, avgAngleError = parseFile2("localizationError.txt")
    
  print "total measurements: " + str(counter)
  print keys
  for i in xrange(numPlots):
    data1[i] = data1[i]/counter
    data2[i] = data2[i]/counter

  drawHistogram(data1, keys)
  drawHistogram2(data2, keys)

def drawAvgStdvsLocError(name='localizationError.txt'):
  loc,std = parseFile(name,mode=1)
  plt.title('LocError vs avgStd Scatter')
  plt.xlabel('avgStd')
  plt.ylabel('LocError')

  plt.scatter(std, loc)
  plt.grid()
  plt.savefig("scatter.png")
  
  
if __name__ == '__main__':
  import sys
  if len(sys.argv) > 1:
    multFiles(sys.argv[1])
  else:
    multFiles()