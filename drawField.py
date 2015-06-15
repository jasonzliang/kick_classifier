#!/usr/bin/python
'''
Created on Jul 27, 2014

@author: jason
'''

from classify import *
import random,os,math
from PIL import Image,ImageDraw

startpoint=(160,140)
endpoint=(1760,2540)
fieldDim=(endpoint[0]-startpoint[0],endpoint[1]-startpoint[1])
myImage = Image.open("pictures/field.png")

def transformPt(pt):
  woffset = int(((pt[0] + 15.0)/30.0)*fieldDim[1])
  hoffset = int((abs(pt[1] - 10.0)/20.0)*fieldDim[0])  
  return (startpoint[1] + woffset, startpoint[0] + hoffset)

def drawLineSegment(draw, startpt, endpt, color, w=8):
  linestart = transformPt(startpt)
  lineend = transformPt(endpt)
  draw.line([linestart, lineend], fill=color, width=w)
  
def drawCircle(draw, pt, color, r=10):
  pt = transformPt(pt)
  draw.ellipse((pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r), fill=color)

def drawField_helper(oppArray, me, ball, kickTarget, success, afterKick, outname):
  workingImg = myImage.copy()
  draw = ImageDraw.Draw(workingImg)
    
  kickV = (kickTarget[0]-ball[0], kickTarget[1]-ball[1])
  kickAngle = math.atan2(kickV[1], kickV[0])
  kickPt2 = (ball[0] + 2*math.cos(kickAngle), ball[1] + 2*math.sin(kickAngle))
  drawCircle(draw, ball, color='blue', r=12)
  drawLineSegment(draw, ball, kickPt2, color='blue')
  drawCircle(draw, kickTarget, color="blue")
  drawCircle(draw, afterKick, color='yellow')
#   print transformPt(kickTarget)
  
  myAngle = math.radians(me[2])
  pt2 = (me[0] + math.cos(myAngle)/3, me[1] + math.sin(myAngle)/3)
  drawCircle(draw, me, color='green')
  drawLineSegment(draw, me, pt2, color='green')
  
  for opp in oppArray:
    oppAng = math.radians(opp[2])
    pt2 = (opp[0] + math.cos(oppAng)/3, opp[1] + math.sin(oppAng)/3)
    drawCircle(draw, opp, color='red')
    drawLineSegment(draw, opp, pt2, color='red')
#   drawCircle(draw, (-15,-10), color="yellow")
#   drawCircle(draw, (-15, 10), color="yellow")
#   drawCircle(draw, (15,-10), color="yellow")
#   drawCircle(draw, (15,10), color="yellow")


#   if success:
#     draw.rectangle((0,0, startpoint[1], startpoint[0]), fill="green")
#   else:
#     draw.rectangle((0,0, startpoint[1], startpoint[0]), fill="red")
    
  del draw
  workingImg.save(outname, "png")
  

def drawField(filename='type0_apollo.txt', max_count=20, kickType=10):
  w = open(filename)
  lines = w.readlines()
  random.shuffle(lines)
  number = range(max_count*2)
  random.shuffle(number)
  success_counter = 0
  failure_counter = 0
  for index, line in enumerate(lines):
    if success_counter >= max_count and failure_counter >= max_count:
      break
      
    gameState, ballMovement = line.rstrip().split("#")
    oppArray, me, ball, kickTarget, agentType = preprocessGameState(gameState)
    
    if len(oppArray) == 0 or kickTarget[2] != kickType:
      continue
    kickSuccess = parseKickSuccess(ballMovement)
    x, immedAfterKick, x, afterKick, x, kickMode = ballMovement.split(":")
    afterKick = [float(x) for x in afterKick.split()]
    
    if kickSuccess and success_counter >= max_count:
      continue
    if not kickSuccess and failure_counter >= max_count:
      continue
    
    print "drawing image number " + str(success_counter + failure_counter)
    
    if kickSuccess:
      successString = "success"
      success_counter += 1
    else:
      successString = "failure"
      failure_counter += 1
      
    outname =  filename[:-4] + "_kickType" + str(int(kickTarget[2])) + "_image" + str(number[success_counter + failure_counter - 1]) + ".png"
    foldername = "pictures/" + filename[:-4] + "_kickType" + str(int(kickTarget[2]))
    if not os.path.exists(foldername):
      os.mkdir(foldername)
    outname = os.path.join(foldername, outname)
    with open(outname[:-4] + '.txt', 'wb') as f:
      f.write(successString+'\n')
    drawField_helper(oppArray, me, ball, kickTarget, kickSuccess, afterKick, outname)
    
def drawFieldBatch(filenames=['type0_fc.txt', 'type0_apollo.txt', 'type1_fc.txt', 'type1_apollo.txt', 
                             'type2_fc.txt', 'type2_apollo.txt'], kickTypes=[7,10], max_count=30):
  for filename in filenames:
    for kickType in kickTypes:
      print filename + ", kickType:" + str(kickType)
      drawField(filename, max_count, kickType)
      
if __name__ == '__main__':
    drawFieldBatch()