import sys
import os
# import client
import glob
# import util
from PIL import Image, ImageDraw
import cv2
import cv2.cv as cv
import numpy as np
import math
import linecache
from scipy.stats.stats import pearsonr
from scipy.interpolate import griddata
from scipy import sparse
import matplotlib
import matplotlib.pyplot as pyplot
# import gpuclient
import cPickle
import random

def makecells(dbname):
  util.makecells(lat=0,lon=0,length=2,inputDir=dbname,distance=1,radius=1,outputDir=dbname+'/cells-100/')

# generates a txt file that matches query images to db images 
def createqdbmap(matchdir = 'debug_mall_query_2013_2_v3_new_NewparkMall_4_mask',querydir='eric_mall_query_2013_2'):
  blacklist = processBlackList(querydir)
  print blacklist
  rawdict = loadMatchDict()
  f = open(querydir + '/querydbmatch.txt', 'wb')
#  g = open('mall_query_2013/infomap.txt', 'wb')
  for path in glob.glob(matchdir + '/*.jpg'):
    queryimg = path.split('/')[-1].split('_')[-2]
    dbimg = path.split('/')[-1].split('_')[-1]
    os.system('cp /media/8074D55274D54B92/v3_new_NewparkMall_4_mask/' + dbimg + " eric_mall_db_2013_newdataset")
    os.system('cp /media/8074D55274D54B92/v3_new_NewparkMall_4_mask/' + dbimg[:-4]+'sift.txt' + " eric_mall_db_2013_newdataset")
    if queryimg not in blacklist:
      f.write(queryimg + ';' + dbimg + '\n')
#      g.write(queryimg + ';' + rawdict[queryimg] + '\n')
  f.close()
#  g.close()

def pad(s):
  if type(s) is int:
    s = str(s)
  while len(s) < 5:
    s = '0' + s
  return s
  
def processBlackList(querydir):
  f = open(querydir + '/blacklist.txt', 'rb')
  blacklist = f.readlines()
  for i,value in enumerate(blacklist):
    value = value.rstrip()
    value = '0,0-' + pad(value) + '.jpg'
    blacklist[i] = value
  f.close()
  
  return blacklist

def loadMatchDict(namemap='v3_new_NewparkMall_4_mask/namemap.txt'):
  matchdict = {}
  f = open(namemap, 'rb')
  for line in f:
    queryimg, rawimg = line.rstrip().split(';')
    matchdict[queryimg] = rawimg
  return matchdict

def loadMatchDict2(namemap='v3_new_NewparkMall_4_mask/namemap.txt'):
  matchdict = {}
  f = open(namemap, 'rb')
  for line in f:
    oname, dbimg = line.rstrip().split(';')
    matchdict[dbimg] = oname
  return matchdict

def drawCameraPoints(cameraFile='chaoranCory_db/left_cameraA.txt'):
  data = np.loadtxt(cameraFile)
  xlist = data[:,1]
  ylist = data[:,3]
  pyplot.scatter(xlist,ylist)
  pyplot.show()

def drawHistogram(accuracy='accuracy.txt', outname=None):
#  accuracy = '/home/jason/Desktop/image_localize/project/src/temp/old_results/before ransac iteration reduction/' + accuracy
  points = []
  f = open(accuracy, 'rb')
  for line in f:
    a,b,c,d,error = line.rstrip().split(';')
    points.append(float(error))
  f.close()
  x = range(int(max(points))+1)
  x.append(int(max(points))+1)
#  print x
  pyplot.title('Pose Estimation Location Performance')
  pyplot.xlabel('Error in Meters')
  pyplot.ylabel('Fraction of Images')
  pyplot.hist(points, bins=x, cumulative=True, normed=True, color='green')
  if outname != None:
    pyplot.savefig(outname)
  else:
    pyplot.show()
  
def drawYawHistogram(accuracy='yaw.txt'):
#  accuracy = '/home/jason/Desktop/image_localize/project/src/temp/old_results/before ransac iteration reduction/' + accuracy
  points = []
  f = open(accuracy, 'rb')
  for line in f:
    img,yawerror = line.rstrip().split(';')
    if np.isnan(float(yawerror)):
      continue
    points.append(float(yawerror))
  f.close()
  
#  x = range(int(max(points))+1)
#  x.append(int(max(points))+1)
  print points
  pyplot.title('Pose Estimation Yaw Performance')
  pyplot.xlabel('Error in Degrees')
  pyplot.ylabel('Fraction of Images')
  pyplot.hist(points, bins = range(0, int(max(points))+5, 5), cumulative=True, normed=True, color='green')
  pyplot.show()
  
def drawHistogram2(accuracy='poseRunInfo.txt', newFile=True):
#  accuracy = '/home/jason/Desktop/image_localize/project/src/temp/old_results/before ransac iteration reduction/' + accuracy
  points = []
  f = open(accuracy, 'rb')
  if newFile:
    for line in f:
      error = float(line.rstrip().split(';')[4])
      if error != 9999:
        points.append(float(error))
#      if float(error) < 1.0:
#        os.system('cp eric_mall_query_2013/' + a + ' queryimages')
  else:
    for line in f:
      if line.find('Error') != -1:
        value = float(line.rstrip().split()[-1])
        points.append(value)
  f.close()
  print points
  x = range(int(max(points))+1)
  x.append(int(max(points))+1)
  h, l = np.histogram(points, bins=x)
  h = h/float(len(points))
  pyplot.title('Pose Estimation Location Performance')
  pyplot.xlabel('Error in Meters')
  pyplot.ylabel('Fraction of Images')
  pyplot.bar(l[:-1], h, width=1)
  pyplot.show()
  
def drawYawHistogram2(accuracy='yaw.txt'):
#  accuracy = '/home/jason/Desktop/image_localize/project/src/temp/old_results/before ransac iteration reduction/' + accuracy
  points = []
  f = open(accuracy, 'rb')
  for line in f:
    img,yawerror = line.rstrip().split(';')
    if np.isnan(float(yawerror)):
      continue
    points.append(float(yawerror))
  f.close()
  
#  x = range(int(max(points))+1)
#  x.append(int(max(points))+1)
  print range(0, int(max(points))+5, 5)
  h, l = np.histogram(points, bins=range(0, int(max(points))+5, 5))
  h = h/float(len(points))
  print h, l
  pyplot.title('Pose Estimation Yaw Performance')
  pyplot.xlabel('Error in Degrees')
  pyplot.ylabel('Fraction of Images')
  pyplot.bar(l[:-1], h, width=5)
  pyplot.show()
  
def plotLocations(accuracy = 'accuracy.txt'):
  gx = []
  gy = []
  cx = []
  cy = []
  f = open(accuracy, 'rb')
  for line in f:
    name,dbloc,groundtruth,calcloc,error = line.rstrip().split(';')
    groundtruth = groundtruth.strip('[]').split()
    calcloc = calcloc.strip('[]').split()
    gx.append(groundtruth[0]); gy.append(groundtruth[1])
    cx.append(calcloc[0]); cy.append(calcloc[1])
  f.close()
  
  pyplot.plot(gx, gy, 'bo')
  pyplot.plot(cx, cy, 'ro')
  pyplot.show()
  
def generateInfoFilesWrapper(dbdir='chaoranCory_db'):
  for image in glob.glob(dbdir + '/*.jpg'):
    image = image.split('/')[-1]
    generateInfoFiles_new(dbimg = image, dbdir = dbdir, offset=-20, camerafile = dbdir+'/%s_cameraA.txt')

def generateInfoFiles_new(dbimg='0,0-00038.jpg', dbdir='dummy_db', offset=180, 
                          camerafile='temp/cory/%s_aaron.txt', fov = math.degrees(math.atan(float(1024)/float(612)))*2):
  dbnum = int(dbimg[4:-4])
  oname, dbimg = linecache.getline(dbdir+'/namemap.txt', dbnum).rstrip().split(';')
  location = oname[:-4].split('_')[-1]
  number = int(oname[:-4].split('_')[-2][5:])
  if number - 9 < 1: return
  
  imagename = dbimg[:-4]

  print imagename
  print str(number) + location
  
  camerafile2 = camerafile % location
  camera = linecache.getline(camerafile2, number - 9).strip().split()
  roll,pitch,yaw = (math.degrees(float(camera[4])), math.degrees(float(camera[5])), math.degrees(float(camera[6])))
  x,y,z = (float(camera[1]), float(camera[2]), float(camera[3]))
  #this shift in yaw puts everything in Aaron's coordinate frame
  yaw = (yaw + offset) % 360
  print "yaw after: " + str(yaw)
    
  myString = "{'is-known-occluded': False, 'url': {'href': ''}, 'field-of-view': %s, 'image-size': {'width': 2048, 'height': 2448}, 'view-direction': {'yaw': %s, 'pitch': %s, 'roll': %s}, 'view-location': {'lat': %s, 'alt': %s, 'lon': %s}, 'location': {'x': %s, 'y': %s, 'z': %s}, 'id': 'cory_image'}"
  myString = myString % (fov, yaw, pitch, roll, '0.0', '0.0', '0.0', x,y,z)
  f = open(dbdir +'/'+ imagename + '.info', 'wb')
  f.write(myString)
  f.close()
  
def depthMapAccuracy():
  
  def loadNameMap():
    matchdict = {}
    f = open('eric_mall_query_2013_2/querydbmatch.txt', 'rb')
    for line in f:
      queryimg, oimg = line.rstrip().split(';')
      matchdict[queryimg] = oimg
    return matchdict
  matchdict = loadNameMap()
  
#  g = open('temp/accuracy_full.txt')
#  f = open('temp/accuracy_sparse.txt').readlines()
  g = open('accuracy_0.2.txt')
  f = open('accuracy_2.0.txt').readlines()
  
  h = open('comparisonalpha.txt', 'wb')
  h.write('imagename;alpha=0.2;alpha=2.0\n')
  for i, line in enumerate(g):
#    if i != 10: continue
    img,b,c,d,error1 = line.rstrip().split(';')
    img2,b,c,d,error2 = f[i].rstrip().split(';')
    assert img == img2
    dbimg = matchdict[img]
#    depth_error = str(compareDepthMap(imagename=dbimg)[0])
    error1 = float(error1); error2 = float(error2)
#    if error1 > error2:
#      loc_error = error1/error2
#    else:
#      loc_error = error2/error1
#    h.write(img + ';' +str(error1) +';' + str(error2) + ';' + depth_error + '\n')
    h.write(img + ';' +str(error1) +';' + str(error2)  + '\n')
  h.close()
  
def compareDepthMap(basedir='depth_comp', imagename=None):
  error = []
  if imagename != None:
    iter = [basedir+ '/' + imagename[:-4] + '_sparse.pkl']
  else:
    iter = glob.glob(basedir + '/*.pkl')
  for matchfile in iter:
#    if matchfile != 'depth_comp/0,0-00869.pkl':
#      continue
    dbimg = matchfile.split('/')[-1][:-11] + '.jpg'
    matches = cPickle.load(open(matchfile))
    sparse_map = np.asarray(Image.open('eric_mall_db_2013_newdataset_sparse/' + dbimg[:-4] + '-depth.png'))
    full_map = np.asarray(Image.open('eric_mall_db_2013_newdataset/' + dbimg[:-4] + '-depth.png'))
    match_idx = np.nonzero(matches['imask'])[0]
    depth_comp = []

#    for i in xrange(500):
#      x = random.randint(0, 2047)
#      y = random.randint(0, 2447)
#      depth_comp.append((
#                   sparse_map[y, x],
#                   full_map[y, x],
#                   sparse_map[y, x]/
#                   float(full_map[y, x])
#                   ))
      
    for idx in match_idx:
      x,y = matches['d2d'][idx,:]
      if sparse_map[y, x] > full_map[y, x]:
        ratio = sparse_map[y,x] / float(full_map[y,x])
      else:
        ratio = float(full_map[y,x])/sparse_map[y,x]
      depth_comp.append((
                         sparse_map[y, x],
                         full_map[y, x],
                         ratio
                         ))

#    for tuple in depth_comp:
#      print tuple[2]
    temp = [tuple[2] for tuple in depth_comp]
    print "Average Error For " + dbimg
    print np.mean(temp)
    error.append(np.mean(temp))
#    print depth_comp
#  print error
  print "Average Error For All Images"
  print np.mean(error)
  
  if len(error) == 1:
    return error
  else:
    return None
  
def makeposedirs(querydbmatch='debug_combined_query_v3_new_NewparkMall_4_mask/querydbmatch.txt', querydir='combined_query', dbdir='combined_db'):
  try:
    os.mkdir(querydir)
  except:
    pass
  
  try:
    os.mkdir(dbdir)
  except:
    pass
  
  matchdict = loadMatchDict(querydbmatch)
  
  for queryimg,dbimg in matchdict.items():
    print queryimg, dbimg
    queryimg = queryimg+'.jpg'
    os.system('cp v3_new_NewparkMall_4_mask/' + dbimg + ' ' + dbdir)
    os.system('cp v3_new_NewparkMall_4_mask/' + dbimg[:-4]+'sift.txt' + ' ' + dbdir)
#    gpuclient.extract_features(querydir+'/'+queryimg, querydir+'/'+queryimg[:-4]+'sift.txt')

    os.system('cp v3_new_NewparkMall_4_mask/' + dbimg[:-4]+'-depth.png' + ' ' + dbdir)
    os.system('cp v3_new_NewparkMall_4_mask/' + dbimg[:-4]+'.info' + ' ' + dbdir)
    
  os.system('cp v3_new_NewparkMall_4_mask/namemap.txt' + ' ' + dbdir)
  os.system('cp ' + querydbmatch + ' ' + querydir)
  os.system('cp ' + querydbmatch.split('/')[0]+'/summary.txt' + ' ' + querydir)


def gpuSiftLocal(basedir='false_query_2013', port="7777"):
  for image in glob.glob(basedir + '/*.jpg'):
    image = os.path.abspath(image)
    print "processing: ", image
    siftname = image[:-4] + 'sift.txt'
    if os.path.exists(siftname) and os.path.getsize(siftname) > 0:
      continue
    os.system('/home/jason/Desktop/image_localize/SiftGPU/bin/siftgpu_remote localhost ' + port +  ' ' + image + ' ' + siftname)

def affineSift(basedir='xaffine_v3_new_NewparkMall_4_mask'):
  
  def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
        
  def reformat(siftname):
    f = open(siftname)
    x = f.readlines()
    f.close()
    g = open(siftname, 'wb')
    g.write(x[0])
    for line in x[1:]:
      infolist = line.rstrip().split()
      geom = infolist[:4]
      vector = infolist[4:]
      g.write(' '.join(geom) + '\n')
      for chunk in chunks(vector, 20):
        g.write(' '.join(chunk) + '\n')
    g.close()
      
      
  for image in glob.glob(basedir + '/*.jpg'):
    print "processing: ", image
    image_png = image[:-4] + '.png'
    siftname = image[:-4] + 'sift.txt'
    if os.path.exists(siftname) and os.path.getsize(siftname) > 0:
      continue
    os.system('convert ' + image + ' ' + image_png)
    os.system('./demo_ASIFT ' + image_png  + ' ' + siftname + ' >/dev/null')
    os.system('rm '+ image_png)
    reformat(siftname)
    

def reduceImages(basedir='v3_new_NewparkMall_4_mask', newdir='xaffine_v3_new_NewparkMall_4_mask', dist=4):
  if not os.path.exists(newdir):
    os.system('mkdir ' + newdir)
    os.system('cp ' + basedir + '/namemap.txt ' + newdir+'/')
    
  matchdict = loadMatchDict2()
  for path in glob.glob(basedir+'/*.jpg'):
    image = path.split('/')[-1]
    oname = matchdict[image]
    onum = int(oname.split('_')[-2][5:])
    path = os.path.abspath(path)
    siftpath = path[:-4] + 'sift.txt'
    if onum % dist == 0:
      os.system('ln -s ' + path + ' ' + newdir + '/')
      os.system('ln -s ' + siftpath + ' ' + newdir + '/')
      
  makecells(newdir)
  
#def generateInfoMap_GD(inputfile='temp/gdinfomap.txt', infofile='jason_mall_query_5_2013_2/infomap.txt', gdfile='jason_mall_query_5_2013_2/groundtruth.txt'):
#  '''Generates a infomap.txt and namemap.txt from gdinfomap.txt'''
#  g = open(infofile, 'wb')
#  g.write('queryimg;rawimg;yaw;pitch;roll\n')
#  h = open(gdfile, 'wb')
#  with open(inputfile) as f:
#    for line in f.readlines()[1:]:
#      qimg, x, y, yaw, p, r, gdyaw = line.rstrip().split(';')
#      qimg = 'RKH00' + qimg + '.jpg'
#      g.write(qimg + ';N/A;' + yaw + ';' + p + ';' + r + '\n')
#      h.write(qimg + ';' + x + ';' + y + ';' + gdyaw + '\n')
#  g.close(); h.close()

#def generateDummyFiles(modulo=4, dbdir= 'dummy_db'):
#  def pad(counter, amount):
#    x = str(counter)
#    while len(x) < amount:
#      x = '0' + x
#    return x
#  counter = 1
#  os.system('mkdir ' + dbdir)
#  namemap = open(dbdir+'/namemap.txt', 'wb')
#  posefiles=[('temp/cory/f2left_nick.txt', 'left'),('temp/cory/f2right_nick.txt', 'right')]
#  for posefile, location in posefiles:
#    t = open(posefile)
#    for line in t:
#      camerainfo = line.rstrip().split()
#      if int(camerainfo[0]) % modulo != 0:
#        continue
#      os.system('touch ' + dbdir + '/0,0-' + pad(counter, 5) + '.jpg')
#      namemap.write('Camera_' + location + '_Image' + pad(int(camerainfo[0]), 6) + '_' +  location + '.jpg;0,0-' + pad(counter, 5) + '.jpg\n')
#      counter+=1
#  namemap.close()
  
#def convertToNewFormat(gdfile = 'temp/groundtruth_head.txt', yawgd = 'temp/yaw_gd.txt', 
#                       output='eric_mall_query_2013_2/groundtruth.txt'):
#  temp = {}
#  for line in open(gdfile).readlines():
#    qimg, x, y = line.rstrip().split(';')
#    temp[qimg] = [x,y]
#  for line in open(yawgd).readlines():
#    qimg, yaw = line.rstrip().split(';')
#    try:
#      temp[qimg].append(yaw)
#    except:
#      pass
#  f = open(output, 'wb')
#  for qimg, data in temp.items():
#    x,y,yaw = data
#    f.write(qimg + ';' + x + ';' + y + ';' + yaw + '\n')
#    
#  f.close()
    
def loadPoseRunInfo(f='predict/poseRunInfo.txt', plot=('hconf', 'error')):
  legend = ['error', 'inliers', 'rperr', 'nmat', 'rsiter', 'iconf', 'hconf']
  mat = np.loadtxt(open(f),delimiter=";",skiprows=1, usecols=(5,6,8,9,10,11,12,13))
  mat = mat[~np.isnan(mat).any(axis=1)]
  print mat.shape
  xaxis = legend.index(plot[0])
  yaxis = legend.index(plot[1])
#  pyplot.scatter(mat[:,xaxis], mat[:,yaxis])
#  pyplot.savefig(str(plot) + '.png')
  
  temp = []
  for i in xrange(1,7):
    temp.append((i, legend[i], abs(pearsonr(mat[:,0], mat[:,i])[0])))
  for value in sorted(temp, key=lambda x: x[2]):
    print value
  
def makeTrainingTestFile(f='poseRunInfo.txt', errorThres=4, testFileFract=0, errorPos=4, featurePos=[5,8,9,10,11,12,13], seed=None):    

  if seed != None:
    random.seed(seed)
    
  f = open(f)
  g = open('predict/training_file_jason.txt', 'wb')
  h = open('predict/testing_file_jason.txt', 'wb')
  for line in f.readlines()[1:]:
    temp = line.rstrip().split(';')
    error = temp[errorPos]
    qimg = temp[0]
    
    if float(error) < errorThres:
      label = '1 '
    else:
      label = '0 '
      
    writestring = []
    for index, i in enumerate(featurePos):
      if i != 5:
        temp[i] = str(float(temp[i])/(float(temp[-1])))
      writestring.append(str(index+1) + ':' + temp[i])
    writestring = label + ' '.join(writestring) + ' #' + qimg + '\n'
    
    if random.random() < testFileFract:
      h.write(writestring)
    else:
      g.write(writestring)
  
  g.close(); f.close(); h.close()
   
 
#Rotates and adds mask to the images.
def rotateImg(dbdir='left', myiD=110732692):
  b = Image.open(dbdir + '/right_mask.png')
  c = Image.open(dbdir +'/left_mask.png')
  for i, path in enumerate(glob.glob(dbdir + '/*.jpg')):
    print path
    id = int(path.split('/')[-1].split('_')[1])
    a = Image.open(path)
    if id == myiD: #Right
      a = a.rotate(-90)
      a.paste(b, (0,0), b)
    else: #Left
      a = a.rotate(90)
      a.paste(c, (0,0), c)
    a.save(path)

#Takes in a directory with the rotated and masked images, copies them to a new directory where they are renamed and subsampled.
def reduceandrename(dbdir='left', newdir='left_db', subSampleFactor=3, myId=110732692):
  try:
    os.mkdir(newdir)
  except:
    pass
  
  f = open(newdir + '/namemap.txt', 'wb')
  c = 1
  for path in sorted(glob.glob(dbdir + '/*.jpg')):
    num = int(path.split('_')[-1][8:-4])
    if num % subSampleFactor != 0:
      continue
    os.system('cp ' + os.path.abspath(path) + ' ' + newdir + '/0,0-' + pad(c) + '.jpg')
    id = int(path.split('/')[-1].split('_')[1])
    if id == myId:
      f.write(path.split('/')[-1][:-4] + '_right.jpg;0,0-' + pad(c) + '.jpg\n')
    else:
      f.write(path.split('/')[-1][:-4] + '_left.jpg;0,0-' + pad(c) + '.jpg\n')
    c += 1
  
    
if __name__ == '__main__':
#   makecells('chaoranCory_db')
  drawHistogram2()
#  drawCameraPoints()
#  rotateImg()
#  makeposedirs()
#  gpuSiftLocal()
#  from predict import find_best_classifier
#  from collections import Counter
#  x = Counter()
#  scores = []
#  for i in xrange(50):
#    makeTrainingTestFile()
#    score, c = find_best_classifier.main()
#    x[c] += 1
#    scores.append(score)
#  print "Average: ", np.mean(scores)
#  print "Std: ", np.std(scores)
#  import operator
#  sorted_x = sorted(x.iteritems(), key=operator.itemgetter(1), reverse=True)
#  print "Best: ", sorted_x[0]
#  print "All: ", sorted_x
#  generateInfoFilesWrapper()

