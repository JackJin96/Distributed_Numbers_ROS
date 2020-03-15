from multiprocessing import Process
import sys
import datetime

starttime = datetime.datetime.now()
rocket = 0

def func1():
  global rocket
  print 'start func1'
  while rocket < sys.maxint:
    rocket += 1
    print "func1: %d" % rocket
    endtime = datetime.datetime.now()
    if endtime - starttime >= datetime.timedelta(seconds = 0.1):
      break
  print 'end func1'

def func2():
  global rocket
  print 'start func2'
  while rocket < sys.maxint:
    rocket += 1
    print "func2: %d" % rocket
    endtime = datetime.datetime.now()
    if endtime - starttime >= datetime.timedelta(seconds = 0.1):
      break
  print 'end func2'

if __name__=='__main__':
  p1 = Process(target = func1)
  p1.start()
  p2 = Process(target = func2)
  p2.start()
  func1()
  func2()