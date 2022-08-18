import cv2
import numpy
# from pykeyboard import PyKeyboard
#
# k = PyKeyboard()
#
# def diffImg(t0, t1, t2):
#   d1 = cv2.absdiff(t2, t1)
#   d2 = cv2.absdiff(t1, t0)
#   return cv2.bitwise_and(d1, d2)
#
#
# cam = cv2.VideoCapture(0)
# ret = cam.set(3,320)
# ret = cam.set(4,240)
# winName = "Movement Indicator"
# cv2.namedWindow(winName, cv2.CV_WINDOW_AUTOSIZE)
#
# # Read three images first:
# #cam.read()[1]			Required for Windows/Older laptops
# #cam.read()[1]			Required for Windows/Older laptops
# t_minus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
# t = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
# t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
# change1 = [0,0]
#
# while True:
#   cv2.imshow( winName, diffImg(t_minus, t, t_plus) )
#
#   # Read next image
#   t_minus = t
#   t = t_plus
#   t_plus = cv2.cvtColor(cam.read()[1], cv2.COLOR_RGB2GRAY)
#   image1 = numpy.array(diffImg(t_minus, t, t_plus))
#   left = 0
#   right = 0
#
#   right = cv2.countNonZero(image1[:,:119])
#   left = cv2.countNonZero(image1[:,179:])
#   #
#
#
#   if right>15000:
#       change1[0] = 1
#       print "right"
#       if change1[1] == 1:
#           prev = [0,1]
#   if left>19000 and right<14900:
#       change1[1] = 1
#       print "left"
#       if change1[0] == 1:
#           prev = [1,0]
#   if change1[0] == 1 and change1[1] == 1:
#       if prev == [0,1]:
#           print "left to right"
#           k.tap_key("Left")			#k.left_key for Windows
#       elif prev == [1,0]:
#           print "right to left"
#           k.tap_key("Right")		#k.right_key for Windows
#       change1 = [0,0]
#
#   key = cv2.waitKey(10)
#   if key == 27:
#     cv2.destroyWindow(winName)
#     break
#
# print "Goodbye"
