from Header import *
# from PrepareDataset import *
from Model import *
import time
import math
import matplotlib.pyplot as plt


lower_sign=np.array([59,168,71])
upper_sign=np.array([128,255,255])


def filter(image, kernel_er, kernel_di):

	kernel_1 = np.ones((kernel_er,kernel_er),np.uint8)
	kernel_2 = np.ones((kernel_di,kernel_di),np.uint8)
	image = cv2.dilate(image,kernel_1,iterations = 1)
	image = cv2.erode(image,kernel_2,iterations = 1)
	return image

def Probability(z):
	z_exp=[math.exp(i) for i in z]
	softmax=[i/sum(z_exp) for i in z_exp]
	return max(softmax)

def DetectSign(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower_sign, upper_sign)
	mask=filter(mask,3,1)

	_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if len(contours) == 0:
		print('NO SIGN')
		return 0
	else:
		for idx in range(len(contours)):
			mask_fill = cv2.fillPoly(mask, pts =[contours[idx]], color=255)
		cv2.imshow('mask_fill',mask_fill)
		_, contours, hierarchy = cv2.findContours(mask_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		for idx in range(len(contours)):
			x, y, w, h = cv2.boundingRect(contours[idx])
			if w*h<150:
				continue
				print('NO SIGN')
			else:
				print('Area : ' , w*h)
				sign=img[y:y+h, x:x+w].copy()
				cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
				continue
		cv2.imshow('img ', cv2.resize(img, (0,0), fx=2, fy=2))
		cv2.imshow('sign', sign)
		return sign


model = Alex()
serializers.load_npz('D:\\Project\\CuocDuaSo2018\\Traffic_Signs_Detection_And_Recognition\\Result\\model_epoch-35', model) #load file resule





path='D:\\Project\\CuocDuaSo2018\\Control\\Frame\\163.jpg'
img=cv2.imread(path)
cv2.imshow('img',img)

DetectSign(img)

# start=time.time()
img=img.astype(np.float64)
image = cv2.resize(img,(48, 48))
image=np.rollaxis(image, 2, -3)
image *= (1.0 / 255.0)
image=np.asarray(image,dtype=np.float32)
# start=time.time()
y = model(image[None, ...])
result= y.data.argmax(axis=1)[0]
if result==0:
	print('LEFT')
if result ==1:
	print('RIGHT')
# print('Time predict 1 image: ', time.time()-start)
print('label image ', result)
print('Probability: ',Probability(y.data[0]))
cv2.waitKey()


