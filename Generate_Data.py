import numpy as np
import os
import cv2
from shutil import move
def GenerateData(Images):
	k=-1
	for root, dirs, files in os.walk(Images):
		if k==-1:
			k=k+1
			continue
		listImage = os.listdir(root)
		folder=root.split('\\')
		Image_Dir='.\\Evaluation\\'+folder[-1]
		if not os.path.exists(Image_Dir): os.makedirs(Image_Dir)
		i = 0
		while i <= len(listImage)-5:
			move(root+'\\'+ listImage[i], Image_Dir+'\\'+listImage[i])
			i += 5

def CreateCSV(Data):
	for root, dirs , files in os.walk(Data):
                for file in files:
                    EvaluationDataSet=open(Data+'Evaluation.csv','a')
                    TrainingDataSet=open(Data+'Training.csv', 'a')
                    
                    path=os.path.join(root, file)
                    
                    k=path.split('\\')
                    k=int(str(k[-1][:3]))-1
                    string = path +' '+str(k)
                    print(k,'   ',string)
                    if 'Evaluation' in string:
                        EvaluationDataSet.write(string)
                        EvaluationDataSet.write('\n')
                        EvaluationDataSet.close()
                    if 'Training' in string:
                        TrainingDataSet.write(string)
                        TrainingDataSet.write('\n')
                        TrainingDataSet.close()

def ResizeImage(Data):
    for root, dirs , files in os.walk(Data):
                for file in files:
                    path=os.path.join(root, file)
                    img=cv2.imread(path)
                    img = cv2.resize(img,(200, 200), interpolation = cv2.INTER_AREA)
                    cv2.imwrite(path,img)
                    print(path)

Images="D:\\HocTap\\3rdYear\\MachineLearningInComputerVision\\Caltech256\\Images\\"
Data="D:\\HocTap\\3rdYear\\MachineLearningInComputerVision\\Caltech256\\Data\\"
ResizeImage(Data)
# GenerateData(Images)
