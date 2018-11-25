from Header import *

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def crop_center(img,cropx,cropy):
    y,x,z = img.shape  #z: RGB z=3
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)   
    return img[starty:starty+cropy,startx:startx+cropx]

def SaveImgAfterCrop(img,Address):
    folderSaveImg='D:\\DetectsBrokenObject\\Dataset\\' 
    img=crop_center(img,512,512)
    if 'TrainingDataSetScale\\DAMAGE' in Address :
        folderSaveImg=folderSaveImg + 'TrainingDataSetScale\\DAMAGE\\'
    elif 'TrainingDataSetScale\\NORMAL' in Address:
        folderSaveImg=folderSaveImg + 'TrainingDataSetScale\\NORMAL\\'
    elif 'EvaluationDataSetScale\\DAMAGE' in Address:
        folderSaveImg=folderSaveImg + 'EvaluationDataSetScale\\DAMAGE\\'
    elif 'EvaluationDataSetScale\\NORMAL' in Address:
        folderSaveImg=folderSaveImg + 'EvaluationDataSetScale\\NORMAL\\'
    elif 'TestingDataSetScale\\DAMAGE' in Address:
        folderSaveImg=folderSaveImg + 'TestingDataSetScale\\DAMAGE\\'
    else: # TestingDataSetScale\\NORMAL
        folderSaveImg=folderSaveImg + 'TestingDataSetScale\\NORMAL\\'

    namefile='.jpg'  # Define file imgage output
    global k 
    k=k+1
    addr=  folderSaveImg + str(k) + namefile
    img = Image.fromarray(img)
    img.save(addr)

path="E:\\MyData\\Project\\AI_DocumentLayoutAnalysis\\Data_org\\Data2\\"
def Rename(path):
    i=192
    for filename in os.listdir(path): 
        dst =str(i) + ".jpg"
        src =path+ filename 
        dst ="E:\\MyData\\Project\\AI_DocumentLayoutAnalysis\\Data_org\\Data\\"+ dst 
          
        # # rename() function will 
        # # rename all the files 
        os.rename(src, dst) 
        i += 1
        print(dst)
# Rename(path)
  
def CreatFileCSV():
    k=0
    for Add in AddressDataset:
        for i in range(4):
            AddData=Add+'\\'+str(i)
            for root, dirs , files in os.walk(AddData):
                for file in files:
                    if file.endswith(".jpg"):
                        EvaluationDataSet=open('E:\\MyData\Project\\AI_DocumentLayoutAnalysis\\Dataset\\Evaluation.csv','a')
                        TrainingDataSet=open('E:\\MyData\Project\\AI_DocumentLayoutAnalysis\\Dataset\\Training.csv', 'a')
                        
                        path=os.path.join(root, file)
                        string = path +' '+str(i)
                        k=k+1
                        print(k,'   ',string)
                        if 'Evaluation' in string:
                            EvaluationDataSet.write(string)
                            EvaluationDataSet.write('\n')
                            EvaluationDataSet.close()
                        if 'Training' in string:
                            TrainingDataSet.write(string)
                            TrainingDataSet.write('\n')
                            TrainingDataSet.close()
def SplitingData():
    k=0
    for i in range(4): # i = 0 1 2 3 
        for root, dirs , files in os.walk(AddressDatasetORG[i]):
            for file in files:
                if file.endswith(".jpg"):
                    string=os.path.join(root, file)
                    k=k+1
                    if k%5==0 :
                        if k%5==0:
                            k=0
                        dst=AddressDataset[1] +'\\'+ str(root[len(root)-2])+'\\'+file
                    else:
                        dst=AddressDataset[0] +'\\'+ str(root[len(root)-2])+'\\'+file
                    copyfile(string, dst)

def ReadImage(): 
    k=1
    global TargetTrainImgList
    global TargetTestImgList
    global TargetValImgList

    for Add in AddressDataset:
        for i in range(4):
            AddData=Add+'\\'+str(i)
            for root, dirs , files in os.walk(AddData):
                for file in files:
                    if file.endswith(".jpg"):
                        string=os.path.join(root, file)
                        print(string)
                        
                        # CreatFileCSV(string)
                        
                    img=mpimg.imread(string)
                    print(k,'   img shape', img.shape)
                    k=k+1
                    img=rgb2gray(img)
                    img *=(1.0 / 255.0)
                    #SaveImgAfterCrop(img,AddressImgBeforeScale[i]) #If not done crop

                    img=np.asarray(img,dtype=np.float32)
                    #print('type(img):',img[0].dtype)
                    img=np.reshape(img,[1,512,512])

                    

                    if AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\TrainingDataSetScale\\DAMAGE':
                        TrainImgList.append(img)
                        TargetTrainImgList.append(0)
                    elif AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\TrainingDataSetScale\\NORMAL':
                        TrainImgList.append(img)
                        TargetTrainImgList.append(1)
                    elif AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\EvaluationDataSetScale\\DAMAGE':
                        ValImgList.append(img)
                        TargetValImgList.append(0)
                    elif AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\EvaluationDataSetScale\\NORMAL':
                        ValImgList.append(img)
                        TargetValImgList.append(1)
                    elif AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\TestingDataSetScale\\DAMAGE':
                        TestImgList.append(img)
                        TargetTestImgList.append(0)
                    else:
                        TestImgList.append(img)
                        TargetTestImgList.append(1)
                    # # no=gc.collect()
                    # print('no gc collect:',no)


    TargetTrainImgList = np.asarray(TargetTrainImgList,dtype=np.int32)
    TargetTestImgList = np.asarray(TargetTestImgList,dtype=np.int32)
    TargetValImgList = np.asarray(TargetValImgList,dtype=np.int32)\

    Mytrain=datasets.TupleDataset(TrainImgList, TargetTrainImgList)
    Myval=datasets.TupleDataset(ValImgList, TargetValImgList)
    Mytest=datasets.TupleDataset(TestImgList, TargetTestImgList)
    return Mytrain,Myval,Mytest

# SplitingData() 
# ReadImage()
# CreatFileCSV()

class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path):
        self.base = chainer.datasets.LabeledImageDataset(path)

    def __len__(self):
        return len(self.base)

    def get_example(self, i):  #    Train on image chanel=3
       
        # It reads the i-th image/label pair and return a preprocessed image.
        image, label = self.base[i]
        image=np.rollaxis(image, 0, 3)
        image = image[...,::-1] #BGR to RGB
        image = cv2.resize(image,(48, 48))  #(width,height)
        # print('height, width, channels' , image.shape)
        # exit()
        image=np.rollaxis(image, 2, -3) # input model chanel-height-width
        image *= (1.0 / 255.0) # Scale to [0, 1]
        image=np.asarray(image,dtype=np.float32)
        return image, label
    # def get_example(self, i): #chanel =1
       
    #     # It reads the i-th image/label pair and return a preprocessed image.
    #     image, label = self.base[i]

    #     image=np.rollaxis(image, 0, 3)
    #     # print(image.shape)
        
    #     # #image = cv2.resize(image,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_LINEAR)
    #     # image = cv2.resize(image,(140, 140), interpolation = cv2.INTER_AREA)
    #     # #image=crop_center(image,224,224)

    #     image=rgb2gray(image)
    #     # print(image.shape)
    #     w,h=image.shape
    #     # # print('w:  ', w , '    h: ', h)
    #     # # exit()
    #     image *= (1.0 / 255.0) # Scale to [0, 1]
        
    #     image=np.asarray(image,dtype=np.float32)
        
    #     image=np.reshape(image,[1,w,h])
    #     # print('type(img):',image[0].dtype)
    #     return image, label
