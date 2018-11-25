from PrepareDataset import *
from Model import *
from Header import *
import Resnet50
import densenet
train = PreprocessedDataset(AddressDatasetCSV[0])
val = PreprocessedDataset(AddressDatasetCSV[1])

batchsize=64

train_iter=iterators.SerialIterator(train,batchsize)
test_iter=iterators.SerialIterator(val,batchsize,False,False)



#train_iter = chainer.iterators.MultiprocessIterator(train, batchsize)
#test_iter = chainer.iterators.MultiprocessIterator(val, batchsize, repeat=False,shuffle=False )

#model = resnet50.ResNeXt50() 
#model=Alexnet() ## Don't working on Gray image

model=Alex()
model=L.Classifier(model)
gpu_id = 0  # Set to -1 if you use CPU
if gpu_id >= 0:
    model.to_gpu(gpu_id)  # If you use CPU, comment out this line
max_epoch=500


optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)#
# 0.0001 -> 0.001 -> 0.1
# optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)

optimizer.setup(model)

updater=training.updaters.StandardUpdater(train_iter,optimizer,device=gpu_id)
# trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='D:\\HocTap\\3rdYear\\MachineLearningInComputerVision\\Caltech256\\Result_Caltech256')
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='drive/My Drive/GG_Colab/Traffic_Signs_Recognition/Result') #save result model
trainer.extend(extensions.LogReport()) #save loss and accuracy
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}')) #save snapshot epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id)) 
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy','validation/main/loss', 'validation/main/accuracy', 'elapsed_time'])) # Print in the screen
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png')) # Draw loss.png and save in dir
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png')) # Draw loss.png and save in dir
trainer.extend(extensions.dump_graph('main/loss'))
# chainer.serializers.load_npz('drive/My Drive/GG_Colab/Traffic_Signs_Recognition/Result/snapshot_epoch-9', trainer)
print('Training')
trainer.run()


