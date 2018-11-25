from Header import *
class MLP(Chain):     #first Model
    def __init__(self, n_mid_units=100, n_out=256):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
class VGG16Net(chainer.Chain):
    def __init__(self, train=True):
        super(VGG16Net, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 64, 3, stride=1, pad=1)
            self.conv2=L.Convolution2D(None, 64, 3, stride=1, pad=1)

            self.conv3=L.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv4=L.Convolution2D(None, 128, 3, stride=1, pad=1)

            self.conv5=L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv6=L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv7=L.Convolution2D(None, 256, 3, stride=1, pad=1)

            self.conv8=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv9=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv10=L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.conv11=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv12=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv13=L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.fc14=L.Linear(None, 4096)
            self.fc15=L.Linear(None, 4096)
            self.fc16=L.Linear(None, 4)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 2, stride=2)

        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv4(h))), 2, stride=2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv7(h))), 2, stride=2)

        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv10(h))), 2, stride=2)

        h = F.relu(self.conv11(h))
        h = F.relu(self.conv12(h))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv13(h))), 2, stride=2)

        h = F.dropout(F.relu(self.fc14(h)))
        h = F.dropout(F.relu(self.fc15(h)))
        h = self.fc16(h)

        return h
class VGG19Net(chainer.Chain):
    def __init__(self, train=True):
        super(VGG19Net, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 64, 3, stride=1, pad=1)
            self.conv2=L.Convolution2D(None, 64, 3, stride=1, pad=1)

            self.conv3=L.Convolution2D(None, 128, 3, stride=1, pad=1)
            self.conv4=L.Convolution2D(None, 128, 3, stride=1, pad=1)

            self.conv5=L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv6=L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv7=L.Convolution2D(None, 256, 3, stride=1, pad=1)
            self.conv8=L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.conv9=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv10=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv11=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv12=L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.conv13=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv14=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv15=L.Convolution2D(None, 512, 3, stride=1, pad=1)
            self.conv16=L.Convolution2D(None, 512, 3, stride=1, pad=1)

            self.fc17=L.Linear(None, 4096)
            self.fc18=L.Linear(None, 4096)
            self.fc19=L.Linear(None, 4)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 2, stride=2)

        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv4(h))), 2, stride=2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv8(h))), 2, stride=2)

        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.relu(self.conv11(h))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv12(h))), 2, stride=2)

        h = F.relu(self.conv13(h))
        h = F.relu(self.conv14(h))
        h = F.relu(self.conv15(h))
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv16(h))), 2, stride=2)

        h = F.dropout(F.relu(self.fc17(h)))
        h = F.dropout(F.relu(self.fc18(h)))
        h = self.fc19(h)

        return h
class LeNet(chainer.Chain):
    def __init__(self, train=True):
        super(LeNet, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None, 6, 5, stride=1)
            self.conv2=L.Convolution2D(None, 16, 5, stride=1)
            self.fc3=L.Linear(None, 120)
            self.fc4=L.Linear(None, 64)
            self.fc5=L.Linear(None, 4)

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.sigmoid(self.conv1(x))), 2, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.sigmoid(self.conv2(h))), 2, stride=2)
        h = F.sigmoid(self.fc3(h))
        h = F.sigmoid(self.fc4(h))
        h = self.fc5(h)

        return h
class LogisticRegressionModel(Chain):

    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        with self.init_scope():
            self.w = chainer.Parameter(initializer=chainer.initializers.Normal())
            self.w.initialize([3, 1])

    def __call__(self, x, t):
        # Call the loss function
        return self.loss(x, t)

    def predictor(self, x):
        # Predict given an input (a, b, 1)
        z = F.matmul(x, self.w)
        return 1. / (1. + F.exp(-z))

    def loss(self, x, t):
        # Compute the loss for a given input (a, b, 1) and target
        y = self.predictor(x)
        loss = -t * F.log(y) - (1 - t) * F.log(1 - y)
        reporter_module.report({'loss': loss.data[0, 0]}, self)
        reporter_module.report({'w': self.w[0, 0]}, self)
        return loss
class Alex(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        super(Alex, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  96, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256,  5, pad=2)
            self.conv3 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv4 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv5 = L.Convolution2D(None, 256,  3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 3)

    def __call__ (self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        return h
