from os import listdir
from os.path import isfile, join
import random

from PIL import Image
import numpy as np
import os

def load_image(filename):
    im_frame = Image.open(filename, 'r')
    pixels = np.array(im_frame.getdata())
    pixels = pixels[:, -1]
    return pixels / 255

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def save(filename, array):
    if os.path.isfile("model/" + filename + ".npy"):
        os.remove("model/" + filename + ".npy")
    np.save('model/' + filename, array)

def load(filename):
    if os.path.isfile("model/" + filename + ".npy"):
        result = np.load('model/' + filename + ".npy")
        return result
    return None

class NNetwork:
    def __init__(self):
        self.count = 0
        self.images = None

        self.digit = None
        self.expected = None
        self.inputAct = None
        self.inputConn = None
        self.inputConnGrad = None
        self.inputBias = None
        self.inputBiasGrad = None

        self.middleAct = None
        self.middleConn = None
        self.middleConnGrad = None
        self.middleBias = None
        self.middleBiasGrad = None
        self.middleDelta = None

        self.outputAct = None
        self.outputConn = None
        self.outputConnGrad = None
        self.outputBias = None
        self.outputBiasGrad = None
        self.outputDelta = None

        self.expected = None

    def loadimgs(self):
        self.images = []
        for digit in range(10):
            imgfiles = [f for f in listdir(join('data', str(digit))) if isfile(join('data', str(digit), f))]
            for file in imgfiles:
                data = load_image(join('data', str(digit), file))
                self.images.append((digit, data))
            print(str(len(imgfiles)) + " loaded for digit " + str(digit))
        random.shuffle(self.images)
        print(str(len(self.images)) + " images loaded")

    def initprocdata(self):
        self.count = 0
        self.inputConnGrad = np.zeros((16, 28*28))

        self.middleDelta = np.zeros((3, 16))
        self.middleConnGrad = np.zeros((2, 16, 16))
        self.middleBiasGrad = np.zeros((3, 16))

        self.outputDelta = np.zeros((10))
        self.outputConnGrad = np.zeros((10, 16))
        self.outputBiasGrad = np.zeros((10))

    def initparams(self):
        precision = 1000.0

        self.inputConn = np.random.randint(-precision, precision, size=(16, 28*28)) / precision

        self.middleAct = np.zeros((3, 16))
        self.middleConn = np.random.randint(-precision, precision, size=(2, 16, 16)) / precision
        self.middleBias = np.random.randint(-precision, precision, size=(3, 16)) / precision

        self.outputAct = np.zeros((10))
        self.outputConn = np.random.randint(-precision, precision, size=(10, 16)) / precision
        self.outputBias = np.random.randint(-precision, precision, size=(10)) / precision

    def train(self):
        percent = 0
        rate = 0.001
        while percent < 1.:
            self.initprocdata()
            allCost = 0
            for img in self.images:
                self.run(img)
                self.delta()
                self.gradient()
                allCost += self.cost()
            self.update(rate)
            percent += rate
            cost = allCost / len(self.images)
            print(str(round(percent * 100)) + "%, cost: " + str(cost))

    def run(self, img):
        self.inputAct = img[1]
        self.digit = img[0]
        self.expected = np.zeros((10))
        self.expected[self.digit] = 1.

        self.middleAct[0] = sigmoid(np.matmul(self.inputConn, self.inputAct) + self.middleBias[0])
        self.middleAct[1] = sigmoid(np.matmul(self.middleConn[0], self.middleAct[0]) + self.middleBias[1])
        self.middleAct[2] = sigmoid(np.matmul(self.middleConn[1], self.middleAct[1]) + self.middleBias[2])
        self.outputAct = sigmoid(np.matmul(self.outputConn, self.middleAct[2]) + self.outputBias)

    def cost(self):
        return np.sum(np.square(self.expected - self.outputAct))

    def delta(self):
        self.outputDelta = 2 * (self.expected - self.outputAct) * self.outputAct * (1 - self.outputAct)
        self.middleDelta[2] = np.matmul(np.transpose(self.outputConn), self.outputDelta) * self.middleAct[2] * (1 - self.middleAct[2])
        self.middleDelta[1] = np.matmul(np.transpose(self.middleConn[1]), self.middleDelta[2]) * self.middleAct[1] * (1 - self.middleAct[1])
        self.middleDelta[0] = np.matmul(np.transpose(self.middleConn[0]), self.middleDelta[1]) * self.middleAct[0] * (1 - self.middleAct[0])

    def gradient(self):
        self.outputConnGrad = self.outputConnGrad + np.outer(self.outputDelta, self.middleAct[2])
        self.middleConnGrad[1] = self.middleConnGrad[1] + np.outer(self.middleDelta[2], self.middleAct[1])
        self.middleConnGrad[0] = self.middleConnGrad[0] + np.outer(self.middleDelta[1], self.middleAct[0])
        self.inputConnGrad = self.inputConnGrad + np.outer(self.middleDelta[0], self.inputAct)

        self.outputBiasGrad = self.outputBiasGrad + self.outputDelta
        self.middleBiasGrad = self.middleBiasGrad + self.middleDelta
        self.count = self.count + 1

    def update(self, rate):
        self.outputConn = self.outputConn + (rate * (self.outputConnGrad / self.count))
        self.outputBias = self.outputBias + (rate * (self.outputBiasGrad / self.count))

        self.middleConn = self.middleConn + (rate * (self.middleConnGrad / self.count))
        self.middleBias = self.middleBias + (rate * (self.middleBiasGrad / self.count))

        self.inputConn = self.inputConn + (rate * (self.inputConnGrad / self.count))


    def save(self):
        save("inputConn", self.inputConn)

        save("middleConn", self.middleConn)
        save("middleBias", self.middleBias)

        save("outputConn", self.outputConn)
        save("outputBias", self.outputBias)

    def load(self):
        self.inputConn = load("inputConn")
        self.inputConnGrad = np.zeros((16, 28 * 28))

        self.middleAct = np.zeros((3, 16))
        self.middleConn = load("middleConn")
        self.middleConnGrad = np.zeros((2, 16, 16))
        self.middleBias = load("middleBias")
        self.middleBiasGrad = np.zeros((3, 16))
        self.middleDelta = np.zeros((3, 16))

        self.outputAct = np.zeros((10))
        self.outputConn = load("outputConn")
        self.outputConnGrad = np.zeros((10, 16))
        self.outputBias = load("outputBias")
        self.outputBiasGrad = np.zeros((10))
        self.outputDelta = np.zeros((10))

nn = NNetwork()
nn.loadimgs()
nn.initparams()
nn.train()
nn.save()

nn.load()
nn.run(nn.images[0])
print(nn.digit)
print(nn.outputAct)
print(nn.cost())