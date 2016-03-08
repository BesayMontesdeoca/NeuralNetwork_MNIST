# -*- coding: utf-8 -*-
import Layer
import LoadMnist
import Graphics
import numpy as np

# Método que para el vector pasado por parámetro(que se correspondería con el vector salida de alguna capa)
# calcula la función sigmoide de cada uno de sus elementos.
def sigmoid(neuralOut):
    for i in range(neuralOut.shape[0]):
        neuralOut[i] = 1/(1+np.exp(-neuralOut[i]))
    return neuralOut

# Método que calcula el error cuadrático entre los dos vectores pasados por parámetro
def MSE(pry, trY):
    return np.sum((trY-pry)**2)/trY.shape[0]

# Método donde tiene lugar la actulización de los pesos de la red, primero calculando los respectivos
# deltas de cada capa
def update(layerH, layerO, trX, trY, h, pyx, alpha):
    deltaW_o = pyx*(1-pyx)*(trY-pyx)
    deltaW_h = h*(1-h)*np.dot(deltaW_o,layerO.getWeight())
    layerH.setWeight(layerH.getWeight() + alpha*np.dot(np.transpose(deltaW_h),np.array([trX])))
    layerO.setWeight(layerO.getWeight() + alpha*np.dot(np.transpose(deltaW_o),h))
    layerH.setBias(layerH.getBias() + alpha*deltaW_h)
    layerO.setBias(layerO.getBias() + alpha*deltaW_o)

# Método donde se produce la propagación de la información hacia delante, en definitiva, donde se
# genera la salida de la red
def model(X, layerH, layerO):
    h = sigmoid(np.dot(X, np.transpose(layerH.getWeight()))+layerH.getBias())
    pyx = sigmoid(np.dot(h, np.transpose(layerO.getWeight()))+layerO.getBias())
    return pyx, h

# Método que entrena la red un número de veces determinada.
def train(nTrain, trX, trY, layerH, layerO):
    trainCost = []
    hitPercentage = []
    for n in range(nTrain):
        cost = 0
        for i in range(trX.shape[0]):
            [py_x, h] = model(trX[i], layerH, layerO)
            cost += MSE(py_x, trY[i])
            update(layerH, layerO, trX[i], trY[i], h, py_x, 0.1)
        trainCost.append(cost)
        print("Train: {}\nCost: {}".format(n, cost))
        hitPercentage.append(test(teX, teY, layerH, layerO))
        print ""
    return trainCost, hitPercentage

# Método que compara las salida de la red con las etiquetas para calcular
# la calidad de la red y obtener su tasa de acierto
def test(teX, teY, layerH, layerO):
    cont = 0
    for i in range(teX.shape[0]):
        [prx, h] = model(teX[i], layerH, layerO)
        # print("{} -- {}".format(np.argmax(prx), np.argmax(teY[i])))
        if(np.argmax(prx) == np.argmax(teY[i])):
            cont += 1
    hitPercentage = (cont/float(teX.shape[0]))*float(100)
    print("Percentage of hits: {} %".format(hitPercentage))
    return hitPercentage


print "Loading Mnist data ..."
[trX, teX, trY, teY] = LoadMnist.mnist(60000,10000)
print "Done\nTraining network ..."
nTrain = 10
layerH = Layer.Layer(55, 784)
layerO = Layer.Layer(10, 55)
print("Number of Trains: {}".format(nTrain))
[trainCost, hitPercentage] = train(nTrain, trX, trY, layerH, layerO)
Graphics.costHitsGraphica(trainCost, hitPercentage)
