import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import sys
import getpass
import numpy as np
import os
import csv
from ANYexo_generate_train_dataset import dataSet


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = np.inf
        self.delta = delta

    def check(self, val_loss):
        # print("val loss: ", val_loss, " best: ", self.best_score)
        if val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stop")
                return True
        else:
            self.best_score = val_loss
            self.counter = 0
        return False


class mlp(nn.Module):
    def __init__(self, input_dim):
        super(mlp, self).__init__()

        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.softsign(self.fc1(x))
        x = F.softsign(self.fc2(x))
        x = F.softsign(self.fc3(x))
        x = self.fc4(x)
        return x


def train(data, parameters, logdir):
    if torch.cuda.is_available():
        dev = "cuda:0"
        print("GPU found")
    else:
        dev = "cpu"
    device = torch.device(dev)

    input_ml_np, target_ml_np, dataStatistics = data.createDataSet(parameters["hist_len"], parameters["hist_stride"])

    input_train, input_val, target_train, target_val = train_test_split(input_ml_np, target_ml_np)
    input_ml = torch.from_numpy(input_train.astype(np.float32)).to(device)
    target_ml = torch.from_numpy(target_train.astype(np.float32)).to(device)
    input_test = torch.from_numpy(input_val.astype(np.float32)).to(device)
    target_test = torch.from_numpy(target_val.astype(np.float32)).to(device)

    # training configuration
    nbFolds = parameters["nbFolds"]
    shuffling_seed = 0
    training_patience = parameters["training_patience"]
    batch_size = parameters["batch_size"]
    l2_reg = parameters["l2_reg"]
    num_epochs = parameters["num_epochs"]

    model = mlp(input_ml.size()[1]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2_reg)
    earlyStop = EarlyStopping(patience=training_patience*batch_size)

    for step in range(num_epochs):

        permutation = torch.randperm(input_ml.size()[0])

        for i in range(0, input_ml.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            x, y = input_ml[indices], target_ml[indices]
            yPred = model(x)
            loss = loss_fn(yPred, y)
            earlyStopFlag = earlyStop.check(loss.item())
            if earlyStopFlag:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if earlyStopFlag:
            torch.save(model.state_dict(), logdir + "/model")
            break
        if step % 1 == 0:
            print(step, loss.item())

    torch.save(model.state_dict(), logdir + "/model")
    mseLoss = nn.MSELoss()
    maeLoss = nn.L1Loss()
    yTrainPredict = model(input_ml)
    MSEtrain = mseLoss(yTrainPredict, target_ml).item()
    MAEtrain = maeLoss(yTrainPredict, target_ml).item()

    yTestPredict = model(input_test)
    MSEtest = mseLoss(yTestPredict, target_test).item()
    MAEtest = maeLoss(yTestPredict, target_test).item()

    # save training config log file
    print("logdir : ", logdir)
    fs = open(logdir + '/log.txt', 'w+')

    fs.write('strides and past timesteps\n\n')
    fs.write('hist len: ' + str(parameters["hist_len"]) + '\n')
    fs.write('hist stride: ' + str(parameters["hist_stride"]) + '\n\n')

    fs.write('shuffling seed: ' + str(shuffling_seed) + '\n')
    fs.write('training patience (val_los): ' + str(training_patience) + '\n')
    fs.write('training batch size: ' + str(batch_size) + '\n')
    fs.write('l2 regularization: ' + str(l2_reg) + '\n\n')
    fs.write('num epochs: ' + str(step))
    if earlyStopFlag:
        fs.write("  early stop")
    fs.write('\n\nFolds number for K-fold split: ' + str(nbFolds) + '\n\n')

    fs.write('inputs1 means:\n' + str(dataStatistics["input1Mean"]) + '\n')
    fs.write('inputs2 means:\n' + str(dataStatistics["input2Mean"]) + '\n')
    fs.write('inputs1 stds:\n' + str(dataStatistics["input1Std"]) + '\n')
    fs.write('inputs2 stds:\n' + str(dataStatistics["input2Std"]) + '\n')
    fs.write('target means:\n' + str(dataStatistics["targetMean"]) + '\n')
    fs.write('target stds:\n' + str(dataStatistics["targetStd"]) + '\n')

    fs.write('\n\nVALIDATION SET \t\t MSE:' + str(MSEtest) + '\t\t\tMAE: ' + str(MAEtest))

    fs.write('\n\nTRAINING SET \t\t MSE:' + str(MSEtrain) + '\t\t\tMAE: ' + str(MAEtrain))

    fs.write('\n\n\n-------------------------------\n\n')
    fs.write('\nModel Summary:\n\n')
    original_stdout = sys.stdout
    sys.stdout = fs
    print(model)
    sys.stdout = original_stdout

    fs.close()

    return MAEtrain, MSEtrain, MAEtest, MSEtest


def savePerformance(performance, path):
    sortedValMAE = sorted(performance, key=lambda a_entry: a_entry[2])
    sortedValMSE = sorted(performance, key=lambda a_entry: a_entry[3])
    with open(path + '/sortedAccodingToMAE.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['MAE Training', 'MSE Training', 'MAE Val', 'MSE Val', "Training Index"])
        writer.writerows(sortedValMAE)

    with open(path + '/sortedAccodingToMSE.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['MAE Training', 'MSE Training', 'MAE Val', 'MSE Val', "Training Index"])
        writer.writerows(sortedValMSE)


if __name__ == '__main__':
    # savePath = "/home/" + getpass.getuser() + "/master_thesis/skid_steer/actuator_model/actuator_model_speed_trainings/20210729_wholeData"
    # datasetPath =  "/home/" + getpass.getuser() + "/master_thesis/skid_steer/data/dataSet.pkl"

    # savePath = "/home/" + getpass.getuser() + "/Documents/Anyexo/ANYexo_training_data/traineddata/HWTest12.23/GED"
    # datasetPath = "/home/" + getpass.getuser() + "/Documents/Anyexo/ANYexo_training_data/untraineddata/HWTest12.23/400HzGED/dataSet.pkl"
    savePath = "/home/" + getpass.getuser() + "/Documents/Anyexo/ANYexo_training_data/traineddata/5258/GHC"
    datasetPath = "/home/" + getpass.getuser() + "/Documents/Anyexo/ANYexo_training_data/untraineddata/5258/400HzGHC/dataSet.pkl"

    data = dataSet(datasetPath)

    histList = [1]
    striList = [16]
    l2CoefList = [0.001]#tune it to zero

    performance = []
    index = 0.001
    totalIter = len(histList)*len(striList)*len(l2CoefList)
    params = []

    for histLen, histStride, l2Coef in zip(histList, striList, l2CoefList):
        params.append([histLen, histStride, l2Coef, index])
        print("index ", index, " / ", totalIter)
        parameters = {"hist_len": histLen,  "hist_stride": histStride,
                      "training_patience": 100000000, "batch_size": 1000, "l2_reg": l2Coef, "nbFolds": 10,
                      "num_epochs": 100, "saving_period": 500}

        logdir = savePath + "/trained_models/" + str(index) +'BS1000_numepo100_stride16_histlen1_400Hz'
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        maeTrainAvg, mseTrainAvg, maeValAvg, mseValAvg = train(data, parameters, logdir)
        performance.append([maeTrainAvg, mseTrainAvg, maeValAvg, mseValAvg, index])
        savePerformance(performance, savePath)
        index += 1

    with open(savePath + '/grid.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['hist_len', 'hist_stride', 'l2Coef', "Training Index"])
        writer.writerows(params)
