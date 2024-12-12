# -*- coding: utf-8 -*-

import os
import sys
import argparse
from math import degrees, radians, cos, sin

#import geojson
import numpy as np
from datetime import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.append('../')
from utils.dataset import read_data
from utils.dataset import MinMaxNormObj
from utils.SAGCN import SAGCN
from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
import time
from torch.utils.tensorboard import SummaryWriter
# from math import cos, sin, atan2, sqrt, pi ,radians, degrees
# from geopy.distance import geodesic

torch.manual_seed(22)

parse = argparse.ArgumentParser()
parse.add_argument('-cuda', type=int, default=1)
parse.add_argument('-traffic', type=str, default='internet')
parse.add_argument('-batch_size', type=int, default=16, help='batch size')
parse.add_argument('-epoch_size', type=int, default=3, help='epochs')
parse.add_argument('-close_size', type=int, default=25)  # *******
parse.add_argument('-predict_size', type=int, default=1)
parse.add_argument('-nb_flow', type=int, default=1)
parse.add_argument('-height', type=int, default=100)
parse.add_argument('-width', type=int, default=100)
parse.add_argument('-meta', type=int, default=0)
parse.add_argument('-cross', type=int, default=0)
parse.add_argument('-cluster', type=int, default=3)  # default-3
parse.add_argument('-loss', type=str, default='l1', help='l1 | l2')
parse.add_argument('-lr', type=float, default=1e-3)
parse.add_argument('-weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parse.add_argument('-rows', nargs='+', type=int, default=[40, 45])
parse.add_argument('-cols', nargs='+', type=int, default=[40, 45])
parse.add_argument('-test_row', type=int, default=10, help='test row')
parse.add_argument('-test_col', type=int, default=18, help='test col')
parse.add_argument('-last_kernel', type=int, default=1)
parse.add_argument('-period_size', type=int, default=0)
parse.add_argument('-trend_size', type=int, default=0)
parse.add_argument('-test_size', type=int, default=24 * 7)
parse.add_argument('-fusion', type=int, default=1)
parse.add_argument('-transfer', type=int, default=0)
parse.add_argument('-crop', dest='crop', action='store_true')
parse.add_argument('-no-crop', dest='crop', action='store_false')
parse.set_defaults(crop=True)
parse.add_argument('-train', dest='train', action='store_true')
parse.add_argument('-no-train', dest='train', action='store_false')
parse.set_defaults(train=True)
parse.add_argument('-l2', dest='l2', help='weight decay', type=float, default=1e-4)
parse.add_argument('-adam', dest='adam', help='use adam. Not recommended', action='store_true')
parse.add_argument('-save_dir', type=str, default='results')
parse.add_argument('-server', type=int, default=1) #0为本地路径，1为服务器路径
#0表示不使用gcn,1表示空间transform替换为gcn，2表示加权gcn和空间transformere,3表示gcn+空间transform
parse.add_argument('-gcn_flag', type=int, default=3)




opt = parse.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt.save_dir = '{}/{}'.format(opt.save_dir, opt.traffic)
path_name = 'results_data'
if not os.path.exists(path_name):
    os.makedirs(path_name)
else:
    print('path already exists.')
file_name = "GLSTTN"
hdf5_file = os.path.join(path_name, "%s1207.h5" % file_name)


def get_optim(lr):
    if opt.adam:
        optimizer = optim.Adam(model.parameters(), weight_decay=opt.l2, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(model.parameters(), weight_decay=opt.l2, lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[0.5 * opt.epoch_size, 0.75 * opt.epoch_size ],
                                                     gamma=0.1)
    return optimizer, scheduler

def log(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
        print("mkdir -p " + os.path.dirname(fname))
        os.system("mkdir -p " + os.path.dirname(fname))
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()

EPOCH_NUM = 0
def train_epoch(data_type='train'):
    total_loss = 0

    if data_type == 'train':
        model.train()
        data = train_loader
    if data_type == 'valid':
        model.eval()
        data = valid_loader
    if (opt.close_size > 0) & (opt.meta == 1) & (opt.cross == 1):
        for idx, (c, meta, cross, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            x = c.float().to(device)
            meta = meta.float().to(device)
            cross = cross.float().to(device)
            target_var = target.float().to(device)
            pred= model(x, meta=meta, cross=cross)
            pred = pred[:, 0, :, :].unsqueeze(1)
            loss = criterion(pred, target_var)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    elif (opt.close_size > 0) & (opt.meta == 1):
        for idx, (x, meta, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            x = x.float().to(device)
            meta = meta.float().to(device)
            target_var = target.float().to(device)
            pred = model(x, meta=meta)
            pred = pred[:, 0, :, :].unsqueeze(1)
            loss = criterion(pred, target_var)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    elif (opt.close_size > 0) & (opt.cross == 1):
        for idx, (x, cross, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            x = x.float().to(device)
            cross = cross.float().to(device)
            target_var = target.float().to(device)
            pred = model(x, cross=cross)
            pred = pred[:, 0, :, :].unsqueeze(1)
            loss = criterion(pred, target_var)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    elif opt.close_size > 0:
        start = time.time()
        for idx, (batch, target) in enumerate(data):
            optimizer.zero_grad()
            model.zero_grad()
            x = batch.float().to(device)
            y = target.float().to(device)
            pred= model(x)
            pred = pred[:, 0, :, :].unsqueeze(1)
            loss = criterion(pred, y)
            total_loss += loss.item()
            if data_type == 'train':
                loss.backward()
                optimizer.step()
        if data_type == 'train':
            time_per_EPOCH = (time.time() - start)
            print("{:.2f}s/epoch for training, {:.2f}m/epoch for training".format(time_per_EPOCH, time_per_EPOCH / 60))
            start = time.time()
    return total_loss


def train():
    os.system("mkdir -p " + opt.save_dir)
    best_valid_loss = 10
    train_loss, valid_loss = [], []
    for i in range(opt.epoch_size):
        # scheduler.step()
        train_loss.append(train_epoch('train'))
        valid_loss.append(train_epoch('valid'))
        scheduler.step()

        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]
            torch.save({'epoch': i, 'model': model, 'train_loss': train_loss,
                        'valid_loss': valid_loss}, opt.model_filename + '1207.model')
            torch.save(optimizer, opt.model_filename + '1207.optim')
        torch.save(model.state_dict(), opt.model_filename + '1207_' + str(i) + '.pt')
        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                      'best_valid_loss: {:0.6f}, lr: {:0.5f}').format((i + 1), opt.epoch_size,
                                                                      train_loss[-1],
                                                                      valid_loss[-1],
                                                                      best_valid_loss,
                                                                      opt.lr)
        sw.add_scalar('training_loss', train_loss[-1], i + 1)
        # state_dict = model.state_dict()
        # fs0_weight = state_dict['transformer.transformer.spatial_layers.0.fs.weight']
        # fs1_weight = state_dict['transformer.transformer.spatial_layers.1.fs.weight']
        # fs2_weight = state_dict['transformer.transformer.spatial_layers.2.fs.weight']
        # fg0_weight = state_dict['transformer.transformer.spatial_layers.0.fg.weight']
        # fg1_weight = state_dict['transformer.transformer.spatial_layers.1.fg.weight']
        # fg2_weight = state_dict['transformer.transformer.spatial_layers.2.fg.weight']
        # sw.add_histogram(tag='model_fs0_weight',values= fs0_weight,global_step=i)
        # sw.add_histogram(tag='model_fs1_weight',values= fs1_weight,global_step=i)
        # sw.add_histogram(tag='model_fs2_weight',values= fs2_weight,global_step=i)
        #
        # sw.add_histogram(tag='model_fg0_weight',values= fg0_weight,global_step=i)
        # sw.add_histogram(tag='model_fg1_weight',values= fg1_weight,global_step=i)
        # sw.add_histogram(tag='model_fg2_weight',values= fg2_weight,global_step=i)

        if i % 2 == 0:
            print(log_string)
        log(opt.model_filename + '1207.log', log_string)


def predict(test_type='train'):
    predictions = []
    ground_truth = []
    loss = []
    model.eval()
    model.load_state_dict(torch.load(opt.model_filename + '1207_'+ str(opt.epoch_size-1) + '.pt'))
    if test_type == 'train':
        data = train_loader
    elif test_type == 'test':
        data = test_loader
    elif test_type == 'valid':
        data = valid_loader
    with torch.no_grad():
        if (opt.close_size > 0) & (opt.meta == 1) & (opt.cross == 1):
            for idx, (c, meta, cross, target) in enumerate(data):
                optimizer.zero_grad()
                model.zero_grad()
                x = c.float().to(device)
                meta = meta.float().to(device)
                cross = cross.float().to(device)
                target_var = target.float().to(device)
                pred = model(x, meta=meta, cross=cross)
                pred = pred[:, 0, :, :].unsqueeze(1)
                predictions.append(pred.data.cpu())
                ground_truth.append(target.data)

                loss.append(criterion(pred, target_var).item())
        elif (opt.close_size > 0) & (opt.meta == 1):
            for idx, (x, meta, target) in enumerate(data):
                optimizer.zero_grad()
                model.zero_grad()
                x = x.float().to(device)
                meta = meta.float().to(device)
                y = target.float().to(device)
                pred = model(x, meta=meta)
                pred = pred[:, 0, :, :].unsqueeze(1)
                predictions.append(pred.data.cpu())
                ground_truth.append(target.data)
                loss.append(criterion(pred, y).item())
        elif (opt.close_size > 0) & (opt.cross == 1):
            for idx, (x, cross, target) in enumerate(data):
                optimizer.zero_grad()
                model.zero_grad()
                x = x.float().to(device)
                cross = cross.float().to(device)
                y = target.float().to(device)
                pred = model(x, cross=cross)
                pred = pred[:, 0, :, :].unsqueeze(1)
                predictions.append(pred.data.cpu())
                ground_truth.append(target.data)
                loss.append(criterion(pred, y).item())
        elif opt.close_size > 0:
            for idx, (c, target) in enumerate(data):
                optimizer.zero_grad()
                model.zero_grad()
                x = c.float().to(device)
                y = target.float().to(device)
                pred = model(x)
                pred = pred[:, 0, :, :].unsqueeze(1)
                predictions.append(pred.data.cpu())
                ground_truth.append(target.data)
                loss.append(criterion(pred, y).item())

    final_predict = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)
    print(
        "Shape of final prediction is {}, shape of ground truth is {}".format(final_predict.shape, ground_truth.shape))
    ground_truth = mmn.inverse_transform(ground_truth)
    final_predict = mmn.inverse_transform(final_predict)
    return final_predict, ground_truth


def train_valid_split(dataloader, test_size=0.2, shuffle=True, random_seed=0):
    length = len(dataloader)
    indices = list(range(0, length))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    if type(test_size) is float:
        split = int(np.floor(test_size * length))
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or float'.format(str))
    return indices[split:], indices[:split]
# def center_geolocation(geolocations):
#     x = 0
#     y = 0
#     z = 0
#     lenth = len(geolocations)
#     for lon, lat in geolocations:
#         lon = radians(float(lon))
#         lat = radians(float(lat))
#         x += cos(lat) * cos(lon)
#         y += cos(lat) * sin(lon)
#         z += sin(lat)
#
#     x = float(x / lenth)
#     y = float(y / lenth)
#     z = float(z / lenth)
#
#     return (degrees(atan2(y, x)), degrees(atan2(z, sqrt(x * x + y * y))))

def get_adj(path):
    distance_arr = pd.read_csv(path).values
    print(len(distance_arr))
    A = torch.tensor(distance_arr, dtype=torch.float32)
    return A
if __name__ == '__main__':
    sw = SummaryWriter('./runs1/train1')
    if opt.server == 1:
        path = '../dataset/data_git_version.h5'
        feature_path = '../dataset/crawled_feature.csv'
        geo_path = '../dataset/adj_1.csv'
    else:
        geo_path = "D:\\usersoftware\\KPIpre\\STTN\\data\\adj.csv"
        path = 'D:\\usersoftware\\KPIpre\\STTN\\data\\data_git_version.h5'
        feature_path = 'D:\\usersoftware\\KPIpre\\STTN\\data\\crawled_feature.csv'
    X, X_meta, X_cross, y, label, mmn, A = read_data(path, feature_path, opt)
    A=get_adj(geo_path)

   #mmnObj = MinMaxNormObj()
   # A = mmnObj.fit_transform(Aorg)

    print("X.shape",X.shape)
    print("y.shape",y.shape)
    samples, sequences, channels, height, width = X.shape
    x_train, x_test = X[:-opt.test_size], X[-opt.test_size:]  # (1499,3,3,20,20)
    meta_train, meta_test = X_meta[:-opt.test_size], X_meta[-opt.test_size:]
    cross_train, cross_test = X_cross[:-opt.test_size], X_cross[-opt.test_size:]
    y_tr = y[:-opt.test_size]
    y_te = y[-opt.test_size:]
    prediction_ct = 0
    truth_ct = 0
    attn_s, attn_t = [], []

    opt.model_filename = '{}/GLSTTN'.format(opt.save_dir)
    y_train = y_tr
    y_test = y_te
    if (opt.meta == 1) & (opt.cross == 1):
        train_data = list(zip(*[x_train, meta_train, cross_train, y_train]))
        test_data = list(zip(*[x_test, meta_test, cross_test, y_test]))
    elif (opt.meta == 1) & (opt.cross == 0):
        train_data = list(zip(*[x_train, meta_train, y_train]))
        test_data = list(zip(*[x_test, meta_test, y_test]))
    elif (opt.cross == 1) & (opt.meta == 0):
        train_data = list(zip(*[x_train, cross_train, y_train]))
        test_data = list(zip(*[x_test, cross_test, y_test]))
    elif (opt.meta == 0) & (opt.cross == 0):
        train_data = list(zip(*[x_train, y_train]))
        test_data = list(zip(*[x_test, y_test]))

    train_idx, valid_idx = train_valid_split(train_data, 0.1)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=train_sampler,
                              num_workers=0, pin_memory=True)
    valid_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=valid_sampler,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
    input_shape = X.shape
    meta_shape = X_meta.shape
    cross_shape = X_cross.shape
    gcn_flag = opt.gcn_flag
    model = GLSTTN(gcn_flag,A,input_shape,
                        meta_shape,
                        cross_shape,
                        nb_flows=opt.close_size,
                        fusion=opt.fusion,
                        maps=(opt.meta + opt.cross + 1),
                        flags_meta=opt.meta,
                        flags_cross=opt.cross
                        ).to(device)

    optimizer = optim.Adam(model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[0.5 * opt.epoch_size,
                                                                 0.75 * opt.epoch_size, 0.9 * opt.epoch_size],
                                                     gamma=0.1)
    modelfiy = 0
    if modelfiy==1:
        state_dict = model.state_dict()
        state_dict['transformer.transformer.spatial_layers.0.fs.weight'] = torch.ones((18,18))
        state_dict['transformer.transformer.spatial_layers.1.fs.weight'] = torch.ones((18,18))
        state_dict['transformer.transformer.spatial_layers.2.fs.weight'] = torch.ones((18,18))
        state_dict['transformer.transformer.spatial_layers.0.fg.weight'] = torch.zeros((18,18))
        state_dict['transformer.transformer.spatial_layers.1.fg.weight'] = torch.zeros((18,18))
        state_dict['transformer.transformer.spatial_layers.2.fg.weight'] = torch.zeros((18,18))
        model.load_state_dict(state_dict)
    # summary(model, (25, 1, 20, 20), batch_size=32, device='cpu')
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    if not os.path.isdir(opt.save_dir):
        raise Exception('%s is not a dir' % opt.save_dir)

    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    elif opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    print('Training ...')
    log(opt.model_filename + 'train.log', '[training]')
    if opt.train:
        train()
    pred, truth = predict('test')
    prediction_ct += pred
    truth_ct += truth
    rmse = 'Final RMSE:{:0.4f}'.format(
        metrics.mean_squared_error(prediction_ct.ravel(), truth_ct.ravel()) ** 0.5)
    print(rmse)
    log_string = rmse
    mae = 'Final MAE:{:0.4f}'.format(
        metrics.mean_absolute_error(prediction_ct.ravel(), truth_ct.ravel()))
    print(mae)
    log_string = log_string+'\n'+mae

    Y = truth_ct.ravel()
    Y_hat = prediction_ct.ravel()
    r2 = 'Final R^2 Score: {:.4f}'.format(metrics.r2_score(Y, Y_hat))
    print(r2)
    log_string = log_string+'\n'+r2
    evs = 'Final Variance Score: {:.4f}'.format(metrics.explained_variance_score(Y, Y_hat))
    print(evs)
    log_string = log_string+'\n'+evs

    log(opt.model_filename + 'metric.log', log_string)

    with h5py.File(hdf5_file, 'w') as f:
        f.create_dataset('pred', data=prediction_ct)
        f.create_dataset('truth', data=truth_ct)
