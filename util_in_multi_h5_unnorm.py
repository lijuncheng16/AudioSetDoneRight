import sys, os, os.path, glob
import pickle
from scipy.io import loadmat
import numpy
import h5py
import torch
from torch.autograd import Variable

N_CLASSES = 527
N_WORKERS = 6
local = os.getenv('LOCAL')
local = '/jet/home/billyli/data_folder/data/googleAudioSet'
hf_train_path = os.path.join(local,'data_train.h5')
hf_val_eval_path = os.path.join(local, 'data.h5')

def batch_generator(batch_size, random_seed = 15213):
    rng = numpy.random.RandomState(random_seed)
    all_epochs = list(range(1,12,1))
    all_iter = list(range(1,2501,1))
    while True:
        rng.shuffle(all_epochs)
        rng.shuffle(all_iter)
        for i in all_epochs:
            hf_train = h5py.File(hf_train_path, 'r')
            for j in all_iter:
                key = str(i)+'_'+str(j)
                feat_a = hf_train[key]['audio'][:]
                feat_v = hf_train[key]['video'][:]
                label = hf_train[key]['label'][:] 
                feat_a = feat_a.astype('float32')
                feat_v = feat_v.astype('float32')
                label = label.astype('float32')
#                 yield tuple(Variable(torch.from_numpy(numpy.stack(x))).cuda() for x in zip(*batch))
                yield Variable(torch.from_numpy(feat_a)).cuda(), Variable(torch.from_numpy(feat_v)).cuda(), Variable(torch.from_numpy(label)).cuda()
                del feat_a,feat_v, label
            hf_train.close()

                            
def multi_bulk_load(prefix):
    hf_val_eval = h5py.File(hf_val_eval_path, 'r')
    if prefix == 'GAS_valid':
        feat_a = hf_val_eval['valid']['data']['feat_a'][:]
        feat_v = hf_val_eval['valid']['data']['feat_v'][:]
        labels = hf_val_eval['valid']['data']['labels'][:]
    elif prefix == 'GAS_eval':
        feat_a = hf_val_eval['eval']['data']['feat_a'][:]
        feat_v = hf_val_eval['eval']['data']['feat_v'][:]
        labels = hf_val_eval['eval']['data']['labels'][:]
    else:
        assert('error')
    hf_val_eval.close()
    return feat_a.astype('float32'), feat_v.astype('float32'), labels.astype('float32'), None 
