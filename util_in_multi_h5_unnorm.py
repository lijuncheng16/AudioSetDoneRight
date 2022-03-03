import sys, os, os.path, glob
import pickle
from scipy.io import loadmat
import numpy
import h5py
import torch
from torch.autograd import Variable
import subprocess
import socket

curr_node = socket.gethostname().split('.')[0]
batcmd=f"squeue -u billyli | grep {curr_node}"
curr_slurm = subprocess.check_output(batcmd, shell=True, text=True)
slurm_id = curr_slurm.strip().split(' ')[0]

GAS_FEATURE_DIR = '/jet/home/billyli/data_folder/data/googleAudioSet/pylon5/ir3l68p/kaixinm/cmu-thesis/data/audioset'
DCASE_FEATURE_DIR = '/jet/home/billyli/data_folder/data/dcase'

N_CLASSES = 527
N_WORKERS = 6
# local = os.getenv('LOCAL')
# local = '/jet/home/billyli/data_folder/data/googleAudioSet'
local = f"/local/slurm-{slurm_id}/local/audio"
hf_train_path = os.path.join(local,'data_train.h5')
hf_val_eval_path = os.path.join(local, 'data.h5')
with open(os.path.join(GAS_FEATURE_DIR, 'normalizer.pkl'), 'rb') as f:
    mu, sigma = pickle.load(f, encoding='bytes')


def batch_generator(batch_size, random_seed = 15213):
    rng = numpy.random.RandomState(random_seed)
    rand_int= numpy.random.randint(batch_size, 100)
    all_epochs = list(range(1,12,1))
    all_iter = list(range(1,2501,1))
    while True:
        rng.shuffle(all_epochs)
        rng.shuffle(all_iter)
        for i in all_epochs:
            hf_train = h5py.File(hf_train_path, 'r')
            for j in all_iter:
                key = str(i)+'_'+str(j)
                feat_a = hf_train[key]['audio'][rand_int-batch_size:rand_int]
#                 print('feat_a shape', feat_a.shape)
#                 feat_v = hf_train[key]['video'][:]
                feat_v = hf_train[key]['video'][rand_int-batch_size:rand_int]
                label = hf_train[key]['label'][rand_int-batch_size:rand_int]
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

def bulk_load(prefix):
    feat = []; labels = []; hashes = []
    for filename in sorted(glob.glob(os.path.join(GAS_FEATURE_DIR, '%s_*.mat' % prefix)) +
                           glob.glob(os.path.join(DCASE_FEATURE_DIR, '%s_*.mat' % prefix))):
        data = loadmat(filename)
        feat.append(((data['feat'] - mu) / sigma).astype('float32'))
        labels.append(data['labels'].astype('bool'))
        hashes.append(data['hashes'])
    return numpy.concatenate(feat), numpy.concatenate(labels), numpy.concatenate(hashes)

def unnorm_bulk_load(prefix):
    """
    Very very bad performance, values near 100, and would end up with MAP 0.012, do not use this function, only for testing purpose.
    """
    feat = []; labels = []; hashes = []
    for filename in sorted(glob.glob(os.path.join(GAS_FEATURE_DIR, '%s_*.mat' % prefix)) +
                           glob.glob(os.path.join(DCASE_FEATURE_DIR, '%s_*.mat' % prefix))):
        data = loadmat(filename)
        feat.append((data['feat']).astype('float32'))
        labels.append(data['labels'].astype('bool'))
        hashes.append(data['hashes'])
    return numpy.concatenate(feat), numpy.concatenate(labels), numpy.concatenate(hashes)

