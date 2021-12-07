import sys, os, os.path
import argparse
import numpy as np
from util_out import *
from util_f1 import *
from scipy.io import loadmat, savemat
import json

# Parse input arguments
def mybool(s):
    return s.lower() in ['t', 'true', 'y', 'yes', '1']
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type = str, default = None, required=True)
parser.add_argument('--TALNet', action = 'store_true')              # specify this to evaluate the pre-trained TALNet model
parser.add_argument('--embedding_size', type = int, default = 1024) # this is the embedding size after a pooling layer
                                                                    # after a non-pooling layer, the embeddings size will be twice this much
parser.add_argument('--n_conv_layers', type = int, default = 10)
parser.add_argument('--n_trans_layers', type = int, default = 1)
parser.add_argument('--kernel_size', type = str, default = '3')     # 'n' or 'nxm'
parser.add_argument('--n_pool_layers', type = int, default = 5)     # the pooling layers will be inserted uniformly into the conv layers
                                                                    # the should be at least 2 and at most 6 pooling layers
                                                                    # the first two pooling layers will have stride (2,2); later ones will have stride (1,2)
parser.add_argument('--batch_norm', type = mybool, default = True)
parser.add_argument('--dropout', type = float, default = 0.0)
parser.add_argument('--pooling', type = str, default = 'lin', choices = ['max', 'ave', 'lin', 'exp', 'att'])
parser.add_argument('--batch_size', type = int, default = 250)
parser.add_argument('--ckpt_size', type = int, default = 1000)      # how many batches per checkpoint
parser.add_argument('--optimizer', type = str, default = 'adam', choices = ['adam', 'sgd'])
parser.add_argument('--init_lr', type = float, default = 1e-3)
parser.add_argument('--weight_decay', type = float, default = 0)
parser.add_argument('--beta1', type = float, default = 0.9)
parser.add_argument('--beta2', type = float, default = 0.999)
parser.add_argument('--lr_patience', type = int, default = 3)
parser.add_argument('--lr_factor', type = float, default = 0.8)
parser.add_argument('--random_seed', type = int, default = 15213)
parser.add_argument('--ckpt', type = int)
parser.add_argument('--additional_outname', type = str, default = '')
parser.add_argument('--addpos', type = mybool, default = False)
parser.add_argument('--transformer_dropout', type = float, default = 0.5)
parser.add_argument('--gradient_accumulation', type = int, default = 1)
parser.add_argument('--scheduler', type = str, default = 'reduce', choices = ['reduce', 'warmup-decay'])
parser.add_argument('--from_scratch', type = mybool, default = False)
args = parser.parse_args()
if 'x' not in args.kernel_size:
    args.kernel_size = args.kernel_size + 'x' + args.kernel_size

# Locate model file and prepare directories for prediction and evaluation
expid = '%s-embed%d-%dC%dP-kernel%s-%s-drop%.1f-%s-batch%d-ckpt%d-%s-lr%.0e-pat%d-fac%.1f-seed%d-Trans%d-weight-decay%.8f-betas%.3f-%.3f' % (
    args.model_type,
    args.embedding_size,
    args.n_conv_layers,
    args.n_pool_layers,
    args.kernel_size,
    'bn' if args.batch_norm else 'nobn',
    args.dropout,
    args.pooling,
    args.batch_size,
    args.ckpt_size,
    args.optimizer,
    args.init_lr,
    args.lr_patience,
    args.lr_factor,
    args.random_seed,
    args.n_trans_layers,
    args.weight_decay,
    args.beta1, 
    args.beta2
)
expid += args.additional_outname
WORKSPACE = os.path.join('../../workspace/audioset', expid)
PRED_PATH = os.path.join(WORKSPACE, 'pred')
if not os.path.exists(PRED_PATH): os.makedirs(PRED_PATH)
EVAL_PATH = os.path.join(WORKSPACE, 'eval')
if not os.path.exists(EVAL_PATH): os.makedirs(EVAL_PATH)
if args.TALNet:
    WORKSPACE = '../../workspace/audioset/embed1024-10C5P-kernel3x3-bn-drop0.0-lin-batch250-ckpt1000-adam-lr1e-03-pat3-fac0.8-seed15213'
    PRED_PATH = os.path.join(WORKSPACE, 'pred')
    EVAL_PATH = os.path.join(WORKSPACE, 'eval')
    MODEL_FILE = os.path.join(WORKSPACE, 'model', 'checkpoint15.pt')
    PRED_FILE = os.path.join(PRED_PATH, 'checkpoint15.mat')
    EVAL_FILE = os.path.join(EVAL_PATH, 'checkpoint15.txt')
#else:
    #MODEL_FILE = os.path.join(WORKSPACE, 'model', 'checkpoint%d.pt' % args.ckpt)
    #PRED_FILE = os.path.join(PRED_PATH, 'checkpoint%d.mat' % args.ckpt)
    #EVAL_FILE = os.path.join(EVAL_PATH, 'checkpoint%d.txt' % args.ckpt)
#with open(EVAL_FILE, 'w'):
#    pass

def write_log(s):
   print(s)
   with open(EVAL_FILE, 'a') as f:
       f.write(s + '\n')

if False:
    # Load saved predictions, no need to use GPU
    data = loadmat(PRED_FILE)
    dcase_thres = data['dcase_thres'].ravel()
    dcase_test_y = data['dcase_test_y']
    dcase_test_frame_y = data['dcase_test_frame_y']
    dcase_test_outputs = []
    dcase_test_outputs.append(data['dcase_test_global_prob'])
    dcase_test_outputs.append(data['dcase_test_frame_prob'])
    if args.pooling == 'att':
        dcase_test_outputs.append(data['dcase_test_frame_att'])
    gas_eval_y = data['gas_eval_y']
    gas_eval_global_prob = data['gas_eval_global_prob']
else:
    import torch
    import torch.nn as nn
    from torch.optim import *
    from torch.optim.lr_scheduler import *
    from torch.autograd import Variable
    from Net_mModal import Net, NewNet, TransformerEncoder, Transformer, MMTEncoder, LateFusion, videoModel, SuperLateFusion
    from util_in_multi_h5_unnorm import *
    from util_out import *
    from util_f1 import *
    from AudioResNet import resnet50

    # Load model
    args.kernel_size = tuple(int(x) for x in args.kernel_size.split('x'))
    #model = Net(args).cuda()
    if args.model_type == 'TAL-trans':
        model = TransformerEncoder(args).cuda()
    elif args.model_type == 'MMTLF':
        model = LateFusion(args).cuda()
    elif args.model_type == 'VM':
        model = videoModel(args).cuda()
    elif args.model_type == 'SPMMTLF':
        model = SuperLateFusion(args).cuda()
    else:
        print ('model type not recognized')
        exit(0)
    #model.load_state_dict(torch.load(MODEL_FILE)['model'])
    model.eval()

    # Load DCASE data
    #dcase_valid_x, dcase_valid_y, _ = bulk_load('DCASE_valid')
    #dcase_test_x, dcase_test_y, dcase_test_hashes = bulk_load('DCASE_test')
    #dcase_test_frame_y = load_dcase_test_frame_truth()
    #DCASE_CLASS_IDS = [318, 324, 341, 321, 307, 310, 314, 397, 325, 326, 323, 319, 14, 342, 329, 331, 316]

    # Predict on DCASE data
    #dcase_valid_global_prob = model.predict(dcase_valid_x, verbose = False)[:, DCASE_CLASS_IDS]
    #dcase_thres = optimize_micro_avg_f1(dcase_valid_global_prob, dcase_valid_y)
    #dcase_test_outputs = model.predict(dcase_test_x, verbose = True)
    #dcase_test_outputs = tuple(x[..., DCASE_CLASS_IDS] for x in dcase_test_outputs)

    # Load GAS data
    gas_valid_x1, gas_valid_x2, gas_valid_y, _ = multi_bulk_load('GAS_valid')
    gas_eval_x1, gas_eval_x2, gas_eval_y, _ = multi_bulk_load('GAS_eval')
    #numpy.save('100labels.npy', gas_eval_y[0:100])
    #exit(0)
    print (gas_valid_y.shape)
    print (gas_eval_y.shape)
    # Predict on GAS data
    global_prob_a, _ = model.predict(gas_eval_x1, gas_eval_x2, verbose=False)
    global_prob_v = np.load('/pylon5/ir3l68p/billyli/DayLongAudio/audioset_py3/results_file/eval_output_best.npy')
    gv_map, gv_mauc, gv_dprime = gas_eval(global_prob_v, gas_eval_y)
    print (gv_map, gv_mauc, gv_dprime)
    print (global_prob_a.shape, global_prob_v.shape)
    global_prob = global_prob_a*0.6+global_prob_v*0.4
    gv_map, gv_mauc, gv_dprime = gas_eval(global_prob, gas_eval_y)
    print (gv_map, gv_mauc, gv_dprime)
    exit(0)
    #global_prob,_ = model.predict(gas_valid_x1, gas_valid_x2, verbose = False)
    global_prob, _ = model.predict(gas_valid_x1, gas_valid_x2, verbose=False)
    gv_map, gv_mauc, gv_dprime = gas_eval(global_prob, gas_valid_y)
    print (global_prob.shape)
    print (gv_map, gv_mauc, gv_dprime)
    global_prob, _ = model.predict(gas_eval_x1, gas_eval_x2, verbose=False)
    gv_map, gv_mauc, gv_dprime = gas_eval(global_prob, gas_eval_y)
    print (global_prob.shape)
    print (gv_map, gv_mauc, gv_dprime)
    exit(0)
    thres = optimize_micro_avg_f1(global_prob, gas_valid_y)
    print (thres.shape)
    print (thres)
    confusion_matrix = np.zeros((527, 527))
    for i in range(len(global_prob)):
        sample_pred = np.where((global_prob[i] > thres) == True)[0]
        sample_gold = np.where(gas_valid_y[i]==1)[0]
        miss_pred = [p for p in sample_pred if p not in sample_gold]
        miss_gold = [g for g in sample_gold if g not in sample_pred]
        for g in miss_gold:
            for p in miss_pred:
                confusion_matrix[g][p] += 1
    total_nums = gas_valid_y.sum(axis=0)
    #numpy.save('total_nums.npy', total_nums)
    numpy.save('VM_confusion.npy', confusion_matrix)
    exit(0)
    # Save predictions
    data = {}
    # data['dcase_thres'] = dcase_thres
    # data['dcase_test_hashes'] = dcase_test_hashes
    # data['dcase_test_y'] = dcase_test_y
    # data['dcase_test_frame_y'] = dcase_test_frame_y
    # data['dcase_test_global_prob'] = dcase_test_outputs[0]
    # data['dcase_test_frame_prob'] = dcase_test_outputs[1]
    # if args.pooling == 'att':
    #    data['dcase_test_frame_att'] = dcase_test_outputs[2]
    data['gas_eval_hashes'] = gas_eval_hashes
    data['gas_eval_y'] = gas_eval_y
    data['gas_eval_global_prob'] = gas_eval_global_prob
    savemat(PRED_FILE, data)

# Evaluation on DCASE 2017
# write_log('Performance on DCASE 2017:')
# write_log('')
# write_log('           ||          ||            Task A (recording level)           ||                       Task B (1-second segment level)                       ')
# write_log('     CLASS ||    THRES ||   TP |   FN |   FP |  Prec. | Recall |     F1 ||   TP |   FN |   FP |  Prec. | Recall |     F1 |  Sub |  Del |  Ins |     ER ')
# FORMAT1 = ' Micro Avg ||          || %#4d | %#4d | %#4d | %6.02f | %6.02f | %6.02f || %#4d | %#4d | %#4d | %6.02f | %6.02f | %6.02f | %#4d | %#4d | %#4d | %6.02f '
# FORMAT2 = ' %######9d || %8.0006f || %#4d | %#4d | %#4d | %6.02f | %6.02f | %6.02f || %#4d | %#4d | %#4d | %6.02f | %6.02f | %6.02f |      |      |      |        '
# SEP     = ''.join('+' if c == '|' else '-' for c in FORMAT1)
# write_log(SEP)

# dcase_test_y and dcase_test_frame_y are inconsistent in some places
# so when you evaluate Task A, use a "fake_dcase_test_frame_y" derived from dcase_test_y
# fake_dcase_test_frame_y = numpy.tile(numpy.expand_dims(dcase_test_y, 1), (1, 100, 1))

# # Micro-average performance across all classes
# res_taskA = dcase_sed_eval(dcase_test_outputs, args.pooling, dcase_thres, fake_dcase_test_frame_y, 100, verbose = True)
# res_taskB = dcase_sed_eval(dcase_test_outputs, args.pooling, dcase_thres, dcase_test_frame_y, 10, verbose = True)
# write_log(FORMAT1 % (res_taskA.TP, res_taskA.FN, res_taskA.FP, res_taskA.precision, res_taskA.recall, res_taskA.F1,
#                      res_taskB.TP, res_taskB.FN, res_taskB.FP, res_taskB.precision, res_taskB.recall, res_taskB.F1,
#                      res_taskB.sub, res_taskB.dele, res_taskB.ins, res_taskB.ER))
# write_log(SEP)

# # Class-wise performance
# N_CLASSES = dcase_test_outputs[0].shape[-1]
# for i in range(N_CLASSES):
#     outputs = [x[..., i:i+1] for x in dcase_test_outputs]
#     res_taskA = dcase_sed_eval(outputs, args.pooling, dcase_thres[i], fake_dcase_test_frame_y[..., i:i+1], 100, verbose = True)
#     res_taskB = dcase_sed_eval(outputs, args.pooling, dcase_thres[i], dcase_test_frame_y[..., i:i+1], 10, verbose = True)
#     write_log(FORMAT2 % (i, dcase_thres[i],
#                          res_taskA.TP, res_taskA.FN, res_taskA.FP, res_taskA.precision, res_taskA.recall, res_taskA.F1,
#                          res_taskB.TP, res_taskB.FN, res_taskB.FP, res_taskB.precision, res_taskB.recall, res_taskB.F1))

# # Evaluation on Google Audio Set
write_log('')
write_log('Performance on Google Audio Set:')
write_log('')
write_log("   CLASS ||    AP |   AUC |    d' ")
FORMAT  = ' %00007s || %5.3f | %5.3f |%6.03f '
SEP     = ''.join('+' if c == '|' else '-' for c in FORMAT)
write_log(SEP)

classwise = []
classwise_dict = []
N_CLASSES = gas_eval_global_prob.shape[-1]
for i in range(N_CLASSES):
    a, b, c = gas_eval(gas_eval_global_prob[:,i], gas_eval_y[:,i])     # AP, AUC, dprime
    classwise_dict.append({'mAP':a, 'AUC':b, 'dprime':c, 'id':i})
    classwise.append((a, b, c))
map, mauc = numpy.array(classwise).mean(axis = 0)[:2]
print (map, mauc, dprime(mauc))
with open('bestclasswise.jsonl', 'w') as fout:
    for c in classwise_dict:
        json.dump(c, fout)
        fout.write('\n')
write_log(FORMAT % ('Average', map, mauc, dprime(mauc)))
write_log(SEP)
for i in range(N_CLASSES):
    write_log(FORMAT % ((str(i),) + classwise[i]))
