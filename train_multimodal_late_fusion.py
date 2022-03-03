import sys, os, os.path, time
import argparse
import numpy
import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.autograd import Variable
from Net_mModal import Net, NewNet, TransformerEncoder, Transformer, MMTEncoder, LateFusion, videoModel, SuperLateFusion
from util_in_multi_h5_unnorm import *
from util_out import *
from util_f1 import *
from AudioResNet import resnet50, wide_resnet50_2
from AST import ASTModel
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
#from transformers import  AdamW, get_linear_schedule_with_warmup
torch.backends.cudnn.benchmark = True
from torch.optim.lr_scheduler import LambdaLR
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# Parse input arguments
def mybool(s):
    return s.lower() in ['t', 'true', 'y', 'yes', '1']
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type = str, default = None, required=True)
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
parser.add_argument('--pooling', type = str, default = 'lin', choices = ['max', 'ave', 'lin', 'exp', 'att', 'h-att', 'all'])
parser.add_argument('--batch_size', type = int, default = 250)
parser.add_argument('--ckpt_size', type = int, default = 1000)      # how many batches per checkpoint
parser.add_argument('--optimizer', type = str, default = 'adam', choices = ['adam', 'sgd', 'adamw'])
parser.add_argument('--init_lr', type = float, default = 1e-3)
parser.add_argument('--weight_decay', type = float, default = 0)
parser.add_argument('--beta1', type = float, default = 0.9)
parser.add_argument('--beta2', type = float, default = 0.999)
parser.add_argument('--lr_patience', type = int, default = 3)
parser.add_argument('--lr_factor', type = float, default = 0.8)
parser.add_argument('--max_ckpt', type = int, default = 30)
parser.add_argument('--random_seed', type = int, default = 15213)
parser.add_argument('--additional_outname', type = str, default = '')
parser.add_argument('--continue_from_ckpt', type = str, default = None)
parser.add_argument('--warmup_steps', type = int, default = 1000)
parser.add_argument('--gradient_accumulation', type = int, default = 1)
parser.add_argument('--scheduler', type = str, default = 'reduce', choices = ['reduce', 'warmup-decay','multistepLR'])
parser.add_argument('--addpos', type = mybool, default = False)
parser.add_argument('--transformer_dropout', type = float, default = 0.5)
parser.add_argument('--from_scratch', type = mybool, default = False)
parser.add_argument('--fusion_module', type = int, default = 0) # 0 for early fusion, 1 for mid fusion 1 
args = parser.parse_args()
if 'x' not in args.kernel_size:
    args.kernel_size = args.kernel_size + 'x' + args.kernel_size

numpy.random.seed(args.random_seed)

# Prepare log file and model directory
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
WORKSPACE = os.path.join('../../workspace/ICASSP2021_tune', expid)
MODEL_PATH = os.path.join(WORKSPACE, 'model')
if not os.path.exists(MODEL_PATH): os.makedirs(MODEL_PATH)
LOG_FILE = os.path.join(WORKSPACE, 'train.log')
os.system("cp -r train_multimodal_late_fusion.py %s" % os.path.join(WORKSPACE, 'train_multimodal_late_fusion.py'))
os.system("cp -r Net_mModal.py %s" % os.path.join(WORKSPACE, 'Net_mModal.py'))
os.system("cp -r AudioResNet.py %s" % os.path.join(WORKSPACE, 'AudioResNet.py'))
with open(LOG_FILE, 'w'):
    pass

def write_log(s):
    timestamp = time.strftime('%m-%d %H:%M:%S')
    msg = '[' + timestamp + '] ' + s
    print (msg)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Load data
write_log('Loading data ...')
train_gen = batch_generator(batch_size = args.batch_size, random_seed = args.random_seed)
gas_valid_x1, gas_valid_x2, gas_valid_y, _ = multi_bulk_load('GAS_valid')
gas_eval_x1, gas_eval_x2, gas_eval_y, _ = multi_bulk_load('GAS_eval')
# dcase_valid_x, dcase_valid_y, _ = bulk_load('DCASE_valid')
# dcase_test_x, dcase_test_y, _ = bulk_load('DCASE_test')
# dcase_test_frame_truth = load_dcase_test_frame_truth()
# DCASE_CLASS_IDS = [318, 324, 341, 321, 307, 310, 314, 397, 325, 326, 323, 319, 14, 342, 329, 331, 316]
print('data loaded')
#TODO normalize the data here
print(gas_valid_x1.shape)
print(gas_valid_x2.shape)
print(gas_valid_y.shape)
print(gas_eval_x1.shape)
print(gas_eval_x2.shape)
print(gas_eval_y.shape)

# Build model
args.kernel_size = tuple(int(x) for x in args.kernel_size.split('x'))
if args.model_type == 'TAL':
    model = Net(args).cuda()
elif args.model_type == 'TAL-trans':
    model = TransformerEncoder(args).cuda()
elif args.model_type == 'TAL-new':
    model = NewNet(args).cuda()
elif args.model_type == 'Trans':
    model = Transformer(args).cuda()
elif args.model_type == 'resnet':
    model = resnet50().cuda()
elif args.model_type == 'wide_resnet':
    model = wide_resnet50_2().cuda()
elif args.model_type == 'MMT':
    model = MMTEncoder(args).cuda()
elif args.model_type == 'MMTLF':
    model = LateFusion(args).cuda()
elif args.model_type == 'VM':
    model = videoModel(args).cuda()
elif args.model_type == 'AST':
    model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False).cuda()
else:
    print ('model type not recognized')
    exit(0)
count = count_parameters(model)
print (count)

if args.optimizer == 'sgd':
    optimizer = SGD(model.parameters(), lr = args.init_lr, momentum = 0.9, nesterov = True)
elif args.optimizer == 'adam':
    optimizer = Adam(model.parameters(), lr = args.init_lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    optimizer = AdamW(model.parameters(), lr = args.init_lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
if args.scheduler == 'reduce':
    scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = args.lr_factor, patience = args.lr_patience) if args.lr_factor < 1.0 else None
elif args.scheduler == 'warmup-decay':
    t_total = args.ckpt_size * args.max_ckpt / args.gradient_accumulation
    #     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total) Deprecated API style
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps = t_total)
elif args.scheduler == 'multistepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2,3,4,5], gamma=0.5, last_epoch=-1)


else:
    print ('scheduler type not recognized')
    exit(0)
criterion = nn.BCELoss()
if args.model_type =='AST':
    criterion = nn.BCEWithLogitsLoss()
start_ckpt = 1
if args.continue_from_ckpt != None:
    prev_ckpt = torch.load(args.continue_from_ckpt)
    start_ckpt = prev_ckpt['epoch']
    scheduler.load_state_dict(prev_ckpt['scheduler'])
    model.load_state_dict(prev_ckpt['model'])
    optimizer.load_state_dict(prev_ckpt['optimizer'])
    write_log('Loading model from %s' % args.continue_from_ckpt)
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    model = nn.DataParallel(model)
# Train model
write_log('Training model ...')
write_log('                            ||       GAS_VALID       ||        GAS_EVAL       || D_VAL ||              DCASE_TEST               ')
write_log(" CKPT |    LR    |  Tr.LOSS ||  MAP  |  MAUC |   d'  ||  MAP  |  MAUC |   d'  || Gl.F1 || Gl.F1 | Fr.ER | Fr.F1 | 1s.ER | 1s.F1 ")
FORMAT  = ' %#4d | %8.0003g | %8.0006f || %5.3f | %5.3f |%6.03f || %5.3f | %5.3f |%6.03f || %5.3f || %5.3f | %5.3f | %5.3f | %5.3f | %5.3f '
SEP     = ''.join('+' if c == '|' else '-' for c in FORMAT)
write_log(SEP)
tb_writer = SummaryWriter(os.path.join('runs', args.additional_outname))
global_step = 0
for checkpoint in range(start_ckpt, args.max_ckpt + start_ckpt):
    # Train for args.ckpt_size batches
    model.train()
    train_loss = 0
    import gc
    gc.collect()
    for batch in range(1, args.ckpt_size + 1):
        x1, x2, y = next(train_gen)
#         print(f'loaded batch{batch}, size: {x1.shape, x2.shape, y.shape}')
        if args.model_type in ['TAL-trans', 'TAL', 'resnet','wide_resnet']:
            global_prob = model(x1)[0]
            global_prob = global_prob.float()
#             print(global_prob)
        elif args.model_type == 'AST':
            global_prob = model(x1)[0]
#             print(global_prob.shape)
        elif args.model_type == 'VM':
            global_prob = model(x2)[0]
        else:
            global_prob = model(x1, x2)[0]
        if args.model_type != 'AST':
            global_prob.clamp_(min = 1e-7, max = 1 - 1e-7)
        loss = criterion(global_prob, y)
        if args.gradient_accumulation > 1:
            loss = loss / args.gradient_accumulation
        if n_gpu > 1:
            loss = loss.mean()
        train_loss += loss.item()
        if numpy.isnan(train_loss) or numpy.isinf(train_loss): break
        loss.backward()
        global_step += 1
        if global_step % args.gradient_accumulation == 0:
            optimizer.step()
            if args.scheduler == 'warmup-decay':
                scheduler.step() 
            elif args.scheduler == 'multistepLR':
                scheduler.step() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.zero_grad()
        if batch % 500 == 0:
            sys.stderr.write('Checkpoint %d, Batch %d / %d, avg train loss = %f\r' % \
                            (checkpoint, batch, args.ckpt_size, train_loss / batch))
            tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
            tb_writer.add_scalar('loss', train_loss / batch, global_step)
            
        del x1,x2, y, global_prob, loss         # This line and next line: to save GPU memory
        torch.cuda.empty_cache()            # I don't know if they're useful or not
    train_loss /= args.ckpt_size

    # Evaluate model
    model.eval()
    sys.stderr.write('Evaluating model on GAS_VALID ...\r')
    if args.model_type in ['TAL-trans', 'TAL']:
        global_prob,_ = model.predict(gas_valid_x1, verbose = False)
    elif args.model_type in ['resnet','wide_resnet','AST']:
        global_prob = model.predict(gas_valid_x1, verbose = False)
    elif args.model_type == 'VM':
        global_prob,_ = model.predict(gas_valid_x2, verbose = False)
    else:
        global_prob,_ = model.predict(gas_valid_x1, gas_valid_x2, verbose = False)
#     print(global_prob.shape, gas_valid_y.shape)
    gv_map, gv_mauc, gv_dprime = gas_eval(global_prob, gas_valid_y)
    tb_writer.add_scalar('GAS_valid_Accuracy', gv_map, global_step)
    sys.stderr.write('Evaluating model on GAS_EVAL ... \r')
    if args.model_type in ['TAL-trans', 'TAL']:
        global_prob,_ = model.predict(gas_eval_x1, verbose = False)
    elif args.model_type in ['resnet','wide_resnet','AST']:
        global_prob = model.predict(gas_eval_x1, verbose = False)
    elif args.model_type == 'VM':
        global_prob,_ = model.predict(gas_eval_x2, verbose = False)
    else:
        global_prob,_ = model.predict(gas_eval_x1, gas_eval_x2, verbose = False)
    ge_map, ge_mauc, ge_dprime = gas_eval(global_prob, gas_eval_y)
    tb_writer.add_scalar('GAS_test_Accuracy', ge_map, global_step)
#     sys.stderr.write('Evaluating model on DCASE_VALID ...\r')
#     global_prob = model.predict(dcase_valid_x, verbose = False)[:, DCASE_CLASS_IDS]
#     thres = optimize_micro_avg_f1(global_prob, dcase_valid_y)
#     dv_f1 = f1(global_prob >= thres, dcase_valid_y)
#     sys.stderr.write('Evaluating model on DCASE_TEST ... \r')
#     outputs = model.predict(dcase_test_x, verbose = True)
#     outputs = tuple(x[..., DCASE_CLASS_IDS] for x in outputs)
#     dt_f1 = f1(outputs[0] >= thres, dcase_test_y)
#     dt_frame_er, dt_frame_f1 = dcase_sed_eval(outputs, args.pooling, thres, dcase_test_frame_truth, 1)
#     dt_1s_er, dt_1s_f1 = dcase_sed_eval(outputs, args.pooling, thres, dcase_test_frame_truth, 10)

    # Write log
    write_log(FORMAT % (
        checkpoint, optimizer.param_groups[0]['lr'], train_loss,
        gv_map, gv_mauc, gv_dprime,
        ge_map, ge_mauc, ge_dprime,
        0, 0, 0, 0, 0, 0
#         dv_f1, dt_f1, dt_frame_er, dt_frame_f1, dt_1s_er, dt_1s_f1
    ))
    
#     for name, param in model.named_parameters():
#                 tb_writer.add_histogram(name, param, global_step)
#                 tb_writer.add_histogram('{}.grad'.format(name), param.grad, global_step)
    # write_log(FORMAT % (
    #     checkpoint, optimizer.param_groups[0]['lr'], train_loss,
    #     gv_map, gv_mauc, gv_dprime,
    #     ge_map, ge_mauc, ge_dprime,
    #     0, 0, 0, 0, 0, 0
    # ))

    # Abort if training has gone mad
    if numpy.isnan(train_loss) or numpy.isinf(train_loss):
        write_log('Aborted.')
        break

    # Save model. Too bad I can't save the scheduler
    MODEL_FILE = os.path.join(MODEL_PATH, 'checkpoint%d.pt' % checkpoint)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':checkpoint+1, 'scheduler':scheduler.state_dict()}
    sys.stderr.write('Saving model to %s ...\r' % MODEL_FILE)
    torch.save(state, MODEL_FILE)

    # Update learning rate
    if args.scheduler == 'reduce':
        scheduler.step(gv_map)

# model.eval()
# sys.stderr.write('Fusion model on GAS_VALID ...\r')
# if args.model_type == 'TAL-trans':
#     global_prob,_ = model.predict(gas_valid_x1, verbose = False)
# elif args.model_type == 'VM':
#     global_prob,_ = model.predict(gas_valid_x2, verbose = False)
# else:
#     global_prob,_ = model.predict(gas_valid_x1, gas_valid_x2, verbose = False)
# #pickle.dump(_, open('fusion_valid_hidden_might_not_best.pkl', 'wb'))
# sys.stderr.write('Fusion model on GAS_EVAL ... \r')
# if args.model_type == 'TAL-trans':
#     global_prob,_ = model.predict(gas_eval_x1, verbose = False)
# elif args.model_type == 'VM':
#     global_prob,_ = model.predict(gas_eval_x2, verbose = False)
# else:
#     global_prob,_ = model.predict(gas_eval_x1, gas_eval_x2, verbose = False)
#pickle.dump(_, open('fusion_valid_hidden_might_not_best.pkl', 'wb'))
        
write_log('DONE!')
tb_writer.close()

