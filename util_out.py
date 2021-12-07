from scipy import stats
import numpy

def roc(pred, truth):
    data = numpy.array(sorted(zip(pred, truth), reverse = True))
    pred, truth = data[:,0], data[:,1]
    TP = truth.cumsum()
    FP = (1 - truth).cumsum()
    mask = numpy.concatenate([numpy.diff(pred) < 0, numpy.array([True])])
    TP = numpy.concatenate([numpy.array([0]), TP[mask]])
    FP = numpy.concatenate([numpy.array([0]), FP[mask]])
    return TP, FP

def ap_and_auc(pred, truth):
    TP, FP = roc(pred, truth)
    auc = ((TP[1:] + TP[:-1]) * numpy.diff(FP)).sum() / (2 * TP[-1] * FP[-1])
    precision = TP[1:] / (TP + FP)[1:]
    weight = numpy.diff(TP)
    ap = (precision * weight).sum() / TP[-1]
    return ap, auc

def dprime(auc):
    return stats.norm().ppf(auc) * numpy.sqrt(2.0)

def gas_eval(pred, truth):
    if truth.ndim == 1:
        ap, auc = ap_and_auc(pred, truth)
    else:
        ap, auc = numpy.array([ap_and_auc(pred[:,i], truth[:,i]) for i in range(truth.shape[1]) if truth[:,i].any()]).mean(axis = 0)
    return ap, auc, dprime(auc)

def dcase_sed_eval(outputs, pooling, thres, truth, seg_len, verbose = False):
    pred = outputs[1].reshape((-1, seg_len, outputs[1].shape[-1]))
    if pooling == 'max':
        seg_prob = pred.max(axis = 1)
    elif pooling == 'ave':
        seg_prob = pred.mean(axis = 1)
    elif pooling == 'lin':
        seg_prob = (pred * pred).sum(axis = 1) / pred.sum(axis = 1)
    elif pooling == 'exp':
        seg_prob = (pred * numpy.exp(pred)).sum(axis = 1) / numpy.exp(pred).sum(axis = 1)
    elif pooling == 'att':
        att = outputs[2].reshape((-1, seg_len, outputs[2].shape[-1]))
        seg_prob = (pred * att).sum(axis = 1) / att.sum(axis = 1)
    elif pooling == 'h-att':
        seg_prob = (pred * pred).sum(axis = 1) / pred.sum(axis = 1)
    elif pooling == 'all':
        max_prob = pred.max(axis = 1)
        max_prob = numpy.expand_dims(max_prob, axis=2)
        ave_prob = pred.mean(axis=1)
        ave_prob = numpy.expand_dims(ave_prob, axis=2)
        lin_prob = (pred * pred).sum(axis = 1) / pred.sum(axis = 1)
        lin_prob = numpy.expand_dims(lin_prob, axis=2)
        exp_prob = (pred * numpy.exp(pred)).sum(axis = 1) / numpy.exp(pred).sum(axis = 1)
        exp_prob = numpy.expand_dims(exp_prob, axis=2)
        att = outputs[2].reshape((-1, seg_len, outputs[2].shape[-1]))
        att_prob = (pred * att).sum(axis = 1) / att.sum(axis = 1)
        att_prob = numpy.expand_dims(att_prob, axis=2)
        all_prob = numpy.concatenate([max_prob, ave_prob, lin_prob, exp_prob, att_prob], axis=2)
        global_weights = numpy.transpose(outputs[3], (0, 2, 1))
        global_weights = numpy.expand_dims(global_weights, axis=1)
        global_weights = numpy.repeat(global_weights, outputs[1].shape[1]/seg_len, axis=1)
        global_weights =  global_weights.reshape((-1, global_weights.shape[-2], global_weights.shape[-1]))
        # print ('pred', pred.shape)
        # print ('max_prob', max_prob.shape)
        # print ('all_prob', all_prob.shape)
        # print ('output1', outputs[1].shape)
        # print ('outputs3', outputs[3].shape)
        # print ('global_weights', global_weights.shape)
        seg_prob = (all_prob * global_weights).sum(axis=2)
        

    pred = seg_prob >= thres
    truth = truth.reshape((-1, seg_len, truth.shape[-1])).max(axis = 1)

    if not verbose:
        Ntrue = truth.sum(axis = 1)
        Npred = pred.sum(axis = 1)
        Ncorr = (truth & pred).sum(axis = 1)
        Nmiss = Ntrue - Ncorr
        Nfa = Npred - Ncorr

        error_rate = 1.0 * numpy.maximum(Nmiss, Nfa).sum() / Ntrue.sum()
        f1 = 2.0 * Ncorr.sum() / (Ntrue + Npred).sum()
        return error_rate, f1
    else:
        class Object(object):
            pass
        res = Object()
        res.TP = (truth & pred).sum()
        res.FN = (truth & ~pred).sum()
        res.FP = (~truth & pred).sum()
        res.precision = 100.0 * res.TP / (res.TP + res.FP)
        res.recall = 100.0 * res.TP / (res.TP + res.FN)
        res.F1 = 200.0 * res.TP / (2 * res.TP + res.FP + res.FN)
        res.sub = numpy.minimum((truth & ~pred).sum(axis = 1), (~truth & pred).sum(axis = 1)).sum()
        res.dele = res.FN - res.sub
        res.ins = res.FP - res.sub
        res.ER = 100.0 * (res.sub + res.dele + res.ins) / (res.TP + res.FN)
        return res
