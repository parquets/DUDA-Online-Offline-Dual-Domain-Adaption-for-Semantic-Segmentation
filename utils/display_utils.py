import numpy as np

def print_loss_dict(loss_dict, iter_cnt):
    res = ''
    for loss_name, loss_value in loss_dict.items():
        res += ', {}: {:.6f}'.format(loss_name, loss_value/iter_cnt)
    return res

def print_iou_list(iou_list):
    res = ''
    for i, iou in enumerate(iou_list):
        res += ', {}: {:.6f}'.format(i, iou)
    return res

def print_top(result, metrics, top=0.1):
    res = np.array([x[metrics] for x in result])
    res = np.sort(res)
    # top = int(len(res) * 0.1) + 1
    top = 1
    return res[-top:].mean()

def result_list2dict(iou_list, metrics):
    res = {}
    for i, iou in enumerate(iou_list):
        res[metrics+str(i)] = iou
    return res

def itv2time(iItv):
    h = int(iItv//3600)
    sUp_h = iItv-3600*h
    m = int(sUp_h//60)
    sUp_m = sUp_h-60*m
    s = int(sUp_m)
    return "{}h {:0>2d}min".format(h,m,s)