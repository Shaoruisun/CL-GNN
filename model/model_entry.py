#from model.best.fcn import DTIPredictorV4_V2
#from model.best.fcn import RTMScore
#from model.better.fcn import Resnet101Fcn
#from model.sota.fcn import LightFcn
import torch.nn as nn
import torch
from model.sota.fcn import ECIF_GNN

#device = torch.device('cuda:0')
def select_model(args):
    type2model = {
        #'resnet50_fcn': CustomFcn(args),
        'ECIF_GNN': ECIF_GNN(dropout=args.dropout)#in_feats=args.hidden_dim0, hidden_size=args.hidden_dim,dropout=args.dropout),
        #'deeplabv3_fcn': DeepLabv3Fcn(args),
        #'mobilnetv3_fcn': LightFcn(args)
    }
    model = type2model[args.model_type]
    return model


def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model