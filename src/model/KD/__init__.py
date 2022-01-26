from model.KD.L2 import L2
from model.KD.LSH import LSH

def make_KD_opt(args):
    if args.distill == 'none':
        return None
    KD_opt = dict()
    KD_opt['distill'] = args.distill
    KD_opt['beta'] = args.beta
    KD_opt['hash_num'] = args.hash_num
    KD_opt['std'] = args.std
    KD_opt['LSH_loss'] = args.LSH_loss
    KD_opt['LSH_weight'] = args.LSH_weight
    return KD_opt

def create_KD_loss(opt):
    if opt['distill'] == 'l2':
        kd = L2(opt['beta'])
    elif opt['distill'] == 'lsh':
        kd = LSH(opt['feature_dim'], opt['hash_num'], std=opt['std'], 
            with_l2=False, LSH_loss=opt['LSH_loss'], beta=opt['beta'], 
            load_weight=opt['LSH_weight'])
    elif opt['distill'] == 'lshl2':
        kd = LSH(opt['feature_dim'], opt['hash_num'], std=opt['std'], 
            with_l2=True, LSH_loss=opt['LSH_loss'], beta=opt['beta'],
            load_weight=opt['LSH_weight'])
    return kd