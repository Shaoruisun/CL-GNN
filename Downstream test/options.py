import argparse
import os


def parse_common_args(parser):
    parser.add_argument('--dataset',default='' , help='2007,2013 or 2016')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--model_type', type=str, default='ECIF_GNN', help='used in model_entry.py')
    parser.add_argument('--data_type', type=str, default='PDBbind2020', help='used in data_entry.py')
    parser.add_argument('--save_prefix', type=str, default='pref', help='some comment for model or test result dir')
    parser.add_argument('--load_model_path', type=str, default='/home/zyj/RTMScore复现/checkpoints/ECIF_GNN_temp_0.5/4999_000000.pth',help='model path for pretrain or test')
    parser.add_argument('--load_not_strict', action='store_true', help='allow to load only common state dicts')
    parser.add_argument('--val_list', type=str, default='/data/davis/data_test.csv',
                        help='val list in train, test list path in test')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--block_num', type=int, default=3)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--save_path', type=str,default='/home/zyj/RTMScore复现/save_path/')
    
    
    parser.add_argument('-d', '--dir', default="/home/zyj/RTMScore复现/data/general-set/",
                   help='The directory to store the protein-ligand complexes.')
    parser.add_argument('-c', '--cutoff', default=None, type=float,
                   help='the cutoff to determine the pocket')
    parser.add_argument('-o', '--outprefix', default="out",
                   help='The output bin file.')
    parser.add_argument('-usH', '--useH', default=False, action="store_true",
                   help='whether to use the explicit H atoms.')
    parser.add_argument('-uschi', '--use_chirality', default=False, action="store_true",
                   help='whether to use chirality.')
    parser.add_argument('-p', '--parallel', default=False, action="store_true",
                   help='whether to obtain the graphs in parallel (When the dataset is too large,\
						 it may be out of memory when conducting the parallel mode).')
    return parser


def parse_train_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')#0.0005
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.9, type=float, metavar='M',
                        help='beta parameters for adam')#0.85
    parser.add_argument('--weight-decay', '--wd', default=10**-6, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
    parser.add_argument('--train_list', type=str, default='/data/davis/data_train.csv')
    parser.add_argument('--batch_size', type=int, default=128)#50
    parser.add_argument('--valnum', type=int, default=20)
    parser.add_argument('--hidden_dim0', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=4096)
    parser.add_argument('--n_gaussians', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.00)
    parser.add_argument('--dist_threhold', type=float, default=7.)
    return parser


def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--save_viz',default=True, action='store_true', help='save viz result in eval or not')
    parser.add_argument('--result_dir', type=str, default=' ', help='leave blank, auto generated')
    return parser


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args


def get_train_model_dir(args):
    model_dir = os.path.join('checkpoints', args.model_type + '_' + args.save_prefix)
    if not os.path.exists(model_dir):
        os.system('mkdir -p ' + model_dir)
    args.model_dir = model_dir


def get_test_result_dir(args):
    ext = os.path.basename(args.load_model_path).split('.')[-1]
    model_dir = args.load_model_path.replace(ext, '')
    val_info = os.path.basename(os.path.dirname(args.val_list)) + '_' + os.path.basename(args.val_list.replace('.txt', ''))
    result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
    if not os.path.exists(result_dir):
        os.system('mkdir -p ' + result_dir)
    args.result_dir = result_dir


def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def prepare_train_args():
    args = get_train_args()
    get_train_model_dir(args)
    save_args(args, args.model_dir)
    return args


def prepare_test_args():
    args = get_test_args()
    get_test_result_dir(args)
    save_args(args, args.result_dir)
    return args


if __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args()
