import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='clueweb09',
                        help='Choose a dataset from {robust04, clueweb09}')
                        
    parser.add_argument('--fold', type=int, default=3,
                        help='run fold 1~5')
    parser.add_argument('--doc_len', type=int, default=300,
                        help='DOC_PAD_LEN')
    parser.add_argument('--qrl_len', type=int, default=4,
                        help='QRL_PAD_LEN')

    parser.add_argument('--model', nargs='?', default='GRMM',
                        help='Specify the name of model.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='')
    parser.add_argument('--verbose', type=int, default=1,
                        help='')
    parser.add_argument('--seed', type=int, default=1,
                        help='')

    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epoch.')
    parser.add_argument('--idf', type=int, default=1,
                        help='idf_flag')
    parser.add_argument('--layers', type=int, default=2,
                        help='num of layers')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--topk', type=int, default=40,
                        help='topk')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--dp', type=float, default=0,
                        help='Dropout rate.')
    parser.add_argument('--l2', type=float, default=0,
                        help='l2 norm.')

    return parser.parse_args()
