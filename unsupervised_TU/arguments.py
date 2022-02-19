import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', default='NCI1', help='NCI1,PTC_MR,IMDB-BINARY,IMDB-MULTI,REDDIT-BINARY')
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)
    parser.add_argument('--device', default='cuda:6', type=str, help='gpu device ids')
    parser.add_argument('--lr', dest='lr', type=float, default= 0.01,
            help='Learning rate.')
    parser.add_argument('--alpha', default=1.2, type=float, help='stregnth for regularization')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32, help='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    # Random
    parser.add_argument('--eta', type=float, default=1.0, help='0.1, 1.0, 10, 100, 1000')
    parser.add_argument('--batch_size', type=int, default=128, help='128, 256, 512, 1024')     

    return parser.parse_args()

