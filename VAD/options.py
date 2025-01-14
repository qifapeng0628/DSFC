import argparse
parser = argparse.ArgumentParser(description='Robust Backbone for WSVAD by DSFC')
parser.add_argument('--device', type=int, default=0, help='GPU ID')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--model_name', default='model_single', help=' ')
parser.add_argument('--loss_type', default='DMIL_C', type=str,  help='the type of n_pair loss, max_min_2, max_min, attention, attention_median, attention_H_L or max')
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--inference_model', type=str, default="\VAD\shanghaitech.pkl")
parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--testing_path', type=str, default=None, help='time file for test model')
parser.add_argument('--testing_model', type=str, default=None, help='iteration name for testing model')
parser.add_argument('--feature_size', type=int, default=1408, help='size of feature (default: 1408)')
parser.add_argument('--batch_size',  type=int, default=1, help='number of samples in one iteration')
parser.add_argument('--sample_size',  type=int, default=10, help='number of samples in one iteration')
parser.add_argument('--sample_step', type=int, default=1, help='')
parser.add_argument('--dataset_name', type=str, default='shanghaitech8', help='')
parser.add_argument('--dataset_path', type=str, default='C:\\Users\\pengq\\Desktop\\my_workSH\\', help='path to dir contains anomaly datasets')
parser.add_argument('--feature_modal', type=str, default='combine', help='features from different input, options contain rgb, flow , combine')
parser.add_argument('--max-seqlen', type=int, default=300, help='shanghaitech and UBnormal set to 300,ucf_crime set to 2000')
parser.add_argument('--Lambda', type=str, default='1_20', help='')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max_epoch', type=int, default=10, help='maximum iteration to train (default: 50000)')
parser.add_argument('--feature_pretrain_model', type=str, default='i3d', help='type of feature to be used I3D or C3D (default: I3D)')
parser.add_argument('--feature_layer', type=str, default='fc6', help='fc6 or fc7')
parser.add_argument('--k', type=int, default=10, help='value of k')
parser.add_argument('--plot', type=int, default=1, help='whether plot the video anomalous map on testing')
# parser.add_argument('--rank', type=int, default=0, help='')
# parser.add_argument('--loss_instance_type', type=str, default='weight', help='mean, weight, weight_center or individual')
# parser.add_argument('--MIL_loss_type', type=str, default='CE', help='CE or MSE')
parser.add_argument('--larger_mem', type=int, default=0, help='')
# parser.add_argument('--u_ratio', type=int, default=10, help='')
# parser.add_argument('--anomaly_smooth', type=int, default=1,
#                     help='type of smooth function, all or normal')
# parser.add_argument('--sparise_term', type=int, default=1,
#                     help='type of smooth function, all or normal')
# parser.add_argument('--attention_type', type=str, default='softmax',
#                     help='type of normalization of attention vector, softmax or sigmoid')
# parser.add_argument('--confidence', type=float, default=0, help='anomaly sample threshold')
parser.add_argument('--snapshot', type=int, default=100, help='anomaly sample threshold')
# parser.add_argument('--ps', type=str, default='normal_loss_mean')
parser.add_argument('--label_type', type=str, default='unary')
parser.add_argument('--lambda2', type=int, default=1800, help='')
parser.add_argument('--inference', type=str, default='C:\\Users\\pengq\\Desktop\\my_workSH\\shbest\\iter_300.pkl', help='')


