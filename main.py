from StyleGAN import StyleGAN
import argparse
from utils import *

"""parsing and configuration"""


def parse_args():
    desc = "Tensorflow implementation of StyleGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train, test, eval, draw]')
    parser.add_argument('--draw', type=str, default='uncurated', help='[uncurated, style_mix, truncation_trick]')
    parser.add_argument('--dataset', type=str, default='ffhq', help='The dataset name what you want to generate')

    parser.add_argument('--iteration', type=int, default=1200, help='The number of images (K) used in train phase')
    parser.add_argument('--max_iteration', type=int, default=25000, help='The number of images (K) for last resolution')
    parser.add_argument('--resume_snapshot', type=int, default=-1, help='Specify which checkpoint to load')

    parser.add_argument('--batch_size_test', type=int, default=1, help='The size of batch in the test phase')
    parser.add_argument('--gpu_num', type=int, default=1, help='The number of gpu')

    parser.add_argument('--progressive', type=str2bool, default=True, help='use progressive training')
    parser.add_argument('--sn', type=str2bool, default=False, help='use spectral normalization')
    parser.add_argument('--use_noise', type=str2bool, default=False, help='use noise')
    parser.add_argument('--use_instance_norm', type=str2bool, default=True, help='use instance norm')
    parser.add_argument('--use_style_mod', type=str2bool, default=True, help='use style mod')

    # AC-GAN
    parser.add_argument('--inp_res', type=int, default=4, help='downscaled input resolution')
    parser.add_argument('--style_res', type=int, default=32, help='differentiating coarse (inclusive) and fine codes')
    parser.add_argument('--input_layer_type', type=str, default='const', help='input layer type: [const, down, const_var]')
    parser.add_argument('--labels_fine', type=str, default='', help='fine styles, e.g. "0,1,2" ')
    parser.add_argument('--labels_coarse', type=str, default='', help='coarse styles, e.g. "3,4,5" ')
    parser.add_argument('--D_mode', type=str, default='separate', help='how to apply AC-GAN [separate, together]')
    parser.add_argument('--cond_weight', type=float, default=1.0, help='gamma in the AC-GAN condition term')
    parser.add_argument('--labels_keep_rate', type=float, default=0.5, help='label rate for semi-supervised training')
    parser.add_argument('--cond_type', type=str, default='L2', help='the AC-GAN condition type [L1, L2]')
    parser.add_argument('--use_z', type=str2bool, default=True, help='use latent z, overridden by latent_size=0')

    # Metrics
    parser.add_argument('--FactorVAE', type=str2bool, default=True, help='if using the metric FactorVAE')
    parser.add_argument('--FID', type=str2bool, default=True, help='if using the metric FID')
    parser.add_argument('--MIG', type=str2bool, default=True, help='if using the metric MIG')
    parser.add_argument('--L2', type=str2bool, default=True, help='if using the metric L2')

    parser.add_argument('--img_size', type=int, default=1024, help='The target size of image')
    parser.add_argument('--num_channels', type=int, default=3, help='The number of image channels')
    parser.add_argument('--test_num', type=int, default=16, help='The number of generating images in the test phase')

    parser.add_argument('--seed', type=int, default=1, help='seed in the draw phase')

    parser.add_argument('--data_dir', type=str, default='data', help='Directory to load dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Dir to save the checkpoints')
    parser.add_argument('--test_dir', type=str, default='tests', help='Dir to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Dir to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples', help='Dir to save the samples on training')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --batch_size
    try:
        assert args.batch_size_test >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        gan = StyleGAN(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train':
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test':
            gan.test()
            print(" [*] Test finished!")

        if args.phase == 'eval':
            gan.eval_disentanglement()
            print(" [*] Test finished!")

        if args.phase == 'draw':
            gan.draw_uncurated_result_figure()
            print(" [*] Un-curated finished!")


if __name__ == '__main__':
    main()
