import time
import math
from ops import *
from utils import *
import numpy as np
import functools
import PIL.Image
from tqdm import tqdm
from metric import *
from data.dataset import make_input_fn, get_named_ground_truth_data, make_labels_fn


class StyleGAN(object):

    def __init__(self, sess, args):
        self.phase = args.phase
        self.progressive = args.progressive
        self.model_name = "StyleGAN"
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.test_dir = args.test_dir
        self.log_dir = args.log_dir
        self.data_dir = args.data_dir

        # AC-GAN
        self.inp_res = args.inp_res
        self.input_layer_type = args.input_layer_type
        self.use_noise = args.use_noise
        self.use_instance_norm = args.use_instance_norm
        self.use_style_mod = args.use_style_mod

        self.D_mode = args.D_mode
        self.cond_weight = args.cond_weight
        self.cond_type = args.cond_type
        self.use_z = args.use_z
        self.style_res = args.style_res

        self.labels_fine = args.labels_fine
        self.labels_coarse = args.labels_coarse
        self.labels_fine_list = list(map(int, self.labels_fine.split(','))) if self.labels_fine != '' else []
        self.labels_coarse_list = list(map(int, self.labels_coarse.split(','))) if self.labels_coarse != '' else []
        self.fine_label_size = len(self.labels_fine_list)
        self.coarse_label_size = len(self.labels_coarse_list)

        self.labels_keep_rate = args.labels_keep_rate

        # Metrics
        self.FactorVAE = args.FactorVAE
        self.FID = args.FID
        self.MIG = args.MIG
        self.L2 = args.L2

        self.iteration = args.iteration * 1000
        self.max_iteration = args.max_iteration * 1000
        self.resume_snapshot = args.resume_snapshot

        self.batch_size_test = args.batch_size_test
        self.img_size = args.img_size
        self.num_channels = args.num_channels

        """ Hyper-parameter """
        self.start_res = args.inp_res * 2
        self.resolutions = resolution_list(self.img_size, self.inp_res)  # inp_res=4: [4, 8, 16, 32, 64, 128, 256, 512, 1024 ...]
        self.featuremaps = featuremap_list(self.img_size, self.inp_res)  # inp_res=4: [512, 512, 512, 512, 256, 128, 64, 32, 16, ...]

        if not self.progressive:
            self.resolutions = [self.resolutions[-1]]
            self.featuremaps = [self.featuremaps[-1]]
            self.start_res = self.resolutions[-1]

        self.gpu_num = args.gpu_num

        self.z_dim = 512
        self.w_dim = 512
        self.n_mapping = 8

        self.w_ema_decay = 0.995  # Decay for tracking the moving average of W during training
        self.truncation_psi = 0.7  # Style strength multiplier for the truncation trick
        self.truncation_cutoff = 8  # Number of layers for which to apply the truncation trick

        self.batch_size_base = 4
        self.learning_rate_base = 0.001

        self.train_with_trans = {4: False, 8: False, 16: True, 32: True, 64: True, 128: True, 256: True, 512: True,
                                 1024: True}
        self.batch_sizes = get_batch_sizes(self.gpu_num)

        self.end_iteration = get_end_iteration(self.iteration, self.max_iteration, self.train_with_trans,
                                               self.resolutions, self.start_res)

        self.g_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        self.d_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

        self.sn = args.sn

        self.print_freq = {4: 1000, 8: 1000, 16: 4000, 32: 8000, 64: 16000, 128: 16000, 256: 16000, 512: 16000, 1024: 16000}
        self.save_freq = {4: 3000, 8: 3000, 16: 12000, 32: 24000, 64: 48000, 128: 48000, 256: 48000, 512: 48000, 1024: 48000}

        self.print_freq.update((x, y // self.gpu_num) for x, y in self.print_freq.items())
        self.save_freq.update((x, y // self.gpu_num) for x, y in self.save_freq.items())

        self.test_num = args.test_num
        self.seed = args.seed

        # Obtain the dataset.
        self.dataset = get_named_ground_truth_data(self.data_dir, self.labels_fine_list, self.labels_coarse_list,
                                                   self.dataset_name)

        self.sample_dir = os.path.join(self.model_dir, self.sample_dir)
        check_folder(self.sample_dir)

        self.log_dir = os.path.join(self.model_dir, self.log_dir)
        check_folder(self.log_dir)

        self.checkpoint_dir = os.path.join(self.model_dir, self.checkpoint_dir)
        check_folder(self.checkpoint_dir)

        self.test_dir = os.path.join(self.model_dir, self.test_dir)
        check_folder(self.test_dir)

        self.logger = Logger(file_name=os.path.join(self.log_dir, "log_{}.txt".format(self.phase)),
                             file_mode="w", should_flush=True)

        # set metrics
        self.metrics = group_metrics(labels_list=self.labels_fine_list+self.labels_coarse_list,
                                     is_FactorVAE=self.FactorVAE, is_FID=self.FID, is_MIG=self.MIG, is_L2=self.L2)
        for metric in self.metrics:
                metric.set_model(self)

        print()

        print("##### Information #####")
        print("# dataset : ", self.dataset_name)
        print("# dataset number : ", self.dataset.num_samples)
        print("# dataset image size : ", self.dataset.data_shape)

        print("# labels_fine_list : ", str(self.labels_fine_list))
        print("# labels_coarse_list : ", str(self.labels_coarse_list))
        print("# gpu : ", self.gpu_num)
        print("# batch_size in train phase : ", self.batch_sizes)
        print("# batch_size in test phase : ", self.batch_size_test)

        print("# input resolution : ", self.inp_res)
        print("# start resolution : ", self.start_res)
        print("# target resolution : ", self.img_size)
        print("# iteration per resolution : ", self.iteration)

        print("# progressive training : ", self.progressive)
        print("# spectral normalization : ", self.sn)

        print()

    ##################################################################################
    # Generator
    ##################################################################################

    def g_mapping(self, label, z, n_broadcast, style_name='', use_z=True, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('g_mapping_{}'.format(style_name), reuse=reuse):
            # normalize input first
            x = pixel_norm(z)

            z_size = z.shape[1].value
            label_size = label.shape[1].value
            if label_size > 0:
                # Embed labels and concatenate them with latents.
                with tf.variable_scope('LabelConcat'):
                    w = tf.get_variable('weight', shape=[label_size, z_size],
                                        initializer=tf.initializers.random_normal())
                    y = tf.matmul(label, tf.cast(w, 'float32'))
                    x = tf.concat([x, y], axis=1) if use_z else y

            # run through mapping network
            for i in range(self.n_mapping):
                with tf.variable_scope('FC_{:d}'.format(i)):
                    x = fully_connected(x, units=self.w_dim, gain=np.sqrt(2), lrmul=0.01, sn=self.sn)
                    x = apply_bias(x, lrmul=0.01)
                    x = lrelu(x, alpha=0.2)

            # broadcast to n_layers
            with tf.variable_scope('Broadcast'):
                x = tf.tile(x[:, np.newaxis], [1, n_broadcast, 1])

        return x

    def g_synthesis(self, images_in, w_bc_fine, w_bc_coarse, alpha, resolutions, featuremaps, num_channels,
                    inp_res=4, input_layer_type='const', use_noise=False, use_instance_norm=True, use_style_mod=True,
                    reuse=tf.AUTO_REUSE):
        with tf.variable_scope('g_synthesis', reuse=reuse):
            coarse_styles, middle_styles, fine_styles = get_style_class(resolutions, featuremaps)
            layer_index = 2

            """ initial layer """
            res = resolutions[0]
            n_f = featuremaps[0]

            assert res == inp_res
            w_broadcasted = w_bc_fine if res >= self.style_res else w_bc_coarse
            x = synthesis_const_downscaled_block(images_in, w_broadcasted, inp_res, input_layer_type, resolutions[-1],
                                                 n_f, use_noise, use_instance_norm, use_style_mod, self.sn)

            """ remaining layers """
            if self.progressive:
                images_out = torgb(x, res, num_channels, sn=self.sn)  # Note: 8x8 block takes in 4x4 input

                # Coarse style [4 ~ 8]
                # pose, hair, face shape
                coarse_styles.pop(res, None)
                for res, n_f in coarse_styles.items():
                    w_broadcasted = w_bc_fine if res >= self.style_res else w_bc_coarse
                    x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, use_noise, use_instance_norm,
                                        use_style_mod, sn=self.sn)
                    img = torgb(x, res, num_channels, sn=self.sn)
                    images_out = upscale2d(images_out)
                    images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

                    layer_index += 2

                # Middle style [16 ~ 32]
                # facial features, eye  TODO: maybe we can mix fine and coarse styles here
                middle_styles.pop(res, None)
                for res, n_f in middle_styles.items():
                    w_broadcasted = w_bc_fine if res >= self.style_res else w_bc_coarse
                    x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, use_noise, use_instance_norm,
                                        use_style_mod, sn=self.sn)
                    img = torgb(x, res, num_channels, sn=self.sn)
                    images_out = upscale2d(images_out)
                    images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

                    layer_index += 2

                # Fine style [64 ~ 1024]
                # color scheme
                fine_styles.pop(res, None)
                for res, n_f in fine_styles.items():
                    w_broadcasted = w_bc_fine if res >= self.style_res else w_bc_coarse
                    x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, use_noise, use_instance_norm,
                                        use_style_mod, sn=self.sn)
                    img = torgb(x, res, num_channels, sn=self.sn)
                    images_out = upscale2d(images_out)
                    images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

                    layer_index += 2

            else:
                for res, n_f in zip(resolutions[1:], featuremaps[1:]):
                    w_broadcasted = w_bc_fine if res >= self.style_res else w_bc_coarse
                    x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, use_noise, use_instance_norm,
                                        use_style_mod, sn=self.sn)

                    layer_index += 2
                images_out = torgb(x, resolutions[-1], num_channels, sn=self.sn)

            return tf.nn.tanh(images_out)  # TODO: test

    def generator(self, images_in, labels_in, z, alpha, target_img_size, num_channels, inp_res, input_layer_type,
                  use_noise, use_instance_norm, use_style_mod, use_z, is_training=True, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("generator", reuse=reuse):
            resolutions = resolution_list(target_img_size, inp_res)
            featuremaps = featuremap_list(target_img_size, inp_res)

            w_avg_fine = tf.get_variable('w_avg_fine', shape=[self.w_dim],
                                         dtype=tf.float32, initializer=tf.initializers.zeros(),
                                         trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_TOWER)
            w_avg_coarse = tf.get_variable('w_avg_coarse', shape=[self.w_dim],
                                           dtype=tf.float32, initializer=tf.initializers.zeros(),
                                           trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_TOWER)

            """ fine and coarse labels """
            assert labels_in.shape[1].value == self.fine_label_size + self.coarse_label_size, labels_in.shape[1].value
            labels_fine = labels_in[:, :self.fine_label_size]
            labels_coarse = labels_in[:, self.fine_label_size:]
            assert labels_fine.shape[1].value == self.fine_label_size
            assert labels_coarse.shape[1].value == self.coarse_label_size

            """ mapping layers """
            n_broadcast = len(resolutions) * 2
            w_bc_fine = self.g_mapping(labels_fine, z, n_broadcast, style_name='fine', use_z=use_z)
            w_bc_coarse = self.g_mapping(labels_coarse, z, n_broadcast, style_name='coarse', use_z=use_z)

            if is_training:
                """ apply regularization techniques on training """
                # update moving average of w
                w_bc_fine = self.update_moving_average_of_w(w_bc_fine, w_avg_fine, style_name='fine')
                w_bc_coarse = self.update_moving_average_of_w(w_bc_coarse, w_avg_coarse, style_name='coarse')

            else:
                """ apply truncation trick on evaluation """
                w_bc_fine = self.truncation_trick(n_broadcast, w_bc_fine, w_avg_fine, self.truncation_psi,
                                                  style_name='fine')
                w_bc_coarse = self.truncation_trick(n_broadcast, w_bc_coarse, w_avg_coarse, self.truncation_psi,
                                                    style_name='coarse')

            """ synthesis layers """
            x = self.g_synthesis(images_in, w_bc_fine, w_bc_coarse, alpha, resolutions, featuremaps, num_channels,
                                 inp_res, input_layer_type, use_noise, use_instance_norm, use_style_mod)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, alpha, target_img_size, mode='', reuse=tf.AUTO_REUSE):
        with tf.variable_scope("discriminator", reuse=reuse):
            resolutions = resolution_list(target_img_size, 4)
            featuremaps = featuremap_list(target_img_size, 4)

            r_resolutions = resolutions[::-1]
            r_featuremaps = featuremaps[::-1]

            labels_fine = tf.zeros_like(tf.layers.flatten(x_init))[:, :0]  # [batch_size, 0]

            """ set inputs """
            x = fromrgb(x_init, r_resolutions[0], r_featuremaps[0], self.sn)

            """ stack discriminator blocks """
            for index, (res, n_f) in enumerate(zip(r_resolutions[:-1], r_featuremaps[:-1])):
                res_next = r_resolutions[index + 1]
                n_f_next = r_featuremaps[index + 1]

                x = discriminator_block(x, res, n_f, n_f_next, self.sn)

                if self.progressive:  # and index == 0
                    x_init = downscale2d(x_init)
                    y = fromrgb(x_init, res_next, n_f_next, self.sn)
                    x = smooth_transition(y, x, res, r_resolutions[0], alpha)

                if self.fine_label_size > 0 and mode == 'separate' and res == self.style_res:  # at 32x32 block
                    x_flat = tf.layers.flatten(x)
                    with tf.variable_scope('Dense_Fine'):
                        labels_fine = fully_connected(x_flat, units=self.fine_label_size, gain=1.0, lrmul=1.0, sn=self.sn)
                        labels_fine = apply_bias(labels_fine, lrmul=1.0)

            """ last block """
            res = r_resolutions[-1]
            n_f = r_featuremaps[-1]

            logit, labels_from_last = discriminator_last_block(
                x, self.fine_label_size, self.coarse_label_size, target_img_size, res, self.style_res, self.inp_res,
                n_f, n_f, mode, self.sn)

            labels = tf.concat([labels_fine, labels_from_last], axis=1)

            assert self.inp_res <= target_img_size
            if target_img_size < self.style_res:
                assert labels.shape[1].value == self.coarse_label_size, labels.shape[1].value
            else:
                if self.inp_res < self.style_res:
                    assert labels.shape[1].value == self.fine_label_size + self.coarse_label_size, labels.shape[1].value
                else:
                    assert labels.shape[1].value == self.fine_label_size, labels.shape[1].value

            return logit, labels

    ##################################################################################
    # Technical skills
    ##################################################################################

    def update_moving_average_of_w(self, w_broadcasted, w_avg, style_name=''):
        with tf.variable_scope('WAvg_{}'.format(style_name)):
            batch_avg = tf.reduce_mean(w_broadcasted[:, 0], axis=0)
            update_op = tf.assign(w_avg, lerp(batch_avg, w_avg, self.w_ema_decay))

            with tf.control_dependencies([update_op]):
                w_broadcasted = tf.identity(w_broadcasted)

        return w_broadcasted

    def truncation_trick(self, n_broadcast, w_broadcasted, w_avg, truncation_psi, style_name=''):
        with tf.variable_scope('truncation_{}'.format(style_name)):
            layer_indices = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_indices.shape, dtype=np.float32)
            coefs = tf.where(layer_indices < self.truncation_cutoff, truncation_psi * ones, ones)
            w_broadcasted = lerp(w_avg, w_broadcasted, coefs)

        return w_broadcasted

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph """
        if self.phase == 'train':
            self.d_loss_per_res = {}
            self.g_loss_per_res = {}
            self.label_pen_reals_per_res = {}
            self.label_pen_fakes_per_res = {}
            self.generator_optim = {}
            self.discriminator_optim = {}
            self.alpha_summary_per_res = {}
            self.d_summary_per_res = {}
            self.g_summary_per_res = {}
            self.label_reals_summary_per_res = {}
            self.label_fakes_summary_per_res = {}

            self.real_images_per_res = {}
            self.fake_images_per_res = {}
            self.labels_real_per_res = {}
            self.z_per_res = {}

            self.real_images_save_per_res = {}
            self.fake_images_save_per_res = {}
            self.labels_save_per_res = {}
            self.z_save_per_res = {}

            self.images_infer_per_res = {}
            self.labels_infer_per_res = {}

            self.image_class_pre_res = {}

            # Create a numpy random state. We will sample the random seeds for training and evaluation from this.
            random_state = np.random.RandomState(self.seed)

            for res in self.resolutions[self.resolutions.index(self.start_res):]:
                g_loss_per_gpu = []
                d_loss_per_gpu = []
                label_pen_reals_per_gpu = []
                label_pen_fakes_per_gpu = []
                real_images_per_gpu = []
                fake_images_per_gpu = []
                labels_real_per_gpu = []
                z_per_gpu = []
                fake_images_save_per_gpu = []
                labels_infer_per_gpu = []

                batch_size = self.batch_sizes.get(res, self.batch_size_base)
                global_step = tf.get_variable('global_step_{}'.format(res), shape=[], dtype=tf.float32,
                                              initializer=tf.initializers.zeros(),
                                              trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_TOWER)
                alpha_const, zero_constant = get_alpha_const(self.iteration // 2, batch_size * self.gpu_num, global_step)

                # smooth transition variable
                do_train_trans = self.train_with_trans[res]

                alpha = tf.get_variable('alpha_{}'.format(res), shape=[], dtype=tf.float32,
                                        initializer=tf.initializers.ones() if do_train_trans else tf.initializers.zeros(),
                                        trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_TOWER)

                if do_train_trans:
                    alpha_assign_op = tf.assign(alpha, alpha_const)
                else:
                    alpha_assign_op = tf.assign(alpha, zero_constant)

                # placeholders
                real_images_save_per_gpu = tf.placeholder(
                    dtype=tf.float32, shape=[None, self.num_channels, res, res],
                    name='real_image_save_{}'.format(res))
                real_images_save_list = tf.split(real_images_save_per_gpu, num_or_size_splits=self.gpu_num)
                labels_save_per_gpu = tf.placeholder(
                    dtype=tf.float32, shape=[None, self.fine_label_size + self.coarse_label_size],
                    name='labels_in_save_{}'.format(res))
                labels_save_list = tf.split(labels_save_per_gpu, num_or_size_splits=self.gpu_num)
                z_save_per_gpu = tf.placeholder(
                    dtype=tf.float32, shape=[None, self.z_dim],
                    name='z_save_{}'.format(res))
                z_save_list = tf.split(z_save_per_gpu, num_or_size_splits=self.gpu_num)

                images_infer_per_gpu = tf.placeholder(
                    dtype=tf.float32, shape=[None, self.num_channels, res, res],
                    name='image_infer_{}'.format(res))
                images_infer_list = tf.split(images_infer_per_gpu, num_or_size_splits=self.gpu_num)

                image_class = ImageData(res, self.num_channels)

                with tf.control_dependencies([alpha_assign_op]):
                    for gpu_id in range(self.gpu_num):
                        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):

                                load_dataset = make_input_fn(self.dataset, image_class, random_state.randint(2 ** 32), gpu_id)
                                # images in NCHW, labels in [N, fine_size + coarse_size]
                                real_img, labels_in_real, labels_real_mask = load_dataset(batch_size, shuffle=False)  # TODO: shuffle

                                sampling_labels = make_labels_fn(self.dataset, random_state.randint(2 ** 31), gpu_id)
                                # labels in [N, fine_size + coarse_size]
                                labels_in_fake = sampling_labels(batch_size, shuffle=False)  # TODO: shuffle

                                z = tf.random_normal(shape=[batch_size, self.z_dim], seed=self.seed)
                                real_img = smooth_crossfade(real_img, alpha)

                                fake_img = self.generator(real_img, labels_in_fake, z, alpha, res, self.num_channels,
                                                          self.inp_res, self.input_layer_type, self.use_noise,
                                                          self.use_instance_norm, self.use_style_mod, self.use_z)

                                # used for saving and testing
                                real_img_save = real_images_save_list[gpu_id]
                                labels_in_save = labels_save_list[gpu_id]
                                z_save = z_save_list[gpu_id]
                                fake_img_save = self.generator(real_img_save, labels_in_save, z_save, alpha, res,
                                                               self.num_channels, self.inp_res, self.input_layer_type,
                                                               self.use_noise, self.use_instance_norm, self.use_style_mod,
                                                               self.use_z, is_training=True)

                                real_logit, real_labels = self.discriminator(real_img, alpha, res, self.D_mode)
                                fake_logit, fake_labels = self.discriminator(fake_img, alpha, res, self.D_mode)

                                # used for inference from images
                                img_infer = images_infer_list[gpu_id]
                                _, labels_infer = self.discriminator(img_infer, alpha, res, self.D_mode)

                                # compute loss
                                d_loss, g_loss, label_penalty_reals, label_penalty_fakes \
                                    = compute_loss(real_img, labels_in_real, labels_real_mask, labels_in_fake,
                                                   self.fine_label_size, self.coarse_label_size, real_logit, fake_logit,
                                                   real_labels, fake_labels, self.style_res, self.inp_res, self.cond_weight,
                                                   self.cond_type, self.labels_keep_rate, self.seed)

                                d_loss_per_gpu.append(d_loss)
                                g_loss_per_gpu.append(g_loss)
                                label_pen_reals_per_gpu.append(label_penalty_reals)
                                label_pen_fakes_per_gpu.append(label_penalty_fakes)
                                real_images_per_gpu.append(real_img)
                                fake_images_per_gpu.append(fake_img)
                                labels_real_per_gpu.append(labels_in_real)
                                z_per_gpu.append(z)
                                fake_images_save_per_gpu.append(fake_img_save)
                                labels_infer_per_gpu.append(labels_infer)

                print("Create graph for {} resolution".format(res))

                # prepare appropriate training vars
                d_vars, g_vars = filter_trainable_variables(res, self.inp_res, self.style_res)

                print('g_vars: ', g_vars)
                print('d_vars: ', d_vars)

                d_loss = tf.reduce_mean(d_loss_per_gpu)
                g_loss = tf.reduce_mean(g_loss_per_gpu)

                label_penalty_reals = tf.reduce_mean(label_pen_reals_per_gpu)
                label_penalty_fakes = tf.reduce_mean(label_pen_fakes_per_gpu)

                d_lr = self.d_learning_rates.get(res, self.learning_rate_base)
                g_lr = self.g_learning_rates.get(res, self.learning_rate_base)

                if self.gpu_num == 1:
                    colocate_grad = False
                else:
                    colocate_grad = True

                d_optim = tf.train.AdamOptimizer(d_lr, beta1=0.0, beta2=0.99, epsilon=1e-8). \
                    minimize(d_loss, var_list=d_vars, colocate_gradients_with_ops=colocate_grad)

                g_optim = tf.train.AdamOptimizer(g_lr, beta1=0.0, beta2=0.99, epsilon=1e-8). \
                    minimize(g_loss, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=colocate_grad)

                self.discriminator_optim[res] = d_optim
                self.generator_optim[res] = g_optim

                self.d_loss_per_res[res] = d_loss
                self.g_loss_per_res[res] = g_loss

                self.label_pen_reals_per_res[res] = label_penalty_reals
                self.label_pen_fakes_per_res[res] = label_penalty_fakes

                self.real_images_per_res[res] = tf.concat(real_images_per_gpu, axis=0)
                self.fake_images_per_res[res] = tf.concat(fake_images_per_gpu, axis=0)
                self.labels_real_per_res[res] = tf.concat(labels_real_per_gpu, axis=0)
                self.z_per_res[res] = tf.concat(z_per_gpu, axis=0)

                self.real_images_save_per_res[res] = real_images_save_per_gpu
                self.fake_images_save_per_res[res] = tf.concat(fake_images_save_per_gpu, axis=0)
                self.labels_save_per_res[res] = labels_save_per_gpu
                self.z_save_per_res[res] = z_save_per_gpu

                self.images_infer_per_res[res] = images_infer_per_gpu
                self.labels_infer_per_res[res] = tf.concat(labels_infer_per_gpu, axis=0)

                self.image_class_pre_res[res] = image_class

                """ Summary """
                self.alpha_summary_per_res[res] = tf.summary.scalar("alpha/alpha_{}".format(res), alpha)

                self.d_summary_per_res[res] = tf.summary.scalar("d_loss/d_loss_{}".format(res), self.d_loss_per_res[res])
                self.g_summary_per_res[res] = tf.summary.scalar("g_loss/g_loss_{}".format(res), self.g_loss_per_res[res])

                self.label_reals_summary_per_res[res] = tf.summary.scalar("label_real/label_real_{}".format(res),
                                                                          self.label_pen_reals_per_res[res])
                self.label_fakes_summary_per_res[res] = tf.summary.scalar("label_fake/label_fake_{}".format(res),
                                                                          self.label_pen_fakes_per_res[res])

        else:
            """" Testing """
            random_state = np.random.RandomState(self.seed)
            image_class = ImageData(self.img_size, self.num_channels)
            self.load_dataset = make_input_fn(self.dataset, image_class, random_state.randint(2 ** 32), gpu_id=0)
            self.real_img_test, self.labels_in_test, _ = self.load_dataset(self.batch_size_test, shuffle=False)
            self.test_z = tf.random_normal(shape=[self.batch_size_test, self.z_dim])
            alpha = tf.constant(0.0, dtype=tf.float32, shape=[])
            self.fake_img_test = self.generator(self.real_img_test, self.labels_in_test, self.test_z, alpha, self.img_size,
                                              self.num_channels, self.inp_res, self.input_layer_type, self.use_noise,
                                              self.use_instance_norm, self.use_style_mod, self.use_z, is_training=False)
            _, self.labels_infer_test = self.discriminator(self.fake_img_test, alpha, self.img_size, self.D_mode)

    ##################################################################################
    # Train
    ##################################################################################

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=15)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # restore check-point if it exits (the latest one)
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:

            start_res_idx = get_checkpoint_res(checkpoint_counter, self.batch_sizes, self.iteration,
                                               self.start_res, self.img_size, self.gpu_num,
                                               self.end_iteration, self.train_with_trans)

            if not self.progressive:
                start_res_idx = 0

            start_batch_idx = checkpoint_counter

            for res_idx in range(self.resolutions.index(self.start_res), start_res_idx):
                res = self.resolutions[res_idx]
                batch_size_per_res = self.batch_sizes.get(res, self.batch_size_base) * self.gpu_num

                if self.train_with_trans[res]:
                    if res == self.img_size:
                        iteration = self.end_iteration
                    else:
                        iteration = self.iteration
                else:
                    iteration = self.iteration // 2

                if start_batch_idx - (iteration // batch_size_per_res) < 0:
                    break
                else:
                    start_batch_idx = start_batch_idx - (iteration // batch_size_per_res)

            counter = checkpoint_counter
            print(" [*] Load SUCCESS")

        else:
            start_res_idx = self.resolutions.index(self.start_res)
            start_batch_idx = 0
            counter = 1
            print(" [!] Load failed...")

        start_time = time.time()
        for current_res_num in range(start_res_idx, len(self.resolutions)):

            current_res = self.resolutions[current_res_num]
            batch_size_per_res = self.batch_sizes.get(current_res, self.batch_size_base) * self.gpu_num

            if self.progressive:
                if self.train_with_trans[current_res]:

                    if current_res == self.img_size:
                        current_iter = self.end_iteration // batch_size_per_res
                    else:
                        current_iter = self.iteration // batch_size_per_res
                else:
                    current_iter = (self.iteration // 2) // batch_size_per_res

            else:
                current_iter = self.end_iteration

            # save real samples
            print('Starting running real_images_save...')
            real_samples_np, labels_real_np, z_np = self.sess.run([self.real_images_per_res[current_res],
                                                             self.labels_real_per_res[current_res],
                                                             self.z_per_res[current_res]])
            print('real_samples_np size: {}, type: {}'.format(real_samples_np.shape, real_samples_np.dtype))
            print('labels_real_np size: {}, type: {}'.format(labels_real_np.shape, labels_real_np.dtype))
            print('z_np size: {}, type: {}'.format(z_np.shape, z_np.dtype))
            manifold_h = int(np.floor(np.sqrt(batch_size_per_res)))
            manifold_w = int(np.floor(np.sqrt(batch_size_per_res)))
            print('Starting saving real samples...')
            save_images(real_samples_np[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                        './{}/real_img_{:04d}.png'.format(self.sample_dir, current_res))

            if current_res == self.resolutions[start_res_idx] or current_res == self.img_size:
                print('Starting evaluating metrics...')
                for metric in self.metrics:
                    scores_dict = metric.evaluate(self.dataset, current_res, log_func=print)
                    eval_msg = metric.name
                    for eval_name, score in scores_dict.items():
                        eval_msg += ', ' + eval_name + ': ' + str(score)
                    print(eval_msg)

            print('Starting training...')
            for idx in range(start_batch_idx, current_iter):

                # update D network
                _, summary_d_per_res, d_loss, summary_label_pen_reals_per_res = \
                    self.sess.run([self.discriminator_optim[current_res], self.d_summary_per_res[current_res],
                                   self.d_loss_per_res[current_res], self.label_reals_summary_per_res[current_res]])

                self.writer.add_summary(summary_d_per_res, idx)
                self.writer.add_summary(summary_label_pen_reals_per_res, idx)

                # update G network
                _, summary_g_per_res, summary_alpha, g_loss, summary_label_pen_fakes_per_res = \
                    self.sess.run([self.generator_optim[current_res], self.g_summary_per_res[current_res],
                                   self.alpha_summary_per_res[current_res], self.g_loss_per_res[current_res],
                                   self.label_fakes_summary_per_res[current_res]])

                self.writer.add_summary(summary_g_per_res, idx)
                self.writer.add_summary(summary_alpha, idx)
                self.writer.add_summary(summary_label_pen_fakes_per_res, idx)

                # display training status
                counter += 1

                if np.mod(idx + 1, self.print_freq[current_res]) == 0:
                    msg = "Current res: [%4d] [%6d/%6d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % \
                          (current_res, idx, current_iter, time.time() - start_time, d_loss, g_loss)
                    print(msg)

                if np.mod(idx + 1, 2 * self.save_freq[current_res]) == 0:
                    fake_samples_np = self.generate_with_control(real_samples_np, labels_real_np, z_np)
                    print('fake samples shape: {}'.format(fake_samples_np.shape))

                    save_images(fake_samples_np[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './{}/fake_img_{:04d}_{:06d}_{:06d}.png'.format(
                                    self.sample_dir, current_res, idx + 1, counter))

                if np.mod(idx + 1, 4 * self.save_freq[current_res]) == 0:
                    self.save(self.checkpoint_dir, counter)

                    if current_res == self.img_size:
                        """only evaluate metrics in the last resolution here"""
                        print('Starting evaluating metrics...')
                        for metric in self.metrics:
                            scores_dict = metric.evaluate(self.dataset, current_res, log_func=print)
                            eval_msg = metric.name
                            for eval_name, score in scores_dict.items():
                                eval_msg += ', ' + eval_name + ': ' + str(score)
                                summary = tf.Summary(
                                    value=[tf.Summary.Value(
                                        tag="metric_{}/{}/{}".format(current_res, metric.name, eval_name),
                                        simple_value=score)])
                                self.writer.add_summary(summary, idx)
                            print(eval_msg)

            # After an epoch, start_batch_idx is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_idx = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            print('Starting evaluating metrics...')
            for metric in self.metrics:
                scores_dict = metric.evaluate(self.dataset, current_res, log_func=print)
                eval_msg = metric.name
                for eval_name, score in scores_dict.items():
                    eval_msg += ', ' + eval_name + ': ' + str(score)
                print(eval_msg)

        # save model for final step
        self.save(self.checkpoint_dir, counter)
        self.logger.close()

    def inference_from(self, images, res, batch_size):
        # resize img to res shape and NHWC -> NCHW
        images = self.image_class_pre_res[res].images_processing_np(images)
        assert images.shape[-1] == res, images.shape[-1]
        assert images.shape[0] > 0, images.shape[0]
        latents = []
        for i in range(int(math.ceil(float(images.shape[0]) / batch_size))):
            sub_latents = self.sess.run(
                self.labels_infer_per_res[res],
                feed_dict={self.images_infer_per_res[res]: images[i * batch_size: (i + 1) * batch_size]})
            latents.append(sub_latents)
        # print('Total latents inferred at res {}: {}'.format(res, len(latents)))
        return np.concatenate(latents, axis=0)

    def inference_test_from(self, images, res, batch_size):
        # resize img to res shape and NHWC -> NCHW
        images = self.image_class_pre_res[res].images_processing_np(images)
        assert images.shape[-1] == res, images.shape[-1]
        assert images.shape[0] > 0, images.shape[0]
        latents = []
        for i in range(int(math.ceil(float(images.shape[0]) / batch_size))):
            sub_latents = self.sess.run(
                self.labels_infer_per_res[res],
                feed_dict={self.images_infer_per_res[res]: images[i * batch_size: (i + 1) * batch_size]})
            latents.append(sub_latents)
        # print('Total latents inferred at res {}: {}'.format(res, len(latents)))
        return np.concatenate(latents, axis=0)

    def generate_randomly(self, res, num_images):
        """output is numpy.array representing images (NCHW, [-1, 1])"""
        batch_size = self.batch_sizes[res] * self.gpu_num
        images = []
        for begin in range(0, num_images, batch_size):
            end = min(begin + batch_size, num_images)
            images.append(self.sess.run(self.fake_images_per_res[res])[:end - begin])
        images = np.concatenate(images, axis=0)
        assert images.shape[0] == num_images, images.shape
        return images

    def generate_with_control(self, real_samples_np, labels_real_np, z_np):
        """output is numpy.array representing images (NCHW, [-1, 1])"""
        assert real_samples_np.shape[0] == labels_real_np.shape[0] == z_np.shape[0]
        num_images = real_samples_np.shape[0]
        if self.phase == 'train':
            res = real_samples_np.shape[-1]
            batch_size = self.batch_sizes[res] * self.gpu_num
            images = []
            for begin in range(0, num_images, batch_size):
                end = min(begin + batch_size, num_images)
                fake_samples_np = self.sess.run(
                    self.fake_images_save_per_res[res],
                    feed_dict={self.real_images_save_per_res[res]: real_samples_np[begin: end],
                               self.labels_save_per_res[res]: labels_real_np[begin: end],
                               self.z_save_per_res[res]: z_np[begin: end]})
                images.append(fake_samples_np)
            images = np.concatenate(images, axis=0)
            assert images.shape[0] == num_images, images.shape
            return images
        else:
            batch_size = self.batch_size_test
            images = []
            labels = []
            for begin in range(0, num_images, batch_size):
                end = min(begin + batch_size, num_images)
                fake_samples_np, labels_infer_np = self.sess.run(
                    [self.fake_img_test, self.labels_infer_test],
                    feed_dict={self.real_img_test: real_samples_np[begin: end],
                               self.labels_in_test: labels_real_np[begin: end],
                               self.test_z: z_np[begin: end]})
                images.append(fake_samples_np)
                labels.append(labels_infer_np)
            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
            assert images.shape[0] == num_images, images.shape
            assert labels.shape[0] == num_images, labels.shape
            return images, labels

    def sample_randomly(self, res, num_images):
        """output is numpy.array representing images (NCHW, [-1, 1])"""
        batch_size = self.batch_sizes[res] * self.gpu_num
        images = []
        for begin in range(0, num_images, batch_size):
            end = min(begin + batch_size, num_images)
            images.append(self.sess.run(self.real_images_per_res[res])[:end - begin])
        images = np.concatenate(images, axis=0)
        assert images.shape[0] == num_images, images.shape
        return images

    @property
    def model_dir(self):

        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        if self.progressive:
            progressive = '_progressive'
        else:
            progressive = ''

        if self.use_z:
            use_z = '_z'
        else:
            use_z = ''

        if self.use_noise:
            use_noise = '_noise'
        else:
            use_noise = ''

        if self.use_instance_norm:
            use_instance_norm = '_instnorm'
        else:
            use_instance_norm = ''

        if self.use_style_mod:
            use_style_mod = '_instnorm'
        else:
            use_style_mod = ''

        return "results/{}_{}_{}to{}_inpRes{}_styleRes{}_{}_{}_Fine{}_Coarse{}{}{}{}{}{}{}".format(
            self.model_name, self.dataset_name, self.start_res, self.img_size, self.inp_res,
            self.style_res, self.D_mode, self.input_layer_type, self.labels_fine,
            self.labels_coarse, progressive, sn, use_z, use_noise, use_instance_norm, use_style_mod)

    def save(self, checkpoint_dir, step):
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir, resume_snapshot=-1):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if resume_snapshot > 0:  # specify which checkpoint to load
            if ckpt and len(ckpt.all_model_checkpoint_paths) > 0:
                for model_checkpoint_path in ckpt.all_model_checkpoint_paths:
                    ckpt_name = os.path.basename(model_checkpoint_path)
                    count = int(ckpt_name.split('-')[-1])
                    if count == resume_snapshot:
                        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                        print(" [*] Success to read {}".format(ckpt_name))
                        return True, resume_snapshot
                print(" [*] Failed to find a checkpoint")
                return False, 0

        else:  # load the latest checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                counter = int(ckpt_name.split('-')[-1])
                print(" [*] Success to read {}".format(ckpt_name))
                return True, counter
            else:
                print(" [*] Failed to find a checkpoint")
                return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        image_frame_dim = int(np.floor(np.sqrt(self.batch_size_test)))

        # real images (testing)
        real_img_test_np = self.sess.run(self.real_img_test)
        save_images(real_img_test_np[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    '{}/test_real_img_{}.png'.format(self.test_dir, self.img_size))

        for i in tqdm(range(self.test_num)):

            if self.batch_size_test == 1:
                seed = np.random.randint(low=0, high=10000)
                random_state = np.random.RandomState(seed)
                image_class = ImageData(self.img_size, self.num_channels)
                load_dataset = make_input_fn(self.dataset, image_class, random_state.randint(2 ** 32), gpu_id=0)
                real_img_test, labels_in_test, _ = load_dataset(self.batch_size_test, shuffle=False)

                test_z = tf.cast(np.random.RandomState(seed).normal(size=[self.batch_size_test, self.z_dim]), tf.float32)
                alpha = tf.constant(0.0, dtype=tf.float32, shape=[])
                self.fake_img_test = self.generator(real_img_test, labels_in_test, test_z, alpha, self.img_size,
                                                    self.num_channels, self.inp_res, self.input_layer_type,
                                                    self.use_noise, self.use_instance_norm, self.use_style_mod,
                                                    self.use_z, is_training=False)
                samples = self.sess.run(self.fake_img_test)

                save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                            '{}/test_fake_img_{}_{}_{}.png'.format(self.test_dir, self.img_size, i, seed))

            else:
                samples = self.sess.run(self.fake_img_test)
                print('Maximum values in samples: ', np.max(samples))
                print('Minimum values in samples: ', np.min(samples))

                save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                            '{}/test_fake_img_{}_{}.png'.format(self.test_dir, self.img_size, i))

    def eval_disentanglement(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir, self.resume_snapshot)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        num_save = 4
        assert self.batch_size_test >= num_save

        image_frame_dim = int(np.floor(np.sqrt(num_save)))

        eval_dir = os.path.join(self.test_dir, 'random_seed%d' % self.seed)
        os.makedirs(eval_dir, exist_ok=True)

        # real images (testing)
        real_img_np, labels_in_np, z_np = self.sess.run([self.real_img_test, self.labels_in_test, self.test_z])
        save_images(real_img_np[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    '{}/test_real_img_{}_{:06d}.png'.format(eval_dir, self.img_size, self.resume_snapshot))

        # interpolate labels
        labels_list = self.labels_fine_list + self.labels_coarse_list
        labels_inf_vars_mean_dict = {}
        labels_inf_diffs_mean_dict = {}
        for index in range(len(labels_list)):
            labels_inf_vars_mean, labels_inf_diffs_mean = interpolate_labels(
                self.generate_with_control, real_img_np, labels_in_np, z_np, index, labels_list, eval_dir,
                resume_snapshot=self.resume_snapshot, num_save=num_save)
            labels_inf_vars_mean_dict['factor{}'.format(labels_list[index])] = labels_inf_vars_mean
            labels_inf_diffs_mean_dict['factor{}'.format(labels_list[index])] = labels_inf_diffs_mean
        print('labels_inf_diffs_mean_dict: ', labels_inf_vars_mean_dict)
        print('labels_inf_diffs_mean_dict: ', labels_inf_diffs_mean_dict)

        # randomize latent z
        randomize_latent(self.generate_with_control, real_img_np, labels_in_np, z_np, eval_dir,
                         resume_snapshot=self.resume_snapshot)

        if self.dataset_name == 'isaac3d':
            eval_est_dir = os.path.join(self.test_dir, 'random_seed%d_test' % self.seed)
            os.makedirs(eval_est_dir, exist_ok=True)

            import glob
            image_test_dir = os.path.join(self.data_dir, 'Isaac3D_v1', 'test_images')
            images_test = sorted(glob.glob(os.path.join(image_test_dir, '*.png')))[:num_save]
            images_test = np.array(images_test)
            image_class = ImageData(self.img_size, self.num_channels)
            images_test = image_class.images_processing_np(images_test)

            save_images(images_test[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        '{}/test_real_img_{}_{:06d}.png'.format(eval_est_dir, self.img_size, self.resume_snapshot))

            # interpolate labels
            labels_list = self.labels_fine_list + self.labels_coarse_list
            labels_in_np_test = np.load(os.path.join(image_test_dir, 'test_labels.npy'))[:, labels_list]

            for index in range(len(labels_list)):
                interpolate_labels(self.generate_with_control, images_test, labels_in_np_test, z_np, index, labels_list,
                                   eval_est_dir, resume_snapshot=self.resume_snapshot, num_save=num_save)
            # randomize latent z
            randomize_latent(self.generate_with_control, images_test, labels_in_np, z_np, eval_est_dir,
                             resume_snapshot=self.resume_snapshot)

    def calculate_interpolation_variance(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir, self.resume_snapshot)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")



    def draw_uncurated_result_figure(self):
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        test_dir = os.path.join(self.test_dir, 'paper_figure')
        check_folder(test_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        lods = [0, 1, 2, 2, 3, 3]
        seed = 3291
        rows = 3
        cx = 0
        cy = 0

        alpha = tf.constant(0.0, dtype=tf.float32, shape=[])

        batch_size_test = sum(rows * 2 ** lod for lod in lods)
        if self.seed:
            latents = tf.cast(
                np.random.RandomState(seed).normal(size=[batch_size_test, self.z_dim]), tf.float32)
        else:
            latents = tf.cast(np.random.normal(size=[batch_size_test, self.z_dim]), tf.float32)
        real_img_test, labels_in_test = self.load_dataset(batch_size_test, shuffle=False)
        images = self.sess.run(
            self.generator(real_img_test, labels_in_test, latents, alpha, self.img_size, self.num_channels,
                           self.inp_res, self.input_layer_type, self.use_noise,
                           self.use_instance_norm, self.use_style_mod, self.use_z, is_training=False))

        for i in range(len(images)):
            images[i] = adjust_dynamic_range(images[i], drange_in=[-1, 1], drange_out=[0, 255])
            images[i] = images[i].transpose((1, 2, 0))  # CHW -> HWC

        canvas = PIL.Image.new('RGB', (sum(self.img_size // 2 ** lod for lod in lods), self.img_size * rows), 'white')
        image_iter = iter(list(images))

        for col, lod in enumerate(lods):
            for row in range(rows * 2 ** lod):
                image = PIL.Image.fromarray(np.uint8(next(image_iter)), 'RGB')

                image = image.crop((cx, cy, cx + self.img_size, cy + self.img_size))
                image = image.resize((self.img_size // 2 ** lod, self.img_size // 2 ** lod), PIL.Image.ANTIALIAS)
                canvas.paste(image,
                             (sum(self.img_size // 2 ** lod for lod in lods[:col]), row * self.img_size // 2 ** lod))

        canvas.save('{}/figure02-uncurated.png'.format(test_dir))

