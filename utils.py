import numpy as np
import os, sys
from glob import glob
from ops import lerp

import tensorflow as tf
import tensorflow.contrib.slim as slim
import PIL.Image
from scipy import ndimage

from typing import Any, List, Tuple, Union


class ImageData:

    def __init__(self, img_size, num_channels):
        self.img_size = img_size
        self.num_channels = num_channels

    def image_processing_tf(self, image_file, label, label_mask):
        """We do nothing about label, dealing an image"""
        if image_file.shape.ndims <= 1:
            # it is a filename
            assert image_file.dtype == tf.string, image_file
            x = tf.read_file(image_file)
            img = tf.image.decode_jpeg(x, channels=self.num_channels, dct_method='INTEGER_ACCURATE')  # uint8
        else:
            # it is an image numpy
            assert image_file.dtype == tf.float32
            img = image_file
            assert img.shape[-1].value == self.num_channels
        img = tf.cast(preprocess_fit_train_image(img, self.img_size), tf.float32)

        return img, label, label_mask

    def images_processing_np(self, image_files):
        """Dealing a batch of images"""
        if image_files.ndim <= 1:
            # it is a numpy array of filenames
            images = []
            for image_file in image_files:
                assert isinstance(image_file, str), image_file
                img = PIL.Image.open(image_file).resize((self.img_size, self.img_size), PIL.Image.BILINEAR)  # uint8
                img = adjust_dynamic_range(np.array(img, dtype=np.float32), drange_in=[0, 255], drange_out=[-1, 1])
                images.append(img)
            images = np.array(images)
        else:
            # it is a numpy array of images
            factor = int(image_files.shape[1] / self.img_size)
            assert factor >= 1, factor
            images = ndimage.zoom(image_files, zoom=(1, 1./factor, 1./factor, 1), order=1)  # spline op: bilinear
            images = adjust_dynamic_range(np.array(images, dtype=np.float32), drange_in=[0, 255], drange_out=[-1, 1])
        assert images.ndim == 4
        images = np.transpose(images, [0, 3, 1, 2])  # NHWC -> NCHW
        return images


def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def random_flip_left_right(image):
    s = tf.shape(image)
    mask = tf.random_uniform([1, 1, 1], 0.0, 1.0)
    mask = tf.tile(mask, [s[0], s[1], s[2]])  # [c, h, w]
    image = tf.where(mask < 0.5, image, tf.reverse(image, axis=[1]))
    return image


def smooth_crossfade(images, alpha):
    s = tf.shape(images)
    y = tf.reshape(images, [-1, s[1], s[2] // 2, 2, s[3] // 2, 2])  # images are in NCHW
    y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
    y = tf.tile(y, [1, 1, 1, 2, 1, 2])
    y = tf.reshape(y, [-1, s[1], s[2], s[3]])
    images = lerp(images, y, alpha)
    return images


def preprocess_fit_train_image(image, res):
    assert image.shape.ndims == 3
    image = tf.image.resize_images(image, size=[res, res], method=tf.image.ResizeMethod.BILINEAR)
    image = adjust_dynamic_range(image, drange_in=[0, 255], drange_out=[-1, 1])
    image = tf.transpose(image, [2, 0, 1])  # HWC -> CHW
    # image = random_flip_left_right(image)
    return image


def load_data(dataset_name):
    x = glob(os.path.join("./dataset", dataset_name, '*.*'))
    return x


def save_images(images, size, image_path):
    c = images.shape[1]
    img = merge(images, size)
    assert img.ndim == 3
    if c == 1:
        img = img[0]  # grayscale 1HW => HW
    else:
        img = img.transpose((1, 2, 0))   # CHW -> HWC
    img = adjust_dynamic_range(img, drange_in=[-1, 1], drange_out=[0, 255])
    # print('images.ndim', images.ndim)
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    fmt = 'RGB' if img.ndim == 3 else 'L'
    PIL.Image.fromarray(img, fmt).save(image_path)


def merge(images, size):
    h, w = images.shape[2], images.shape[3]
    c = images.shape[1]
    img = np.zeros((c, h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[:, h * j:h * (j + 1), w * i:w * (i + 1)] = image

    return img


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def str2bool(x):
    return x.lower() in ('true')


def get_checkpoint_res(checkpoint_counter, batch_sizes, iteration, start_res, end_res,
                       gpu_num, end_iteration, do_trans):
    batch_sizes_key = list(batch_sizes.keys())

    start_index = batch_sizes_key.index(start_res)

    iteration_per_res = []

    for res, bs in batch_sizes.items():

        if do_trans[res]:
            if res == end_res:
                iteration_per_res.append(end_iteration // (bs * gpu_num))
            else:
                iteration_per_res.append(iteration // (bs * gpu_num))
        else:
            iteration_per_res.append((iteration // 2) // (bs * gpu_num))

    iteration_per_res = iteration_per_res[start_index:]

    for i in range(len(iteration_per_res)):

        checkpoint_counter = checkpoint_counter - iteration_per_res[i]

        if checkpoint_counter < 1:
            return i + start_index


class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


##################################################################################
# Evaluation operations
##################################################################################

def interpolate_labels(gen_fn, real_img_np, labels_in_np, z_np, index, labels_list, eval_dir, resume_snapshot,
                       num_points=10, num_save=4, min_val=0., max_val=1.):
    assert labels_in_np.shape[-1] > index, labels_in_np.shape[-1]
    assert labels_in_np.shape[0] >= num_save, labels_in_np.shape[0]
    labels_interpolated = np.copy(labels_in_np)

    grid_mix_real_fake = [real_img_np[:num_save, :, :, :]]
    labels_infs = []
    labels_inf_diffs = []
    for i in range(num_points):
        v = min_val + i * 1. / (num_points - 1) * (max_val - min_val)
        print('labels index: {} with v={}, min_val={}, max_val={}'.format(index, v, min_val, max_val))
        labels_interpolated[:, index] = v
        fake_img_interpolated, labels_inf_interpolated = gen_fn(real_img_np, labels_interpolated, z_np)
        grid_mix_real_fake.append(fake_img_interpolated[:num_save, :, :, :])
        labels_infs.append(labels_inf_interpolated[:, index])
        labels_inf_diffs.append(np.abs(labels_inf_interpolated[:, index] - labels_in_np[:, index]))

    grid_mix_real_fake = [np.array(x) for x in zip(*grid_mix_real_fake)]
    grid_mix_real_fake = np.array(grid_mix_real_fake).reshape([-1] + list(real_img_np.shape[1:]))
    assert grid_mix_real_fake.shape[0] == num_save * (num_points + 1)

    save_images(grid_mix_real_fake, [num_save, (num_points + 1)],
                os.path.join(eval_dir, 'mix-labels%d-min%.2fmax%.2f-%06d.png'
                             % (labels_list[index], min_val, max_val, resume_snapshot)))

    assert len(labels_infs) == num_points
    labels_infs = np.array(labels_infs)
    assert labels_infs.ndim == 2
    labels_inf_vars_mean = np.mean(np.var(labels_infs, axis=0))
    labels_inf_diffs_mean = np.mean(np.max(labels_infs, axis=0) - np.min(labels_infs, axis=0))
    return labels_inf_vars_mean, labels_inf_diffs_mean


def randomize_latent(gen_fn, real_img_np, labels_in_np, z_np, eval_dir, resume_snapshot, num_points=10, num_save=4):
    z_np_rand = np.copy(z_np)
    labels_in_np_const = np.array([labels_in_np[0, :] for _ in labels_in_np])

    grid_mix_real_fake = [real_img_np[:num_save, :, :, :]]
    for i in range(num_points):
        v = np.random.randn(*z_np_rand.shape)
        print('latent code value [0,:5]:\n {}'.format(v[0, :5]))
        z_np_rand[:] = v
        fake_img_rand, _ = gen_fn(real_img_np, labels_in_np_const, z_np_rand)
        grid_mix_real_fake.append(fake_img_rand[:num_save, :, :, :])

    grid_mix_real_fake = [np.array(x) for x in zip(*grid_mix_real_fake)]
    grid_mix_real_fake = np.array(grid_mix_real_fake).reshape([-1] + list(real_img_np.shape[1:]))
    assert grid_mix_real_fake.shape[0] == num_save * (num_points + 1)

    save_images(grid_mix_real_fake, [num_save, (num_points + 1)],
                os.path.join(eval_dir, 'mix-z-%06d.png' % resume_snapshot))

