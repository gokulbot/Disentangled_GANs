import numpy as np
import os
import glob
from tensorflow import gfile
import tensorflow as tf
from six.moves import range
from utils import adjust_dynamic_range


class GroundTruthData(object):
    """Abstract class for data sets that are two-step generative models."""

    @property
    def num_samples(self):
        raise NotImplementedError()

    @property
    def num_factors(self):
        raise NotImplementedError()

    @property
    def factors_num_values(self):
        raise NotImplementedError()

    @property
    def observation_shape(self):
        raise NotImplementedError()

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        raise NotImplementedError()

    def sample_observations_from_factors(self, latent_factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        raise NotImplementedError()

    def sample(self, num, random_state):
        """Sample a batch of factors Y and observations X."""
        latent_factors = self.sample_factors(num, random_state)
        images, labels, labels_mask = self.sample_observations_from_factors(latent_factors, random_state)  # images in NHWC
        return images, labels, labels_mask, latent_factors

    def sample_images(self, num, random_state):
        """Sample a batch of observations X."""
        return self.sample(num, random_state)[0]

    def sample_labels(self, num, random_state):
        """Sample a batch of labels."""
        return self.sample(num, random_state)[1]


class SplitDiscreteStateSpace(object):
    """State space with factors split between latent variable and observations."""

    def __init__(self, factor_sizes, latent_factor_indices):
        self.factor_sizes = factor_sizes
        self.num_factors = len(self.factor_sizes)
        self.latent_factor_indices = latent_factor_indices
        self.observation_factor_indices = [i for i in range(self.num_factors) if i not in self.latent_factor_indices]

    @property
    def num_latent_factors(self):
        return len(self.latent_factor_indices)

    def sample_latent_factors(self, num, random_state):
        """Sample a batch of the latent factors."""
        factors = np.zeros(shape=(num, len(self.latent_factor_indices)), dtype=np.int64)
        for pos, i in enumerate(self.latent_factor_indices):
            factors[:, pos] = self._sample_factor(i, num, random_state)
        return factors

    def sample_all_factors(self, latent_factors, random_state):
        """Samples the remaining factors based on the latent factors."""
        num_samples = latent_factors.shape[0]
        all_factors = np.zeros(shape=(num_samples, self.num_factors), dtype=np.int64)
        all_factors[:, self.latent_factor_indices] = latent_factors
        # Complete all the other factors
        for i in self.observation_factor_indices:
            all_factors[:, i] = self._sample_factor(i, num_samples, random_state)
        return all_factors

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factor_sizes[i], size=num)


# ---------------------------------------------------------------------

def tf_dataset_from_ground_truth_data(ground_truth_data, random_seed):
    """Generate a tf.data.DataSet from ground_truth data."""

    def generator():
        # We need to hard code the random seed so that the data set can be reset.
        random_state = np.random.RandomState(random_seed)
        while True:
            images, labels, labels_mask, _ = ground_truth_data.sample(1, random_state)
            yield images[0], labels[0], labels_mask[0]

    data_types = (tf.string, tf.float32, tf.float32) if len(ground_truth_data.observation_shape) == 0 \
        else (tf.float32, tf.float32, tf.float32)  # image filenames or arrays
    return tf.data.Dataset.from_generator(
        generator, data_types,
        output_shapes=(ground_truth_data.observation_shape, [ground_truth_data.num_factors], []))


def make_input_fn(ground_truth_data, image_class, seed, gpu_id, num_batches=None):
    """Creates an input function for the experiments."""

    def load_dataset(batch_size, shuffle=True):
        dataset = tf_dataset_from_ground_truth_data(ground_truth_data, seed)
        # pre-processing
        dataset = dataset.map(image_class.image_processing_tf, num_parallel_calls=64)  # equal to number of cpus

        if shuffle:
            dataset = dataset.shuffle(ground_truth_data.num_samples)

        # We need to drop the remainder as otherwise we lose the batch size in the
        # tensor shape. This has no effect as our data set is infinite.
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=16)  # need to tune for optimization

        if num_batches is not None:
            dataset = dataset.take(num_batches)

        dataset = dataset.apply(tf.data.experimental.prefetch_to_device(device='/gpu:{}'.format(gpu_id)))
        return dataset.make_one_shot_iterator().get_next()

    return load_dataset


def tf_labels_from_ground_truth_data(ground_truth_data, random_seed):
    """Generate a tf.data.DataSet from ground_truth data."""

    def generator():
        # We need to hard code the random seed so that the data set can be reset.
        random_state = np.random.RandomState(random_seed)
        while True:
            labels = ground_truth_data.sample_labels(1, random_state)
            yield labels[0]

    return tf.data.Dataset.from_generator(
        generator, tf.float32,
        output_shapes=([ground_truth_data.num_factors]))


def make_labels_fn(ground_truth_data, seed, gpu_id, num_batches=None):
    """Creates an input function for the experiments."""

    def sample_labels(batch_size, shuffle=True):
        labels_dst = tf_labels_from_ground_truth_data(ground_truth_data, seed)

        if shuffle:
            labels_dst = labels_dst.shuffle(ground_truth_data.num_samples)

        # We need to drop the remainder as otherwise we lose the batch size in the
        # tensor shape. This has no effect as our data set is infinite.
        labels_dst = labels_dst.batch(batch_size, drop_remainder=True)
        labels_dst = labels_dst.prefetch(buffer_size=16)  # need to tune for optimization

        if num_batches is not None:
            labels_dst = labels_dst.take(num_batches)

            labels_dst = labels_dst.apply(tf.data.experimental.prefetch_to_device(device='/gpu:{}'.format(gpu_id)))
        return labels_dst.make_one_shot_iterator().get_next()

    return sample_labels


# ---------------------------------------------------------------------
class DSprites(GroundTruthData):
    """DSprites dataset.

    The data set was originally introduced in "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework" and can be downloaded from
    https://github.com/deepmind/dsprites-dataset.

    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    """

    def __init__(self, data_path, labels_fine_list=[], labels_coarse_list=[]):
        # By default, all factors (including shape) are considered ground truth factors.
        self.labels_fine_list = labels_fine_list
        self.labels_coarse_list = labels_coarse_list
        self.latent_factor_indices = self.labels_fine_list + self.labels_coarse_list

        self.fine_factor_indices = [0, 1]
        self.coarse_factor_indices = [2, 3, 4]
        if not set(self.labels_fine_list).issubset(set(self.fine_factor_indices)):
            print("[warning]: labels_fine_list is not a subset of fine ground-truth list")
        if not set(self.labels_coarse_list).issubset(set(self.coarse_factor_indices)):
            print("[warning]: labels_coarse_list is not a subset of coarse ground-truth list")

        self.data_shape = [64, 64, 1]
        # Load the data so that we can sample from it.
        dsprites_file = os.path.join(data_path, 'dsprites', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        with gfile.Open(dsprites_file, "rb") as data_file:
            # Data was saved originally using python2, so we need to set the encoding.
            data = np.load(data_file, encoding="latin1", allow_pickle=True)
            self.images = np.array(data["imgs"])
            self.images = adjust_dynamic_range(self.images, drange_in=[0, 1], drange_out=[0, 255])  # TODO
            self.factor_sizes = np.array(data["metadata"][()]["latents_sizes"], dtype=np.int64)
        self.full_factor_sizes = [3, 6, 40, 32, 32]
        self.factor_bases = np.prod(self.full_factor_sizes) / np.cumprod(self.full_factor_sizes)
        self.state_space = SplitDiscreteStateSpace(self.full_factor_sizes, self.latent_factor_indices)

        self.labels_mask = np.random.uniform(0., 1., size=len(self.images))  # for semi-supervised learning

    @property
    def num_samples(self):
        return self.images.shape[0]

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return [self.full_factor_sizes[i] for i in self.latent_factor_indices]

    @property
    def observation_shape(self):
        return self.data_shape

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        latent_factors = self.state_space.sample_latent_factors(num, random_state)
        return latent_factors

    def sample_observations_from_factors(self, latent_factors, random_state):
        """Sample a batch of observations X and labels given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(latent_factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        all_labels = np.divide(all_factors, self.full_factor_sizes)  # normalize all_factors => all_labels
        labels = all_labels[:, self.latent_factor_indices]
        images = np.expand_dims(self.images[indices].astype(np.float32), axis=3)

        labels_mask = self.labels_mask[indices]
        assert labels_mask.ndim == 1 and labels_mask.shape[0] == images.shape[0]
        return images, labels, labels_mask

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.full_factor_sizes[i], size=num)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
class FFHQ(GroundTruthData):
    """FFHQ dataset.

    The data set was originally introduced in StyleGAN and can be downloaded from
    https://github.com/NVlabs/ffhq-dataset.

    There is no ground-truth factors of variation
    """

    def __init__(self, data_path, label_size_fine=0, label_size_coarse=0):
        self.label_size_fine = label_size_fine
        self.label_size_coarse = label_size_coarse
        self.images = glob.glob(os.path.join(data_path, 'ffhq', '*.png'))
        self.data_shape = [1024, 1024, 3]

    @property
    def num_samples(self):
        assert isinstance(self.images, list)
        return len(self.images)

    @property
    def num_factors(self):
        return 0

    @property
    def observation_shape(self):
        """It is because we use image filenames"""
        return []

    def sample_factors(self, num, random_state):
        """There is no labels in FFHQ and all_factors are image indices"""
        latent_factors = random_state.randint(self.num_samples, size=num)
        return latent_factors

    def sample_observations_from_factors(self, latent_factors, random_state):
        indices = latent_factors
        labels = np.zeros([indices.shape[0], 0])
        images = np.array([self.images[index] for index in indices], dtype=str)
        labels_mask = np.zeros([indices.shape[0]])
        return images, labels, labels_mask
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
class pinkroom(GroundTruthData):
    """pinkroom dataset.

    The data set is created for supervised disentanglement - with the dataset name:
    - images: 'resized_pink_room_camera4_train/images/xxxxxx.png'
    - labels: 'resized_pink_room_camera4_train/train-rpt.labels'

    The ground-truth factors of variation are (in the default setting):
    0 - lighting intensity (5 different values)
    1 - lighting direction phi (10 different values)
    2 - lighting direction theta (10 different values)
    3 - camera position x (8 different values)
    4 - camera position y (8 different values)
    5 - camera position z (8 different values)
    """

    def __init__(self, data_path, labels_fine_list=[], labels_coarse_list=[]):
        # By default, all factors (including shape) are considered ground truth factors.
        self.labels_fine_list = labels_fine_list
        self.labels_coarse_list = labels_coarse_list
        self.latent_factor_indices = self.labels_fine_list + self.labels_coarse_list

        self.fine_factor_indices = [0, 1, 2]
        self.coarse_factor_indices = [3, 4, 5]
        if not set(self.labels_fine_list).issubset(set(self.fine_factor_indices)):
            print("[warning]: labels_fine_list is not a subset of fine ground-truth list")
        if not set(self.labels_coarse_list).issubset(set(self.coarse_factor_indices)):
            print("[warning]: labels_coarse_list is not a subset of coarse ground-truth list")

        self.data_shape = [1024, 1024, 3]
        # Load the data so that we can sample from it.
        pinkroom_dir = os.path.join(data_path, 'resized_pink_room_camera4_train')
        self.images = sorted(glob.glob(os.path.join(pinkroom_dir, 'images', '*.png')))

        self.labels = np.load(os.path.join(pinkroom_dir, 'train-rpt.labels'))
        self.factor_sizes = [5, 10, 10, 8, 8, 8]
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes, self.latent_factor_indices)

        self.labels_mask = np.random.uniform(0., 1., size=len(self.images))  # for semi-supervised learning

    @property
    def num_samples(self):
        assert isinstance(self.images, list)
        return len(self.images)

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return [self.factor_sizes[i] for i in self.latent_factor_indices]

    @property
    def observation_shape(self):
        """It is because we use image filenames"""
        return []

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        latent_factors = self.state_space.sample_latent_factors(num, random_state)
        return latent_factors

    def sample_observations_from_factors(self, latent_factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(latent_factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        assert indices.ndim == 1, indices
        all_labels = self.labels[indices].astype(np.float32)
        labels = all_labels[:, self.latent_factor_indices]
        images = np.array([self.images[index] for index in indices], dtype=str)

        labels_mask = self.labels_mask[indices]
        assert labels_mask.ndim == 1 and labels_mask.shape[0] == images.shape[0]
        return images, labels, labels_mask

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factor_sizes[i], size=num)

# ---------------------------------------------------------------------
class Isaac3D(GroundTruthData):
    """pinkroom dataset.

    The data set is created for supervised disentanglement - with the dataset name:
    - images: 'Isaac3D_v1/images/xxxxxx.png'
    - labels: 'Isaac3D_v1/labels.npy'

    The ground-truth factors of variation are (in the default setting):
    0 - object shape (3 different values, discrete)
    1 - robot horizontal move (8 different values)
    2 - robot vertical move (5 different values)
    3 - camera height (4 different values)
    4 - object scale (4 different values)
    5 - lighting intensity (4 different values)
    6 - lighting direction (6 different values)
    7 - object color (4 different values)
    8 - wall color (4 different values)
    """

    def __init__(self, data_path, labels_fine_list=[], labels_coarse_list=[]):
        # By default, all factors (including shape) are considered ground truth factors.
        self.labels_fine_list = labels_fine_list
        self.labels_coarse_list = labels_coarse_list
        self.latent_factor_indices = self.labels_fine_list + self.labels_coarse_list

        self.fine_factor_indices = [5, 6, 7, 8]
        self.coarse_factor_indices = [0, 1, 2, 3, 4]
        if not set(self.labels_fine_list).issubset(set(self.fine_factor_indices)):
            print("[warning]: labels_fine_list is not a subset of fine ground-truth list")
        if not set(self.labels_coarse_list).issubset(set(self.coarse_factor_indices)):
            print("[warning]: labels_coarse_list is not a subset of coarse ground-truth list")

        self.data_shape = [512, 512, 3]
        # Load the data so that we can sample from it.
        isaac3d_dir = os.path.join(data_path, 'Isaac3D_v1')
        self.images = sorted(glob.glob(os.path.join(isaac3d_dir, 'images', '*.png')))

        self.labels = np.load(os.path.join(isaac3d_dir, 'labels.npy'))
        self.factor_sizes = [3, 8, 5, 4, 4, 4, 6, 4, 4]
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes, self.latent_factor_indices)

        self.labels_mask = np.random.uniform(0., 1., size=len(self.images))  # for semi-supervised learning

    @property
    def num_samples(self):
        assert isinstance(self.images, list)
        return len(self.images)

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return [self.factor_sizes[i] for i in self.latent_factor_indices]

    @property
    def observation_shape(self):
        """It is because we use image filenames"""
        return []

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        latent_factors = self.state_space.sample_latent_factors(num, random_state)
        return latent_factors  # latent_factors are in the order: fine + coarse

    def sample_observations_from_factors(self, latent_factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        # all_factors are in the original order [0, 1, 2, ...]
        all_factors = self.state_space.sample_all_factors(latent_factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        assert indices.ndim == 1, indices
        all_labels = self.labels[indices].astype(np.float32)
        labels = all_labels[:, self.latent_factor_indices]  # labels are in the order: [fine, coarse]
        images = np.array([self.images[index] for index in indices], dtype=str)

        labels_mask = self.labels_mask[indices]
        assert labels_mask.ndim == 1 and labels_mask.shape[0] == images.shape[0]
        return images, labels, labels_mask

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factor_sizes[i], size=num)


# ---------------------------------------------------------------------
class Falcor3D(GroundTruthData):
    """pinkroom dataset.

    The data set is created for supervised disentanglement - with the dataset name:
    - images: 'resized_pink_room_camera5_train/images/xxxxxx.png'
    - labels: 'resized_pink_room_camera5_train/train-rec.labels'

    The ground-truth factors of variation are (in the default setting):
    0 - lighting intensity (5 different values)
    1 - lighting direction x_l (6 different values)
    2 - lighting direction y_l (6 different values)
    3 - lighting direction z_l (6 different values)
    4 - camera position x_c (6 different values)
    5 - camera position y_c (6 different values)
    6 - camera position z_c (6 different values)
    """

    def __init__(self, data_path, labels_fine_list=[], labels_coarse_list=[], data_id=5):
        # By default, all factors (including shape) are considered ground truth factors.
        self.labels_fine_list = labels_fine_list
        self.labels_coarse_list = labels_coarse_list
        self.latent_factor_indices = self.labels_fine_list + self.labels_coarse_list
        self.data_id = data_id

        self.fine_factor_indices = [0, 1, 2, 3]
        self.coarse_factor_indices = [4, 5, 6]
        if not set(self.labels_fine_list).issubset(set(self.fine_factor_indices)):
            print("[warning]: labels_fine_list is not a subset of fine ground-truth list")
        if not set(self.labels_coarse_list).issubset(set(self.coarse_factor_indices)):
            print("[warning]: labels_coarse_list is not a subset of coarse ground-truth list")

        if self.data_id == 7:
            self.data_shape = [1024, 1024, 3]
        else:
            self.data_shape = [512, 512, 3]

        # Load the data so that we can sample from it.
        pinkroom_dir = os.path.join(data_path, 'resized_pink_room_camera{}_train'.format(self.data_id))
        self.images = sorted(glob.glob(os.path.join(pinkroom_dir, 'images', '*.png')))

        self.labels = np.load(os.path.join(pinkroom_dir, 'train-rec.labels'))
        self.factor_sizes = [5, 6, 6, 6, 6, 6, 6]
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        self.state_space = SplitDiscreteStateSpace(self.factor_sizes, self.latent_factor_indices)

        self.labels_mask = np.random.uniform(0., 1., size=len(self.images))  # for semi-supervised learning

    @property
    def num_samples(self):
        assert isinstance(self.images, list)
        return len(self.images)

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return [self.factor_sizes[i] for i in self.latent_factor_indices]

    @property
    def observation_shape(self):
        """It is because we use image filenames"""
        return []

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        latent_factors = self.state_space.sample_latent_factors(num, random_state)
        return latent_factors

    def sample_observations_from_factors(self, latent_factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(latent_factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        assert indices.ndim == 1, indices
        all_labels = self.labels[indices].astype(np.float32)
        labels = all_labels[:, self.latent_factor_indices]
        images = np.array([self.images[index] for index in indices], dtype=str)

        labels_mask = self.labels_mask[indices]
        assert labels_mask.ndim == 1 and labels_mask.shape[0] == images.shape[0]
        return images, labels, labels_mask

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factor_sizes[i], size=num)


# ---------------------------------------------------------------------
def get_named_ground_truth_data(data_path, labels_fine_list, labels_coarse_list, name):
    """Returns ground truth data set based on name."""

    if name == "dsprites":
        return DSprites(data_path, labels_fine_list, labels_coarse_list)
    elif name == 'ffhq':
        return FFHQ(data_path, labels_fine_list, labels_coarse_list)
    elif name == 'pinkroom':
        return pinkroom(data_path, labels_fine_list, labels_coarse_list)
    elif name == 'isaac3d':
        return Isaac3D(data_path, labels_fine_list, labels_coarse_list)
    elif name == 'falcor3d_5':
        return Falcor3D(data_path, labels_fine_list, labels_coarse_list, data_id=5)
    elif name == 'falcor3d_6':
        return Falcor3D(data_path, labels_fine_list, labels_coarse_list, data_id=6)
    elif name == 'falcor3d_7':
        return Falcor3D(data_path, labels_fine_list, labels_coarse_list, data_id=7)
    else:
        raise ValueError("Invalid data set name.")
