import numpy as np
import tensorflow as tf
import time
from absl import logging
import functools
import sklearn
from tensorflow.python.ops import array_ops
tfgan = tf.contrib.gan


# the base class for metrics
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.model = None

    def set_model(self, model):
        self.model = model

    def evaluate(self, dataset_obj, res, batch_size):
        raise NotImplementedError


# ----------------------------------------------------------------------------


class FactorVAEMetric(Metric):
    def __init__(self, num_train=5000, num_eval=2000,
                 num_variance_estimate=10000, random_state=np.random.RandomState(123), prune_dims_thres=0.05,
                 *args, **kwargs):
        super(FactorVAEMetric, self).__init__(*args, **kwargs)
        self.num_train = num_train
        self.num_eval = num_eval
        self.num_variance_estimate = num_variance_estimate
        self.random_state = random_state
        self.prune_dims_thres = prune_dims_thres

    def evaluate(self, ground_truth_data, res, batch_size=64, log_func=logging.info):
        log_func("------------------------ FactorVAE ------------------------")
        log_func("Computing global variances to standardize.")
        start_time = time.time()
        global_variances = self._compute_variances(
            ground_truth_data, res, batch_size, self.model.inference_from,
            self.num_variance_estimate, self.random_state, log_func)
        log_func("time collapsed: {}".format(time.time() - start_time))
        active_dims = self._prune_dims(global_variances, self.prune_dims_thres)
        scores_dict = {}

        if not active_dims.any():
            scores_dict["train_accuracy"] = 0.
            scores_dict["eval_accuracy"] = 0.
            scores_dict["num_active_dims"] = 0
            return scores_dict

        log_func("Generating training set.")
        start_time = time.time()
        training_votes = self._generate_training_batch(
            ground_truth_data, res, batch_size, self.model.inference_from, self.num_train,
            self.random_state, global_variances, active_dims, log_func)
        classifier = np.argmax(training_votes, axis=0)
        other_index = np.arange(training_votes.shape[1])

        log_func("Evaluate training set accuracy.")
        train_accuracy = np.sum(training_votes[classifier, other_index]) * 1. / np.sum(training_votes)
        logging.info("Training set accuracy: %.2g", train_accuracy)
        log_func("time collapsed: {}".format(time.time() - start_time))

        log_func("Generating evaluation set.")
        start_time = time.time()
        eval_votes = self._generate_training_batch(
            ground_truth_data, res, batch_size, self.model.inference_from, self.num_eval,
            self.random_state, global_variances, active_dims, log_func)

        log_func("Evaluate evaluation set accuracy.")
        eval_accuracy = np.sum(eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)
        logging.info("Evaluation set accuracy: %.2g", eval_accuracy)
        log_func("time collapsed: {}".format(time.time() - start_time))

        scores_dict["train_accuracy"] = train_accuracy
        scores_dict["eval_accuracy"] = eval_accuracy
        scores_dict["num_active_dims"] = len(active_dims)
        return scores_dict

    def _prune_dims(self, variances, threshold=0.):
        """Mask for dimensions collapsed to the prior."""
        scale_z = np.sqrt(variances)
        return scale_z >= threshold

    def _compute_variances(self, ground_truth_data, res, batch_size, representation_function,
                           num_variance, random_state, log_func):
        """Computes the variance for each dimension of the representation."""
        images = ground_truth_data.sample_images(num_variance, random_state)
        log_func("In _compute_variance, the images nums are: {}".format(images.shape[0]))
        representations = representation_function(images, res, batch_size)
        log_func("In _compute_variance, the presentations nums are: {}".format(representations.shape[0]))
        return np.var(representations, axis=0, ddof=1)

    def _generate_training_sample(self, ground_truth_data, res, batch_size, representation_function,
                                  random_state, global_variances, active_dims):
        """Sample a single training sample based on a mini-batch of ground-truth data."""
        # Select random coordinate to keep fixed.
        factor_index = random_state.randint(ground_truth_data.num_factors)
        # Sample two mini batches of latent variables.
        factors = ground_truth_data.sample_factors(batch_size, random_state)
        # Fix the selected factor across mini-batch.
        factors[:, factor_index] = factors[0, factor_index]
        # Obtain the observations.
        images, _, _ = ground_truth_data.sample_observations_from_factors(factors, random_state)
        representations = representation_function(images, res, batch_size)
        local_variances = np.var(representations, axis=0, ddof=1)
        argmin = np.argmin(
            local_variances[active_dims] / global_variances[active_dims]
        )
        return factor_index, argmin

    def _generate_training_batch(self, ground_truth_data, res, batch_size, representation_function,
                                 num_points, random_state, global_variances, active_dims, log_func):
        """Sample a set of training samples based on a batch of ground-truth data."""
        votes = np.zeros((ground_truth_data.num_factors, global_variances.shape[0]), dtype=np.int64)
        for i in range(num_points):
            factor_index, argmin = self._generate_training_sample(
                ground_truth_data, res, batch_size, representation_function, random_state,
                global_variances, active_dims)
            votes[factor_index, argmin] += 1
            if i % 100 == 0:
                log_func("in _generate_training_batch, {}".format(i))
        return votes


# ----------------------------------------------------------------------------

class FID(Metric):
    def __init__(self, num_images=50000, random_state=np.random.RandomState(123), *args, **kwargs):
        super(FID, self).__init__(*args, **kwargs)
        self.num_images = num_images
        self.random_state = random_state

        # build graph to run images through Inception.
        self.inception_images = tf.placeholder(tf.float32, [None, None, None, 3])
        self.activations1 = tf.placeholder(tf.float32, [None, None], name='activations1')
        self.activations2 = tf.placeholder(tf.float32, [None, None], name='activations2')
        self.fcd = tfgan.eval.frechet_classifier_distance_from_activations(self.activations1, self.activations2)

        self.activations = self.inception_activations(images=self.inception_images)

    def evaluate(self, ground_truth_data, res, batch_size=64, log_func=logging.info):
        log_func("------------------------ FID ------------------------")
        scores_dict = {}

        start1_time = time.time()
        act1 = self.get_inception_activations(self.model.sample_randomly, self.inception_images,
                                              self.activations, batch_size, res)
        log_func("time collapsed for real images: {}".format(time.time() - start1_time))
        start2_time = time.time()
        act2 = self.get_inception_activations(self.model.generate_randomly, self.inception_images,
                                              self.activations, batch_size, res)
        log_func("time collapsed for fake images: {}".format(time.time() - start2_time))

        fid = self.model.sess.run(self.fcd, feed_dict={self.activations1: act1, self.activations2: act2})
        log_func("time collapsed for fid: {}".format(time.time() - start1_time))

        scores_dict['score'] = np.real(fid)
        return scores_dict

    def inception_activations(self, images, num_splits=1):
        size = 299
        images = tf.image.resize_bilinear(images, [size, size])
        generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
        activations = tf.map_fn(
            fn=functools.partial(tfgan.eval.run_inception, output_tensor='pool_3:0'),
            elems=array_ops.stack(generated_images_list),
            parallel_iterations=1,
            back_prop=False,
            swap_memory=True,
            name='RunClassifier')
        activations = array_ops.concat(array_ops.unstack(activations), 0)
        return activations

    def get_inception_activations(self, img_generate_fn, inception_images, activations, batch_size, res):
        act = np.zeros([self.num_images, 2048], dtype=np.float32)
        i = 0
        for begin in range(0, self.num_images, batch_size):
            i += 1
            end = min(begin + batch_size, self.num_images)
            images_batch = img_generate_fn(res, end - begin)
            images_batch = images_batch.transpose((0, 2, 3, 1))  # NCHW -> NHWC
            # deal with the case channels=1
            if images_batch.shape[-1] == 1:
                images_batch = np.tile(images_batch, [1, 1, 1, 3])
            assert images_batch.shape[-1] == 3
            act[begin: end] = self.model.sess.run(activations, feed_dict={inception_images: images_batch})
            if i % 300 == 0:
                print("generated {} batches of activations".format(i))
        return act


# ----------------------------------------------------------------------------
class MIG(Metric):
    """Computes the mutual information gap."""
    def __init__(self, num_train=10000, random_state=np.random.RandomState(123),
                 *args, **kwargs):
        super(MIG, self).__init__(*args, **kwargs)
        self.num_train = num_train
        self.random_state = random_state

    def evaluate(self, ground_truth_data, res, batch_size=16, log_func=logging.info):
        log_func("------------------------ MIG ------------------------")
        log_func("Generating training set.")
        start_time = time.time()
        mus_train, ys_train = self.generate_batch_factor_code(
            ground_truth_data, res, batch_size, self.model.inference_from, self.num_train, self.random_state)
        log_func("time collapsed: {}, and mus shape: {}, amd ys shape: {}".format(
            time.time() - start_time, mus_train.shape, ys_train.shape))
        log_func("mus_train some values: {}". format(mus_train[:, :5]))
        log_func("ys_train some values: {}". format(ys_train[:, :5]))
        assert mus_train.shape[1] == self.num_train
        log_func("Computing MIG.")
        start_time = time.time()
        score_dict = self._compute_mig(mus_train, ys_train, log_func)
        log_func("time collapsed: {}".format(time.time() - start_time))
        return score_dict

    def _compute_mig(self, mus_train, ys_train, log_func):
        """Computes score based on both training and testing codes and factors."""
        score_dict = {}
        discretized_mus = histogram_discretize(mus_train)
        m = discrete_mutual_info(discretized_mus, ys_train)
        assert m.shape[0] == mus_train.shape[0]
        assert m.shape[1] == ys_train.shape[0]
        # m is [num_latents, num_factors]
        entropy = discrete_entropy(ys_train)
        sorted_m = np.sort(m, axis=0)[::-1]
        score_dict["discrete_mig"] = np.mean(
            np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
        return score_dict

    def generate_batch_factor_code(self, ground_truth_data, res, batch_size, representation_function,
                                   num_points, random_state):
        """Sample a single training sample based on a mini-batch of ground-truth data.
          representations: Codes (num_codes, num_points)-np array.
          factors: Factors (num_factors, num_points)-np array.
        """
        representations = None
        factors = None
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, batch_size)
            current_images, _, _, cur_latent_factors = ground_truth_data.sample(num_points_iter, random_state)
            if i == 0:
                factors = cur_latent_factors
                representations = representation_function(current_images, res, batch_size)
            else:
                factors = np.vstack((factors, cur_latent_factors))
                representations = np.vstack((representations, representation_function(current_images, res, batch_size)))
            i += num_points_iter
        return np.transpose(representations), np.transpose(factors)


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h


def histogram_discretize(target, num_bins=10):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
    return discretized


# ----------------------------------------------------------------------------

class L2Metric(Metric):
    def __init__(self, num_train=64*10, random_state=np.random.RandomState(123), labels_list=[], *args, **kwargs):
        super(L2Metric, self).__init__(*args, **kwargs)
        self.num_train = num_train
        self.random_state = random_state
        self.labels_list = labels_list

    def evaluate(self, ground_truth_data, res, batch_size=64, log_func=logging.info):
        log_func("------------------------ L2 ------------------------")
        scores_dict = {}

        start_time = time.time()
        preds_train, ys_train = self.generate_batch_pred_code(
            ground_truth_data, res, batch_size, self.model.inference_from, self.num_train, self.random_state)

        # in case that we only do coarse-grained reconstruction in the early training stage
        fine_label_size = ys_train.shape[0] - preds_train.shape[0]
        preds_fine_dumb = np.zeros([fine_label_size, ys_train.shape[1]])
        preds_train = np.concatenate([preds_fine_dumb, preds_train], axis=0)
        assert preds_train.shape[0] == ys_train.shape[0]

        score_per_factor = np.mean(np.abs(preds_train - ys_train), axis=-1, keepdims=False)
        log_func('Total factor size: {}'.format(score_per_factor.shape[0]))
        for i in range(score_per_factor.shape[0]):
            factor_index = self.labels_list[i]
            scores_dict['factor{}'.format(factor_index)] = np.real(score_per_factor[i])
        log_func('total_time: ', np.real(time.time() - start_time))
        return scores_dict

    def generate_batch_pred_code(self, ground_truth_data, res, batch_size, representation_function,
                                   num_points, random_state):
        """Sample a single training sample based on a mini-batch of ground-truth data.
          representations: Codes (num_codes, num_points)-np array.
          labels: Labels generating the codes (num_codes, num_points)-np array.
        """
        representations = None
        labels = None
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, batch_size)
            current_images, current_labels, _, _ = ground_truth_data.sample(num_points_iter, random_state)
            if i == 0:
                labels = current_labels
                representations = representation_function(current_images, res, batch_size)
            else:
                labels = np.vstack((labels, current_labels))
                representations = np.vstack((representations, representation_function(current_images, res, batch_size)))
            i += num_points_iter
        return np.transpose(representations), np.transpose(labels)


# ----------------------------------------------------------------------------
def group_metrics(labels_list=[], is_FactorVAE=True, is_FID=True, is_MIG=True, is_L2=True):
    metrics = []
    if is_FactorVAE and len(labels_list) != 0:
        factVAE = FactorVAEMetric(name='FactorVAE')
        metrics.append(factVAE)

    if is_FID:
        fid50k = FID(name='fid50k')
        metrics.append(fid50k)

    if is_MIG and len(labels_list) != 0:
        mig = MIG(name='MIG')
        metrics.append(mig)

    if is_L2 and len(labels_list) != 0:
        l2Metric = L2Metric(name='L2', labels_list=labels_list)
        metrics.append(l2Metric)

    return metrics

# ----------------------------------------------------------------------------
