# Copyright 2019 Alessandro Davide Ialongo (@ialong)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from numpy.testing import assert_allclose
from numpy.random import randn as mvn
from numpy.linalg import cholesky
import tensorflow as tf
import gpflow as gp
from gpflow.test_util import GPflowTestCase
from gpflow.kullback_leiblers import gauss_kl
from GPt.KL import KL, KL_samples


FLOAT_TYPE = gp.settings.float_type


def choleskify(mats):
    ret = []
    for m in mats:
        if m.ndim == 1:
            m = np.abs(m)
        else:
            m = cholesky(m @ (m.T if m.ndim == 2 else np.transpose(m, [0, 2, 1])))
        ret.append(m)
    return ret


def compare_KLs(sess, feed_dict, mu, Q_chol, P_chols):
    mu_gpflow = tf.transpose(mu) if mu.shape.ndims == 2 else mu[:, None]
    Q_chol_gpflow = Q_chol if Q_chol.shape.ndims == 3 else Q_chol[None, ...]

    KL_gpflow = sess.run(gauss_kl(q_mu=mu_gpflow, q_sqrt=Q_chol_gpflow, K=None), feed_dict=feed_dict)
    KL_gpt = sess.run(KL(mu_diff=mu, Q_chol=Q_chol, P_chol=None, P=None), feed_dict=feed_dict)
    assert_allclose(KL_gpflow, KL_gpt)

    for P_chol in P_chols:
        P_ndims = P_chol.shape.ndims
        P = tf.square(P_chol) if P_ndims == 1 else tf.matmul(P_chol, P_chol, transpose_b=True)

        KL_gpflow = sess.run(gauss_kl(q_mu=mu_gpflow, q_sqrt=Q_chol_gpflow,
                                      K=tf.diag(P) if P_ndims == 1 else P), feed_dict=feed_dict)

        KL_gpt = sess.run(KL(mu_diff=mu, Q_chol=Q_chol, P_chol=P_chol, P=None), feed_dict=feed_dict)
        assert_allclose(KL_gpflow, KL_gpt)
        KL_gpt = sess.run(KL(mu_diff=mu, Q_chol=Q_chol, P_chol=None, P=P), feed_dict=feed_dict)
        assert_allclose(KL_gpflow, KL_gpt)
        KL_gpt = sess.run(KL(mu_diff=mu, Q_chol=Q_chol, P_chol=P_chol, P=P), feed_dict=feed_dict)
        assert_allclose(KL_gpflow, KL_gpt)


class KLTest(GPflowTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed(0)
        tf.set_random_seed(0)
        self.D, self.M, self.n_samples = 11, 53, 7

    def prepare(self):
        mus = [mvn(self.D), mvn(self.M, self.D), mvn(self.D, self.M), mvn(self.M, self.n_samples, self.D)]
        Q_chols = [mvn(self.D, self.D), mvn(self.D, self.M, self.M), mvn(self.M, self.D, self.D)]
        Q_chols = choleskify(Q_chols)
        Q_chols.append(np.abs(mvn(self.M, self.D)))
        P_chols = [mvn(self.D), mvn(self.D, self.D), mvn(self.M), mvn(self.M, self.M), mvn(self.D, self.M, self.M)]
        P_chols = choleskify(P_chols)
        P_chols.append(np.abs(mvn(self.D, self.M)))
        mus = {a.shape: a for a in mus}
        Q_chols = {a.shape: a for a in Q_chols}
        P_chols = {a.shape: a for a in P_chols}
        return mus, Q_chols, P_chols

    def get_feed_dict(self, mus_tf, Qs_tf, Ps_tf):
        shape = lambda ph: tuple(np.array(ph.shape, dtype=int))
        mus, Qs, Ps = self.prepare()
        feed_dict = dict()
        for mu_tf in mus_tf:
            feed_dict[mu_tf] = mus[shape(mu_tf)]
        for Q_tf in Qs_tf:
            feed_dict[Q_tf] = Qs[shape(Q_tf)]
        for P_tf in Ps_tf:
            feed_dict[P_tf] = Ps[shape(P_tf)]
        return feed_dict

    def test_KL_mu_D_Q_DxD(self):
        with self.test_context() as sess:
            mu = tf.placeholder(FLOAT_TYPE, shape=(self.D,))
            Q_chol = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.D))
            P_chol_diag = tf.placeholder(FLOAT_TYPE, shape=(self.D))
            P_chol = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.D))

            feed_dict = self.get_feed_dict([mu], [Q_chol], [P_chol_diag, P_chol])

            compare_KLs(sess, feed_dict, mu, Q_chol, [P_chol_diag, P_chol])

    def test_KL_mu_MxD_Q_DxMxM(self):
        with self.test_context() as sess:
            mu = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.M))
            Q_chol = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.M, self.M))
            P_chol_1D = tf.placeholder(FLOAT_TYPE, shape=(self.M))
            P_chol_2D = tf.placeholder(FLOAT_TYPE, shape=(self.M, self.M))
            P_chol_3D = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.M, self.M))

            feed_dict = self.get_feed_dict([mu], [Q_chol], [P_chol_1D, P_chol_2D, P_chol_3D])

            compare_KLs(sess, feed_dict, mu, Q_chol, [P_chol_1D, P_chol_2D, P_chol_3D])

    def test_KL_samples_mu_2D(self):
        with self.test_context() as sess:
            mu = tf.placeholder(FLOAT_TYPE, shape=(self.M, self.D))
            Q_chol_2D = tf.placeholder(FLOAT_TYPE, shape=(self.M, self.D))
            Q_chol_3D = tf.placeholder(FLOAT_TYPE, shape=(self.M, self.D, self.D))
            P_chol_1D = tf.placeholder(FLOAT_TYPE, shape=(self.D))
            P_chol_2D = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.D))

            feed_dict = self.get_feed_dict([mu], [Q_chol_2D, Q_chol_3D], [P_chol_1D, P_chol_2D])

            KL_s_1 = sess.run(tf.reduce_sum(KL_samples(mu, Q_chol_2D, None)), feed_dict)
            KL_s_2 = sess.run(tf.reduce_sum(KL_samples(mu, Q_chol_2D, P_chol_1D)), feed_dict)
            KL_s_3 = sess.run(tf.reduce_sum(KL_samples(mu, Q_chol_2D, P_chol_2D)), feed_dict)
            KL_s_4 = sess.run(tf.reduce_sum(KL_samples(mu, Q_chol_3D, None)), feed_dict)
            KL_s_5 = sess.run(tf.reduce_sum(KL_samples(mu, Q_chol_3D, P_chol_1D)), feed_dict)
            KL_s_6 = sess.run(tf.reduce_sum(KL_samples(mu, Q_chol_3D, P_chol_2D)), feed_dict)

            KL_1 = sess.run(KL(mu, tf.matrix_diag(Q_chol_2D), P_chol=None), feed_dict)
            KL_2 = sess.run(KL(mu, tf.matrix_diag(Q_chol_2D), P_chol=tf.diag(P_chol_1D)), feed_dict)
            KL_3 = sess.run(KL(mu, tf.matrix_diag(Q_chol_2D), P_chol=P_chol_2D), feed_dict)
            KL_4 = sess.run(KL(mu, Q_chol_3D, P_chol=None), feed_dict)
            KL_5 = sess.run(KL(mu, Q_chol_3D, P_chol=tf.diag(P_chol_1D)), feed_dict)
            KL_6 = sess.run(KL(mu, Q_chol_3D, P_chol=P_chol_2D), feed_dict)

            assert_allclose(KL_s_1, KL_1)
            assert_allclose(KL_s_2, KL_2)
            assert_allclose(KL_s_3, KL_3)
            assert_allclose(KL_s_4, KL_4)
            assert_allclose(KL_s_5, KL_5)
            assert_allclose(KL_s_6, KL_6)

    def test_KL_samples_mu_3D(self):
        with self.test_context() as sess:
            mu_3D = tf.placeholder(FLOAT_TYPE, shape=(self.M, self.n_samples, self.D))
            Q_chol_2D = tf.placeholder(FLOAT_TYPE, shape=(self.M, self.D))
            Q_chol_3D = tf.placeholder(FLOAT_TYPE, shape=(self.M, self.D, self.D))
            P_chol_1D = tf.placeholder(FLOAT_TYPE, shape=(self.D))
            P_chol_2D = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.D))

            feed_dict = self.get_feed_dict([mu_3D], [Q_chol_2D, Q_chol_3D], [P_chol_1D, P_chol_2D])

            KL_s_1 = sess.run(KL_samples(mu_3D, Q_chol_2D, None), feed_dict)
            KL_s_2 = sess.run(KL_samples(mu_3D, Q_chol_2D, P_chol_1D), feed_dict)
            KL_s_3 = sess.run(KL_samples(mu_3D, Q_chol_2D, P_chol_2D), feed_dict)
            KL_s_4 = sess.run(KL_samples(mu_3D, Q_chol_3D, None), feed_dict)
            KL_s_5 = sess.run(KL_samples(mu_3D, Q_chol_3D, P_chol_1D), feed_dict)
            KL_s_6 = sess.run(KL_samples(mu_3D, Q_chol_3D, P_chol_2D), feed_dict)

            KL_mu_only_arg = lambda Q_chol, P_chol: lambda mu: KL_samples(mu, Q_chol, P_chol=P_chol)
            map_schema = lambda Q_chol, P_chol: \
                tf.reduce_mean(tf.map_fn(KL_mu_only_arg(Q_chol, P_chol), tf.transpose(mu_3D, [1,0,2])), 0)

            KL_map_1 = sess.run(map_schema(Q_chol_2D, None), feed_dict)
            KL_map_2 = sess.run(map_schema(Q_chol_2D, P_chol_1D), feed_dict)
            KL_map_3 = sess.run(map_schema(Q_chol_2D, P_chol_2D), feed_dict)
            KL_map_4 = sess.run(map_schema(Q_chol_3D, None), feed_dict)
            KL_map_5 = sess.run(map_schema(Q_chol_3D, P_chol_1D), feed_dict)
            KL_map_6 = sess.run(map_schema(Q_chol_3D, P_chol_2D), feed_dict)

            assert_allclose(KL_s_1, KL_map_1)
            assert_allclose(KL_s_2, KL_map_2)
            assert_allclose(KL_s_3, KL_map_3)
            assert_allclose(KL_s_4, KL_map_4)
            assert_allclose(KL_s_5, KL_map_5)
            assert_allclose(KL_s_6, KL_map_6)

    def test_whitening(self):
        with self.test_context() as sess:
            mu = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.M))
            Q_chol = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.M, self.M))
            P_chol = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.M, self.M))

            feed_dict = self.get_feed_dict([mu], [Q_chol], [P_chol])

            KL_black = sess.run(KL(mu, Q_chol, P_chol=P_chol), feed_dict)
            KL_white = sess.run(KL(tf.matrix_triangular_solve(P_chol, mu[:, :, None], lower=True)[..., 0],
                                   tf.matrix_triangular_solve(P_chol, Q_chol, lower=True)), feed_dict)

            assert_allclose(KL_black, KL_white)

    def test_KL_x1_multiseq(self):
        with self.test_context() as sess:
            mu = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.M))
            Q_chol = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.M, self.M))
            P_chol_1D = tf.placeholder(FLOAT_TYPE, shape=(self.M))
            P_chol_2D = tf.placeholder(FLOAT_TYPE, shape=(self.M, self.M))
            P_chol_3D_diag = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.M))
            P_chol_3D = tf.placeholder(FLOAT_TYPE, shape=(self.D, self.M, self.M))

            feed_dict = self.get_feed_dict([mu], [Q_chol], [P_chol_1D, P_chol_2D, P_chol_3D_diag, P_chol_3D])

            KL_1 = sess.run(KL(mu, Q_chol, P_chol=None), feed_dict)
            KL_2 = sess.run(KL(mu, Q_chol, P_chol=P_chol_1D), feed_dict)
            KL_3 = sess.run(KL(mu, Q_chol, P_chol=P_chol_2D), feed_dict)
            KL_4 = sess.run(KL(mu, Q_chol, P_chol=tf.matrix_diag(P_chol_3D_diag)), feed_dict)
            KL_5 = sess.run(KL(mu, Q_chol, P_chol=P_chol_3D), feed_dict)

            KL_map_1 = sess.run(tf.map_fn(lambda a: KL(a[0], a[1], P_chol=None),
                                          (mu, Q_chol), (FLOAT_TYPE)), feed_dict)
            KL_map_2 = sess.run(tf.map_fn(lambda a: KL(a[0], a[1], P_chol=P_chol_1D),
                                          (mu, Q_chol), (FLOAT_TYPE)), feed_dict)
            KL_map_3 = sess.run(tf.map_fn(lambda a: KL(a[0], a[1], P_chol=P_chol_2D),
                                          (mu, Q_chol), (FLOAT_TYPE)), feed_dict)
            KL_map_4 = sess.run(tf.map_fn(lambda a: KL(a[0], a[1], P_chol=a[2]),
                                          (mu, Q_chol, P_chol_3D_diag), (FLOAT_TYPE)), feed_dict)
            KL_map_5 = sess.run(tf.map_fn(lambda a: KL(a[0], a[1], P_chol=a[2]),
                                          (mu, Q_chol, P_chol_3D), (FLOAT_TYPE)), feed_dict)

            assert_allclose(KL_1, KL_map_1.sum())
            assert_allclose(KL_2, KL_map_2.sum())
            assert_allclose(KL_3, KL_map_3.sum())
            assert_allclose(KL_4, KL_map_4.sum())
            assert_allclose(KL_5, KL_map_5.sum())


if __name__ == '__main__':
    tf.test.main()