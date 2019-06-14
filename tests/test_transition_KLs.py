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
from gpflow import mean_functions as mean_fns
from gpflow.conditionals import conditional, Kuu, Kuf
from GPt.gpssm import GPSSM
from GPt.emissions import GaussianEmissions
from GPt.KL import KL, KL_samples


class TransitionKLsTest(GPflowTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 0
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.T, self.D, self.E = 11, 3, 2
        self.n_samples, self.n_ind_pts = int(1e5), 4
        self.white = True

    def prepare(self):
        Y = np.random.randn(self.T, self.D)
        Q_diag = np.random.randn(self.E) ** 2.
        kern = [gp.kernels.RBF(self.E, ARD=True) for _ in range(self.E)]
        for k in kern: k.lengthscales = np.random.rand(self.E)
        for k in kern: k.variance = np.random.rand()
        Z = np.random.randn(self.E, self.n_ind_pts, self.E)
        mean_fn = mean_fns.Linear(np.random.randn(self.E, self.E), np.random.randn(self.E))
        Umu = np.random.randn(self.E, self.n_ind_pts)
        Ucov_chol = np.random.randn(self.E, self.n_ind_pts, self.n_ind_pts)
        Ucov_chol = np.linalg.cholesky(np.matmul(Ucov_chol, np.transpose(Ucov_chol, [0, 2, 1])))
        qx1_mu = np.random.randn(self.E)
        qx1_cov = np.random.randn(self.E, self.E)
        qx1_cov = qx1_cov @ qx1_cov.T
        As = np.random.randn(self.T-1, self.E)
        bs = np.random.randn(self.T-1, self.E)
        Ss = np.random.randn(self.T-1, self.E) ** 2.
        m = GPSSM(self.E, Y, inputs=None, emissions=None, px1_mu=None, px1_cov=None,
                  kern=kern, Z=Z, n_ind_pts=None, mean_fn=mean_fn,
                  Q_diag=Q_diag, Umu=Umu, Ucov_chol=Ucov_chol,
                  qx1_mu=qx1_mu, qx1_cov=qx1_cov, As=As, bs=bs, Ss=Ss, n_samples=self.n_samples, seed=self.seed)
        _ = m.compute_log_likelihood()
        return m

    def test_transition_KLs_MC(self):
        with self.test_context() as sess:
            shape = [self.T - 1, self.n_samples, self.E]
            X_samples = tf.placeholder(gp.settings.float_type, shape=shape)
            feed_dict = {X_samples: np.random.randn(*shape)}

            m = self.prepare()
            f_mus, f_vars = conditional(tf.reshape(X_samples, [-1, self.E]),
                                        m.Z, m.kern, m.Umu.constrained_tensor, white=self.white,
                                        q_sqrt=m.Ucov_chol.constrained_tensor)
            f_mus += m.mean_fn(tf.reshape(X_samples, [-1, self.E]))

            gpssm_KLs = m._build_transition_KLs(tf.reshape(f_mus, [m.T - 1, m.n_samples, m.latent_dim]),
                                                tf.reshape(f_vars, [m.T - 1, m.n_samples, m.latent_dim]))

            f_samples = f_mus + tf.sqrt(f_vars) * tf.random_normal(
                [(self.T - 1) * self.n_samples, self.E], dtype=gp.settings.float_type, seed=self.seed)

            q_mus = m.As.constrained_tensor[:, None, :] * tf.reshape(f_samples, shape) \
                    + m.bs.constrained_tensor[:, None, :]
            q_mus = tf.reshape(q_mus, [-1, self.E])
            q_covs = tf.reshape(tf.tile(
                m.S_chols.constrained_tensor[:, None, :], [1, self.n_samples, 1]), [-1, self.E])
            mc_KLs = KL_samples(q_mus - f_samples, Q_chol=q_covs, P_chol=m.Q_sqrt.constrained_tensor)
            mc_KLs = tf.reduce_mean(tf.reshape(mc_KLs, shape[:-1]), -1)

            assert_allclose(*sess.run([gpssm_KLs, mc_KLs], feed_dict=feed_dict), rtol=0.5*1e-2)

    def test_transition_KLs_extra_trace(self):
        with self.test_context() as sess:
            shape = [self.T - 1, self.n_samples, self.E]
            X_samples = tf.placeholder(gp.settings.float_type, shape=shape)
            feed_dict = {X_samples: np.random.randn(*shape)}

            m = self.prepare()
            f_mus, f_vars = conditional(tf.reshape(X_samples, [-1, self.E]),
                                        m.Z, m.kern, m.Umu.constrained_tensor, white=self.white,
                                        q_sqrt=m.Ucov_chol.constrained_tensor)
            f_mus += m.mean_fn(tf.reshape(X_samples, [-1, self.E]))

            gpssm_KLs = m._build_transition_KLs(tf.reshape(f_mus, [m.T - 1, m.n_samples, m.latent_dim]),
                                                tf.reshape(f_vars, [m.T - 1, m.n_samples, m.latent_dim]))

            q_mus = m.As.constrained_tensor[:, None, :] * tf.reshape(f_mus, shape) \
                  + m.bs.constrained_tensor[:, None, :]
            q_mus = tf.reshape(q_mus, [-1, self.E])
            q_covs = tf.reshape(tf.tile(
                m.S_chols.constrained_tensor[:, None, :], [1, self.n_samples, 1]), [-1, self.E])
            trace_KLs = KL_samples(q_mus - f_mus, Q_chol=q_covs, P_chol=m.Q_sqrt.constrained_tensor)
            trace_KLs = tf.reduce_mean(tf.reshape(trace_KLs, shape[:-1]), -1)

            trace_KLs += 0.5 * tf.reduce_mean(tf.reduce_sum(
                (tf.square(m.As.constrained_tensor - 1.)[:, None, :] * tf.reshape(f_vars, shape))
                / tf.square(m.Q_sqrt.constrained_tensor), -1), -1)

            assert_allclose(*sess.run([gpssm_KLs, trace_KLs], feed_dict=feed_dict))

    def test_factorized_transition_KLs(self):
        def KL_sampled_mu_and_Q_diag_P(mu_diff, Q_chol, P_chol):
            """
            :param mu_diff: NxSxD
            :param Q_chol: NxSxD
            :param P_chol: D
            :return: N
            """
            D = tf.shape(mu_diff)[-1]
            assert mu_diff.shape.ndims is not None
            assert Q_chol.shape.ndims is not None
            assert P_chol.shape.ndims is not None

            mahalanobis = mu_diff / P_chol
            mahalanobis = tf.reduce_sum(tf.square(mahalanobis), -1)
            mahalanobis = tf.reduce_mean(mahalanobis, -1)

            trace = Q_chol / P_chol
            trace = tf.reduce_sum(tf.square(trace), -1)
            trace = tf.reduce_mean(trace, -1)

            constant = tf.cast(D, dtype=mu_diff.dtype)
            log_det_P = 2. * tf.reduce_sum(tf.log(tf.abs(P_chol)))
            log_det_Q = 2. * tf.reduce_mean(tf.reduce_sum(tf.log(tf.abs(Q_chol)), -1), -1)
            double_KL = trace + mahalanobis - constant + log_det_P - log_det_Q
            return 0.5 * double_KL

        with self.test_context() as sess:
            m = self.prepare()
            with gp.params_as_tensors_for(m):
                _, f_mus, f_vars, xcov_chols = sess.run(m._build_linear_time_q_sample(
                    return_f_moments=True, return_x_cov_chols=True, sample_f=False, sample_u=False))

                gpssm_KLs = sess.run(m._build_transition_KLs(tf.constant(f_mus), tf.constant(f_vars)))

                diff_term = tf.reduce_sum(tf.reduce_mean(tf.constant(f_vars), -2) * m.As / tf.square(m.Q_sqrt), -1)
                diff_term += tf.reduce_sum(tf.log(tf.abs(m.S_chols)), 1)
                diff_term -= tf.reduce_sum(tf.reduce_mean(tf.log(tf.abs(tf.constant(xcov_chols))), -2), -1)

                gpssm_KLs += sess.run(diff_term)

                gpssm_factorized_KLs = sess.run(m._build_factorized_transition_KLs(
                    tf.constant(f_mus), tf.constant(f_vars), tf.constant(xcov_chols)))

                assert_allclose(gpssm_KLs, gpssm_factorized_KLs)

                gpssm_factorized_KLs_2 = sess.run(KL_sampled_mu_and_Q_diag_P(
                    m.As[:, None, :] * f_mus + m.bs[:, None, :] - f_mus,
                    tf.constant(xcov_chols),
                    m.Q_sqrt))
                gpssm_factorized_KLs_2 += 0.5 * np.mean(np.sum(f_vars / np.square(sess.run(m.Q_sqrt)), -1), -1)

                assert_allclose(gpssm_factorized_KLs, gpssm_factorized_KLs_2)


if __name__ == '__main__':
    tf.test.main()
