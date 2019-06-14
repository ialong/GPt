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
import pytest
from numpy.testing import assert_allclose
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import gpflow as gp
from gpflow import params_as_tensors_for
from gpflow.test_util import GPflowTestCase
from gpflow import mean_functions as mean_fns
from gpflow.conditionals import conditional, Kuu, Kuf
from GPt.gpssm import GPSSM


def general_prepare(self):
    Y = np.random.randn(self.T, self.D)
    inputs = np.random.randn(self.T - 1, self.input_dim) if self.input_dim > 0 else None
    Q_diag = np.random.randn(self.E) ** 2.
    kern = [gp.kernels.RBF(self.E + self.input_dim, ARD=True) for _ in range(self.E)]
    for k in kern: k.lengthscales = np.random.rand(self.E + self.input_dim) * 2.
    for k in kern: k.variance = np.random.rand()
    Z = np.random.randn(self.E, self.n_ind_pts, self.E + self.input_dim)
    mean_fn = mean_fns.Linear(np.random.randn(self.E, self.E), np.random.randn(self.E))
    Umu = np.random.randn(self.E, self.n_ind_pts)
    Ucov_chol = np.random.randn(self.E, self.n_ind_pts, self.n_ind_pts)
    Ucov_chol = np.linalg.cholesky(np.matmul(Ucov_chol, np.transpose(Ucov_chol, [0, 2, 1])))
    qx1_mu = np.random.randn(self.E)
    qx1_cov = np.random.randn(self.E, self.E)
    qx1_cov = qx1_cov @ qx1_cov.T
    As = np.random.randn(self.T - 1, self.E)
    bs = np.random.randn(self.T - 1, self.E)
    Ss = np.random.randn(self.T - 1, self.E) ** 2.
    m = GPSSM(self.E, Y, inputs=inputs, emissions=None, px1_mu=None, px1_cov=None,
              kern=kern, Z=Z, n_ind_pts=None, mean_fn=mean_fn,
              Q_diag=Q_diag, Umu=Umu, Ucov_chol=Ucov_chol,
              qx1_mu=qx1_mu, qx1_cov=qx1_cov, As=As, bs=bs, Ss=Ss, n_samples=self.n_samples, seed=self.seed)
    _ = m.compute_log_likelihood()
    return m


class FactorizedSamplingTest(GPflowTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 0
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.T, self.D, self.E, self.input_dim = 11, 3, 2, 0
        self.n_samples, self.n_ind_pts = int(1e3), 4
        self.white = True

    def prepare(self):
        return general_prepare(self)

    def test_X_samples(self):
        with self.test_context() as sess:
            shape = [self.T, self.n_samples, self.E]

            m = self.prepare()

            qe_samples = tfd.MultivariateNormalDiag(loc=tf.zeros(shape[1:], dtype=gp.settings.float_type))
            qe_samples = sess.run(qe_samples.sample(self.T, seed=self.seed))
            X_tmin1 = tf.placeholder(gp.settings.float_type, shape=shape[1:])
            Kzz = sess.run(Kuu(m.Z, m.kern, jitter=gp.settings.numerics.jitter_level))
            Kzz_inv = np.linalg.inv(np.linalg.cholesky(Kzz)) if self.white else np.linalg.inv(Kzz)  # E x M x M
            X_samples_np = np.zeros(shape)
            X_samples_np[0] = m.qx1_mu.value + qe_samples[0] @ m.qx1_cov_chol.value.T
            for t in range(self.T-1):
                Kzx = sess.run(Kuf(m.Z, m.kern, X_tmin1), feed_dict={X_tmin1:X_samples_np[t]})  # E x M x N
                Kxx = sess.run(m.kern.Kdiag(X_tmin1, full_output_cov=False), feed_dict={X_tmin1:X_samples_np[t]})  # N x E
                mean_x = sess.run(m.mean_fn(X_tmin1), feed_dict={X_tmin1:X_samples_np[t]})
                Kzz_invKzx = np.matmul(Kzz_inv, Kzx)  # E x M x N
                mu = mean_x + np.sum(Kzz_invKzx * m.Umu.value[..., None], 1).T  # N x E
                mu = m.As.value[t] * mu + m.bs.value[t]
                if self.white:
                    cov = np.matmul(np.transpose(m.Ucov_chol.value, [0, 2, 1]), Kzz_invKzx)
                    cov = np.sum(np.square(cov) - np.square(Kzz_invKzx), 1)
                else:
                    cov = np.matmul(m.Ucov_chol.value, np.transpose(m.Ucov_chol.value, [0, 2, 1])) - Kzz  # E x M x M
                    cov = np.sum(np.matmul(cov, Kzz_invKzx) * Kzz_invKzx, 1)
                cov = Kxx + cov.T
                cov = np.square(m.As.value[t]) * cov + np.square(m.S_chols.value[t])  # N x E
                X_samples_np[t+1] = mu + qe_samples[t+1] * np.sqrt(cov)

            X_samples_tf = sess.run(m._build_linear_time_q_sample(sample_u=False))[0]

            assert_allclose(X_samples_tf, X_samples_np)

    def test_X_F_samples(self):
        with self.test_context() as sess:
            shape = [self.T, self.n_samples, self.E]

            m = self.prepare()

            qe_samples = tfd.MultivariateNormalDiag(loc=tf.zeros(shape[1:], dtype=gp.settings.float_type))
            qe_samples_X = sess.run(qe_samples.sample(self.T, seed=self.seed))
            qe_samples_F = sess.run(qe_samples.sample(self.T-1, seed=self.seed))
            X_tmin1 = tf.placeholder(gp.settings.float_type, shape=shape[1:])
            Kzz = sess.run(Kuu(m.Z, m.kern, jitter=gp.settings.numerics.jitter_level))
            Kzz_inv = np.linalg.inv(np.linalg.cholesky(Kzz)) if self.white else np.linalg.inv(Kzz)  # E x M x M
            X_samples_np = np.zeros(shape)
            X_samples_np[0] = m.qx1_mu.value + qe_samples_X[0] @ m.qx1_cov_chol.value.T
            F_samples_np = np.zeros([self.T-1] + shape[1:])
            for t in range(self.T-1):
                Kzx = sess.run(Kuf(m.Z, m.kern, X_tmin1), feed_dict={X_tmin1:X_samples_np[t]})  # E x M x N
                Kxx = sess.run(m.kern.Kdiag(X_tmin1, full_output_cov=False), feed_dict={X_tmin1:X_samples_np[t]})  # N x E
                mean_x = sess.run(m.mean_fn(X_tmin1), feed_dict={X_tmin1:X_samples_np[t]})
                Kzz_invKzx = np.matmul(Kzz_inv, Kzx)  # E x M x N
                mu = mean_x + np.sum(Kzz_invKzx * m.Umu.value[..., None], 1).T  # N x E
                if self.white:
                    cov = np.matmul(np.transpose(m.Ucov_chol.value, [0, 2, 1]), Kzz_invKzx)
                    cov = np.sum(np.square(cov) - np.square(Kzz_invKzx), 1)
                else:
                    cov = np.matmul(m.Ucov_chol.value, np.transpose(m.Ucov_chol.value, [0, 2, 1])) - Kzz  # E x M x M
                    cov = np.sum(np.matmul(cov, Kzz_invKzx) * Kzz_invKzx, 1)
                cov = Kxx + cov.T
                F_samples_np[t] = mu + qe_samples_F[t] * np.sqrt(cov)

                x_mu = m.As.value[t] * F_samples_np[t] + m.bs.value[t]
                X_samples_np[t+1] = x_mu + qe_samples_X[t+1] * m.S_chols.value[t]

            X_samples_tf, F_samples_tf = sess.run(m._build_linear_time_q_sample(sample_f=True, sample_u=False))

            assert_allclose(X_samples_tf, X_samples_np)
            assert_allclose(F_samples_tf, F_samples_np)

    def test_f_moments(self):
        with self.test_context() as sess:
            m = self.prepare()
            X_samples, F_samples, f_mus, f_vars = sess.run(
                m._build_linear_time_q_sample(return_f_moments=True, sample_f=True, sample_u=False))
            f_mus_batch, f_vars_batch = conditional(tf.reshape(X_samples[:-1], [-1, self.E]),
                                                    m.Z, m.kern, m.Umu.constrained_tensor, white=self.white,
                                                    q_sqrt=m.Ucov_chol.constrained_tensor)
            f_mus_batch += m.mean_fn(tf.reshape(X_samples[:-1], [-1, self.E]))

            f_mus_batch = sess.run(f_mus_batch).reshape(self.T - 1, self.n_samples, self.E)
            f_vars_batch = sess.run(f_vars_batch).reshape(self.T - 1, self.n_samples, self.E)

            assert_allclose(f_mus, f_mus_batch)
            assert_allclose(f_vars, f_vars_batch)

            X_samples_2, f_mus_2, f_vars_2 = sess.run(
                m._build_linear_time_q_sample(return_f_moments=True, sample_f=False, sample_u=False))
            f_mus_batch_2, f_vars_batch_2 = conditional(tf.reshape(X_samples_2[:-1], [-1, self.E]),
                                                        m.Z, m.kern, m.Umu.constrained_tensor, white=self.white,
                                                        q_sqrt=m.Ucov_chol.constrained_tensor)
            f_mus_batch_2 += m.mean_fn(tf.reshape(X_samples_2[:-1], [-1, self.E]))

            f_mus_batch_2 = sess.run(f_mus_batch_2).reshape(self.T - 1, self.n_samples, self.E)
            f_vars_batch_2 = sess.run(f_vars_batch_2).reshape(self.T - 1, self.n_samples, self.E)

            assert_allclose(f_mus_2, f_mus_batch_2)
            assert_allclose(f_vars_2, f_vars_batch_2)


class uDependenceSamplingTest(GPflowTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 0
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.T, self.D, self.E, self.input_dim = 11, 3, 2, 0
        self.n_samples, self.n_ind_pts = int(1e3), 4
        self.white = True

    def prepare(self):
        return general_prepare(self)

    def test_X_samples(self):
        with self.test_context() as sess:
            shape = [self.T, self.n_samples, self.E]

            m = self.prepare()

            qe_samples = tfd.MultivariateNormalDiag(loc=tf.zeros(shape[1:], dtype=gp.settings.float_type))
            qe_samples = sess.run(qe_samples.sample(self.T, seed=self.seed))
            U_samples_np = sess.run(tfd.MultivariateNormalDiag(loc=tf.zeros(
                [self.E, self.n_ind_pts, self.n_samples], dtype=gp.settings.float_type)).sample(seed=self.seed))
            U_samples_np = m.Umu.value[:, :, None] + np.matmul(m.Ucov_chol.value, U_samples_np)

            X_tmin1 = tf.placeholder(gp.settings.float_type, shape=shape[1:])
            Kzz = sess.run(Kuu(m.Z, m.kern, jitter=gp.settings.numerics.jitter_level))
            Kzz_inv = np.linalg.inv(np.linalg.cholesky(Kzz)) if self.white else np.linalg.inv(Kzz)  # E x M x M
            X_samples_np = np.zeros(shape)
            X_samples_np[0] = m.qx1_mu.value + qe_samples[0] @ m.qx1_cov_chol.value.T
            for t in range(self.T-1):
                Kzx = sess.run(Kuf(m.Z, m.kern, X_tmin1), feed_dict={X_tmin1:X_samples_np[t]})  # E x M x N
                Kxx = sess.run(m.kern.Kdiag(X_tmin1, full_output_cov=False), feed_dict={X_tmin1:X_samples_np[t]})  # N x E
                mean_x = sess.run(m.mean_fn(X_tmin1), feed_dict={X_tmin1:X_samples_np[t]})
                Kzz_invKzx = np.matmul(Kzz_inv, Kzx)  # E x M x N
                mu = mean_x + np.sum(Kzz_invKzx * U_samples_np, 1).T  # N x E
                mu = m.As.value[t] * mu + m.bs.value[t]
                if self.white:
                    cov = np.sum(np.square(Kzz_invKzx), 1)
                else:
                    cov = np.sum(Kzz_invKzx * Kzx, 1)
                cov = Kxx - cov.T
                cov = np.square(m.As.value[t]) * cov + np.square(m.S_chols.value[t])  # N x E
                X_samples_np[t+1] = mu + qe_samples[t+1] * np.sqrt(cov)

            X_samples_tf, U_samples_tf = sess.run(m._build_linear_time_q_sample(sample_u=True, return_u=True))

            assert_allclose(X_samples_tf, X_samples_np)
            assert_allclose(U_samples_tf, U_samples_np)

    def test_X_F_samples(self):
        with self.test_context() as sess:
            shape = [self.T, self.n_samples, self.E]

            m = self.prepare()

            qe_samples = tfd.MultivariateNormalDiag(loc=tf.zeros(shape[1:], dtype=gp.settings.float_type))
            qe_samples_X = sess.run(qe_samples.sample(self.T, seed=self.seed))
            qe_samples_F = sess.run(qe_samples.sample(self.T-1, seed=self.seed))
            U_samples_np = sess.run(tfd.MultivariateNormalDiag(loc=tf.zeros(
                [self.E, self.n_ind_pts, self.n_samples], dtype=gp.settings.float_type)).sample(seed=self.seed))
            U_samples_np = m.Umu.value[:, :, None] + np.matmul(m.Ucov_chol.value, U_samples_np)

            X_tmin1 = tf.placeholder(gp.settings.float_type, shape=shape[1:])
            Kzz = sess.run(Kuu(m.Z, m.kern, jitter=gp.settings.numerics.jitter_level))
            Kzz_inv = np.linalg.inv(np.linalg.cholesky(Kzz)) if self.white else np.linalg.inv(Kzz)  # E x M x M
            X_samples_np = np.zeros(shape)
            X_samples_np[0] = m.qx1_mu.value + qe_samples_X[0] @ m.qx1_cov_chol.value.T
            F_samples_np = np.zeros([self.T-1] + shape[1:])
            for t in range(self.T-1):
                Kzx = sess.run(Kuf(m.Z, m.kern, X_tmin1), feed_dict={X_tmin1:X_samples_np[t]})  # E x M x N
                Kxx = sess.run(m.kern.Kdiag(X_tmin1, full_output_cov=False), feed_dict={X_tmin1:X_samples_np[t]})  # N x E
                mean_x = sess.run(m.mean_fn(X_tmin1), feed_dict={X_tmin1:X_samples_np[t]})
                Kzz_invKzx = np.matmul(Kzz_inv, Kzx)  # E x M x N
                mu = mean_x + np.sum(Kzz_invKzx * U_samples_np, 1).T  # N x E
                if self.white:
                    cov = np.sum(np.square(Kzz_invKzx), 1)
                else:
                    cov = np.sum(Kzz_invKzx * Kzx, 1)
                cov = Kxx - cov.T
                F_samples_np[t] = mu + qe_samples_F[t] * np.sqrt(cov)

                x_mu = m.As.value[t] * F_samples_np[t] + m.bs.value[t]
                X_samples_np[t+1] = x_mu + qe_samples_X[t+1] * m.S_chols.value[t]

            X_samples_tf, F_samples_tf, U_samples_tf = sess.run(
                m._build_linear_time_q_sample(sample_f=True, sample_u=True, return_u=True))

            assert_allclose(X_samples_tf, X_samples_np)
            assert_allclose(F_samples_tf, F_samples_np)
            assert_allclose(U_samples_tf, U_samples_np)

    def test_f_moments(self):
        with self.test_context() as sess:
            m = self.prepare()
            X_samples, F_samples, f_mus, f_vars, U_samples = sess.run(
                m._build_linear_time_q_sample(return_f_moments=True, sample_f=True, sample_u=True, return_u=True))

            X_samples_2, f_mus_2, f_vars_2, U_samples_2 = sess.run(
                m._build_linear_time_q_sample(return_f_moments=True, sample_f=False, sample_u=True, return_u=True))

            def single_t_moments(X, U_samples):
                f_mu, f_var = conditional(X, m.Z, m.kern, tf.constant(U_samples, dtype=gp.settings.float_type),
                                          q_sqrt=None, white=self.white)
                f_mu += m.mean_fn(X)
                return f_mu, f_var

            f_mus_batch, f_vars_batch = sess.run(
                tf.map_fn(lambda X: single_t_moments(X, U_samples),
                          tf.constant(X_samples[:-1], dtype=gp.settings.float_type),
                          dtype=(gp.settings.float_type, gp.settings.float_type)))

            f_mus_batch_2, f_vars_batch_2 = sess.run(
                tf.map_fn(lambda X: single_t_moments(X, U_samples_2),
                          tf.constant(X_samples_2[:-1], dtype=gp.settings.float_type),
                          dtype=(gp.settings.float_type, gp.settings.float_type)))

            assert_allclose(f_mus, f_mus_batch)
            assert_allclose(f_vars, f_vars_batch)

            assert_allclose(f_mus_2, f_mus_batch_2)
            assert_allclose(f_vars_2, f_vars_batch_2)


class JointSamplingTest(GPflowTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 0
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.T, self.D, self.E, self.input_dim = 6, 4, 3, 0
        self.n_samples, self.n_ind_pts = int(1e1), 10
        self.white = True

    def prepare(self):
        return general_prepare(self)

    def test_joint_samples(self):
        with self.test_context() as sess:
            shape = [self.T, self.n_samples, self.E]

            m = self.prepare()

            white_samples_X = tfd.MultivariateNormalDiag(
                loc=tf.zeros((self.n_samples, self.T, self.E), dtype=gp.settings.float_type)).sample(seed=self.seed)
            white_samples_X = np.transpose(sess.run(white_samples_X), [1, 0, 2])

            white_samples_F = tfd.MultivariateNormalDiag(
                loc=tf.zeros((self.n_samples, self.T - 1, self.E), dtype=gp.settings.float_type)).sample(seed=self.seed)
            white_samples_F = np.transpose(sess.run(white_samples_F), [1, 0, 2])

            X_buff = tf.placeholder(gp.settings.float_type, shape=[None, self.E])
            Kzz = sess.run(Kuu(m.Z, m.kern, jitter=gp.settings.numerics.jitter_level))
            Kzz_inv = np.linalg.inv(np.linalg.cholesky(Kzz)) if self.white else np.linalg.inv(Kzz)  # E x M x M
            X_samples_np = np.zeros(shape)
            F_samples_np = np.zeros([self.T - 1] + shape[1:])
            f_mus_np = np.zeros([self.T - 1] + shape[1:])
            f_vars_np = np.zeros([self.T - 1] + shape[1:])

            X_samples_np[0] = white_samples_X[0] @ m.qx1_cov_chol.value.T + m.qx1_mu.value

            Kzx = sess.run(Kuf(m.Z, m.kern, X_buff), feed_dict={X_buff: X_samples_np[0]})  # E x M x N
            Kxx = sess.run(m.kern.Kdiag(X_buff, full_output_cov=False), feed_dict={X_buff: X_samples_np[0]})  # N x E
            mean_x = sess.run(m.mean_fn(X_buff), feed_dict={X_buff: X_samples_np[0]})  # N x E
            Kzz_invKzx = np.matmul(Kzz_inv, Kzx)  # E x M x N
            f_mus_np[0] = mean_x + np.sum(Kzz_invKzx * m.Umu.value[..., None], 1).T  # N x E
            if self.white:
                f_var = np.matmul(np.transpose(m.Ucov_chol.value, [0, 2, 1]), Kzz_invKzx)
                f_var = np.sum(np.square(f_var) - np.square(Kzz_invKzx), 1)
            else:
                f_var = np.matmul(m.Ucov_chol.value, np.transpose(m.Ucov_chol.value, [0, 2, 1])) - Kzz  # E x M x M
                f_var = np.sum(np.matmul(f_var, Kzz_invKzx) * Kzz_invKzx, 1)
            f_vars_np[0] = Kxx + f_var.T
            F_samples_np[0] = f_mus_np[0] + white_samples_F[0] * np.sqrt(f_vars_np[0])
            X_samples_np[1] = m.As.value[0] * F_samples_np[0] + m.bs.value[0] + white_samples_X[1] * m.S_chols.value[0]

            def single_sample_f_cond(X, F):
                feed_dict = {X_buff: X}
                Kzx = sess.run(Kuf(m.Z, m.kern, X_buff), feed_dict=feed_dict)  # E x M x t+1
                Kxx = sess.run(m.kern.K(X_buff, full_output_cov=False), feed_dict=feed_dict)  # E x t+1 x t+1
                mean_x = sess.run(m.mean_fn(X_buff), feed_dict=feed_dict)  # t+1 x E
                Kzz_invKzx = np.matmul(Kzz_inv, Kzx)  # E x M x t+1
                f_mu_joint = mean_x + np.sum(Kzz_invKzx * m.Umu.value[..., None], 1).T  # t+1 x E
                if self.white:
                    f_cov_joint = np.matmul(np.transpose(m.Ucov_chol.value, [0, 2, 1]), Kzz_invKzx)  # E x M x t+1
                    f_cov_joint = np.matmul(np.transpose(f_cov_joint, [0, 2, 1]), f_cov_joint)  # E x t+1 x t+1
                    f_cov_joint -= np.matmul(np.transpose(Kzz_invKzx, [0, 2, 1]), Kzz_invKzx)  # E x t+1 x t+1
                else:
                    f_cov_joint = np.matmul(m.Ucov_chol.value, np.transpose(m.Ucov_chol.value, [0, 2, 1])) - Kzz  # E x M x M
                    f_cov_joint = np.matmul(np.matmul(np.transpose(Kzz_invKzx, [0, 2, 1]), f_cov_joint), Kzz_invKzx)  # E x t+1 x t+1
                f_cov_joint = Kxx + f_cov_joint  # E x t+1 x t+1

                C_F_inv_C_F_ft = np.linalg.solve(f_cov_joint[:, :-1, :-1], f_cov_joint[:, :-1, -1:None])[:, :, 0]  # E x t
                F_min_Fmu = F - f_mu_joint[:-1]
                f_mu = f_mu_joint[-1] + np.sum(C_F_inv_C_F_ft * F_min_Fmu.T, -1)  # E
                f_var = f_cov_joint[:, -1, -1] - np.sum(C_F_inv_C_F_ft * f_cov_joint[:, :-1, -1], -1)  # E
                return f_mu, f_var

            for t in range(1, self.T-1):
                for n in range(self.n_samples):
                    f_mus_np[t, n], f_vars_np[t, n] = single_sample_f_cond(X_samples_np[:t+1, n], F_samples_np[:t, n])

                    F_samples_np[t, n] = f_mus_np[t, n] + white_samples_F[t, n] * np.sqrt(f_vars_np[t, n])

                    X_samples_np[t+1, n] = m.As.value[t] * F_samples_np[t, n] + m.bs.value[t] \
                                           + white_samples_X[t+1, n] * m.S_chols.value[t]

            X_samples_tf, F_samples_tf, f_mus_tf, f_vars_tf = sess.run(
                m._build_cubic_time_q_sample(return_f_moments=True, sample_u=False, add_jitter=False))

            assert_allclose(X_samples_tf, X_samples_np)
            assert_allclose(F_samples_tf, F_samples_np)
            assert_allclose(f_mus_tf, f_mus_np)
            assert_allclose(f_vars_tf, f_vars_np)

    def test_joint_samples_sample_u(self):
        with self.test_context() as sess:
            shape = [self.T, self.n_samples, self.E]

            m = self.prepare()

            white_samples_X = tfd.MultivariateNormalDiag(
                loc=tf.zeros((self.n_samples, self.T, self.E), dtype=gp.settings.float_type)).sample(seed=self.seed)
            white_samples_X = np.transpose(sess.run(white_samples_X), [1, 0, 2])

            white_samples_F = tfd.MultivariateNormalDiag(
                loc=tf.zeros((self.n_samples, self.T - 1, self.E), dtype=gp.settings.float_type)).sample(seed=self.seed)
            white_samples_F = np.transpose(sess.run(white_samples_F), [1, 0, 2])

            U_samples_np = sess.run(tfd.MultivariateNormalDiag(
                loc=tf.zeros((self.E, self.n_ind_pts, self.n_samples), dtype=gp.settings.float_type)
            ).sample(seed=self.seed))
            U_samples_np = m.Umu.value[:, :, None] + np.matmul(m.Ucov_chol.value, U_samples_np)

            Kzz = sess.run(Kuu(m.Z, m.kern, jitter=gp.settings.numerics.jitter_level))

            if self.white:
                Kzz_chol = np.linalg.cholesky(Kzz)
                U_samples_np = np.matmul(Kzz_chol, U_samples_np)

            X_buff = tf.placeholder(gp.settings.float_type, shape=[None, self.E])

            X_samples_np = np.zeros(shape)
            F_samples_np = np.zeros([self.T - 1] + shape[1:])
            f_mus_np = np.zeros([self.T - 1] + shape[1:])
            f_vars_np = np.zeros([self.T - 1] + shape[1:])

            X_samples_np[0] = white_samples_X[0] @ m.qx1_cov_chol.value.T + m.qx1_mu.value

            def single_sample_f_cond(K, X, F, U):
                feed_dict = {X_buff: X}
                Kzx = sess.run(Kuf(m.Z, m.kern, X_buff[-1:]), feed_dict=feed_dict)[:, :, 0]  # E x M
                Kxx = sess.run(m.kern.K(X_buff, X_buff[-1:], full_output_cov=False), feed_dict=feed_dict)[:, :, 0]  # E x (t+1)

                K_vector = np.concatenate([Kzx, Kxx], -1)  # E x (M+t+1)
                mean_x = sess.run(m.mean_fn(X_buff), feed_dict=feed_dict)
                UF = (F - mean_x[:-1]).T
                UF = np.concatenate([U, UF], -1)  # E x (M+t)

                Kinv_UF_Kvec = np.linalg.solve(K, np.stack([UF, K_vector[:, :-1]], -1))
                f_mu_f_var = np.sum(K_vector[:, :-1, None] * Kinv_UF_Kvec, -2)

                f_mu = mean_x[-1] + f_mu_f_var[:, 0]
                f_var = K_vector[:, -1] - f_mu_f_var[:, 1]

                K = np.concatenate([K, K_vector[:, :-1, None]], -1)  # E x (M+t) x (M+t+1)
                K = np.concatenate([K, K_vector[:, None, :]], -2)  # E x (M+t+1) x (M+t+1)

                return K, f_mu, f_var

            for n in range(self.n_samples):
                K = Kzz
                for t in range(self.T - 1):
                    K, f_mus_np[t, n], f_vars_np[t, n] = single_sample_f_cond(
                        K, X_samples_np[:t+1, n], F_samples_np[:t, n], U_samples_np[:, :, n])

                    F_samples_np[t, n] = f_mus_np[t, n] + white_samples_F[t, n] * np.sqrt(f_vars_np[t, n])

                    X_samples_np[t + 1, n] = m.As.value[t] * F_samples_np[t, n] + m.bs.value[t] \
                                             + white_samples_X[t + 1, n] * m.S_chols.value[t]

            if self.white:
                U_samples_np = np.linalg.solve(Kzz_chol, U_samples_np)

            X_samples_tf, F_samples_tf, f_mus_tf, f_vars_tf, U_samples_tf = sess.run(
                m._build_cubic_time_q_sample(return_f_moments=True, sample_u=True, return_u=True, add_jitter=False))

            assert_allclose(X_samples_tf, X_samples_np)
            assert_allclose(F_samples_tf, F_samples_np)
            assert_allclose(f_mus_tf, f_mus_np)
            assert_allclose(f_vars_tf, f_vars_np)
            assert_allclose(U_samples_tf, U_samples_np)


if __name__ == '__main__':
    tf.test.main()
