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
import tensorflow as tf
import gpflow as gp
import pandas as pd

from tensorflow_probability import distributions as tfd

from gpflow import Param, params_as_tensors
from gpflow import transforms as gtf
from gpflow import mean_functions as mean_fns
from gpflow.conditionals import conditional
from gpflow.multioutput.features import Kuu, Kuf
from gpflow import settings as gps
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf

from .KL import KL, KL_samples
from .emissions import GaussianEmissions


class GPSSM(gp.models.Model):
    """Gaussian Process State-Space Model base class. Used for sampling, no built-in inference."""
    def __init__(self,
                 latent_dim,
                 Y,
                 inputs=None,
                 emissions=None,
                 px1_mu=None, px1_cov=None,
                 kern=None,
                 Z=None, n_ind_pts=100,
                 mean_fn=None,
                 Q_diag=None,
                 Umu=None, Ucov_chol=None,
                 qx1_mu=None, qx1_cov=None,
                 As=None, bs=None, Ss=None,
                 n_samples=100,
                 seed=None,
                 parallel_iterations=10,
                 jitter=gps.numerics.jitter_level,
                 name=None):

        super().__init__(name=name)

        self.latent_dim = latent_dim
        self.T, self.obs_dim = Y.shape
        self.Y = Param(Y, trainable=False)

        self.inputs = None if inputs is None else Param(inputs, trainable=False)
        self.input_dim = 0 if self.inputs is None else self.inputs.shape[1]

        self.qx1_mu = Param(np.zeros(self.latent_dim) if qx1_mu is None else qx1_mu)
        self.qx1_cov_chol = Param(
            np.eye(self.latent_dim) if qx1_cov is None else np.linalg.cholesky(qx1_cov),
            transform=gtf.LowerTriangular(self.latent_dim, squeeze=True))

        self.As = Param(np.ones((self.T - 1, self.latent_dim)) if As is None else As)
        self.bs = Param(np.zeros((self.T - 1, self.latent_dim)) if bs is None else bs)

        self.Q_sqrt = Param(np.ones(self.latent_dim) if Q_diag is None else Q_diag ** 0.5, transform=gtf.positive)
        if Ss is False:
            self._S_chols = None
        else:
            self.S_chols = Param(np.tile(self.Q_sqrt.value.copy()[None, ...], [self.T - 1, 1]) if Ss is None
                                 else (np.sqrt(Ss) if Ss.ndim == 2 else np.linalg.cholesky(Ss)),
                                 transform=gtf.positive if (Ss is None or Ss.ndim == 2)
                                 else gtf.LowerTriangular(self.latent_dim, num_matrices=self.T - 1, squeeze=False))

        self.emissions = emissions or GaussianEmissions(latent_dim=self.latent_dim, obs_dim=self.obs_dim)

        self.px1_mu = Param(np.zeros(self.latent_dim) if px1_mu is None else px1_mu, trainable=False)
        self.px1_cov_chol = None if px1_cov is None else \
            Param(np.sqrt(px1_cov) if px1_cov.ndim == 1 else np.linalg.cholesky(px1_cov), trainable=False,
                  transform=gtf.positive if px1_cov.ndim == 1 else gtf.LowerTriangular(self.latent_dim, squeeze=True))

        self.n_samples = n_samples
        self.seed = seed
        self.parallel_iterations = parallel_iterations
        self.jitter = jitter

        # Inference-specific attributes (see gpssm_models.py for appropriate choices):
        nans = tf.constant(np.zeros((self.T, self.n_samples, self.latent_dim)) * np.nan, dtype=gps.float_type)
        self.sample_fn = lambda **kwargs: (nans, None)
        self.sample_kwargs = {}
        self.KL_fn = lambda *fs: tf.constant(np.nan, dtype=gps.float_type)

        # GP Transitions:
        self.n_ind_pts = n_ind_pts if Z is None else (Z[0].shape[-2] if isinstance(Z, list) else Z.shape[-2])

        if isinstance(Z, np.ndarray) and Z.ndim == 2:
            self.Z = mf.SharedIndependentMof(gp.features.InducingPoints(Z))
        else:
            Z_list = [np.random.randn(self.n_ind_pts, self.latent_dim + self.input_dim)
                      for _ in range(self.latent_dim)] if Z is None else [z for z in Z]
            self.Z = mf.SeparateIndependentMof([gp.features.InducingPoints(z) for z in Z_list])

        if isinstance(kern, gp.kernels.Kernel):
            self.kern = mk.SharedIndependentMok(kern, self.latent_dim)
        else:
            kern_list = kern or [gp.kernels.Matern32(self.latent_dim + self.input_dim, ARD=True)
                                 for _ in range(self.latent_dim)]
            self.kern = mk.SeparateIndependentMok(kern_list)

        self.mean_fn = mean_fn or mean_fns.Identity(self.latent_dim)
        self.Umu = Param(np.zeros((self.latent_dim, self.n_ind_pts)) if Umu is None else Umu)  # (Lm^-1)(Umu - m(Z))
        LT_transform = gtf.LowerTriangular(self.n_ind_pts, num_matrices=self.latent_dim, squeeze=False)
        self.Ucov_chol = Param(np.tile(np.eye(self.n_ind_pts)[None, ...], [self.latent_dim, 1, 1])
                               if Ucov_chol is None else Ucov_chol, transform=LT_transform)  # (Lm^-1)Lu
        self._Kzz = None

    @property
    def Kzz(self):
        if self._Kzz is None:
            self._Kzz = Kuu(self.Z, self.kern, jitter=self.jitter)  # (latent_dim x) M x M
        return self._Kzz

    @params_as_tensors
    def _build_likelihood(self):
        X_samples, *fs = self._build_sample()
        emissions = self._build_emissions(X_samples)
        KL_X = self._build_KL_X(fs)
        KL_U = self._build_KL_U()
        KL_x1 = self._build_KL_x1()
        return emissions - KL_X - KL_U - KL_x1

    @params_as_tensors
    def _build_sample(self):
        return self.sample_fn(**self.sample_kwargs)

    @params_as_tensors
    def _build_emissions(self, X_samples):
        emissions = self.emissions.logp(X_samples, self.Y[:, None, :])  # T x n_samples
        return tf.reduce_sum(tf.reduce_mean(emissions, -1))

    @params_as_tensors
    def _build_KL_X(self, fs):
        return tf.reduce_sum(self.KL_fn(*fs))

    @params_as_tensors
    def _build_KL_U(self):
        return KL(self.Umu, self.Ucov_chol)

    @params_as_tensors
    def _build_KL_x1(self):
        return KL(self.qx1_mu - self.px1_mu, self.qx1_cov_chol, P_chol=self.px1_cov_chol)

    @params_as_tensors
    def _build_transition_KLs(self, f_mus, f_vars, As=None, bs=None, S_chols=None):
        As = self.As if As is None else As
        bs = self.bs if bs is None else bs
        S_chols = self.S_chols if S_chols is None else S_chols

        const = tf.reduce_sum(tf.log(tf.square(self.Q_sqrt))) - self.latent_dim

        if As.shape.ndims == 2:
            mahalanobis = (As - 1.)[:, None, :] * f_mus
        else:
            mahalanobis = tf.matmul(f_mus, As - tf.eye(self.latent_dim, dtype=gps.float_type),
                                    transpose_b=True)
        mahalanobis += bs[:, None, :]  # (T-1) x n_samples x latent_dim
        mahalanobis = tf.reduce_mean(tf.reduce_sum(tf.square(mahalanobis / self.Q_sqrt), -1), -1)  # T - 1

        mean_f_var = tf.reduce_mean(f_vars, 1)

        if (S_chols.shape.ndims == 2) and (As.shape.ndims == 2):
            trace = tf.square(S_chols) + mean_f_var * tf.square(As - 1.)
        elif As.shape.ndims == 2:
            trace = tf.reduce_sum(tf.square(S_chols), -1) + mean_f_var * tf.square(As - 1.)
        elif S_chols.shape.ndims == 2:
            trace = tf.square(S_chols) + tf.reduce_sum(
                mean_f_var[:, None, :] * tf.square(As - tf.eye(self.latent_dim, dtype=gps.float_type)), -1)
        else:
            trace = tf.reduce_sum(tf.square(S_chols), -1) + tf.reduce_sum(
                mean_f_var[:, None, :] * tf.square(As - tf.eye(self.latent_dim, dtype=gps.float_type)), -1)

        trace = tf.reduce_sum(trace / tf.square(self.Q_sqrt), -1)  # T - 1

        log_det_S = 2. * tf.reduce_sum(tf.log(tf.abs(S_chols if S_chols.shape.ndims == 2
                                                     else tf.matrix_diag_part(S_chols))), -1)  # T - 1

        return 0.5 * (const + mahalanobis + trace - log_det_S)  # T - 1

    @params_as_tensors
    def _build_factorized_transition_KLs(self, f_mus, f_vars, x_cov_chols, As=None, bs=None, S_chols=None):
        As = self.As if As is None else As
        bs = self.bs if bs is None else bs
        S_chols = self.S_chols if S_chols is None else S_chols

        const = tf.reduce_sum(tf.log(tf.square(self.Q_sqrt))) - self.latent_dim

        if As.shape.ndims == 2:
            mahalanobis = (As - 1.)[:, None, :] * f_mus
        else:
            mahalanobis = tf.matmul(f_mus, As - tf.eye(self.latent_dim, dtype=gps.float_type),
                                    transpose_b=True)
        mahalanobis += bs[:, None, :]  # (T-1) x n_samples x latent_dim
        mahalanobis = tf.reduce_mean(tf.reduce_sum(tf.square(mahalanobis / self.Q_sqrt), -1), -1)  # T - 1

        is_diag_xcov_chol = (S_chols.shape.ndims == 2) and (As.shape.ndims == 2)

        if is_diag_xcov_chol:
            trace = f_vars + tf.square(x_cov_chols)
        else:
            trace = f_vars + tf.reduce_sum(tf.square(x_cov_chols), -1)
        trace = tf.reduce_mean(tf.reduce_sum(trace / tf.square(self.Q_sqrt), -1), -1)  # T - 1

        log_det_x_covs = 2. * tf.reduce_mean(tf.reduce_sum(tf.log(tf.abs(
            x_cov_chols if is_diag_xcov_chol else tf.matrix_diag_part(x_cov_chols))), -1), -1)  # T - 1

        return 0.5 * (const + mahalanobis + trace - log_det_x_covs)  # T - 1

    @params_as_tensors
    def _build_transition_KLs_from_samples(self, F_samples, As=None, bs=None, S_chols=None):
        As = self.As if As is None else As
        bs = self.bs if bs is None else bs
        S_chols = self.S_chols if S_chols is None else S_chols

        if As.shape.ndims == 2:
            mu_diff = (As - 1.)[:, None, :] * F_samples
        else:
            mu_diff = tf.matmul(F_samples, As - tf.eye(self.latent_dim, dtype=gps.float_type),
                                transpose_b=True)
        mu_diff += bs[:, None, :]  # (T-1) x n_samples x latent_dim
        return KL_samples(mu_diff, S_chols, P_chol=self.Q_sqrt)

    @params_as_tensors
    def _build_transition_KLs_X_and_F(self, f_mus, f_vars, KL_F, As=None, bs=None, S_chols=None):
        KL_X = self._build_transition_KLs(f_mus, f_vars, As=As, bs=bs, S_chols=S_chols)
        KL_F_avg = tf.reduce_mean(KL_F, -1)
        return KL_X + KL_F_avg  # T - 1

    @params_as_tensors
    def _build_KL_F_joint(self, X, F, U, Lm, inputs=None):
        T = tf.shape(X)[0]
        n_samples = tf.shape(X)[1]
        shared_kern = isinstance(self.kern, mk.SharedIndependentMok)
        shared_kern_and_Z = shared_kern and isinstance(self.Z, mf.SharedIndependentMof)

        X_tr = tf.transpose(X[:-1], [1, 0, 2])  # n_samples x (T-1) x latent_dim
        if inputs is not None:
            X_tr = tf.concat([X_tr, tf.tile(inputs[None, :, :], [n_samples, 1, 1])], -1)
        F_tr = tf.transpose(F, [1, 0, 2])  # n_samples x (T-1) x latent_dim
        U_tr = tf.transpose(U, [2, 0, 1])[..., None]  # n_samples x latent_dim x M x 1

        n_mean_inputs = self.mean_fn.input_dim if hasattr(self.mean_fn, "input_dim") else self.latent_dim
        mean_fn_X = self.mean_fn(tf.reshape(X_tr, [-1, self.latent_dim + self.input_dim])[:, :n_mean_inputs])
        mean_fn_X = tf.reshape(mean_fn_X, [n_samples, T - 1, self.latent_dim])  # n_samples x (T-1) x latent_dim
        K_fn = lambda x: self.kern.kern.K(x) if shared_kern else lambda x: self.kern.K(x, full_output_cov=False)
        Kxx = tf.map_fn(K_fn, X_tr)  # n_samples x (latent_dim x) (T-1) x (T-1)

        # (latent_dim x) M x n_samples*(T-1):
        Kzx = Kuf(self.Z, self.kern, tf.reshape(X_tr, [-1, self.latent_dim + self.input_dim]))
        Kzx_shape = [self.n_ind_pts, n_samples, T - 1]
        Kzx = tf.reshape(Kzx, Kzx_shape if shared_kern_and_Z else [self.latent_dim] + Kzx_shape)
        # n_samples x (latent_dim x) M x (T-1):
        Kzx = tf.transpose(Kzx, [1, 0, 2] if shared_kern_and_Z else [2, 0, 1, 3])

        _Lm = tf.tile(Lm[None, ...], [n_samples, 1, 1] if shared_kern_and_Z else [n_samples, 1, 1, 1])
        LinvK = tf.linalg.triangular_solve(_Lm, Kzx, lower=True)  # n_samples x (latent_dim x) M x (T-1)

        _Kxx = Kxx[:, None, :, :] if (shared_kern and not shared_kern_and_Z) else Kxx
        Cov_p = _Kxx - tf.matmul(LinvK, LinvK, transpose_a=True)  # n_samples (x latent_dim) x (T-1) x (T-1)
        Cov_q = tf.matrix_diag_part(Cov_p)  # n_samples (x latent_dim) x (T-1)

        _LinvK = tf.tile(LinvK[:, None, :, :], [1, self.latent_dim, 1, 1]) if shared_kern_and_Z else LinvK
        # n_samples x (T-1) x latent_dim:
        KinvKu = tf.transpose(tf.matmul(_LinvK, U_tr, transpose_a=True)[..., 0], [0, 2, 1])
        mu_diff = F_tr - mean_fn_X - KinvKu  # n_samples x (T-1) x latent_dim

        if shared_kern_and_Z:
            # n_samples x latent_dim:
            mahalanobis_p = tf.reduce_sum(tf.linalg.solve(Cov_p, mu_diff) * mu_diff, -2)
            mahalanobis_q = tf.reduce_sum(tf.square(mu_diff) / Cov_q[:, :, None], -2)
            Cov_p = Cov_p[:, None, :, :]  # n_samples x 1 x (T-1) x (T-1)
            Cov_q = Cov_q[:, None, :]  # n_samples x 1 x (T-1)
        else:
            # n_samples x latent_dim x (T-1):
            mahalanobis_p = tf.linalg.solve(Cov_p, tf.transpose(mu_diff, [0, 2, 1])[..., None])[..., 0]
            # n_samples x latent_dim:
            mahalanobis_p = tf.einsum('ijk,ikj->ij', mahalanobis_p, mu_diff)
            mahalanobis_q = tf.reduce_sum(tf.square(mu_diff) / tf.transpose(Cov_q, [0, 2, 1]), -2)

        log_p = - 0.5 * (tf.linalg.logdet(Cov_p) + mahalanobis_p)  # n_samples x latent_dim
        log_q = - 0.5 * (tf.reduce_sum(tf.log(tf.abs(Cov_q)), -1) + mahalanobis_q)  # n_samples x latent_dim
        return tf.reduce_sum(log_q - log_p, -1)  # n_samples

    @params_as_tensors
    def _build_linear_time_q_sample(self, return_f_moments=False, return_x_cov_chols=False,
                                    sample_f=False, return_f=False, sample_u=True, return_u=False,
                                    compute_KL_F=False, return_Lm=False,
                                    T=None, inputs=None, qx1_mu=None, qx1_cov_chol=None, x1_samples=None,
                                    As=None, bs=None, S_chols=None, Lm=None):
        T = self.T if T is None else T
        inputs = self.inputs if inputs is None else inputs
        qx1_mu = self.qx1_mu if qx1_mu is None else qx1_mu
        qx1_cov_chol = self.qx1_cov_chol if qx1_cov_chol is None else qx1_cov_chol
        As = self.As if As is None else As
        bs = self.bs if bs is None else bs
        S_chols = self.S_chols if S_chols is None else S_chols
        n_samples = self.n_samples if x1_samples is None else int(x1_samples.shape[0])
        n_mean_inputs = self.mean_fn.input_dim if hasattr(self.mean_fn, "input_dim") else self.latent_dim
        differentiate = x1_samples is None

        Lm = tf.cholesky(self.Kzz) if Lm is None else Lm

        X_samples = tf.TensorArray(size=T, dtype=gps.float_type, clear_after_read=False,
                                   infer_shape=False, element_shape=(n_samples, self.latent_dim))
        if sample_f:
            F_samples = tf.TensorArray(size=T - 1, dtype=gps.float_type, clear_after_read=False,
                                       infer_shape=False, element_shape=(n_samples, self.latent_dim))
        if return_f_moments:
            f_mus = tf.TensorArray(size=T - 1, dtype=gps.float_type, clear_after_read=False,
                                   infer_shape=False, element_shape=(n_samples, self.latent_dim))
            f_vars = tf.TensorArray(size=T - 1, dtype=gps.float_type, clear_after_read=False,
                                    infer_shape=False, element_shape=(n_samples, self.latent_dim))

        is_diag_xcov = (S_chols.shape.ndims == 2) if sample_f else \
            ((S_chols.shape.ndims == 2) and (As.shape.ndims == 2))

        if return_x_cov_chols:
            x_cov_chols = tf.TensorArray(size=T - 1, dtype=gps.float_type, clear_after_read=False,
                                         infer_shape=False, element_shape=
                                         (n_samples, self.latent_dim) if is_diag_xcov
                                         else (n_samples, self.latent_dim, self.latent_dim))
        if sample_u:
            U_samples = tfd.MultivariateNormalDiag(loc=tf.zeros(
                (self.latent_dim, self.n_ind_pts, n_samples), dtype=gps.float_type))
            U_samples = U_samples.sample(seed=self.seed)
            U_samples = self.Umu[:, :, None] + tf.matmul(self.Ucov_chol, U_samples)

        white_samples = tfd.MultivariateNormalDiag(loc=tf.zeros(
            (n_samples, self.latent_dim), dtype=gps.float_type))

        white_samples_X = white_samples.sample(T, seed=self.seed)
        if x1_samples is not None:
            X_samples = X_samples.write(0, x1_samples)
        else:
            if qx1_cov_chol.shape.ndims == 1:
                x1_noise = white_samples_X[0] * qx1_cov_chol
            else:
                x1_noise = tf.matmul(white_samples_X[0], qx1_cov_chol, transpose_b=True)
            X_samples = X_samples.write(0, qx1_mu + x1_noise)

        if sample_f: white_samples_F = white_samples.sample(T - 1, seed=self.seed)

        shared_kern = isinstance(self.kern, mk.SharedIndependentMok)
        shared_kern_and_Z = Lm.shape.ndims == 2 or (shared_kern and isinstance(self.Z, mf.SharedIndependentMof))
        if compute_KL_F:
            assert sample_f and sample_u
            Unwhitened_U = tf.matmul(tf.tile(Lm[None, ...], [self.latent_dim, 1, 1]) if shared_kern_and_Z else Lm,
                                     U_samples)
            U_un_tr = tf.transpose(Unwhitened_U, [2, 0, 1])  # n_samples x latent_dim x M
            KL_F = tf.TensorArray(size=T - 1, dtype=gps.float_type, clear_after_read=False,
                                  infer_shape=False, element_shape=(n_samples))

        def _loop_body(*args):
            t, X = args[:2]
            if sample_f: F = args[2]
            if return_f_moments: f_mus, f_vars = args[3:5] if sample_f else args[2:4]
            if return_x_cov_chols: x_cov_chols = args[-2] if compute_KL_F else args[-1]
            if compute_KL_F: KL_F = args[-1]

            x_t = X.read(t)  # n_samples x latent_dim
            if inputs is not None:
                x_t = tf.concat([x_t, tf.tile(inputs[t][None, :], [n_samples, 1])], -1)

            if sample_u:
                f_mu, f_var = conditional(x_t, self.Z, self.kern, U_samples, q_sqrt=None, white=True, Lm=Lm)
            else:
                f_mu, f_var = conditional(x_t, self.Z, self.kern, self.Umu, q_sqrt=self.Ucov_chol, white=True, Lm=Lm)
            f_mu += self.mean_fn(x_t[:, :n_mean_inputs])
            f_var = tf.abs(f_var)

            if sample_f:
                f_t = f_mu + tf.sqrt(f_var) * white_samples_F[t]  # n_samples x latent_dim
                F = F.write(t, f_t)
                f_mu_or_t = f_t
                tiling = [n_samples, 1] if is_diag_xcov else [n_samples, 1, 1]
                x_cov_chol = tf.tile(S_chols[t][None, ...], tiling)
            else:
                f_mu_or_t = f_mu
                if is_diag_xcov:
                    x_cov_chol = tf.sqrt(tf.square(S_chols[t]) + f_var * tf.square(As[t]))  # (n_samples x latent_dim)
                elif As.shape.ndims == 2:
                    x_cov_chol = tf.matmul(S_chols[t], S_chols[t], transpose_b=True)
                    x_cov_chol += tf.matrix_diag(f_var * tf.square(As[t]))
                    x_cov_chol = tf.cholesky(x_cov_chol)  # (n_samples x latent_dim x latent_dim)
                elif S_chols.shape.ndims == 2:
                    x_cov_chol = tf.diag(tf.square(S_chols[t]))
                    x_cov_chol += tf.tensordot(f_var[:, None, :] * As[t], As[t], axes=[[2], [1]])
                    x_cov_chol = tf.cholesky(x_cov_chol)  # (n_samples x latent_dim x latent_dim)
                else:
                    x_cov_chol = tf.matmul(S_chols[t], S_chols[t], transpose_b=True)
                    x_cov_chol += tf.tensordot(f_var[:, None, :] * As[t], As[t], axes=[[2], [1]])
                    x_cov_chol = tf.cholesky(x_cov_chol)  # (n_samples x latent_dim x latent_dim)

            x_tplus1 = bs[t] + ((As[t] * f_mu_or_t) if As.shape.ndims == 2
                                else tf.matmul(f_mu_or_t, As[t], transpose_b=True))  # n_samples x latent_dim
            x_tplus1 += (white_samples_X[t + 1] * x_cov_chol) if is_diag_xcov \
                else tf.reduce_sum(x_cov_chol * white_samples_X[t + 1][:, None, :], -1)
            X = X.write(t + 1, x_tplus1)

            if return_f_moments:
                f_mus, f_vars = f_mus.write(t, f_mu), f_vars.write(t, f_var)
            if return_x_cov_chols:
                x_cov_chols = x_cov_chols.write(t, x_cov_chol)
            if compute_KL_F:
                X_to_tp1 = tf.transpose(X.gather(tf.range(t + 1)), [1, 0, 2])  # n_samples x (t+1) x latent_dim
                if inputs is not None:
                    X_to_tp1 = tf.concat([X_to_tp1, tf.tile(inputs[:t + 1][None, :, :], [n_samples, 1, 1])], -1)
                F_to_t = tf.transpose(F.gather(tf.range(t)), [1, 2, 0])  # n_samples x latent_dim x t
                mean_fn_X = self.mean_fn(
                    tf.reshape(X_to_tp1, [-1, self.latent_dim + self.input_dim])[:, :n_mean_inputs])
                mean_fn_X = tf.reshape(mean_fn_X, [n_samples, t + 1, self.latent_dim])  # n_samples x (t+1) x latent_dim
                mean_fn_X_to_t = tf.transpose(mean_fn_X[:, :-1], [0, 2, 1])  # n_samples x latent_dim x t
                mean_fn_x_t = mean_fn_X[:, -1]  # n_samples x latent_dim

                K_fn = lambda x: self.kern.kern.K(x) if shared_kern else lambda x: self.kern.K(x, full_output_cov=False)
                Kxx = tf.map_fn(K_fn, X_to_tp1)  # n_samples x (latent_dim x) (t+1) x (t+1)

                # (latent_dim x) M x n_samples*(t+1):
                Kzx = Kuf(self.Z, self.kern, tf.reshape(X_to_tp1, [-1, self.latent_dim + self.input_dim]))
                Kzx_shape = [self.n_ind_pts, n_samples, t + 1]
                Kzx = tf.reshape(Kzx, Kzx_shape if shared_kern_and_Z else [self.latent_dim] + Kzx_shape)
                # n_samples x (latent_dim x) M x (t+1):
                Kzx = tf.transpose(Kzx, [1, 0, 2] if shared_kern_and_Z else [2, 0, 1, 3])

                _Kxx = tf.tile(Kxx[:, None, :, :], [1, self.latent_dim, 1, 1]) \
                    if (shared_kern and not shared_kern_and_Z) else Kxx

                _Kzz = tf.tile(self.Kzz[None, ...], [n_samples, 1, 1] if shared_kern_and_Z else [n_samples, 1, 1, 1])
                Kzx_dperm = [0, 2, 1] if shared_kern_and_Z else [0, 1, 3, 2]
                Kxz_xz = tf.concat([
                    tf.concat([_Kxx[..., :-1, :-1], tf.transpose(Kzx[..., :, :-1], Kzx_dperm)], -1),
                    tf.concat([Kzx[..., :, :-1], _Kzz], -1)], -2)  # n_samples x (latent_dim x) (t + M) x (t + M)
                # n_samples x (latent_dim x) (t + M) x 1:
                Kxz_xtp1 = tf.concat([_Kxx[..., :-1, -1:], Kzx[..., :, -1:]], -2)
                Kxtp1_xtp1 = Kxx[..., -1, -1]  # n_samples (x latent_dim)

                KinvK_p = tf.linalg.solve(Kxz_xz, Kxz_xtp1)[..., 0]  # n_samples x (latent_dim x) (t + M)
                var_p = tf.abs(Kxtp1_xtp1 - tf.reduce_sum(KinvK_p * Kxz_xtp1[..., 0], -1))  # n_samples (x latent_dim)
                F_and_U = tf.concat([F_to_t - mean_fn_X_to_t, U_un_tr], -1) # n_samples x latent_dim x (t + M)
                mu_p = tf.reduce_sum(F_and_U * (KinvK_p[:, None, :] if shared_kern_and_Z else KinvK_p), -1)
                mu_p += mean_fn_x_t  # n_samples x latent_dim
                KinvK_q = tf.linalg.solve(_Kzz, Kzx[..., :, -1:])[..., 0]  # n_samples x (latent_dim x) M
                var_q = tf.abs(Kxtp1_xtp1 - tf.reduce_sum(KinvK_q * Kzx[..., :, -1], -1))  # n_samples (x latent_dim)
                mu_q = tf.reduce_sum(U_un_tr * (KinvK_q[:, None, :] if shared_kern_and_Z else KinvK_q), -1)
                mu_q += mean_fn_x_t  # n_samples x latent_dim
                if shared_kern_and_Z:
                    var_p = var_p[:, None]  # n_samples x 1
                    var_q = var_q[:, None]  # n_samples x 1
                log_p = - 0.5 * (tf.log(var_p) + tf.square(f_t - mu_p) / var_p)  # n_samples x latent_dim
                log_q = - 0.5 * (tf.log(var_q) + tf.square(f_t - mu_q) / var_q)  # n_samples x latent_dim
                KL_f_t = tf.reduce_sum(log_q - log_p, -1)  # n_samples
                KL_F = KL_F.write(t, KL_f_t)

            ret_values = [t + 1, X]
            if sample_f: ret_values += [F]
            if return_f_moments: ret_values += [f_mus, f_vars]
            if return_x_cov_chols: ret_values += [x_cov_chols]
            if compute_KL_F: ret_values += [KL_F]
            return ret_values

        _loop_vars = [0, X_samples]
        if sample_f: _loop_vars += [F_samples]
        if return_f_moments: _loop_vars += [f_mus, f_vars]
        if return_x_cov_chols: _loop_vars += [x_cov_chols]
        if compute_KL_F: _loop_vars += [KL_F]

        result = tf.while_loop(
            cond=lambda t, *args: t < (T - 1),
            body=_loop_body,
            loop_vars=_loop_vars,
            back_prop=differentiate,
            parallel_iterations=self.parallel_iterations)

        ret_values = tuple(r.stack() for r in result[1:])
        if sample_f and not return_f: ret_values = ret_values[:1] + ret_values[2:]
        if sample_u and return_u: ret_values += (U_samples,)
        if return_Lm: ret_values += (Lm,)
        return ret_values

    @params_as_tensors
    def _build_cubic_time_q_sample(self, return_f_moments=False, return_f=True,
                                   sample_u=False, return_u=False, add_jitter=True, inverse_chol=False,
                                   T=None, inputs=None, qx1_mu=None, qx1_cov_chol=None, x1_samples=None,
                                   As=None, bs=None, S_chols=None, Lm=None):
        T = self.T if T is None else T
        inputs = self.inputs if inputs is None else inputs
        if inputs is not None:
            inputs = tf.concat([inputs, tf.zeros((1, tf.shape(inputs)[-1]), dtype=gps.float_type)], 0)
        qx1_mu = self.qx1_mu if qx1_mu is None else qx1_mu
        qx1_cov_chol = self.qx1_cov_chol if qx1_cov_chol is None else qx1_cov_chol
        As = self.As if As is None else As
        bs = self.bs if bs is None else bs
        S_chols = self.S_chols if S_chols is None else S_chols
        n_samples = self.n_samples if x1_samples is None else int(x1_samples.shape[0])
        n_mean_inputs = self.mean_fn.input_dim if hasattr(self.mean_fn, "input_dim") else self.latent_dim
        differentiate = x1_samples is None

        Lm = tf.cholesky(self.Kzz) if Lm is None else Lm
        shared_kern = isinstance(self.kern, mk.SharedIndependentMok)
        shared_kern_and_Z = Lm.shape.ndims == 2 or (shared_kern and isinstance(self.Z, mf.SharedIndependentMof))

        if sample_u:
            U_samples = tfd.MultivariateNormalDiag(loc=tf.zeros(
                (self.latent_dim, self.n_ind_pts, n_samples), dtype=gps.float_type))
            U_samples = U_samples.sample(seed=self.seed)
            U_samples = self.Umu[:, :, None] + tf.matmul(self.Ucov_chol, U_samples)

        white_samples_X = tfd.MultivariateNormalDiag(
            loc=tf.zeros((n_samples, T, self.latent_dim), dtype=gps.float_type)).sample(seed=self.seed)

        white_samples_F = tfd.MultivariateNormalDiag(
            loc=tf.zeros((n_samples, T - 1, self.latent_dim), dtype=gps.float_type)).sample(seed=self.seed)
        white_samples_F = tf.transpose(white_samples_F, [0, 2, 1])

        if x1_samples is None:
            x1_samples = qx1_mu + ((white_samples_X[:, 0] * qx1_cov_chol) if qx1_cov_chol.shape.ndims == 1
                                   else tf.matmul(white_samples_X[:, 0], qx1_cov_chol, transpose_b=True))

        if inputs is not None:
            x1_samples = tf.concat([x1_samples, tf.tile(inputs[:1], [n_samples, 1])], -1)

        if sample_u:
            f1_mu, f1_var = conditional(x1_samples, self.Z, self.kern, U_samples,
                                        q_sqrt=None, white=True, Lm=Lm)
        else:
            f1_mu, f1_var = conditional(x1_samples, self.Z, self.kern, self.Umu,
                                        q_sqrt=self.Ucov_chol, white=True, Lm=Lm)

        f1_mu += self.mean_fn(x1_samples[:, :n_mean_inputs])
        f1_var = tf.abs(f1_var)
        f1_samples = f1_mu + tf.sqrt(f1_var) * white_samples_F[:, :, 0]

        if sample_u: U_samples = tf.transpose(U_samples, [2, 0, 1])  # n_samples x latent_dim x M

        def single_trajectory(args):
            if sample_u:
                U_samples_n = args[-1]
                args = args[:-1]
            x1_samples_n, f1_samples_n, white_samples_X_n, white_samples_F_n, f1_mu_n, f1_var_n = args

            x2_samples_n = bs[0] + ((As[0] * f1_samples_n) if As.shape.ndims == 2
                                    else tf.reduce_sum(As[0] * f1_samples_n, -1))  # latent_dim
            x2_samples_n += (white_samples_X_n[1] * S_chols[0]) if S_chols.shape.ndims == 2 \
                else tf.reduce_sum(S_chols[0] * white_samples_X_n[1], -1)
            if inputs is not None:
                x2_samples_n = tf.concat([x2_samples_n, inputs[1]], 0)
            X_samples_n = tf.stack([x1_samples_n, x2_samples_n], 0)  # 2 x latent_dim

            F_samples_n = tf.TensorArray(size=T - 1, dtype=gps.float_type, clear_after_read=False,
                                         infer_shape=False, element_shape=(self.latent_dim,))
            F_samples_n = F_samples_n.write(0, f1_samples_n)

            if return_f_moments:
                f_mus = tf.TensorArray(size=T - 1, dtype=gps.float_type, clear_after_read=False,
                                       infer_shape=False, element_shape=(self.latent_dim,))
                f_vars = tf.TensorArray(size=T - 1, dtype=gps.float_type, clear_after_read=False,
                                        infer_shape=False, element_shape=(self.latent_dim,))
                f_mus = f_mus.write(0, f1_mu_n)
                f_vars = f_vars.write(0, f1_var_n)

            Kzx = Kuf(self.Z, self.kern, X_samples_n[:1])  # (latent_dim x) M x 1
            Lm_inv_Kzx = tf.matrix_triangular_solve(Lm, Kzx, lower=True)  # (latent_dim x) M x 1

            F_cov_chol = f1_var_n[0] if (shared_kern_and_Z and sample_u) else f1_var_n  # () or latent_dim
            F_cov_chol = tf.sqrt((F_cov_chol + self.jitter) if add_jitter else F_cov_chol)
            if inverse_chol: F_cov_chol = 1. / F_cov_chol
            F_cov_chol = F_cov_chol[..., None, None]  # (latent_dim x) 1 x 1

            def _loop_body(t, X, F, Lm_inv_Kzx, F_cov_chol, f_mus=None, f_vars=None):
                if shared_kern:
                    Kx1_to_tp1_xtp1 = self.kern.kern.K(X, X[-1:])[..., 0]  # t+1
                else:
                    Kx1_to_tp1_xtp1 = self.kern.K(X, X[-1:], full_output_cov=False)[..., 0]  # latent_dim x (t+1)

                Kzxtp1 = Kuf(self.Z, self.kern, X[-1][None, :])  # (latent_dim x) M x 1
                Lm_inv_Kzxtp1 = tf.matrix_triangular_solve(Lm, Kzxtp1, lower=True)[..., 0]  # (latent_dim x) M

                f_tp1_marg_mu = self.mean_fn(X[-1:, :n_mean_inputs])[0]  # () or latent_dim
                if sample_u:
                    f_tp1_marg_mu += tf.reduce_sum(Lm_inv_Kzxtp1 * U_samples_n, -1)  # latent_dim
                else:
                    f_tp1_marg_mu += tf.reduce_sum(Lm_inv_Kzxtp1 * self.Umu, -1)  # latent_dim

                F_cov_tp1 = Kx1_to_tp1_xtp1[..., -1] - tf.reduce_sum(tf.square(Lm_inv_Kzxtp1), -1)  # () or latent_dim

                F_cov_1_to_t_tp1 = Kx1_to_tp1_xtp1[..., :-1]  # (latent_dim x) t
                F_cov_1_to_t_tp1 -= tf.reduce_sum(Lm_inv_Kzx * Lm_inv_Kzxtp1[..., None], -2)  # (latent_dim x) t

                if not sample_u:
                    Uchol_Lm_inv_Kzxtp1 = tf.reduce_sum(self.Ucov_chol * Lm_inv_Kzxtp1[..., None], -2)  # latent_dim x M
                    F_cov_tp1 += tf.reduce_sum(tf.square(Uchol_Lm_inv_Kzxtp1), -1)  # latent_dim
                    # latent_dim x M:
                    Ucov_Lm_inv_Kzxtp1 = tf.reduce_sum(self.Ucov_chol * Uchol_Lm_inv_Kzxtp1[:, None, :], -1)
                    F_cov_1_to_t_tp1 += tf.reduce_sum(Lm_inv_Kzx * Ucov_Lm_inv_Kzxtp1[:, :, None], -2)  # latent_dim x t

                if inverse_chol:
                    F_chol_inv_F_1_to_t_tp1 = tf.reduce_sum(
                        F_cov_chol * F_cov_1_to_t_tp1[..., None, :], -1)  # (latent_dim x) t
                else:
                    F_chol_inv_F_1_to_t_tp1 = tf.matrix_triangular_solve(
                        F_cov_chol, F_cov_1_to_t_tp1[..., None], lower=True)[..., 0]  # (latent_dim x) t
                # latent_dim:
                f_tp1_mu = f_tp1_marg_mu + tf.reduce_sum(F_chol_inv_F_1_to_t_tp1 * white_samples_F_n[:, :t], -1)

                f_tp1_var = F_cov_tp1 - tf.reduce_sum(tf.square(F_chol_inv_F_1_to_t_tp1), -1)  # () or latent_dim
                f_tp1_var = tf.abs(f_tp1_var)

                f_tp1 = f_tp1_mu + tf.sqrt(f_tp1_var) * white_samples_F_n[:, t]  # latent_dim

                x_tplus2 = bs[t] + ((As[t] * f_tp1) if As.shape.ndims == 2
                                    else tf.reduce_sum(As[t] * f_tp1, -1))  # latent_dim
                x_tplus2 += (S_chols[t] * white_samples_X_n[t + 1]) if S_chols.shape.ndims == 2 \
                    else tf.reduce_sum(S_chols[t] * white_samples_X_n[t + 1], -1)  # latent_dim

                if inputs is not None:
                    x_tplus2 = tf.concat([x_tplus2, inputs[t + 1]], 0)

                X = tf.concat([X, x_tplus2[None, :]], 0)  # (t+2) x latent_dim
                F = F.write(t, f_tp1)

                Lm_inv_Kzx = tf.concat([Lm_inv_Kzx, Lm_inv_Kzxtp1[..., None]], -1)  # (latent_dim x) M x (t+1)

                F_cov_chol_diag = tf.sqrt((f_tp1_var + self.jitter) if add_jitter else f_tp1_var)  # () or latent_dim
                if inverse_chol:
                    F_cov_chol_bottom_offdiag = - tf.reduce_sum(F_chol_inv_F_1_to_t_tp1[..., None] * F_cov_chol, -2)
                    F_cov_chol_bottom_offdiag /= F_cov_chol_diag[..., None]
                    F_cov_chol_diag = 1. / F_cov_chol_diag
                    F_cov_chol_bottom_row = tf.concat([F_cov_chol_bottom_offdiag, F_cov_chol_diag[..., None]], -1)
                else:
                    F_cov_chol_bottom_row = tf.concat([F_chol_inv_F_1_to_t_tp1, F_cov_chol_diag[..., None]], -1)

                padding = [[0, 0], [0, 1]] if (shared_kern_and_Z and sample_u) else [[0, 0], [0, 0], [0, 1]]
                F_cov_chol = tf.pad(F_cov_chol, paddings=padding)  # (latent_dim x) t x (t+1)
                # (latent_dim x) (t+1) x (t+1):
                F_cov_chol = tf.concat([F_cov_chol, F_cov_chol_bottom_row[..., None, :]], -2)

                ret_values = (t + 1, X, F, Lm_inv_Kzx, F_cov_chol)
                if return_f_moments:
                    if shared_kern_and_Z and sample_u:
                        f_tp1_var = tf.tile(f_tp1_var[None], [self.latent_dim])  # latent_dim
                    f_mus, f_vars = f_mus.write(t, f_tp1_mu), f_vars.write(t, f_tp1_var)
                    ret_values += (f_mus, f_vars)
                return ret_values

            _loop_vars = [1, X_samples_n, F_samples_n, Lm_inv_Kzx, F_cov_chol]

            shape_invar_Lm_inv_Kzx = tf.TensorShape([self.n_ind_pts, None]) if shared_kern_and_Z \
                else tf.TensorShape([self.latent_dim, self.n_ind_pts, None])
            shape_invar_F_cov_chol = tf.TensorShape([None, None]) if (shared_kern_and_Z and sample_u) \
                else tf.TensorShape([self.latent_dim, None, None])
            _shape_invariants = [tf.TensorShape([]),
                                 tf.TensorShape([None, self.latent_dim + self.input_dim]),
                                 tf.TensorShape(None),
                                 shape_invar_Lm_inv_Kzx,
                                 shape_invar_F_cov_chol]

            if return_f_moments:
                _loop_vars += [f_mus, f_vars]
                _shape_invariants += [tf.TensorShape(None), tf.TensorShape(None)]

            loop_result = tf.while_loop(
                cond=lambda t, *args: t < (T - 1),
                body=_loop_body,
                loop_vars=_loop_vars,
                shape_invariants=_shape_invariants,
                back_prop=differentiate,
                parallel_iterations=self.parallel_iterations)

            loop_result = loop_result[1:]
            X_traj = loop_result[0]
            if inputs is not None: X_traj = X_traj[:, :self.latent_dim]
            F_traj = loop_result[1].stack()
            if return_f_moments:
                return X_traj, F_traj, loop_result[-2].stack(), loop_result[-1].stack()
            return X_traj, F_traj

        iterables = (x1_samples, f1_samples, white_samples_X, white_samples_F, f1_mu, f1_var)
        if sample_u: iterables += (U_samples,)
        map_fn_result = tf.map_fn(single_trajectory,
                                  iterables,
                                  (gps.float_type,) * (4 if return_f_moments else 2),
                                  back_prop=differentiate,
                                  parallel_iterations=self.parallel_iterations)

        X_samples = tf.transpose(map_fn_result[0], [1, 0, 2])  # T x n_samples x latent_dim
        ret_values = (X_samples,)
        if return_f:
            F_samples = tf.transpose(map_fn_result[1], [1, 0, 2])  # (T-1) x n_samples x latent_dim
            ret_values += (F_samples,)
        if return_f_moments:
            f_mus = tf.transpose(map_fn_result[2], [1, 0, 2])  # (T-1) x n_samples x latent_dim
            f_vars = tf.transpose(map_fn_result[3], [1, 0, 2])  # (T-1) x n_samples x latent_dim
            ret_values += (f_mus, f_vars)
        if sample_u and return_u: ret_values += (tf.transpose(U_samples, [1, 2, 0]),)  # latent_dim x M x n_samples
        return ret_values

    @params_as_tensors
    def _build_predict_f(self, X):
        f_mu, f_var = conditional(X, self.Z, self.kern, self.Umu, q_sqrt=self.Ucov_chol, white=True)
        n_mean_inputs = self.mean_fn.input_dim if hasattr(self.mean_fn, "input_dim") else self.latent_dim
        f_mu += self.mean_fn(X[:, :n_mean_inputs])
        return f_mu, f_var

    def sample(self, T, N=1, x0_samples=None, inputs=None, cubic=True,
               sample_u=False, sample_f=False, return_op=False):
        if x0_samples is None:
            assert len(self.px1_mu.shape) == 1
            noise = tf.random_normal((N, self.latent_dim), dtype=gps.float_type, seed=self.seed)
            if self.px1_cov_chol is not None:
                if len(self.px1_cov_chol.shape) == 1:
                    noise = noise * self.px1_cov_chol.constrained_tensor
                else:
                    noise = tf.matmul(noise, self.px1_cov_chol.constrained_tensor, transpose_b=True)
            x0_samples = self.px1_mu.constrained_tensor + noise
            if inputs is not None:
                inputs = tf.constant(inputs)
            elif self.inputs is not None:
                inputs = self.inputs.constrained_tensor
        else:
            x0_samples = tf.constant(x0_samples)
            inputs = None if inputs is None else tf.constant(inputs)
            T += 1

        sample_fn = self._build_cubic_time_q_sample if cubic else \
            lambda **kwargs: self._build_linear_time_q_sample(sample_f=sample_f, **kwargs)

        X_samples, *fs = sample_fn(T=T, sample_u=sample_u,
                                   inputs=inputs,
                                   x1_samples=x0_samples,
                                   As=tf.ones((T - 1, self.latent_dim), dtype=gps.float_type),
                                   bs=tf.zeros((T - 1, self.latent_dim), dtype=gps.float_type),
                                   S_chols=self.Q_sqrt.constrained_tensor *
                                           tf.ones((T - 1, self.latent_dim), dtype=gps.float_type))

        if return_op:
            return X_samples
        else:
            session = self.enquire_session()
            X_samples = session.run(X_samples)
            Y_samples = self.emissions.sample_conditional(X_samples)
            return X_samples, Y_samples

    def assign(self, dct, **kwargs):
        if isinstance(dct, pd.Series):
            dct = dct.to_dict()
        for k in list(dct.keys()):
            new_key = '/'.join([self.name] + k.split('/')[1:])
            dct[new_key] = dct.pop(k)
        super().assign(dct, **kwargs)
