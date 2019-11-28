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
from gpflow import Param, params_as_tensors
from gpflow import settings as gps
from gpflow import mean_functions as mean_fns
from gpflow.conditionals import conditional, Kuu
from gpflow import transforms as gtf
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf
from gpflow.logdensities import mvn_logp, diag_mvn_logp
from .KL import KL


class BaseGaussianTransitions(gp.Parameterized):
    def __init__(self, dim, input_dim=0, Q=None, name=None):
        super().__init__(name=name)
        self.OBSERVATIONS_AS_INPUT = False
        self.dim = dim
        self.input_dim = input_dim
        if Q is None or Q.ndim == 2:
            self.Qchol = Param(np.eye(self.dim) if Q is None else np.linalg.cholesky(Q),
                               gtf.LowerTriangular(self.dim, squeeze=True))
        elif Q.ndim == 1:
            self.Qchol = Param(Q ** 0.5)

    @params_as_tensors
    def conditional_mean(self, X, inputs=None):
        raise NotImplementedError

    @params_as_tensors
    def conditional_variance(self, X, inputs=None):
        if self.Qchol.shape.ndims == 2:
            Q = tf.matmul(self.Qchol, self.Qchol, transpose_b=True)
            tile_Q = [1, 1]
        else:
            Q = tf.square(self.Qchol)
            tile_Q = [1]
        if X.shape.ndims == 3:
            return tf.tile(Q[None, None, ...], [tf.shape(X)[0], tf.shape(X)[1], *tile_Q])
        else:
            return tf.tile(Q[None, ...], [tf.shape(X)[0], *tile_Q])

    @params_as_tensors
    def conditional(self, X, inputs=None):
        return self.conditional_mean(X, inputs=inputs), self.conditional_variance(X, inputs=inputs)

    @params_as_tensors
    def logp(self, X, inputs=None):
        d = X[..., 1:, :] - self.conditional_mean(X[..., :-1, :], inputs=inputs)
        if self.Qchol.shape.ndims == 2:
            dim_perm = [2, 0, 1] if X.shape.ndims == 3 else [1, 0]
            return mvn_logp(tf.transpose(d, dim_perm), self.Qchol)
        elif self.Qchol.shape.ndims == 1:
            return diag_mvn_logp(d, self.Qchol)

    def sample_conditional(self, N):
        session = self.enquire_session()
        x_tf = tf.placeholder(gp.settings.float_type, shape=[N, self.dim])
        input_tf = None if self.input_dim == 0 else tf.placeholder(gp.settings.float_type,
                                                                   shape=[1, self.input_dim])
        mu_op = self.conditional_mean(x_tf, inputs=input_tf)
        Qchol = self.Qchol.value.copy()

        def sample_conditional_fn(x, input=None):
            feed_dict = {x_tf: x}
            if input is not None: feed_dict[input_tf] = input[None, :]
            mu = session.run(mu_op, feed_dict=feed_dict)
            if Qchol.ndim == 1:
                noise_samples = np.random.randn(*x.shape) * Qchol
            else:
                noise_samples = np.random.randn(*x.shape) @ Qchol.T
            return mu + noise_samples

        return sample_conditional_fn

    @params_as_tensors
    def variational_expectations(self, Xmu, Xcov, inputs=None):
        raise NotImplementedError


class GPTransitions(gp.Parameterized):
    def __init__(self, dim, input_dim=0, kern=None, Z=None, n_ind_pts=100,
                 mean_fn=None, Q_diag=None, Umu=None, Ucov_chol=None,
                 jitter=gps.numerics.jitter_level, name=None):
        super().__init__(name=name)
        self.OBSERVATIONS_AS_INPUT = False
        self.dim = dim
        self.input_dim = input_dim
        self.jitter = jitter

        self.Q_sqrt = Param(np.ones(self.dim) if Q_diag is None else Q_diag ** 0.5, transform=gtf.positive)

        self.n_ind_pts = n_ind_pts if Z is None else (Z[0].shape[-2] if isinstance(Z, list) else Z.shape[-2])

        if isinstance(Z, np.ndarray) and Z.ndim == 2:
            self.Z = mf.SharedIndependentMof(gp.features.InducingPoints(Z))
        else:
            Z_list = [np.random.randn(self.n_ind_pts, self.dim + self.input_dim)
                      for _ in range(self.dim)] if Z is None else [z for z in Z]
            self.Z = mf.SeparateIndependentMof([gp.features.InducingPoints(z) for z in Z_list])

        if isinstance(kern, gp.kernels.Kernel):
            self.kern = mk.SharedIndependentMok(kern, self.dim)
        else:
            kern_list = kern or [gp.kernels.Matern32(self.dim + self.input_dim, ARD=True) for _ in range(self.dim)]
            self.kern = mk.SeparateIndependentMok(kern_list)

        self.mean_fn = mean_fn or mean_fns.Identity(self.dim)
        self.Umu = Param(np.zeros((self.dim, self.n_ind_pts)) if Umu is None else Umu)  # Lm^-1(Umu - m(Z))
        transform = gtf.LowerTriangular(self.n_ind_pts, num_matrices=self.dim, squeeze=False)
        self.Ucov_chol = Param(np.tile(np.eye(self.n_ind_pts)[None, ...], [self.dim, 1, 1])
                               if Ucov_chol is None else Ucov_chol, transform=transform)  # Lm^-1(Ucov_chol)
        self._Kzz = None

    @property
    def Kzz(self):
        if self._Kzz is None:
            self._Kzz = Kuu(self.Z, self.kern, jitter=self.jitter)  # (E x) x M x M
        return self._Kzz

    @params_as_tensors
    def conditional_mean(self, X, inputs=None, Lm=None):
        return self.conditional(X, inputs=inputs, add_noise=False, Lm=Lm)[0]

    @params_as_tensors
    def conditional_variance(self, X, inputs=None, add_noise=True, Lm=None):
        return self.conditional(X, inputs=inputs, add_noise=add_noise, Lm=Lm)[1]

    @params_as_tensors
    def conditional(self, X, inputs=None, add_noise=True, Lm=None):
        N = tf.shape(X)[0]
        if X.shape.ndims == 3:
            X_in = X if inputs is None else tf.concat([X, tf.tile(inputs[None, :, :], [N, 1, 1])], -1)
            X_in = tf.reshape(X_in, [-1, self.dim + self.input_dim])
        else:
            X_in = X if inputs is None else tf.concat([X, tf.tile(inputs[None, :], [N, 1])], -1)
        mu, var = conditional(X_in, self.Z, self.kern, self.Umu, q_sqrt=self.Ucov_chol, white=True, Lm=Lm)
        n_mean_inputs = self.mean_fn.input_dim if hasattr(self.mean_fn, "input_dim") else self.dim
        mu += self.mean_fn(X_in[:, :n_mean_inputs])

        if X.shape.ndims == 3:
            T = tf.shape(X)[1]
            mu = tf.reshape(mu, [N, T, self.dim])
            var = tf.reshape(var, [N, T, self.dim])

        if add_noise:
            var += self.Q_sqrt ** 2.
        return mu, var

    @params_as_tensors
    def logp(self, X, inputs=None, subtract_KL_U=True):
        T = tf.shape(X)[-2]
        mu, var = self.conditional(X[..., :-1, :], inputs=inputs, add_noise=False)  # N x (T-1) x E or (T-1) x E
        logp = diag_mvn_logp(X[..., 1:, :] - mu, self.Q_sqrt)
        trace = tf.reduce_sum(var / tf.square(self.Q_sqrt), -1)
        ret_value = logp - 0.5 * trace
        if subtract_KL_U:
            KL_U = KL(self.Umu, self.Ucov_chol) / tf.cast(T - 1, X.dtype)
            ret_value -= KL_U
        return ret_value

    def sample_conditional(self, N):
        session = self.enquire_session()
        Lm = tf.constant(session.run(tf.cholesky(self.Kzz)))
        x_tf = tf.placeholder(gp.settings.float_type, shape=[N, self.dim])
        input_tf = None if self.input_dim == 0 else tf.placeholder(gp.settings.float_type,
                                                                   shape=[self.input_dim])
        mu_op, var_op = self.conditional(x_tf, inputs=input_tf, add_noise=True, Lm=Lm)

        def sample_conditional_fn(x, input=None):
            feed_dict = {x_tf: x}
            if input is not None: feed_dict[input_tf] = input
            mu, var = session.run([mu_op, var_op], feed_dict=feed_dict)
            return mu + np.sqrt(var) * np.random.randn(*x.shape)
        return sample_conditional_fn

    @params_as_tensors
    def variational_expectations(self, Xmu, Xcov, inputs=None):
        raise NotImplementedError


class QuadraticPeriodicTransitions(BaseGaussianTransitions):
    def __init__(self, dim, input_dim=None, A=None, B=None, C=None, D=None, Q=None, name=None):
        _input_dim = input_dim or dim
        _Q = np.eye(dim) * np.sqrt(10.) if Q is None else Q
        super().__init__(dim=dim, input_dim=_input_dim, Q=_Q, name=name)
        self.A = Param(np.eye(self.dim) * 0.5 if A is None else A)
        self.B = Param(np.eye(self.dim) * 25. if B is None else B)
        self.C = Param(np.eye(self.dim) * 8.0 if C is None else C)
        self.D = Param(np.eye(self.dim, self.input_dim) * 1.2 if D is None else D)

    @params_as_tensors
    def conditional_mean(self, X, inputs):
        if X.shape.ndims == 3:
            _X = tf.reshape(X, [-1, tf.shape(X)[-1]])  # (n_samples*(T-1))xD
            Xmu = tf.matmul(_X, self.A, transpose_b=True) + \
                  tf.matmul(_X, self.B, transpose_b=True) / (1. + tf.square(_X))
            Xmu = tf.reshape(Xmu, tf.shape(X))
        else:
            Xmu = tf.matmul(X, self.A, transpose_b=True) + \
                  tf.matmul(X, self.B, transpose_b=True) / (1. + tf.square(X))
        Xmu += tf.matmul(tf.cos(tf.matmul(inputs, self.D, transpose_b=True)), self.C, transpose_b=True)
        return Xmu  # (T-1)xD or n_samplesx(T-1)xD


class GARCHParametricTransitions(BaseGaussianTransitions):
    def __init__(self, latent_dim, input_dim, A=None, B=None, C=None, d=None, Q=None, name=None):
        _Q = np.eye(latent_dim) * 0.2 if Q is None else Q
        super().__init__(dim=latent_dim, input_dim=input_dim, Q=_Q, name=name)
        self.OBSERVATIONS_AS_INPUT = True
        self.A = Param(np.eye(self.dim) * 0.2 if A is None else A)
        self.B = Param(np.eye(self.dim, self.input_dim) * (-0.2) if B is None else B)
        self.C = Param(np.eye(self.dim, self.input_dim) * 0.1 if C is None else C)
        self.d = Param(np.zeros(self.dim) + 0 if d is None else d)

    @params_as_tensors
    def conditional_mean(self, X, inputs):
        if X.shape.ndims == 3:
            Xmu = tf.matmul(tf.reshape(X, [-1, tf.shape(X)[-1]]), self.A, transpose_b=True)  # (n_samples*(T-1))xD
            Xmu = tf.reshape(Xmu, tf.shape(X))
        else:
            Xmu = tf.matmul(X, self.A, transpose_b=True)
        Xmu += tf.matmul(inputs, self.B, transpose_b=True) \
               + tf.matmul(tf.square(inputs), self.C, transpose_b=True) + self.d
        return Xmu  # (T-1)xD or n_samplesx(T-1)xD

    @params_as_tensors
    def variational_expectations(self, Xmu, Xcov, inputs):
        T = Xmu._shape_as_list()[0]
        logp = self.logp(Xmu, inputs)
        tiled_A = tf.tile(self.A[None, :, :], [T-1, 1, 1])

        if isinstance(Xcov, tuple):
            trace_factor = - 2 * tf.matmul(tiled_A, Xcov[1])
            Xcov_diag = Xcov[0]
        else:
            trace_factor = 0.
            Xcov_diag = Xcov
        if Xcov_diag.shape.ndims == 2:
            trace_factor = tf.matrix_diag(Xcov_diag[1:]) \
                           + tf.matmul(self.A * Xcov_diag[:-1][:, None, :], tiled_A, transpose_b=True)
        else:
            trace_factor += Xcov_diag[1:] + tf.matmul(tf.matmul(tiled_A, Xcov_diag[:-1]), tiled_A, transpose_b=True)
        trace = tf.trace(tf.cholesky_solve(tf.tile(self.Qchol[None, :, :], [T-1, 1, 1]), trace_factor))
        return logp - 0.5 * trace


class KinkTransitions(BaseGaussianTransitions):
    def __init__(self, dim, a=None, b=None, c=None, D=None, Q=None, name=None):
        _Q = np.eye(dim) * 0.5 if Q is None else Q
        super().__init__(dim=dim, Q=_Q, name=name)
        self.a = Param(np.ones(self.dim) * 0.8 if a is None else a)
        self.b = Param(np.ones(self.dim) * 0.2 if b is None else b)
        self.c = Param(np.ones(self.dim) * 5.0 if c is None else c)
        self.D = Param(np.eye(self.dim) * 2.0 if D is None else D)

    @params_as_tensors
    def conditional_mean(self, X, inputs=None):
        if X.shape.ndims == 3:
            Xmu = tf.matmul(tf.reshape(X, [-1, tf.shape(X)[-1]]), self.D, transpose_b=True)  # (n_samples*(T-1))xD
            Xmu = tf.reshape(Xmu, tf.shape(X))
        else:
            Xmu = tf.matmul(X, self.D, transpose_b=True)  # (T-1)xD
        Xmu = self.a + (self.b + X) * (1. - self.c / (1. + tf.exp(-Xmu)))
        return Xmu  # (T-1)xD or n_samplesx(T-1)xD
