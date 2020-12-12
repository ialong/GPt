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
from gpflow import settings, params_as_tensors, autoflow, kullback_leiblers
from gpflow.logdensities import mvn_logp
from gpflow import mean_functions as mean_fns


class GaussianEmissions(gp.likelihoods.Likelihood):
    def __init__(self, latent_dim=None, obs_dim=None, C=None, R=None, bias=None, name=None):
        super().__init__(name=name)
        self.REQUIRE_FULL_COV = True
        self.latent_dim = C.shape[1] if C is not None else latent_dim

        if (C is None) and (R is None):
            self.obs_dim = obs_dim
        else:
            self.obs_dim = R.shape[0] if R is not None else C.shape[0]

        self.C = gp.Param(np.eye(self.obs_dim, self.latent_dim) if C is None else C)
        self.Rchol = gp.Param(np.eye(self.obs_dim) if R is None else np.linalg.cholesky(R),
                              gp.transforms.LowerTriangular(self.obs_dim, squeeze=True))
        self.bias = gp.Param(np.zeros(self.obs_dim) if bias is None else bias)

    @params_as_tensors
    def conditional_mean(self, X):
        """
        :param X: latent state (T x E) or (n_samples x T x E)
        :return: mu(Y)|X (T x D) or (n_samples x T x D)
        """
        if X.shape.ndims == 3:
            Ymu = tf.matmul(tf.reshape(X, [-1, tf.shape(X)[-1]]), self.C,
                            transpose_b=True) + self.bias
            return tf.reshape(Ymu, [tf.shape(X)[0], tf.shape(X)[1], self.obs_dim])
        else:
            return tf.matmul(X, self.C, transpose_b=True) + self.bias

    @params_as_tensors
    def conditional_variance(self, X):
        """
        :param X: latent state (T x E) or (n_samples x T x E)
        :return: cov(Y)|X (T x D x D) or (n_samples x T x D x D)
        """
        R = tf.matmul(self.Rchol, self.Rchol, transpose_b=True)
        if X.shape.ndims == 3:
            return tf.tile(R[None, None, :, :], [tf.shape(X)[0], tf.shape(X)[1], 1, 1])
        else:
            return tf.tile(R[None, :, :], [tf.shape(X)[0], 1, 1])

    @params_as_tensors
    def logp(self, X, Y):
        """
        :param X: latent state (T x E) or (n_samples x T x E)
        :param Y: observations (T x D)
        :return: \log P(Y|X(n)) (T) or (n_samples x T)
        """
        d = Y - self.conditional_mean(X)
        dim_perm = [2, 0, 1] if X.shape.ndims == 3 else [1, 0]
        return mvn_logp(tf.transpose(d, dim_perm), self.Rchol)

    def sample_conditional(self, X):
        X_in = X if X.ndim == 2 else X.reshape(-1, X.shape[-1])
        noise_samples = np.random.randn(X_in.shape[0], self.obs_dim) @ self.Rchol.value.T
        Y = X_in @ self.C.value.T + self.bias.value + noise_samples
        if X.ndim != 2:
            Y = Y.reshape(*X.shape[:-1], self.obs_dim)
        return Y

    @params_as_tensors
    def predict_mean_and_var(self, Xmu, Xcov):
        assert Xcov.shape.ndims >= 2
        _Xcov = Xcov if Xcov.shape.ndims == 3 else tf.matrix_diag(Xcov)
        Ymu_pred = self.conditional_mean(Xmu)
        C_batch = tf.tile(tf.expand_dims(self.C, 0), [tf.shape(_Xcov)[0], 1, 1])
        Ycov_pred = tf.matmul(self.Rchol, self.Rchol, transpose_b=True) \
                    + tf.matmul(C_batch, tf.matmul(_Xcov, C_batch, transpose_b=True))
        return Ymu_pred, Ycov_pred

    @params_as_tensors
    def variational_expectations(self, Xmu, Xcov, Y):
        assert Xcov.shape.ndims >= 2
        _Xcov = Xcov if Xcov.shape.ndims == 3 else tf.matrix_diag(Xcov)
        logdet = 2. * tf.reduce_sum(tf.log(tf.abs(tf.diag_part(self.Rchol))))
        d = Y - self.conditional_mean(Xmu)
        quad = tf.reduce_sum(tf.square(tf.matrix_triangular_solve(self.Rchol, tf.transpose(d), lower=True)), 0)  # T
        Ctr_Rinv_C = tf.matmul(self.C, tf.cholesky_solve(self.Rchol, self.C), transpose_a=True)
        tr = tf.reduce_sum(Ctr_Rinv_C * _Xcov, [1, 2])  # T
        return -0.5 * (self.obs_dim * np.log(2. * np.pi) + logdet + quad + tr)

    @autoflow((settings.float_type,), (settings.float_type,))
    def compute_predictive_mean_and_var(self, Xmu, Xcov):
        return self.predict_mean_and_var(Xmu, Xcov)

    @autoflow((settings.float_type,), (settings.float_type,), (settings.float_type,))
    def compute_variational_expectations(self, Xmu, Xcov, Y):
        return self.variational_expectations(Xmu, Xcov, Y)


class SingleGPEmissions(gp.likelihoods.Likelihood):
    def __init__(self, latent_dim, Z, mean_function=None, kern=None, likelihood=None, name=None):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.obs_dim = 1
        self.n_ind_pts = Z.shape[0]

        self.mean_function = mean_function or mean_fns.Zero(output_dim=self.obs_dim)
        self.kern = kern or gp.kernels.RBF(self.latent_dim, ARD=True)
        self.likelihood = likelihood or gp.likelihoods.Gaussian()
        self.Z = gp.features.InducingPoints(Z)
        self.Umu = gp.Param(np.zeros((self.n_ind_pts, self.latent_dim)))  # (Lm^-1)(Umu - m(Z))
        self.Ucov_chol = gp.Param(np.tile(np.eye(self.n_ind_pts)[None, ...], [self.obs_dim, 1, 1]),
                                  transform=gp.transforms.LowerTriangular(
                                      self.n_ind_pts, num_matrices=self.obs_dim, squeeze=False))  # (Lm^-1)Lu

    @params_as_tensors
    def conditional(self, X, add_observation_noise=True):
        """
        :param X: latent state (... x E)
        :return: mu(Y)|X (... x D) and var(Y)|X (... x D)
        """
        in_shape = tf.shape(X)
        out_shape = tf.concat([in_shape[:-1], [self.obs_dim]])
        _X = tf.reshape(X, [-1, self.latent_dim])
        mu, var = gp.conditionals.conditional(_X, self.Z, self.kern, self.Umu, q_sqrt=self.Ucov_chol,
                                              full_cov=False, white=True, full_output_cov=False)
        mu += self.mean_function(_X)
        if add_observation_noise:
            var += self.likelihood.variance
        return tf.reshape(mu, out_shape), tf.reshape(var, out_shape)


    @params_as_tensors
    def conditional_mean(self, X):
        """
        :param X: latent state (... x E)
        :return: mu(Y)|X (... x D)
        """
        return self.conditional(X)[0]

    @params_as_tensors
    def conditional_variance(self, X):
        """
        :param X: latent state (... x E)
        :return: var(Y)|X (... x D)
        """
        return self.conditional(X)[1]

    @params_as_tensors
    def logp(self, X, Y):
        """
        :param X: latent state (n_samples x T x E)
        :param Y: observations (n_samples x T x D)
        :return: variational lower bound on \log P(Y|X) (n_samples x T)
        """
        KL = kullback_leiblers.gauss_kl(self.Umu, self.Ucov_chol, None)  # ()
        fmean, fvar = self.conditional(X, add_observation_noise=False)  # (n_samples x T x D) and (n_samples x T x D)
        var_exp = tf.reduce_sum(self.likelihood.variational_expectations(fmean, fvar, Y), -1)  # (n_samples x T)
        return var_exp - KL / tf.cast(tf.shape(X)[1], gp.settings.float_type)


class PolarToCartesianEmissions(GaussianEmissions):
    def __init__(self, R=None, name=None):
        obs_dim = 2
        R_init = np.eye(obs_dim) * 0.1 ** 2 if R is None else R
        super().__init__(latent_dim=obs_dim, obs_dim=obs_dim,
                         C=np.eye(obs_dim), R=R_init, bias=np.zeros(obs_dim), name=name)
        self.C.trainable = False
        self.bias.trainable = False

    @params_as_tensors
    def conditional_mean(self, X):
        return tf.stack([tf.cos(X[..., 0] + 3 / 2 * np.pi),
                         tf.sin(X[..., 0] + 3 / 2 * np.pi)], -1)

    def sample_conditional(self, X):
        conditional_mean = np.stack([
            np.cos(X[..., 0] + 3 / 2 * np.pi),
            np.sin(X[..., 0] + 3 / 2 * np.pi)], -1)
        flat_noise = np.random.randn(np.prod(X.shape[:-1]), self.obs_dim)
        noise_samples = (flat_noise @ self.Rchol.value.T).reshape(*X.shape)
        return conditional_mean + noise_samples

    @params_as_tensors
    def predict_mean_and_var(self, Xmu, Xcov):
        raise NotImplementedError

    @params_as_tensors
    def variational_expectations(self, Xmu, Xcov, Y):
        raise NotImplementedError


class SquaringEmissions(GaussianEmissions):
    def __init__(self, obs_dim, latent_dim=None, C=None, R=None, name=None):
        super().__init__(latent_dim=latent_dim or obs_dim,
                         obs_dim=obs_dim,
                         C=C, R=R, bias=np.zeros(obs_dim), name=name)
        self.bias.trainable = False

    @params_as_tensors
    def conditional_mean(self, X):
        return super().conditional_mean(tf.square(X))

    def sample_conditional(self, X):
        super().sample_conditional(np.square(X))

    @params_as_tensors
    def predict_mean_and_var(self, Xmu, Xcov):
        raise NotImplementedError

    @params_as_tensors
    def variational_expectations(self, Xmu, Xcov, Y):
        raise NotImplementedError


class VolatilityEmissions(gp.likelihoods.Likelihood):
    """
    The volatility likelihood is a zero mean Gaussian likelihood with varying noise:
        p(y|x) = N(y| 0, \exp(x))

    :param inv_link: the link function that is applied to the inputs, it defaults to `tf.exp`
    :type inv_link: a basic TensorFlow function
    """
    def __init__(self, inv_link=tf.exp, name=None):
        super().__init__(name=name)
        self.REQUIRE_FULL_COV = False
        self.inv_link = inv_link

    def conditional_mean(self, X):
        return tf.zeros_like(X)

    def conditional_variance(self, X):
        return self.inv_link(X)

    def logp(self, X, Y):
        return gp.logdensities.gaussian(Y, self.conditional_mean(X), self.conditional_variance(X))

    def sample_conditional(self, X):
        if self.inv_link is tf.exp:
            return np.exp(0.5 * X) * np.random.randn(*X.shape)
        else:
            raise NotImplementedError('Currently only the exponential link function is supported')

    def predict_mean_and_var(self, Xmu, Xcov):
        assert Xcov.shape.ndims >= 2
        Xvar = Xcov if Xcov.shape.ndims == 2 else tf.matrix_diag_part(Xcov)
        mu = self.conditional_mean(Xmu)
        if self.inv_link is tf.exp:
            var = tf.exp(Xmu + Xvar / 2.0)
            return mu, var
        else:
            raise NotImplementedError('Currently only the exponential link function is supported')

    def variational_expectations(self, Xmu, Xcov, Y):
        """
        <log p(y|x)>_NormDist(Xmu, Xcov)
        :param Xmu: Latent function means (TxD)
        :param Xcov: Latent function variances (TxDxD) or (TxD)
        :param Y: Observations (TxD)
        :return: expectations (T)
        """
        assert Xcov.shape.ndims >= 2
        Xvar = Xcov if Xcov.shape.ndims == 2 else tf.matrix_diag_part(Xcov)
        if self.inv_link is tf.exp:
            return -0.5 * tf.reduce_sum(
                np.log(2 * np.pi) + Xmu + tf.square(Y) * tf.exp(-Xmu + Xvar / 2.)
                , 1)
        else:
            raise NotImplementedError('Currently only the exponential link function is supported')


class PriceAndVolatilityEmissions(VolatilityEmissions):
    """
    This is a Volatility likelihood with a non-zero mean and varying noise:
        p(y|x_1, x_2) = N(y| w * x_1 + b, \exp(x_2))

    :param inv_link: the link function that is applied to the inputs, it defaults to `tf.exp`
    :type inv_link: a basic TensorFlow function
    """
    def __init__(self, inv_link=tf.exp, w=1., b=0., name=None):
        super().__init__(inv_link, name=name)
        self.w = gp.Param(w)
        self.b = gp.Param(b)

    @params_as_tensors
    def conditional_mean(self, X):
        return self.w * X[..., 0:1] + self.b

    def conditional_variance(self, X):
        return self.inv_link(X[..., 1:2])

    def sample_conditional(self, X):
        if self.inv_link is tf.exp:
            return np.exp(0.5 * X[..., 1:2]) * np.random.randn(*X.shape[:-1], 1) \
                   + self.w.value * X[..., 0:1] + self.b.value
        else:
            raise NotImplementedError('Currently only the exponential link function is supported')

    @params_as_tensors
    def predict_mean_and_var(self, Xmu, Xcov):
        if Xcov.shape.ndims == 3:
            _Xcov = tf.identity(Xcov)
        else:
            _Xcov = tf.matrix_diag(Xcov)

        Ymu_pred = self.conditional_mean(Xmu)
        if self.inv_link is tf.exp:
            sigma_1 = _Xcov[:, 0, 0][:, None]
            sigma_2 = _Xcov[:, 1, 1][:, None]
            Yvar_pred = sigma_1 * self.w ** 2. + tf.exp(Xmu[:, 1:2] + sigma_2 / 2.)
            return Ymu_pred, Yvar_pred
        else:
            raise NotImplementedError('Currently only the exponential link function is supported')

    @params_as_tensors
    def variational_expectations(self, Xmu, Xcov, Y):
        """
        <log p(y|x)>_NormDist(Xmu, Xcov)
        :param Xmu: Latent function means (Tx2)
        :param Xcov: Latent function variances (Tx2x2) or (Tx2)
        :param Y: Observations (Tx1)
        :return: expectations (T)
        """
        if Xcov.shape.ndims == 3:
            _Xcov = tf.identity(Xcov)
        else:
            _Xcov = tf.matrix_diag(Xcov)

        if self.inv_link is tf.exp:
            sigma_1 = _Xcov[:, 0, 0][:, None]
            sigma_2 = _Xcov[:, 1, 1][:, None]
            cross = _Xcov[:, 0, 1][:, None]
            return -0.5 * (
                    np.log(2 * np.pi) + Xmu[:, 1:2]
                    + (tf.square(Y - self.b - self.w * (Xmu[:, 0:1] - cross)) + sigma_1 * self.w ** 2.)
                    * tf.exp(-Xmu[:, 1:2] + sigma_2 / 2.))
        else:
            raise NotImplementedError('Currently only the exponential link function is supported')
