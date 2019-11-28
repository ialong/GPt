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
from tensorflow_probability import distributions as tfd
import gpflow as gp
from gpflow import settings, params_as_tensors, autoflow
from gpflow import transforms as gtf
from gpflow.logdensities import mvn_logp, sum_mvn_logp, diag_mvn_logp
from .transitions import GPTransitions
from .emissions import GaussianEmissions
from .utils import extract_cov_blocks
from .KL import KL


class SSM(gp.models.Model):
    """State-Space Model base class. Used for sampling, no built-in inference."""
    def __init__(self, X_init, Y, transitions, inputs=None, emissions=None, px1_mu=None, px1_cov=None, name=None):
        super().__init__(name=name)
        self.T_latent, self.latent_dim = X_init.shape
        self.T, self.obs_dim = Y.shape

        self.transitions = transitions
        self.emissions = emissions or GaussianEmissions(self.latent_dim, self.obs_dim)

        self.X = gp.Param(X_init)
        self.Y = gp.Param(Y, trainable=False)
        self.inputs = None if inputs is None else gp.Param(inputs, trainable=False)

        self.px1_mu = gp.Param(np.zeros(self.latent_dim) if px1_mu is None else px1_mu, trainable=False)
        self.px1_cov_chol = gp.Param(np.eye(self.latent_dim) if px1_cov is None
                                     else np.linalg.cholesky(px1_cov), trainable=False,
                                     transform=gtf.LowerTriangular(self.latent_dim, squeeze=True))

    @params_as_tensors
    def _build_likelihood(self):
        log_px1 = sum_mvn_logp((self.X[0] - self.px1_mu)[:, None], self.px1_cov_chol)
        inputs = self.Y[:-1] if self.transitions.OBSERVATIONS_AS_INPUT else self.inputs
        log_pX  = tf.reduce_sum(self.transitions.logp(self.X, inputs))
        log_pY  = tf.reduce_sum(self.emissions.logp(self.X[:self.T], self.Y))
        return log_px1 + log_pX + log_pY

    def sample(self, T, N=1, x0_samples=None, inputs=None):
        N = N if x0_samples is None else x0_samples.shape[0]
        T = T if x0_samples is None else T + 1
        X = np.zeros((N, T, self.latent_dim))
        Y = np.zeros((N, T, self.obs_dim))

        tr_sample_conditional = self.transitions.sample_conditional(N)

        if x0_samples is None:
            X[:, 0] = self.px1_mu.value + np.random.randn(N, self.latent_dim) @ self.px1_cov_chol.value.T
        else:
            X[:, 0] = x0_samples
        Y[:, 0] = self.emissions.sample_conditional(X[:, 0])

        for t in range(T - 1):
            if self.transitions.OBSERVATIONS_AS_INPUT:
                input = Y[:, t]
            elif inputs is None:
                input = None if self.inputs is None else self.inputs.value[t]
            else:
                input = inputs[t]
            X[:, t + 1] = tr_sample_conditional(X[:, t], input=input)
            Y[:, t + 1] = self.emissions.sample_conditional(X[:, t + 1])

        return X, Y


class SSM_AG(SSM):
    """
    Analytic inference Gaussian State-Space Model. The variational posterior over the states q(X) is Gaussian.
    The variational lower bound is computed and optimized in closed form.
    """
    def __init__(self, latent_dim, Y, transitions,
                 T_latent=None, inputs=None, emissions=None,
                 px1_mu=None, px1_cov=None, Xmu=None, Xchol=None, name=None):

        _Xmu = np.zeros((T_latent or Y.shape[0], latent_dim)) if Xmu is None else Xmu
        super().__init__(_Xmu, Y, transitions, inputs, emissions, px1_mu, px1_cov, name=name)

        _Xchol = np.eye(self.T_latent * self.latent_dim) if Xchol is None else Xchol
        if _Xchol.ndim == 1:
            self.Xchol = gp.Param(_Xchol)
        else:
            chol_transform = gtf.LowerTriangular(self.T_latent * self.latent_dim if _Xchol.ndim == 2
                                                 else self.latent_dim,
                                                 num_matrices=1 if _Xchol.ndim == 2 else self.T_latent,
                                                 squeeze=_Xchol.ndim == 2)
            self.Xchol = gp.Param(_Xchol, transform=chol_transform)

    @params_as_tensors
    def _build_likelihood(self):
        transitions = self._build_transition_expectations()
        emissions = self._build_emission_expectations()
        entropy = self._build_entropy()
        x1_cross_entropy = self._build_x1_cross_entropy()
        return transitions + emissions + entropy - x1_cross_entropy

    @params_as_tensors
    def _build_transition_expectations(self):
        inputs = self.Y[:-1] if self.transitions.OBSERVATIONS_AS_INPUT else self.inputs
        if self.Xchol.shape.ndims == 1:
            Xcov = tf.reshape(tf.square(self.Xchol), [self.T_latent, self.latent_dim])
        elif self.Xchol.shape.ndims == 2:
            Xcov = extract_cov_blocks(self.Xchol, self.T_latent, self.latent_dim, return_off_diag_blocks=True)
        elif self.Xchol.shape.ndims == 3:
            Xcov = tf.matmul(self.Xchol, self.Xchol, transpose_b=True)
        return tf.reduce_sum(self.transitions.variational_expectations(self.X, Xcov, inputs))

    @params_as_tensors
    def _build_emission_expectations(self):
        if self.Xchol.shape.ndims == 1:
            Xcov = tf.reshape(tf.square(self.Xchol[:self.T * self.latent_dim]), [self.T, self.latent_dim])  # TxD

        elif self.Xchol.shape.ndims == 2:
            Xcutoff = self.T * self.latent_dim
            if self.emissions.REQUIRE_FULL_COV:
                Xcov = extract_cov_blocks(self.Xchol[:Xcutoff, :Xcutoff], self.T, self.latent_dim)
            else:
                Xcov = tf.reshape(tf.reduce_sum(
                    tf.square(self.Xchol[:Xcutoff, :Xcutoff]), 1), [self.T, self.latent_dim])  # TxD

        elif self.Xchol.shape.ndims == 3:
            if self.emissions.REQUIRE_FULL_COV:
                Xcov = tf.matmul(self.Xchol[:self.T], self.Xchol[:self.T], transpose_b=True)
            else:
                Xcov = tf.reduce_sum(tf.square(self.Xchol[:self.T]), 2)  # TxD

        return tf.reduce_sum(self.emissions.variational_expectations(self.X[:self.T], Xcov, self.Y))

    @params_as_tensors
    def _build_entropy(self):
        const = 0.5 * self.T_latent * self.latent_dim * (1. + np.log(2. * np.pi))
        if self.Xchol.shape.ndims == 1:
            logdet = tf.reduce_sum(tf.log(tf.abs(self.Xchol)))
        else:
            logdet = tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(self.Xchol))))
        return const + logdet

    @params_as_tensors
    def _build_x1_cross_entropy(self):
        logp = sum_mvn_logp((self.X[0] - self.px1_mu)[:, None], self.px1_cov_chol)
        if self.Xchol.shape.ndims == 1:
            qx1_cov_chol = tf.matrix_diag(self.Xchol[:self.latent_dim])
        elif self.Xchol.shape.ndims == 2:
            qx1_cov_chol = self.Xchol[:self.latent_dim, :self.latent_dim]
        elif self.Xchol.shape.ndims == 3:
            qx1_cov_chol = self.Xchol[0]
        p_cov_inv_q_cov = tf.matrix_triangular_solve(self.px1_cov_chol, qx1_cov_chol, lower=True)
        trace = tf.reduce_sum(tf.square(p_cov_inv_q_cov))
        return 0.5 * trace - logp

    # autoflow methods:

    @autoflow()
    def compute_transition_expectations(self):
        return self._build_transition_expectations()

    @autoflow()
    def compute_emission_expectations(self):
        return self._build_emission_expectations()

    @autoflow()
    def compute_entropy(self):
        return self._build_entropy()

    @autoflow()
    def compute_x1_cross_entropy(self):
        return self._build_x1_cross_entropy()


class SSM_SG(SSM_AG):
    """
    Stochastic inference Gaussian State-Space Model. The variational posterior over the states q(X) is Gaussian.
    The variational lower bound is evaluated and optimized by sampling from the posterior q(X).
    """
    def __init__(self, latent_dim, Y, transitions,
                 T_latent=None, inputs=None, emissions=None,
                 px1_mu=None, px1_cov=None, Xmu=None, Xchol=None,
                 n_samples=100, seed=None, name=None):
        super().__init__(latent_dim, Y, transitions, T_latent, inputs, emissions,
                         px1_mu, px1_cov, Xmu, Xchol, name=name)
        self.n_samples = n_samples
        self.seed = seed
        self._qx = None

    @property
    def qx(self):
        if self._qx is None:
            if self.Xchol.shape.ndims == 1:
                self._qx = tfd.MultivariateNormalDiag(
                    loc=tf.reshape(self.X, [-1]), scale_diag=self.Xchol)
            else:
                self._qx = tfd.MultivariateNormalTriL(
                    loc=self.X if self.Xchol.shape.ndims == 3 else tf.reshape(self.X, [-1]),
                    scale_tril=self.Xchol)
        return self._qx

    @params_as_tensors
    def _build_likelihood(self):
        qx_samples = self._build_sample_qx()
        transitions = self._build_transition_expectations(qx_samples)
        emissions = self._build_emission_expectations(qx_samples)
        entropy = super()._build_entropy()
        x1_cross_entropy = super()._build_x1_cross_entropy()
        return transitions + emissions + entropy - x1_cross_entropy

    @params_as_tensors
    def _build_sample_qx(self, n_samples=None):
        qx_samples = self.qx.sample(n_samples or self.n_samples, seed=self.seed)
        if self.Xchol.shape.ndims < 3:
            return tf.reshape(qx_samples, [-1, self.T_latent, self.latent_dim])
        return qx_samples

    @params_as_tensors
    def _build_transition_expectations(self, qx_samples):
        inputs = self.Y[:-1] if self.transitions.OBSERVATIONS_AS_INPUT else self.inputs
        logp = self.transitions.logp(qx_samples, inputs)
        return tf.reduce_mean(tf.reduce_sum(logp, 1))

    @params_as_tensors
    def _build_emission_expectations(self, qx_samples):
        logp = self.emissions.logp(qx_samples[:, :self.T], self.Y)
        return tf.reduce_mean(tf.reduce_sum(logp, 1))

    @params_as_tensors
    def _build_stochastic_entropy(self, qx_samples):
        return - tf.reduce_mean(self._build_density_evaluation(qx_samples))

    @params_as_tensors
    def _build_stochastic_x1_cross_entropy(self, qx1_samples):
        return - tf.reduce_mean(mvn_logp(
            tf.transpose(qx1_samples - self.px1_mu), self.px1_cov_chol))

    @params_as_tensors
    def _build_density_evaluation(self, qx_samples):
        if self.Xchol.shape.ndims < 3:
            return self.qx.log_prob(
                    tf.reshape(qx_samples, [-1, self.T_latent * self.latent_dim]))
        return tf.reduce_sum(self.qx.log_prob(qx_samples), -1)

    # autoflow methods:

    @autoflow((settings.int_type, []))
    def sample_qx(self, n_samples=None):
        return self._build_sample_qx(n_samples or self.n_samples)

    @autoflow((settings.float_type,))
    def evaluate_sample_density(self, qx_samples):
        qx_samples = tf.reshape(qx_samples, [-1, self.T_latent, self.latent_dim])
        return self._build_density_evaluation(qx_samples)

    @autoflow()
    def compute_transition_expectations(self):
        qx_samples = self._build_sample_qx(self.n_samples)
        return self._build_transition_expectations(qx_samples)

    @autoflow()
    def compute_emission_expectations(self):
        qx_samples = self._build_sample_qx(self.n_samples)
        return self._build_emission_expectations(qx_samples)

    @autoflow()
    def compute_stochastic_entropy(self):
        qx_samples = self._build_sample_qx(self.n_samples)
        return self._build_stochastic_entropy(qx_samples)

    @autoflow()
    def compute_stochastic_x1_cross_entropy(self):
        qx_samples = self._build_sample_qx(self.n_samples)
        return self._build_stochastic_x1_cross_entropy(qx_samples[:, 0])

    @autoflow((settings.float_type,))
    def compute_variational_bound_from_samples(self, qx_samples):
        qx_samples = tf.reshape(qx_samples, [-1, self.T_latent, self.latent_dim])
        transitions = self._build_transition_expectations(qx_samples)
        emissions = self._build_emission_expectations(qx_samples)
        entropy = super()._build_entropy()
        x1_cross_entropy = super()._build_x1_cross_entropy()
        return (transitions, emissions, entropy, -x1_cross_entropy)

    @autoflow((settings.float_type,))
    def compute_entropy_from_samples(self, qx_samples):
        qx_samples = tf.reshape(qx_samples, [-1, self.T_latent, self.latent_dim])
        return - tf.reduce_mean(self._build_density_evaluation(qx_samples))


class SSM_SG_MultipleSequences(SSM_SG):
    """Equivalent to SSM_SG but for data which comes as many (potentially variable-length) independent sequences."""
    def __init__(self, latent_dim, Y, transitions,
                 T_latent=None, inputs=None, emissions=None,
                 px1_mu=None, px1_cov=None, Xmu=None, Xchol=None,
                 n_samples=100, batch_size=None, seed=None, name=None):

        super().__init__(latent_dim, Y[0], transitions,
                         T_latent=None, inputs=None, emissions=emissions,
                         px1_mu=px1_mu, px1_cov=None, Xmu=None, Xchol=None,
                         n_samples=n_samples, seed=seed, name=name)

        self.T = [Y_s.shape[0] for Y_s in Y]
        self.T_latent = T_latent or self.T
        self.n_seq = len(self.T)
        self.T_tf = tf.constant(self.T, dtype=gp.settings.int_type)
        self.T_latent_tf = tf.constant(self.T_latent, dtype=gp.settings.int_type)
        self.sum_T = float(sum(self.T))
        self.sum_T_latent = float(sum(self.T_latent))
        self.batch_size = batch_size

        self.Y = gp.ParamList(Y, trainable=False)

        self.inputs = None if inputs is None else gp.ParamList(inputs, trainable=False)

        _Xmu = [np.zeros((T_s, self.latent_dim)) for T_s in self.T_latent] if Xmu is None else Xmu
        self.X = gp.ParamList(_Xmu)

        _Xchol = [np.eye(T_s * self.latent_dim) for T_s in self.T_latent] if Xchol is None else Xchol
        xc_tr = lambda xc: None if xc.ndim == 1 else gtf.LowerTriangular(
            xc.shape[-1], num_matrices=1 if xc.ndim == 2 else xc.shape[0], squeeze=xc.ndim == 2)
        self.Xchol = gp.ParamList([gp.Param(xc, transform=xc_tr(xc)) for xc in _Xchol])

        self.multi_diag_px1_cov = False
        if isinstance(px1_cov, list):  # different prior for each sequence
            _x1_cov = np.stack(px1_cov)
            _x1_cov = np.sqrt(_x1_cov) if _x1_cov.ndim == 2 else np.linalg.cholesky(_x1_cov)
            _transform = None if _x1_cov.ndim == 2 else gtf.LowerTriangular(self.latent_dim, num_matrices=self.n_seq)
            self.multi_diag_px1_cov = _x1_cov.ndim == 2
        elif isinstance(px1_cov, np.ndarray):  # same prior for each sequence
            assert px1_cov.ndim < 3
            _x1_cov = np.sqrt(px1_cov) if px1_cov.ndim == 1 else np.linalg.cholesky(px1_cov)
            _transform = None if px1_cov.ndim == 1 else gtf.LowerTriangular(self.latent_dim, squeeze=True)
        else:
            _x1_cov = np.eye(self.latent_dim)
            _transform = gtf.LowerTriangular(self.latent_dim, squeeze=True)

        self.px1_cov_chol = gp.Param(_x1_cov, trainable=False, transform=_transform)

    @property
    def qx(self):
        if self._qx is None:
            self._qx = []
            for s in range(self.n_seq):
                if self.Xchol[s].shape.ndims == 1:
                    self._qx.append(tfd.MultivariateNormalDiag(
                        loc=tf.reshape(self.X[s], [-1]), scale_diag=self.Xchol[s]))
                else:
                    self._qx.append(tfd.MultivariateNormalTriL(
                        loc=self.X[s] if self.Xchol[s].shape.ndims == 3 else tf.reshape(self.X[s], [-1]),
                        scale_tril=self.Xchol[s]))
        return self._qx

    @params_as_tensors
    def _build_likelihood(self):
        batch_indices = None if self.batch_size is None else \
            tf.random_shuffle(tf.range(self.n_seq), seed=self.seed)[:self.batch_size]

        qx_samples = self._build_sample_qx(batch_indices=batch_indices)

        transitions = self._build_transition_expectations(qx_samples, batch_indices=batch_indices)
        emissions = self._build_emission_expectations(qx_samples, batch_indices=batch_indices)
        entropy = self._build_entropy(batch_indices=batch_indices)
        x1_cross_entropy = self._build_x1_cross_entropy(batch_indices=batch_indices)
        return transitions + emissions + entropy - x1_cross_entropy

    @params_as_tensors
    def _build_sample_qx(self, n_samples=None, batch_indices=None):
        if n_samples is None: n_samples = self.n_samples
        qx_samples = []
        for s in range(self.n_seq if batch_indices is None else self.batch_size):
            b_s = s if batch_indices is None else batch_indices[s]
            list_of_samples = [self.qx[i].sample(n_samples, seed=self.seed) for i in range(self.n_seq)]
            qx_s = self.gather_from_list(list_of_samples, b_s)
            if self.gather_from_list(self.Xchol, b_s).shape.ndims < 3:
                qx_s = tf.reshape(qx_s, [-1, self.T_latent_tf[b_s], self.latent_dim])
            qx_samples.append(qx_s)
        return qx_samples

    @params_as_tensors
    def _build_transition_expectations(self, qx_samples, batch_indices=None):
        logp_kwargs = {'subtract_KL_U': False} if isinstance(self.transitions, GPTransitions) else {}

        tr_expectations = 0.
        for s in range(self.n_seq if batch_indices is None else self.batch_size):
            b_s = s if batch_indices is None else batch_indices[s]
            inputs = self.gather_from_list(self.Y, b_s)[:-1] if self.transitions.OBSERVATIONS_AS_INPUT \
                else (None if self.inputs is None else self.gather_from_list(self.inputs, b_s))
            logp = self.transitions.logp(qx_samples[s], inputs, **logp_kwargs)
            tr_expectations += tf.reduce_mean(tf.reduce_sum(logp, 1))

        if batch_indices is not None:
            sum_T_l_batch = tf.cast(tf.reduce_sum(tf.gather(self.T_latent_tf, batch_indices)), gp.settings.float_type)
            tr_expectations *= (self.sum_T_latent - self.n_seq) / (sum_T_l_batch - self.batch_size)

        if isinstance(self.transitions, GPTransitions):
            KL_U = KL(self.transitions.Umu, self.transitions.Ucov_chol)
            tr_expectations -= KL_U
        return tr_expectations

    @params_as_tensors
    def _build_emission_expectations(self, qx_samples, batch_indices=None):
        em_expectations = 0.
        for s in range(self.n_seq if batch_indices is None else self.batch_size):
            b_s = s if batch_indices is None else batch_indices[s]
            logp = self.emissions.logp(qx_samples[s][:, :self.T_tf[b_s]], self.gather_from_list(self.Y, b_s))
            em_expectations += tf.reduce_mean(tf.reduce_sum(logp, 1))

        if batch_indices is not None:
            sum_T_batch = tf.cast(tf.reduce_sum(tf.gather(self.T_tf, batch_indices)), gp.settings.float_type)
            em_expectations *= self.sum_T / sum_T_batch
        return em_expectations

    @params_as_tensors
    def _build_entropy(self, batch_indices=None):
        entropy = 0.
        for s in range(self.n_seq if batch_indices is None else self.batch_size):
            b_s = s if batch_indices is None else batch_indices[s]
            T_latent_b_s = tf.cast(self.T_latent_tf[b_s], gp.settings.float_type)
            const = 0.5 * T_latent_b_s * self.latent_dim * (1. + np.log(2. * np.pi))
            _Xchol = self.gather_from_list(self.Xchol, b_s)
            if _Xchol.shape.ndims == 1:
                logdet = tf.reduce_sum(tf.log(tf.abs(_Xchol)))
            else:
                logdet = tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(_Xchol))))
            entropy = const + logdet

        if batch_indices is not None:
            sum_T_l_batch = tf.cast(tf.reduce_sum(tf.gather(self.T_latent_tf, batch_indices)), gp.settings.float_type)
            entropy *= self.sum_T_latent / sum_T_l_batch
        return entropy

    @params_as_tensors
    def _build_x1_cross_entropy(self, batch_indices=None):
        diag_px1 = self.px1_cov_chol.shape.ndims == 1 or self.multi_diag_px1_cov
        shared_px1 = (self.px1_cov_chol.shape.ndims < 3) and (not self.multi_diag_px1_cov)

        x1_ce = 0.
        for s in range(self.n_seq if batch_indices is None else self.batch_size):
            b_s = s if batch_indices is None else batch_indices[s]
            _px1_mu = self.px1_mu if self.px1_mu.shape.ndims == 1 else self.px1_mu[b_s]
            _px1_cov_chol = self.px1_cov_chol if shared_px1 else self.px1_cov_chol[b_s]
            _qx1_mu = self.gather_from_list(self.X, b_s)[0]
            _qx1_cov_chol = self.gather_from_list(self.Xchol, b_s)
            assert _qx1_cov_chol.shape.ndims in {1, 2, 3}
            if _qx1_cov_chol.shape.ndims == 1:
                _qx1_cov_chol = _qx1_cov_chol[:self.latent_dim]
                _qx1_cov_chol = _qx1_cov_chol[:, None] if diag_px1 else tf.matrix_diag(_qx1_cov_chol)
            elif _qx1_cov_chol.shape.ndims == 2:
                _qx1_cov_chol = _qx1_cov_chol[:self.latent_dim, :self.latent_dim]
            elif _qx1_cov_chol.shape.ndims == 3:
                _qx1_cov_chol = _qx1_cov_chol[0]

            if diag_px1:
                logp = diag_mvn_logp(_qx1_mu - _px1_mu, _px1_cov_chol)
                trace = tf.reduce_sum(tf.square(_qx1_cov_chol / _px1_cov_chol[:, None]))
            else:
                logp = sum_mvn_logp((_qx1_mu - _px1_mu)[:, None], _px1_cov_chol)
                trace = tf.reduce_sum(tf.square(
                    tf.matrix_triangular_solve(_px1_cov_chol, _qx1_cov_chol, lower=True)))
            x1_ce += 0.5 * trace - logp

        if batch_indices is not None:
            x1_ce *= float(self.n_seq) / float(self.batch_size)
        return x1_ce

    @params_as_tensors
    def _build_stochastic_entropy(self, qx_samples, batch_indices=None):
        entropy = - tf.reduce_sum(tf.reduce_mean(tf.stack(self._build_density_evaluation(qx_samples)), -1))
        if batch_indices is not None:
            sum_T_l_batch = tf.cast(tf.reduce_sum(tf.gather(self.T_latent_tf, batch_indices)), gp.settings.float_type)
            entropy *= self.sum_T_latent / sum_T_l_batch
        return entropy

    @params_as_tensors
    def _build_stochastic_x1_cross_entropy(self, qx1_samples, batch_indices=None):
        diag_px1 = self.px1_cov_chol.shape.ndims == 1 or self.multi_diag_px1_cov
        if self.multi_diag_px1_cov or self.px1_cov_chol.shape.ndims == 3:
            x1_ce = 0.
            for s in range(self.n_seq if batch_indices is None else self.batch_size):
                b_s = s if batch_indices is None else batch_indices[s]
                _px1_mu = self.px1_mu if self.px1_mu.shape.ndims == 1 else self.px1_mu[b_s]
                if diag_px1:
                    _x1_ce = diag_mvn_logp(qx1_samples[s] - _px1_mu, self.px1_cov_chol[b_s])
                else:
                    _x1_ce = mvn_logp(tf.transpose(qx1_samples[s] - _px1_mu), self.px1_cov_chol[b_s])
                x1_ce += tf.reduce_mean(_x1_ce)
        else:
            _px1_mu = self.px1_mu if self.px1_mu.shape.ndims == 1 else self.px1_mu[:, None, :]
            if diag_px1:
                x1_ce = diag_mvn_logp(qx1_samples - _px1_mu, self.px1_cov_chol)
            else:
                x1_ce = mvn_logp(tf.transpose(qx1_samples - _px1_mu, [2, 0, 1]), self.px1_cov_chol)
            x1_ce = tf.reduce_sum(tf.reduce_mean(x1_ce, -1))

        if batch_indices is not None:
            x1_ce *= float(self.n_seq) / float(self.batch_size)
        return - x1_ce

    @params_as_tensors
    def _build_density_evaluation(self, qx_samples, batch_indices=None):
        densities = []
        for s in range(self.n_seq if batch_indices is None else self.batch_size):
            b_s = s if batch_indices is None else batch_indices[s]
            if self.gather_from_list(self.Xchol, b_s).shape.ndims < 3:
                reshaped_samples = tf.reshape(qx_samples[s], [-1, self.T_latent[s] * self.latent_dim])
                list_of_logp = [self.qx[i].log_prob(reshaped_samples) for i in range(self.n_seq)]
                densities.append(self.gather_from_list(list_of_logp, b_s))
            else:
                list_of_logp = [self.qx[i].log_prob(qx_samples[s]) for i in range(self.n_seq)]
                densities.append(tf.reduce_sum(self.gather_from_list(list_of_logp, b_s), -1))
        return densities

    def gather_from_list(self, obj_list, index):
        """
        Warning: if index is not within range it returns first element of obj_list
        """
        if isinstance(index, int):
            return obj_list[index]

        s_getter = lambda s: lambda: obj_list[s]
        recursive_getter = obj_list[0]
        for s in range(1, len(obj_list)):
            recursive_getter = tf.cond(tf.equal(index, s), s_getter(s), lambda: recursive_getter)
        return recursive_getter

    # autoflow methods:

    @autoflow()
    def compute_transition_expectations(self):
        raise NotImplementedError

    @autoflow()
    def compute_emission_expectations(self):
        raise NotImplementedError

    @autoflow()
    def compute_entropy(self):
        raise NotImplementedError

    @autoflow()
    def compute_x1_cross_entropy(self):
        raise NotImplementedError

    @autoflow((settings.int_type, []))
    def sample_qx(self, n_samples=None):
        raise NotImplementedError

    @autoflow((settings.float_type,))
    def evaluate_sample_density(self, qx_samples):
        raise NotImplementedError

    @autoflow()
    def compute_transition_expectations(self):
        raise NotImplementedError

    @autoflow()
    def compute_emission_expectations(self):
        raise NotImplementedError

    @autoflow()
    def compute_stochastic_entropy(self):
        raise NotImplementedError

    @autoflow()
    def compute_stochastic_x1_cross_entropy(self):
        raise NotImplementedError

    @autoflow((settings.float_type,))
    def compute_variational_bound_from_samples(self, qx_samples):
        raise NotImplementedError

    @autoflow((settings.float_type,))
    def compute_entropy_from_samples(self, qx_samples):
        raise NotImplementedError
