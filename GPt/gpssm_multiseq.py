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
from gpflow import Param, ParamList, params_as_tensors
from gpflow import transforms as gtf
from GPt.KL import KL, KL_samples
from GPt.gpssm import GPSSM


class GPSSM_MultipleSequences(GPSSM):
    """Equivalent to GPSSM but for data which comes as many (potentially variable-length) independent sequences."""
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
                 batch_size=None,
                 chunking=False,
                 seed=None,
                 parallel_iterations=10,
                 jitter=gp.settings.numerics.jitter_level,
                 name=None):

        super().__init__(latent_dim, Y[0], inputs=None if inputs is None else inputs[0], emissions=emissions,
                         px1_mu=px1_mu, px1_cov=None, kern=kern, Z=Z, n_ind_pts=n_ind_pts,
                         mean_fn=mean_fn, Q_diag=Q_diag, Umu=Umu, Ucov_chol=Ucov_chol,
                         qx1_mu=qx1_mu, qx1_cov=None, As=None, bs=None, Ss=False if Ss is False else None,
                         n_samples=n_samples, seed=seed, parallel_iterations=parallel_iterations,
                         jitter=jitter, name=name)

        self.T = [Y_s.shape[0] for Y_s in Y]
        self.T_tf = tf.constant(self.T, dtype=gp.settings.int_type)
        self.max_T = max(self.T)
        self.sum_T = float(sum(self.T))
        self.n_seq = len(self.T)
        self.batch_size = batch_size
        self.chunking = chunking

        if self.batch_size is None:
            self.Y = ParamList(Y, trainable=False)
        else:
            _Y = np.stack([np.concatenate([Ys, np.zeros((self.max_T - len(Ys), self.obs_dim))]) for Ys in Y])
            self.Y = Param(_Y, trainable=False)

        if inputs is not None:
            if self.batch_size is None:
                self.inputs = ParamList(inputs, trainable=False)
            else:
                desired_length = self.max_T if self.chunking else self.max_T - 1
                _inputs = [np.concatenate([inputs[s], np.zeros((desired_length - len(inputs[s]), self.input_dim))])
                           for s in range(self.n_seq)]  # pad the inputs
                self.inputs = Param(_inputs, trainable=False)

        if qx1_mu is None:
            self.qx1_mu = Param(np.zeros((self.n_seq, self.latent_dim)))

        self.qx1_cov_chol = Param(np.tile(np.eye(self.latent_dim)[None, ...], [self.n_seq, 1, 1]) if qx1_cov is None
                                  else np.linalg.cholesky(qx1_cov),
                                  transform=gtf.LowerTriangular(self.latent_dim, num_matrices=self.n_seq))


        _As = [np.ones((T_s - 1, self.latent_dim)) for T_s in self.T] if As is None else As
        _bs = [np.zeros((T_s - 1, self.latent_dim)) for T_s in self.T] if bs is None else bs
        if Ss is not False:
            _S_chols = [np.tile(self.Q_sqrt.value.copy()[None, ...], [T_s - 1, 1]) for T_s in self.T] if Ss is None \
                else [np.sqrt(S) if S.ndim == 2 else np.linalg.cholesky(S) for S in Ss]

        if self.batch_size is None:
            self.As = ParamList(_As)
            self.bs = ParamList(_bs)
            if Ss is not False:
                self.S_chols = ParamList([Param(Sc, transform=gtf.positive if Sc.ndim == 2 else
                gtf.LowerTriangular(self.latent_dim, num_matrices=Sc.shape[0])) for Sc in _S_chols])
        else:
            _As = np.stack([np.concatenate([_A, np.zeros((self.max_T - len(_A) - 1, *_A.shape[1:]))]) for _A in _As])
            _bs = np.stack([np.concatenate([_b, np.zeros((self.max_T - len(_b) - 1, self.latent_dim))]) for _b in _bs])
            self.As = Param(_As)
            self.bs = Param(_bs)
            if Ss is not False:
                _S_chols = [np.concatenate([_S, np.zeros((self.max_T - len(_S) - 1, *_S.shape[1:]))])
                            for _S in _S_chols]
                _S_chols = np.stack(_S_chols)
                self.S_chols = Param(_S_chols, transform=gtf.positive if _S_chols.ndim == 3 else \
                    gtf.LowerTriangular(self.latent_dim, num_matrices=(self.n_seq, self.max_T - 1)))

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

        self.px1_cov_chol = None if px1_cov is None else Param(_x1_cov, trainable=False, transform=_transform)

        if self.chunking:
            px1_mu_check = len(self.px1_mu.shape) == 1
            px1_cov_check_1 = not self.multi_diag_px1_cov
            px1_cov_check_2 = self.px1_cov_chol is None or len(self.px1_cov_chol.shape) < 3
            assert px1_mu_check and px1_cov_check_1 and px1_cov_check_2, \
                'Only one prior over x1 allowed for chunking'

    @params_as_tensors
    def _build_likelihood(self):
        batch_indices = None if self.batch_size is None else \
            tf.random_shuffle(tf.range(self.n_seq), seed=self.seed)[:self.batch_size]

        X_samples, fs = self._build_sample(batch_indices=batch_indices)
        emissions = self._build_emissions(X_samples, batch_indices=batch_indices)
        KL_X = self._build_KL_X(fs, batch_indices=batch_indices)
        KL_U = self._build_KL_U()
        KL_x1 = self._build_KL_x1(batch_indices=batch_indices)
        return emissions - KL_X - KL_U - KL_x1

    @params_as_tensors
    def _build_sample(self, batch_indices=None):
        Lm = tf.cholesky(self.Kzz)

        X_samples, fs = [], []
        if self.chunking: f_stitch = []
        for s in range(self.n_seq if batch_indices is None else self.batch_size):
            b_s = s if batch_indices is None else batch_indices[s]
            T_s = self.T[s] if batch_indices is None else self.T_tf[b_s]
            _A, _b, _S_chol = self.As[b_s], self.bs[b_s], self.S_chols[b_s]

            if self.chunking:
                T_s, _A, _b, _S_chol = tf.cond(
                    tf.equal(b_s, self.n_seq - 1),
                    lambda: (T_s, _A, _b, _S_chol),
                    lambda:
                    (T_s + 1,
                     tf.concat([_A, tf.zeros((1, *_A.shape[1:]), dtype=gp.settings.float_type)], 0),
                     tf.concat([_b, tf.zeros((1, self.latent_dim), dtype=gp.settings.float_type)], 0),
                     tf.concat([_S_chol, tf.ones((1, *_S_chol.shape[1:]), dtype=gp.settings.float_type)], 0))
                )

            X_sample, *f = self.sample_fn(T=T_s, inputs=None if self.inputs is None else self.inputs[b_s],
                                          qx1_mu=self.qx1_mu[b_s], qx1_cov_chol=self.qx1_cov_chol[b_s],
                                          As=_A, bs=_b, S_chols=_S_chol, Lm=Lm,
                                          **self.sample_kwargs)
            if self.chunking:
                X_sample = tf.cond(tf.equal(b_s, self.n_seq - 1), lambda: X_sample, lambda: X_sample[:-1])
                f_stitch.append([_f[-1] for _f in f])
                f = [tf.cond(tf.equal(b_s, self.n_seq - 1), lambda: _f, lambda: _f[:-1]) for _f in f]

            X_samples.append(X_sample)
            fs.append(f)

        if self.chunking: fs = [fs, f_stitch]
        return X_samples, fs

    @params_as_tensors
    def _build_emissions(self, X_samples, batch_indices=None):
        emissions = 0.
        for s in range(self.n_seq if batch_indices is None else self.batch_size):
            b_s = s if batch_indices is None else batch_indices[s]
            _Y = self.Y[s] if batch_indices is None else self.Y[b_s, :self.T_tf[b_s]]

            emissions += tf.reduce_sum(tf.reduce_mean(
                self.emissions.logp(X_samples[s], _Y[:, None, :]), -1))

        if batch_indices is not None:
            sum_T_minibatch = tf.cast(tf.reduce_sum(tf.gather(self.T_tf, batch_indices)), gp.settings.float_type)
            emissions *= self.sum_T / sum_T_minibatch
        return emissions

    @params_as_tensors
    def _build_KL_X(self, fs, batch_indices=None):
        if self.chunking: fs, f_stitch = fs

        KL_X = 0.
        for s in range(self.n_seq if batch_indices is None else self.batch_size):
            b_s = s if batch_indices is None else batch_indices[s]
            T_s = self.T_tf[b_s]
            _A = self.As[s] if batch_indices is None else self.As[b_s, :T_s - 1]
            _b = self.bs[s] if batch_indices is None else self.bs[b_s, :T_s - 1]
            _S_chol = self.S_chols[s] if batch_indices is None else self.S_chols[b_s, :T_s - 1]

            KL_X += tf.reduce_sum(self.KL_fn(*fs[s], As=_A, bs=_b, S_chols=_S_chol))

            if self.chunking:
                def KL_stitch(f_stitch_s, b_s):
                    kl = KL_samples((f_stitch_s[0] - self.qx1_mu[b_s + 1])[None, ...],
                                    self.qx1_cov_chol[b_s + 1][None, ...], self.Q_sqrt)[0]
                    if len(f_stitch_s) > 1:
                        kl += 0.5 * tf.reduce_mean(tf.reduce_sum(f_stitch_s[1] / tf.square(self.Q_sqrt), -1))
                    return kl

                if isinstance(b_s, int):
                    if s < self.n_seq - 1:
                        KL_X += KL_stitch(f_stitch[s], b_s)
                else:
                    KL_X += tf.cond(tf.equal(b_s, self.n_seq - 1),
                                    lambda: tf.constant(0., dtype=gp.settings.float_type),
                                    lambda: KL_stitch(f_stitch[s], b_s))

        if batch_indices is not None:
            sum_T_minibatch = tf.cast(tf.reduce_sum(tf.gather(self.T_tf, batch_indices)), gp.settings.float_type)
            if self.chunking:
                KL_X *= (self.sum_T - 1.) / tf.cond(tf.reduce_any(tf.equal(batch_indices, self.n_seq - 1)),
                                                    lambda: sum_T_minibatch - 1., lambda: sum_T_minibatch)
            else:
                KL_X *= (self.sum_T - self.n_seq) / (sum_T_minibatch - self.batch_size)
        return KL_X

    @params_as_tensors
    def _build_KL_x1(self, batch_indices=None):
        """
        qx1_mu: SxE
        qx1_cov_chol: SxExE
        px1_mu: E or SxE
        px1_cov_chol: None or E or ExE or SxE or SxExE
        """
        _P_chol = self.px1_cov_chol if not self.multi_diag_px1_cov else tf.matrix_diag(self.px1_cov_chol)
        if self.chunking:
            _px1_mu = self.px1_mu
            _qx1_mu = self.qx1_mu[0]
            _qx1_cov_chol = self.qx1_cov_chol[0]
        elif batch_indices is None:
            _px1_mu = self.px1_mu
            _qx1_mu = self.qx1_mu
            _qx1_cov_chol = self.qx1_cov_chol
        else:
            _px1_mu = tf.gather(self.px1_mu, batch_indices) if self.px1_mu.shape.ndims == 2 else self.px1_mu
            _qx1_mu = tf.gather(self.qx1_mu, batch_indices)
            _qx1_cov_chol = tf.gather(self.qx1_cov_chol, batch_indices)
            _P_chol = None if self.px1_cov_chol is None else \
                (_P_chol if _P_chol.shape.ndims < 3 else tf.gather(_P_chol, batch_indices))

        KL_x1 = KL(_qx1_mu - _px1_mu, _qx1_cov_chol, P_chol=_P_chol)

        if batch_indices is not None and not self.chunking:
            KL_x1 *= float(self.n_seq) / float(self.batch_size)

        return KL_x1
