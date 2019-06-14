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


import tensorflow as tf
from gpflow import params_as_tensors
from gpflow import settings as gps
from GPt.gpssm import GPSSM
from GPt.gpssm_multiseq import GPSSM_MultipleSequences
from GPt.ssm import SSM_SG, SSM_SG_MultipleSequences
from GPt.transitions import GPTransitions


class GPSSM_VCDT(GPSSM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_fn = self._build_linear_time_q_sample
        self.sample_kwargs = {'return_f_moments': True,
                              'return_x_cov_chols': True,
                              'sample_u': True}
        self.KL_fn = self._build_factorized_transition_KLs


class GPSSM_FactorizedLinear(SSM_SG):
    def __init__(self, latent_dim, Y, inputs=None, emissions=None,
                 px1_mu=None, px1_cov=None,
                 kern=None, Z=None, n_ind_pts=100,
                 mean_fn=None,
                 Q_diag=None,
                 Umu=None, Ucov_chol=None,
                 Xmu=None, Xchol=None,
                 n_samples=100, seed=None,
                 jitter=gps.numerics.jitter_level, name=None):

        transitions = GPTransitions(latent_dim,
                                    input_dim=0 if inputs is None else inputs.shape[1],
                                    kern=kern, Z=Z, n_ind_pts=n_ind_pts,
                                    mean_fn=mean_fn, Q_diag=Q_diag,
                                    Umu=Umu, Ucov_chol=Ucov_chol,
                                    jitter=jitter,
                                    name=None if name is None else name + '/transitions')

        super().__init__(latent_dim, Y, transitions,
                         T_latent=None, inputs=inputs, emissions=emissions,
                         px1_mu=px1_mu, px1_cov=px1_cov, Xmu=Xmu, Xchol=Xchol,
                         n_samples=n_samples,
                         seed=seed, name=name)


class GPSSM_FactorizedNonLinear(GPSSM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_fn = self._build_linear_time_q_sample
        self.sample_kwargs = {'return_f_moments': True,
                              'return_x_cov_chols': True,
                              'sample_u': False}
        self.KL_fn = self._build_factorized_transition_KLs


class GPSSM_Parametric(GPSSM):
    """
    Corresponds to doing inference in a parametric model with prior: p(f(X)|u)p(u).
    This method can often outperform VCDT as it pays no price for being unable to
    approximate the non-parametric posterior and, for sufficiently large numbers of inducing
    points, it can provide a fit which is closer to that of the non-parametric, cubic time method.
    It can thus be useful to check if the ELBO value achieved by this method is similar to the one
    the cubic time method would give, for the same model and variational parameters.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_fn = self._build_linear_time_q_sample
        self.sample_kwargs = {'return_f_moments': True,
                              'sample_u': True}
        self.KL_fn = self._build_transition_KLs


class GPSSM_Cubic(GPSSM):
    """Full non-parametric prior and posterior (with cubic time sampling)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_fn = self._build_cubic_time_q_sample
        self.sample_kwargs = {'return_f_moments': True,
                              'return_f': False,
                              'sample_u': False}
        self.KL_fn = self._build_transition_KLs


# ===== Methods where the posterior transitions are fixed to the prior (A=I, b=0, S=Q) ===== #


class PRSSM(GPSSM):
    def __init__(self, *args, **kwargs):
        if 'As' in kwargs.keys(): kwargs.pop('As')
        if 'bs' in kwargs.keys(): kwargs.pop('bs')
        kwargs['Ss'] = False
        super().__init__(*args, **kwargs)
        self.As.trainable = False
        self.bs.trainable = False
        self.sample_fn = self._build_linear_time_q_sample
        self.sample_kwargs = {'return_f_moments': False,
                              'sample_u': False}
        self.KL_fn = lambda *fs: tf.constant(0., dtype=gps.float_type)

    @property
    def S_chols(self):
        if self._S_chols is None:
            self._S_chols = tf.ones((self.T - 1, self.latent_dim), dtype=gps.float_type) * self.Q_sqrt
        return self._S_chols


class GPSSM_PPT(GPSSM):
    """
    PPT = Parametric, Prior Transitions
    Effectively the same as PRSSM but with the correct sampling scheme:
    explicit sampling and conditioning of the inducing outputs u.
    Also A=I, b=0, S=Q, i.e. the posterior transitions are fixed to the prior.
    Beware that this still corresponds to doing inference w.r.t. a parametric prior p(f(X)|u)p(u).
    """
    def __init__(self, *args, **kwargs):
        if 'As' in kwargs.keys(): kwargs.pop('As')
        if 'bs' in kwargs.keys(): kwargs.pop('bs')
        kwargs['Ss'] = False
        super().__init__(*args, **kwargs)
        self.As.trainable = False
        self.bs.trainable = False
        self.sample_fn = self._build_linear_time_q_sample
        self.sample_kwargs = {'return_f_moments': False,
                              'sample_u': True}
        self.KL_fn = lambda *fs: tf.constant(0., dtype=gps.float_type)

    @property
    def S_chols(self):
        if self._S_chols is None:
            self._S_chols = tf.ones((self.T - 1, self.latent_dim), dtype=gps.float_type) * self.Q_sqrt
        return self._S_chols


class GPSSM_VPT(GPSSM):
    """
    VPT = VCDT, Prior Transitions.
    VCDT inference method but with posterior transitions fixed to the prior (A=I, b=0, S=Q) as in PRSSM.
    """
    def __init__(self, *args, **kwargs):
        if 'As' in kwargs.keys(): kwargs.pop('As')
        if 'bs' in kwargs.keys(): kwargs.pop('bs')
        kwargs['Ss'] = False
        super().__init__(*args, **kwargs)
        self.As.trainable = False
        self.bs.trainable = False
        self.sample_fn = self._build_linear_time_q_sample
        self.sample_kwargs = {'return_f_moments': True,
                              'return_x_cov_chols': True,
                              'sample_u': True}
        self.KL_fn = self._build_factorized_transition_KLs

    @property
    def S_chols(self):
        if self._S_chols is None:
            self._S_chols = tf.ones((self.T - 1, self.latent_dim), dtype=gps.float_type) * self.Q_sqrt
        return self._S_chols


class GPSSM_CPT(GPSSM):
    """
    CPT = Cubic sampling, Prior Transitions.
    Full non-parametric prior and posterior (with cubic time sampling),
    but with posterior transitions fixed to the prior (A=I, b=0, S=Q) as in PRSSM.
    """
    def __init__(self, *args, **kwargs):
        if 'As' in kwargs.keys(): kwargs.pop('As')
        if 'bs' in kwargs.keys(): kwargs.pop('bs')
        kwargs['Ss'] = False
        super().__init__(*args, **kwargs)
        self.As.trainable = False
        self.bs.trainable = False
        self.sample_fn = self._build_cubic_time_q_sample
        self.sample_kwargs = {'return_f_moments': False,
                              'return_f': False,
                              'sample_u': False}
        self.KL_fn = lambda *fs: tf.constant(0., dtype=gps.float_type)

    @property
    def S_chols(self):
        if self._S_chols is None:
            self._S_chols = tf.ones((self.T - 1, self.latent_dim), dtype=gps.float_type) * self.Q_sqrt
        return self._S_chols


# ========================= Multiple Sequences (MS) data - same models as above ========================= #


class GPSSM_MS_VCDT(GPSSM_MultipleSequences):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_fn = self._build_linear_time_q_sample
        self.sample_kwargs = {'return_f_moments': True,
                              'return_x_cov_chols': True,
                              'sample_u': True}
        self.KL_fn = self._build_factorized_transition_KLs


class GPSSM_MS_FactorizedLinear(SSM_SG_MultipleSequences):
    def __init__(self, latent_dim, Y, inputs=None, emissions=None,
                 px1_mu=None, px1_cov=None,
                 kern=None, Z=None, n_ind_pts=100,
                 mean_fn=None,
                 Q_diag=None,
                 Umu=None, Ucov_chol=None,
                 Xmu=None, Xchol=None,
                 n_samples=100, batch_size=None, seed=None,
                 jitter=gps.numerics.jitter_level, name=None):

        transitions = GPTransitions(latent_dim,
                                    input_dim=0 if inputs is None else inputs[0].shape[1],
                                    kern=kern, Z=Z, n_ind_pts=n_ind_pts,
                                    mean_fn=mean_fn, Q_diag=Q_diag,
                                    Umu=Umu, Ucov_chol=Ucov_chol,
                                    jitter=jitter,
                                    name=None if name is None else name + '/transitions')

        super().__init__(latent_dim, Y, transitions,
                         T_latent=None, inputs=inputs, emissions=emissions,
                         px1_mu=px1_mu, px1_cov=px1_cov, Xmu=Xmu, Xchol=Xchol,
                         n_samples=n_samples, batch_size=batch_size,
                         seed=seed, name=name)


class GPSSM_MS_FactorizedNonLinear(GPSSM_MultipleSequences):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_fn = self._build_linear_time_q_sample
        self.sample_kwargs = {'return_f_moments': True,
                              'return_x_cov_chols': True,
                              'sample_u': False}
        self.KL_fn = self._build_factorized_transition_KLs


class GPSSM_MS_Parametric(GPSSM_MultipleSequences):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_fn = self._build_linear_time_q_sample
        self.sample_kwargs = {'return_f_moments': True,
                              'sample_u': True}
        self.KL_fn = self._build_transition_KLs


class GPSSM_MS_Cubic(GPSSM_MultipleSequences):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_fn = self._build_cubic_time_q_sample
        self.sample_kwargs = {'return_f_moments': True,
                              'return_f': False,
                              'sample_u': False}
        self.KL_fn = self._build_transition_KLs


# ===== Methods where the posterior transitions are fixed to the prior (A=I, b=0, S=Q) ===== #


class PRSSM_MS(GPSSM_MultipleSequences):
    def __init__(self, *args, **kwargs):
        if 'As' in kwargs.keys(): kwargs.pop('As')
        if 'bs' in kwargs.keys(): kwargs.pop('bs')
        kwargs['Ss'] = False
        super().__init__(*args, **kwargs)
        self.As.trainable = False
        self.bs.trainable = False
        self.sample_fn = self._build_linear_time_q_sample
        self.sample_kwargs = {'return_f_moments': False,
                              'sample_u': False}

    @property
    def S_chols(self):
        if self._S_chols is None:
            if self.batch_size is None:
                self._S_chols = [tf.ones((self.T[s] - 1, self.latent_dim),
                                         dtype=gps.float_type) * self.Q_sqrt for s in range(self.n_seq)]
            else:
                self._S_chols = tf.ones((self.n_seq, self.max_T - 1, self.latent_dim),
                                        dtype=gps.float_type) * self.Q_sqrt
        return self._S_chols

    @params_as_tensors
    def _build_KL_X(self, fs, batch_indices=None):
        return tf.constant(0., dtype=gps.float_type)


class GPSSM_MS_PPT(GPSSM_MultipleSequences):
    def __init__(self, *args, **kwargs):
        if 'As' in kwargs.keys(): kwargs.pop('As')
        if 'bs' in kwargs.keys(): kwargs.pop('bs')
        kwargs['Ss'] = False
        super().__init__(*args, **kwargs)
        self.As.trainable = False
        self.bs.trainable = False
        self.sample_fn = self._build_linear_time_q_sample
        self.sample_kwargs = {'return_f_moments': False,
                              'sample_u': True}

    @property
    def S_chols(self):
        if self._S_chols is None:
            if self.batch_size is None:
                self._S_chols = [tf.ones((self.T[s] - 1, self.latent_dim),
                                         dtype=gps.float_type) * self.Q_sqrt for s in range(self.n_seq)]
            else:
                self._S_chols = tf.ones((self.n_seq, self.max_T - 1, self.latent_dim),
                                        dtype=gps.float_type) * self.Q_sqrt
        return self._S_chols

    @params_as_tensors
    def _build_KL_X(self, fs, batch_indices=None):
        return tf.constant(0., dtype=gps.float_type)


class GPSSM_MS_VPT(GPSSM_MultipleSequences):
    def __init__(self, *args, **kwargs):
        if 'As' in kwargs.keys(): kwargs.pop('As')
        if 'bs' in kwargs.keys(): kwargs.pop('bs')
        kwargs['Ss'] = False
        super().__init__(*args, **kwargs)
        self.As.trainable = False
        self.bs.trainable = False
        self.sample_fn = self._build_linear_time_q_sample
        self.sample_kwargs = {'return_f_moments': True,
                              'return_x_cov_chols': True,
                              'sample_u': True}
        self.KL_fn = self._build_factorized_transition_KLs

    @property
    def S_chols(self):
        if self._S_chols is None:
            if self.batch_size is None:
                self._S_chols = [tf.ones((self.T[s] - 1, self.latent_dim),
                                         dtype=gps.float_type) * self.Q_sqrt for s in range(self.n_seq)]
            else:
                self._S_chols = tf.ones((self.n_seq, self.max_T - 1, self.latent_dim),
                                        dtype=gps.float_type) * self.Q_sqrt
        return self._S_chols


class GPSSM_MS_CPT(GPSSM_MultipleSequences):
    def __init__(self, *args, **kwargs):
        if 'As' in kwargs.keys(): kwargs.pop('As')
        if 'bs' in kwargs.keys(): kwargs.pop('bs')
        kwargs['Ss'] = False
        super().__init__(*args, **kwargs)
        self.As.trainable = False
        self.bs.trainable = False
        self.sample_fn = self._build_cubic_time_q_sample
        self.sample_kwargs = {'return_f_moments': False,
                              'return_f': False,
                              'sample_u': False}

    @property
    def S_chols(self):
        if self._S_chols is None:
            if self.batch_size is None:
                self._S_chols = [tf.ones((self.T[s] - 1, self.latent_dim),
                                         dtype=gps.float_type) * self.Q_sqrt for s in range(self.n_seq)]
            else:
                self._S_chols = tf.ones((self.n_seq, self.max_T - 1, self.latent_dim),
                                        dtype=gps.float_type) * self.Q_sqrt
        return self._S_chols

    @params_as_tensors
    def _build_KL_X(self, fs, batch_indices=None):
        return tf.constant(0., dtype=gps.float_type)
