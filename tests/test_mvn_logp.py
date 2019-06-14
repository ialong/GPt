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
from scipy.stats import multivariate_normal
from numpy.linalg import cholesky
import tensorflow as tf
import gpflow as gp
from gpflow.test_util import GPflowTestCase
from gpflow.logdensities import mvn_logp, diag_mvn_logp


def compare_logps(sess, mvn_fn, x, mu, L):
    cov_sp = np.eye(x.shape[-1]) if L is None else (L @ L.T if L.ndim == 2 else np.diag(L ** 2.))
    if mu.ndim == 1:
        sp_logp = multivariate_normal.logpdf(x=x, mean=mu, cov=cov_sp)
    elif x.ndim == 2:  # x is TxD and mu is TxD
        sp_logp = np.zeros(x.shape[0])
        for t in range(x.shape[0]):
            sp_logp[t] = multivariate_normal.logpdf(x=x[t], mean=mu[t], cov=cov_sp)
    elif mu.ndim == 2: # x is NxTxD and mu is TxD
        sp_logp = np.zeros(x.shape[:-1])
        for n in range(x.shape[0]):
            for t in range(x.shape[1]):
                sp_logp[n, t] = multivariate_normal.logpdf(x=x[n, t], mean=mu[t], cov=cov_sp)
    elif mu.ndim == 3: # x is NxTxD and mu is NxTxD
        sp_logp = np.zeros(x.shape[:-1])
        for n in range(x.shape[0]):
            for t in range(x.shape[1]):
                sp_logp[n, t] = multivariate_normal.logpdf(x=x[n, t], mean=mu[n, t], cov=cov_sp)

    d = x - mu
    d = d if mvn_fn is diag_mvn_logp \
        else d[:, None] if d.ndim == 1 \
        else d.T if d.ndim == 2 \
        else np.transpose(d, [2, 0, 1])

    d_tf = tf.placeholder(gp.settings.float_type, shape=d.shape if d.ndim == 2 else None)
    L_tf = None if L is None else tf.placeholder(gp.settings.float_type)
    feed_dict = {d_tf: d}
    if L is not None: feed_dict[L_tf] = L
    gp_logp = sess.run(mvn_fn(d_tf, L_tf), feed_dict)

    assert_allclose(sp_logp, gp_logp)


class MvnLogPTest(GPflowTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed(0)
        tf.set_random_seed(0)
        self.S, self.T, self.D = 13, 11, 4

    def prepare_L(self):
        L = mvn(self.D, self.D)
        L = cholesky(L @ L.T)
        return L

    def test_mvn_logp(self):
        L = self.prepare_L()
        with self.test_context() as sess:
            compare_logps(sess, mvn_logp, mvn(self.D), mvn(self.D), L)
            compare_logps(sess, mvn_logp, mvn(self.D), mvn(self.D), None)
            compare_logps(sess, mvn_logp, mvn(self.T, self.D), mvn(self.D), L)
            compare_logps(sess, mvn_logp, mvn(self.T, self.D), mvn(self.D), None)
            compare_logps(sess, mvn_logp, mvn(self.T, self.D), mvn(self.T, self.D), L)
            compare_logps(sess, mvn_logp, mvn(self.T, self.D), mvn(self.T, self.D), None)
            compare_logps(sess, mvn_logp, mvn(self.S, self.T, self.D), mvn(self.D), L)
            compare_logps(sess, mvn_logp, mvn(self.S, self.T, self.D), mvn(self.D), None)
            compare_logps(sess, mvn_logp, mvn(self.S, self.T, self.D), mvn(self.T, self.D), L)
            compare_logps(sess, mvn_logp, mvn(self.S, self.T, self.D), mvn(self.T, self.D), None)
            compare_logps(sess, mvn_logp, mvn(self.S, self.T, self.D), mvn(self.S, self.T, self.D), L)
            compare_logps(sess, mvn_logp, mvn(self.S, self.T, self.D), mvn(self.S, self.T, self.D), None)

    def test_diag_mvn_logp(self):
        L_diag = np.diag(self.prepare_L())
        with self.test_context() as sess:

            compare_logps(sess, diag_mvn_logp, mvn(self.D), mvn(self.D), L_diag)
            compare_logps(sess, diag_mvn_logp, mvn(self.D), mvn(self.D), None)
            compare_logps(sess, diag_mvn_logp, mvn(self.T, self.D), mvn(self.D), L_diag)
            compare_logps(sess, diag_mvn_logp, mvn(self.T, self.D), mvn(self.D), None)
            compare_logps(sess, diag_mvn_logp, mvn(self.T, self.D), mvn(self.T, self.D), L_diag)
            compare_logps(sess, diag_mvn_logp, mvn(self.T, self.D), mvn(self.T, self.D), None)
            compare_logps(sess, diag_mvn_logp, mvn(self.S, self.T, self.D), mvn(self.D), L_diag)
            compare_logps(sess, diag_mvn_logp, mvn(self.S, self.T, self.D), mvn(self.D), None)
            compare_logps(sess, diag_mvn_logp, mvn(self.S, self.T, self.D), mvn(self.T, self.D), L_diag)
            compare_logps(sess, diag_mvn_logp, mvn(self.S, self.T, self.D), mvn(self.T, self.D), None)
            compare_logps(sess, diag_mvn_logp, mvn(self.S, self.T, self.D), mvn(self.S, self.T, self.D), L_diag)
            compare_logps(sess, diag_mvn_logp, mvn(self.S, self.T, self.D), mvn(self.S, self.T, self.D), None)


if __name__ == '__main__':
    tf.test.main()