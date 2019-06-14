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


def block_indices(T, D):
    """
    Returns the indices for diagonal and (upper) off-diagonal
    blocks in an unravelled matrix. Use as:
    diag_blocks = matrix.reshape(-1)[diag_inds].reshape(T,D,D)
    offdiag_blocks = matrix.reshape(-1)[offdiag_inds].reshape(T-1,D,D)
    """
    TD = T * D
    inds = np.tile(np.arange(D), [TD, 1])
    inds += np.arange(TD ** 2, step=TD)[:, None]
    diag_inds = inds + np.repeat(np.arange(TD, step=D), D)[:, None]
    offdiag_inds = diag_inds[:-D] + D
    return diag_inds.ravel(), offdiag_inds.ravel()


def extract_cov_blocks(Xchol, T, D, return_off_diag_blocks=False):
    Xcov = tf.reshape(tf.matmul(Xchol, Xchol, transpose_b=True), [-1])
    diag_inds, offdiag_inds = block_indices(T, D)
    diag_blocks = tf.reshape(tf.gather(Xcov, diag_inds), [T, D, D])
    if return_off_diag_blocks:
        offdiag_blocks = tf.reshape(tf.gather(Xcov, offdiag_inds), [T - 1, D, D])
        return diag_blocks, offdiag_blocks
    return diag_blocks
