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


def KL(mu_diff, Q_chol, *, P_chol=None, P=None):
    """
    :param mu_diff: (DxM) or [D]
    :param Q_chol: (DxMxM) or [DxD]
    :param P_chol: (None or M or MxM or DxMxM) or [None or DxD or D]
    :param P: (None or M or MxM or DxMxM) or [None or DxD or D]
    :return: scalar KL
    """
    mu_ndims = mu_diff.shape.ndims
    assert mu_ndims is not None
    white = P_chol is None and P is None
    if not white:
        P_ndims = (P_chol if P is None else P).shape.ndims
        assert P_ndims is not None

    if white:
        trace = Q_chol
        mahalanobis = mu_diff
    elif P_ndims == 1:
        P_sqrt = tf.sqrt(tf.abs(P)) if P_chol is None else P_chol
        trace = Q_chol / (P_sqrt[:, None] if mu_ndims == 1 else P_sqrt[None, :, None])
        mahalanobis = mu_diff / (P_sqrt if mu_ndims == 1 else P_sqrt[None, :])
        log_det_P = 2. * tf.reduce_sum(tf.log(tf.abs(P_chol))) if P is None else tf.reduce_sum(tf.log(P))
        if mu_ndims == 2:
            log_det_P *= tf.cast(tf.shape(mu_diff)[0], P_sqrt.dtype)
    else:
        D = tf.shape(mu_diff)[0]
        P_chol = tf.cholesky(P) if P_chol is None else P_chol  # DxMxM or MxM or DxD
        tile_P = (P_ndims == 2) and (mu_ndims == 2)

        P_chol_full = tf.tile(P_chol[None, ...], [D, 1, 1]) if tile_P else P_chol  # DxMxM or DxD
        trace = tf.matrix_triangular_solve(P_chol_full, Q_chol, lower=True)  # DxMxM or DxD

        _mu_diff = mu_diff[:, :, None] if P_ndims == 3 else \
            (tf.transpose(mu_diff) if mu_ndims == 2 else mu_diff[:, None])  # DxMx1 or MxD or Dx1
        mahalanobis = tf.matrix_triangular_solve(P_chol, _mu_diff, lower=True)  # DxMx1 or MxD or Dx1

        log_det_P = 2. * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(P_chol))))
        if tile_P:
            log_det_P *= tf.cast(D, P_chol.dtype)

    trace = tf.reduce_sum(tf.square(trace))
    mahalanobis = tf.reduce_sum(tf.square(mahalanobis))
    constant = tf.cast(tf.size(mu_diff, out_type=tf.int64), dtype=mu_diff.dtype)
    log_det_Q = 2. * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(Q_chol))))
    double_KL = trace + mahalanobis - constant - log_det_Q

    if not white:
        double_KL += log_det_P

    return 0.5 * double_KL


def KL_samples(mu_diff, Q_chol, P_chol=None):
    """
    :param mu_diff: NxSxD or NxD
    :param Q_chol: NxDxD or NxD
    :param P_chol: None or DxD or D
    :return: N
    """
    D = tf.shape(mu_diff)[-1]
    assert mu_diff.shape.ndims is not None
    assert Q_chol.shape.ndims is not None
    diag_Q = Q_chol.shape.ndims == 2

    white = P_chol is None
    if not white:
        assert P_chol.shape.ndims is not None
        diag_P = P_chol.shape.ndims == 1

    if white:
        trace = Q_chol
        mahalanobis = mu_diff
    elif diag_P:
        trace = Q_chol / (P_chol if diag_Q else P_chol[:, None])
        mahalanobis = mu_diff / P_chol
        log_det_P = 2. * tf.reduce_sum(tf.log(tf.abs(P_chol)))
    else:
        N = tf.shape(mu_diff)[0]
        trace = tf.matrix_triangular_solve(tf.tile(P_chol[None, ...], [N, 1, 1]),
                                           tf.matrix_diag(Q_chol) if diag_Q else Q_chol, lower=True)  # NxDxD

        if mu_diff.shape.ndims == 2:
            mahalanobis = tf.matrix_triangular_solve(P_chol, tf.transpose(mu_diff), lower=True)  # DxN
        else:
            mahalanobis = tf.transpose(tf.reshape(mu_diff, [-1, D]))
            mahalanobis = tf.matrix_triangular_solve(P_chol, mahalanobis, lower=True)  # Dx(N*S)
            mahalanobis = tf.reshape(mahalanobis, [D, N, -1])  # DxNxS

        log_det_P = 2. * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(P_chol))))

    if white or diag_P:
        mahalanobis = tf.reduce_sum(tf.square(mahalanobis), -1)
    else:
        mahalanobis = tf.reduce_sum(tf.square(mahalanobis), 0)
    if mu_diff.shape.ndims == 3:
        mahalanobis = tf.reduce_mean(mahalanobis, -1)

    trace = tf.square(trace)
    if (not diag_Q) or (diag_Q and not white and not diag_P):
        trace = tf.reduce_sum(trace, -1)
    trace = tf.reduce_sum(trace, -1)

    constant = tf.cast(D, dtype=mu_diff.dtype)
    log_det_Q = 2. * tf.reduce_sum(tf.log(tf.abs(
        Q_chol if diag_Q else tf.matrix_diag_part(Q_chol))), -1)
    double_KL = trace + mahalanobis - constant - log_det_Q

    if not white:
        double_KL += log_det_P

    return 0.5 * double_KL
