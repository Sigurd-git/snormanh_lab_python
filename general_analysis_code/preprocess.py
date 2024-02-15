import numpy as np
import re
from scipy.interpolate import interp1d
from scipy import signal
from scipy.linalg import svd
from math import gcd


def lag(X, lag_num, format):
    """
    X: np array
    lag_num: number of lags
    format: name of dimensions, like 'b c t f ' or 't f'
    this function is used to add lags at t dimension and merge with the f dimension
    its workflow is like this: 'b c t f -> b c t f lag'

    Example:
    X = np.arange(24).reshape(2,3,4)
    print(X)
    X_lag = lag(X,3,'b t f')
    print(X_lag)
    """

    # remove spaces at the beginning and end
    format = format.strip()
    # analyse format, splited by any number of spaces
    format = re.split("\s+", format)

    # find the time dimension
    time_dim = format.index("t")

    # find the feature dimension
    # feature_dim = format.index('f')

    X_lags = []

    lag_before = lag_num >= 0

    if lag_before:
        for i in range(lag_num):
            if i == 0:
                X_lag = X
            else:
                # generate pad matrix
                pad_matrix_shape = list(X.shape)
                pad_matrix_shape[time_dim] = i
                pad_matrix = np.zeros(pad_matrix_shape)
                X_lag = np.concatenate((pad_matrix, X), axis=time_dim)

                # remove the last lag_num samples
                X_lag = np.delete(X_lag, np.s_[-i:], axis=time_dim)

            X_lags.append(X_lag)
    else:
        lag_num = -lag_num
        for i in range(lag_num):
            # generate pad matrix
            pad_matrix_shape = list(X.shape)
            pad_matrix_shape[time_dim] = i
            pad_matrix = np.zeros(pad_matrix_shape)
            X_lag = np.concatenate((X, pad_matrix), axis=time_dim)

            # remove the first lag_num samples
            X_lag = np.delete(X_lag, np.s_[:i], axis=time_dim)

            X_lags.append(X_lag)
    X_lags = np.stack(X_lags, axis=-1)

    return X_lags


def match_lag(X, lag_seqs, format):
    """
    X: np array
    lag_seqs: a list or vector of lag numbers
    format: name of dimensions, like 'b c t f ' or 't f'
    this function is used to add lags at t dimension and merge with the f dimension
    its workflow is like this: 'b c t f -> b c t f lag -> b c t f*lag'

    Example:
    X = np.arange(24).reshape(2,3,4)
    print(X)
    X_lag = match_lag(X,(0,1,2,4),'b t f')
    print(X_lag)
    """

    # remove spaces at the beginning and end
    format = format.strip()
    # analyse format, splited by any number of spaces
    format = re.split("\s+", format)

    # find the time dimension
    time_dim = format.index("t")

    X_lags = []

    for i in lag_seqs:
        if i == 0:
            X_lag = X
        else:
            lag_before = i > 0
            if lag_before:
                # generate pad matrix
                pad_matrix_shape = list(X.shape)
                pad_matrix_shape[time_dim] = i
                pad_matrix = np.zeros(pad_matrix_shape)
                X_lag = np.concatenate((pad_matrix, X), axis=time_dim)

                # remove the last lag_num samples
                X_lag = np.delete(X_lag, np.s_[-i:], axis=time_dim)
            else:
                i = -i
                # generate pad matrix
                pad_matrix_shape = list(X.shape)
                pad_matrix_shape[time_dim] = i
                pad_matrix = np.zeros(pad_matrix_shape)
                X_lag = np.concatenate((X, pad_matrix), axis=time_dim)

                # remove the first lag_num samples
                X_lag = np.delete(X_lag, np.s_[:i], axis=time_dim)
        X_lags.append(X_lag)

    X_lags = np.stack(X_lags, axis=-1)

    return X_lags


def structure_lag(X, lag_num, phoneme_df, format):
    """
    X: np array
    lag_num: the number of lags
    format: name of dimensions, like 'b c t f ' or 't f'
    this function is used to add lags at t dimension and merge with the f dimension
    its workflow is like this: 'b c t f -> b c t f lag -> b c t f*lag'

    Example:
    X = np.arange(24).reshape(2,3,4)
    print(X)
    X_lag = match_lag(X,(0,1,2,4),'b t f')
    print(X_lag)
    """

    # remove spaces at the beginning and end
    format = format.strip()
    # analyse format, splited by any number of spaces
    format = re.split("\s+", format)

    # find the time dimension
    time_dim = format.index("t")

    # find the feature dimension
    feature_dim = format.index("f")

    X_lags = []

    for i in range(lag_num):
        if i == 0:
            X_lag = X
        else:
            lag_before = i > 0
            if lag_before:
                # generate pad matrix
                pad_matrix_shape = list(X.shape)
                pad_matrix_shape[time_dim] = i
                pad_matrix = np.zeros(pad_matrix_shape)
                X_lag = np.concatenate((pad_matrix, X), axis=time_dim)

                # remove the last lag_num samples
                X_lag = np.delete(X_lag, np.s_[-i:], axis=time_dim)
            else:
                lag_num = -lag_num
                # generate pad matrix
                pad_matrix_shape = list(X.shape)
                pad_matrix_shape[time_dim] = i
                pad_matrix = np.zeros(pad_matrix_shape)
                X_lag = np.concatenate((X, pad_matrix), axis=time_dim)

                # remove the first lag_num samples
                X_lag = np.delete(X_lag, np.s_[:i], axis=time_dim)
        X_lags.append(X_lag)

    X_lags = np.concatenate(X_lags, axis=feature_dim)

    return X_lags


def align_time(array, t_origin, t_new, format, interpolate=True, resample=True):
    """
    array: np array, the array to be aligned
    t_origin: original time points, 1d array, corresponding to the array
    t_new: new time points, 1d array
    format: name of dimensions, like 'b c t f ' or 't f', t should be concluded in the format, the other dimensions can be named arbitrarily.
    interpolate: whether to interpolate the array, if False, the new time points will be the nearest time points of the resampled array, and new time points will be included in the return value

    Example:
    X = np.arange(24).reshape(2,3,4)
    t_origin = np.arange(4)
    t_new = np.arange(0,3.9,0.1)
    X_new = align_time(X,t_origin,t_new,'b t f')
    """

    # remove spaces at the beginning and end
    format = format.strip()
    # analyse format, splited by any number of spaces
    format = re.split("\s+", format)
    # find the time dimension
    time_dim = format.index("t")
    # compute origin frequency
    f_0 = (len(t_origin) - 1) / (t_origin[-1] - t_origin[0])

    # compute new frequency
    f_new = (len(t_new) - 1) / (t_new[-1] - t_new[0])
    if np.isnan(f_0):
        f_0 = 1
    if np.isnan(f_new):
        f_new = f_0

    if resample:
        # The number of samples in the resampled signal.
        origin_number = len(t_origin)
        new_number = np.round(origin_number / f_0 * f_new).astype(int)

        gcd_value = gcd(origin_number, new_number)
        up = new_number // gcd_value
        down = origin_number // gcd_value
        array_resample = signal.resample_poly(array, up, down, axis=time_dim)
        t_resample = np.arange(new_number) / f_new + t_origin[0]
    else:
        array_resample = array
        t_resample = t_origin

    t_resample = np.round(t_resample, decimals=8)
    t_new = np.round(t_new, decimals=8)
    # pad the array_resample and t_resample to cover the whole range of t_new
    # pad the array_resample and t_resample
    num_pad_before = np.ceil(
        np.round((t_resample[0] - t_new[0]) * f_new, decimals=8)
    ).astype(int)
    num_pad_after = np.ceil(
        np.round((t_new[-1] - t_resample[-1]) * f_new, decimals=8)
    ).astype(int)
    if num_pad_before > 0:
        pad_matrix_shape = list(array_resample.shape)
        pad_matrix_shape[time_dim] = num_pad_before
        pad_matrix = np.zeros(pad_matrix_shape)
        array_pad = np.concatenate((pad_matrix, array_resample), axis=time_dim)
        t_pad = np.concatenate(
            (np.linspace(t_new[0], t_resample[0], num_pad_before), t_resample), axis=0
        )
        print(f"pad {num_pad_before} before the array")

    if num_pad_after > 0:
        pad_matrix_shape = list(array_resample.shape)
        pad_matrix_shape[time_dim] = num_pad_after
        pad_matrix = np.zeros(pad_matrix_shape)
        if num_pad_before > 0:
            array_pad = np.concatenate((array_pad, pad_matrix), axis=time_dim)
            t_pad = np.concatenate(
                (
                    t_pad,
                    np.linspace(
                        t_resample[-1],
                        t_new[-1] + 1 / f_new,
                        num_pad_after + 1,
                        endpoint=False,
                    )[1:],
                ),
                axis=0,
            )

        else:
            array_pad = np.concatenate((array_resample, pad_matrix), axis=time_dim)
            t_pad = np.concatenate(
                (
                    t_resample,
                    np.linspace(
                        t_resample[-1],
                        t_new[-1] + 1 / f_new,
                        num_pad_after + 1,
                        endpoint=False,
                    )[1:],
                ),
                axis=0,
            )
        print(f"pad {num_pad_after} after the array")

    if num_pad_before <= 0 and num_pad_after <= 0:
        array_pad = array_resample
        t_pad = t_resample

    if interpolate:
        # interpolate
        interp_func = interp1d(t_pad, array_pad, axis=time_dim)
        array_new = interp_func(t_new)
        return array_new
    else:
        return array_pad, t_pad


def generate_onehot_features(
    all_labels,
    onehot_label,
    onehot_onset,
    onehot_offset,
    time_length,
    onset_feature=False,
    sr=100,
):
    """
    This function generates one-hot encoded features for given labels, onsets, and offsets.

    Parameters:
    all_labels (numpy.array): Array of all possible labels.
    onehot_label (numpy.array): Array of labels to be one-hot encoded.
    onehot_onset (numpy.array): Array of onset times for each label in onehot_label.
    onehot_offset (numpy.array): Array of offset times for each label in onehot_label.
    time_length (int): The total time length for the feature tensor.
    onset_feature (bool, optional): If True, only the first True index will be True. Defaults to False.
    sr (int, optional): Sampling rate. Defaults to 100.

    Returns:
    feature_tensor (numpy.array): A 2D array with time_length rows and len(all_labels) columns, filled with one-hot encoded features.
    """
    # make sure all_labels, onehot_label, onehot_onset, onehot_offset are all numpy arrays
    all_labels = np.array(all_labels)
    onehot_label = np.array(onehot_label)
    onehot_onset = np.array(onehot_onset)
    onehot_offset = np.array(onehot_offset)

    feature_tensor = np.full((time_length, len(all_labels)), 0, dtype=np.int8)

    for onehot_index in range(len(onehot_label)):
        onehot = onehot_label[onehot_index]
        onset = onehot_onset[onehot_index]
        offset = onehot_offset[onehot_index]

        t_stim = np.arange(time_length) / sr
        indexs = (onset <= t_stim) & (t_stim <= offset)

        if onset_feature:
            # only leave the first True index to be True
            indexs = np.where(indexs)[0]
            if len(indexs) == 0:
                indexs = np.where((onset <= t_stim))[0][0]
            else:
                indexs = indexs[0]
        feature_tensor[indexs, all_labels == onehot] = 1
    return feature_tensor


def reverse_onehot_features(feature_tensor, all_labels, sr=100):
    """
    This function reverses the one-hot encoded features to obtain the labels, onsets, and offsets.

    Parameters:
    feature_tensor (numpy.array): A 2D array with one-hot encoded features.
    all_labels (numpy.array): Array of all possible labels.
    sr (int, optional): Sampling rate. Defaults to 100.

    Returns:
    onehot_label (numpy.array): Array of labels obtained from the feature_tensor.
    onehot_onset (numpy.array): Array of onset times for each label in onehot_label.
    onehot_offset (numpy.array): Array of offset times for each label in onehot_label.
    """
    # TODO: test it!!!!!
    onehot_label = []
    onehot_onset = []
    onehot_offset = []

    t_stim = np.arange(feature_tensor.shape[0]) / sr

    for label_index, label in enumerate(all_labels):
        column = feature_tensor[:, label_index]
        start_indices = (
            np.where(np.diff(column) == 1)[0] + 1
        )  # +1 because diff gives index of change, but we want the index after
        end_indices = np.where(np.diff(column) == -1)[0]

        # If the column starts with 1, then the onset is at the very beginning
        if column[0] == 1:
            start_indices = np.insert(start_indices, 0, 0)

        # If the column ends with 1, then the offset is at the very end
        if column[-1] == 1:
            end_indices = np.append(end_indices, feature_tensor.shape[0] - 1)

        for s, e in zip(start_indices, end_indices):
            onehot_label.append(label)
            onehot_onset.append(t_stim[s])
            onehot_offset.append(t_stim[e])

    return np.array(onehot_label), np.array(onehot_onset), np.array(onehot_offset)


def generate_gonset_features(onehot_onsets, time_length, sr=100):
    feature_tensor = np.zeros((time_length, 1))

    for onehot_onset in onehot_onsets:
        t_stim = np.arange(time_length) / sr
        indexs = (onehot_onset <= t_stim) & (t_stim <= onehot_onset + 1 / sr)
        feature_tensor[indexs, 0] = 1
    return feature_tensor


def pca_from_svd(F, std_feats, demean_feats, n_components):
    assert not np.isnan(F).any()

    # optionally remove mean and standard deviation
    if demean_feats:
        mF = np.mean(F, axis=0)
    else:
        mF = np.zeros(F.shape[1])
    F_formatted = F - mF

    if std_feats:
        normF = np.std(F, axis=0)
    else:
        normF = np.ones(F.shape[1])
    normF[normF == 0] = 1
    F_formatted = F_formatted / normF

    U, s, Vt = svd(F_formatted, full_matrices=False)

    # select the first n_components
    U = U[:, :n_components]
    s = s[:n_components]

    # reconstruct F
    F_recon = U * s
    weight = Vt[:n_components, :].T
    return F_recon, weight, mF, normF


if __name__ == "__main__":
    # construct a test matrix for lag
    X = np.arange(24).reshape(2, 3, 4)
    print(X)
    X_lag = lag(X, 3, "b t f")

    # construct a test matrix for align_time
    X = np.arange(24).reshape(2, 3, 4)
    t_origin = np.arange(4)
    t_new = np.arange(0, 3.9, 0.1)
    X_new = align_time(X, t_origin, t_new, "b t f")
