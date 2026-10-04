import numpy as np

def get_sfr_mask(img_shape, n_mics, dtype='uint8'):
    # '1' indicates the hole and '0' indicates the valid regions.

    assert n_mics < np.prod(img_shape)
    mask = np.ones(img_shape, dtype=dtype)

    flat_idx = np.random.choice(np.prod(img_shape), size=n_mics, replace=False)
    row, col = np.unravel_index(flat_idx, img_shape)
    mask[row, col] = 0

    mask = np.expand_dims(mask, axis=0)

    return mask