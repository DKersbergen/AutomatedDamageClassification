import numpy as np

def hist_match(after, before):
    """
        Normalisation of images based on histogram matching to the before image.
    Input:
    -----------
        after: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        before: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed after image
    """

    imgsize = after.shape #retrieve array size
    flata = after.ravel() #flatten input array
    flatb = before.ravel() #flatten reference array

    # get the set of unique pixel values and their corresponding indices and
    # counts
    a_values, bin_idx, a_counts = np.unique(flata, return_inverse=True,
                                            return_counts=True)
    b_values, b_counts = np.unique(flatb, return_counts=True)

    # take the cumulative sum of the counts and normalise by the number of pixels
    # to get the empirical CDF for the after and before images
    a_quantiles = np.cumsum(a_counts).astype(np.float64)
    a_quantiles /= a_quantiles[-1]
    b_quantiles = np.cumsum(b_counts).astype(np.float64)
    b_quantiles /= b_quantiles[-1]

    # linear interpolation of pixel values in the before image
    # to correspond most closely to the quantiles in the source image
    interp_b_values = np.interp(a_quantiles, b_quantiles, b_values)

    return interp_b_values[bin_idx].reshape(imgsize)