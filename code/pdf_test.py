from scipy.stats import norm
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt


def combine_mean(means):
    """Combine the mean of each model.
    means is a 2D numpy array where a row is the means for one model.
    Each model is trained with 1 trajectory => N_j = 1 for all j.
    We want to thus sum the means column-wise and normalise by the number of models."""
    return np.sum(means, axis=0) / means.shape[0]


def combine_variance(variances, means, combined_means):
    """Combine the variance of each model.
    variances is a 2D numpy array where a row is the variances for one model.
    means is a 2D numpy array where a row is the means for one model.
    combined_mean is a 1D numpy array with the combined mean for all J models.
    Each model is trained with 1 trajectory => N_j = 1 for all j in J."""
    assert variances.shape == means.shape
    return np.subtract(
        np.sum(np.add(variances, np.square(means)), axis=0) / means.shape[0], 
        np.square(combined_means)
    )


def fusion_mean(means, variances, fusion_variance):
    """Combine the mean of each model.
    means is a 2D numpy array where a row is the means for one model.
    Each model is trained with 1 trajectory => N_j = 1 for all j.
    We want to thus sum the means column-wise and normalise by the number of models."""
    return fusion_variance * (np.sum(means * np.power(variances, -1), axis=0) / means.shape[0])


def fusion_variance(variances):
    """Combine the variance of each model.
    variances is a 2D numpy array where a row is the variances for one model.
    means is a 2D numpy array where a row is the means for one model.
    combined_mean is a 1D numpy array with the combined mean for all J models.
    Each model is trained with 1 trajectory => N_j = 1 for all j in J."""
    return variances.shape[0] / np.sum(np.power(variances, -1), axis=0)

if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2)
    #mean, var, skew, kurt = norm.stats(moments='mvsk')

    n1 = norm(0, np.sqrt(0.2))
    n2 = norm(2, np.sqrt(0.5))
    p = 1e-05
    c = 1000
    x1 = np.linspace(n1.ppf(p), n1.ppf(1-p), c)
    x2 = np.linspace(n2.ppf(p), n2.ppf(1-p), c)
    variances = np.array([n1.var(), n2.var()])
    means = np.array([n1.mean(), n2.mean()])
    n_fusion_var = fusion_variance(variances)
    n_fusion_mean = fusion_mean(means, variances, n_fusion_var)
    n_fusion = norm(n_fusion_mean, np.sqrt(n_fusion_var))
    x_fusion = np.linspace(n_fusion.ppf(p), n_fusion.ppf(1-p), c)

    n_combine_mean = combine_mean(means)
    n_combine_var = combine_variance(variances, means, n_combine_mean)
    n_combine = norm(n_combine_mean, np.sqrt(n_combine_var))
    x_combine = np.linspace(n_combine.ppf(p), n_combine.ppf(1-p), c)

    ax1.plot(x1, n1.pdf(x1), 'r-', lw=1, alpha=0.6, label='norm pdf')
    ax1.plot(x2, n2.pdf(x2), 'b-', lw=1, alpha=0.6, label='test pdf')
    ax1.plot(x_fusion, n_fusion.pdf(x_fusion), 'k--', lw=1, alpha=0.6, label="long boi")
    ax1.set_title("Fusion")

    ax2.plot(x1, n1.pdf(x1), 'r-', lw=1, alpha=0.6, label='norm pdf')
    ax2.plot(x2, n2.pdf(x2), 'b-', lw=1, alpha=0.6, label='test pdf')
    ax2.plot(x_combine, n_combine.pdf(x_combine), 'k--', lw=1, alpha=0.6, label="long boi")
    ax2.set_title("Combine")

    plt.savefig("aggregate_test.png")

    #print(quad(norm.pdf, -4, -3))
    #print(quad(norm.pdf, x[0], x[-1]))
    #print(quad(norm.pdf, z[0], z[-1]))