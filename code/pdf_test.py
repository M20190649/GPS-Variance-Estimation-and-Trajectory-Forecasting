from scipy.stats import norm
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)
    mean, var, skew, kurt = norm.stats(moments='mvsk')
    x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 100)
    y = np.linspace(-4, -3, 100)
    z = np.linspace(-3, 3, 1000)
    ax.plot(x, norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')
    ax.plot(y, norm.pdf(y), 'g-', lw=5, alpha=0.6, label='test pdf')
    ax.plot(z, norm.pdf(z), 'b-', lw=5, alpha=0.6, label="long boi")
    plt.savefig("pdf_test.png")

    print(quad(norm.pdf, -4, -3))
    print(quad(norm.pdf, x[0], x[-1]))
    print(quad(norm.pdf, z[0], z[-1]))