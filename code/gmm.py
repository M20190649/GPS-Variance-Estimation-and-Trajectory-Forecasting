import gpflow
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def gauss(mu, sigma):
    return mu + sigma*np.random.randn(10000)

if __name__ == "__main__":
    x1 = gauss(100, 15)
    x2 = gauss(180, 20)
    x = np.concatenate((x1, x2))
    plt.figure()
    plt.title("Multimodal Bus Stop Histogram")
    #sns.distplot(x, hist=False)
    sns.distplot(x, kde=True)
    plt.savefig("multimodal_bus_stops.png")