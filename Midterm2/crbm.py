import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm
from mpl_toolkits.axes_grid1 import Grid

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class CRBM:
    def __init__(self, input_size, num_filters, filters_size, pooling_size, W=None, vb=None, hb=None, k=None):
        self.num_filters = num_filters
        self.input_size = input_size
        self.input_shape = (input_size, input_size)
        self.filters_shape = (filters_size, filters_size)
        self.pooling_shape = (pooling_size, pooling_size)

        self.W = W if W is not None else np.random.uniform(-1, 1, (self.num_filters, filters_size, filters_size))
        self.vb = vb if vb is not None else np.random.uniform(-1, 1)
        self.hb = hb if hb is not None else np.random.uniform(-1, 1, (self.num_filters))
        self.k = k if k is not None else 1

    def pooling_group_weight(self, H):
        hl, hr, hc = H.shape
        pr, pc = self.pooling_shape
        pool_layer_c = int(np.ceil(float(hc) / pc))
        pool_layer_r = int(np.ceil(float(hr) / pr))
        pool = np.zeros(H.shape, float)
        for k in range(hl):
            for j in range(pool_layer_c):
                for i in range(pool_layer_r):
                    pool[k, i*pr:(i+1)*pr, j*pc:(j+1)*pc] = H[k, i*pr:(i+1)*pr, j*pc:(j+1)*pc].sum(axis=-1).sum(axis=-1)
        return pool

    def pool(self, H):
        hl, hr, hc = H.shape
        pr, pc = self.pooling_shape
        pool_layer_c = int(np.ceil(float(hc) / pc))
        pool_layer_r = int(np.ceil(float(hr) / pr))
        pool = np.zeros((hl, pool_layer_r, pool_layer_c), float)
        for k in range(hl):
            for j in range(pool_layer_c):
                for i in range(pool_layer_r):
                    pool[k, i, j] = H[k, i*pr:(i+1)*pr, j*pc:(j+1)*pc].sum(axis=-1).sum(axis=-1)
        return pool

    def visible_expectation(self, H):
        x = sum(ss.convolve(self.W[k], H[k]) for k in range(self.num_filters))
        x += self.vb
        return sigmoid(x)

    def hidden_expectation(self, V):
        x = np.exp(
            np.array([
                ss.convolve(self.W[k, ::-1, ::-1], V, 'valid') + self.hb[k] for k in range(self.num_filters)        
            ])
        )
        return x / (1. + self.pooling_group_weight(x))

    def pooling_expectation(self, V):
        x = np.exp(
            np.array([
                ss.convolve(self.W[k, ::-1, ::-1], V, 'valid') + self.hb[k] for k in range(self.num_filters)        
            ])
        )
        return 1 - 1. / (1 + self.pool(x))

    def train_cd(self, X, learning_rate=0.01, k=1, target_sparsity=None):
        self.k = k
        n_sample, _, _ = X.shape
        for i in tqdm(range(n_sample)):
            v_0 = X[i]
            q_0 = self.hidden_expectation(v_0)
            hs_0 = np.random.binomial(1, q_0, q_0.shape)
            for s in range(self.k):
                vp_1 = self.visible_expectation(hs_0)
                vs_1 = np.random.binomial(1, vp_1, vp_1.shape)
                q_1 = self.hidden_expectation(vs_1)
                hs_1 = np.random.binomial(1, q_1, q_1.shape)
                p_1 = self.pooling_expectation(vs_1)

            delta_W = (1./(q_0.shape[1]**2)) * np.array([ss.convolve(q_0[k, ::-1, ::-1], v_0, 'valid') - ss.convolve(q_1[k, ::-1, ::-1], vp_1, 'valid') for k in range(self.num_filters)])
            delta_hb = (q_0 - q_1).mean(axis=-1).mean(axis=-1)
            if target_sparsity is not None:
                delta_hb += target_sparsity - self.hidden_expectation(vs_1).mean(axis=-1).mean(axis=-1)
            delta_vb = (v_0 - vp_1).mean().mean()

            self.W += learning_rate*delta_W
            self.hb += learning_rate*delta_hb
            self.vb += learning_rate*delta_vb
    
    def show_filters(self):
        ncols = self.num_filters if self.num_filters < 10 else 10
        nrows = self.num_filters//ncols
        fig = plt.figure(figsize=(ncols, nrows), dpi=100)
        grid = Grid(fig, rect=111, nrows_ncols=(nrows, ncols), axes_pad=0.01)

        for i, ax in enumerate(grid):
            x = self.W[i]
            ax.imshow(x)
            ax.set_axis_off()

        fig.suptitle('CRBM Filters', size=20, y=1.05)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85 + nrows*0.002)
        plt.show()