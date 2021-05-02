import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import subprocess
import glob

from mnist import Mnist
from mpl_toolkits.axes_grid1 import Grid

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class RBM:
    def __init__(self, num_visible, num_hidden, W=None, vb=None, hb=None, k=None):
        self.n_visible = num_visible
        self.n_hidden = num_hidden

        self.W = W if W is not None else np.random.uniform(-1, 1, (num_hidden, num_visible))
        self.vb = vb if vb is not None else np.zeros(num_visible)
        self.hb = hb if hb is not None else np.zeros(num_hidden)
        self.k = k if k is not None else 1
    
    def hidden_expectation(self, V):
        return sigmoid(self.hb + np.dot(V, self.W.T))

    def visible_expectation(self, H):
        return sigmoid(self.vb + np.dot(H, self.W))

    def foward(self, V):
        hp = self.hidden_expectation(V)
        hs = np.random.binomial(1, hp, size=hp.size)
        return hp, hs

    def backward(self, H):
        vp = self.visible_expectation(H)
        vs = np.random.binomial(1, vp, size=vp.size)
        return vp, vs

    def gibbs_sampling(self, V):
        vs = V
        for i in range(self.k):
            hp, hs = self.foward(vs)
            vp, vs = self.backward(hs)
        return hp, hs, vp, vs
    
    def cd(self, X, epoch=1, batch_size=10, k=1, learning_rate=0.01, video=False, verbose=False):
        self.k = k
        n_sample, size = X.shape
        frame_counter=0
        for e in range(epoch):
            for i in range(0, n_sample, batch_size):
                if verbose and i%5000==0:
                    print(f"Epoch: {e} - batch: {i/batch_size}")
                j=i
                batch_W = np.empty((batch_size, self.n_hidden, self.n_visible))
                batch_hb = np.empty((batch_size, self.n_hidden))
                batch_vb = np.empty((batch_size, self.n_visible))
                while j < n_sample and j-i < batch_size:
                    V = X[j]
                    hp, hs, vp_r, vs_r = self.gibbs_sampling(V)
                    hp_r, hs_r = self.foward(vs_r)

                    E_data = np.outer(hp, V)
                    E_model = np.outer(hp_r, vs_r)
                    batch_W[j%batch_size] = E_data - E_model
                    batch_hb[j%batch_size] = hp - hp_r
                    batch_vb[j%batch_size] = V - vs_r
                    j+=1

                # avg gradient over batch
                delta_W = np.mean(batch_W, axis=0)
                delta_hb = np.mean(batch_hb, axis=0)
                delta_vb = np.mean(batch_vb, axis=0)

                self.W += learning_rate*(delta_W) 
                self.hb += learning_rate*(delta_hb)
                self.vb += learning_rate*(delta_vb)

                if video and (i%500==0 if i > 500 else i%100==0):
                    self.show_features(True, frame_counter)
                    frame_counter += 1
        if video:
            self.show_features(True, frame_counter)
            self.create_video()

    def create_video(self):
        # only runs on linux
        os.chdir("./imgs")
        subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', 'hidden_%d.jpg', '-r', '60', '-pix_fmt', 'yuv420p',
            'video.mp4'
        ])
        # for file_name in glob.glob("*.jpg"):
        #     os.remove(file_name)

    def reconstruct(self, V):
        hp, hs = self.foward(V)
        vp, vs = self.backward(hp)
        return vp, vs

    def show_features(self, video=False, index=None):
        maxw = np.amax(self.W)
        minw = np.amin(self.W)
        ncols = self.n_hidden if self.n_hidden < 15 else 15 
        nrows = int(self.n_hidden/ncols)
        fig = plt.figure(figsize=(ncols, nrows), dpi=100)
        grid = Grid(fig, rect=111, nrows_ncols=(nrows, ncols), axes_pad=0.01)

        for i, ax in enumerate(grid):
            x = self.W[i]
            x = (x.reshape(1, -1) - minw)/maxw
            ax.imshow(x.reshape(28,28), cmap=matplotlib.cm.PiYG)
            ax.set_axis_off()

        fig.suptitle('RBM Features', size=20)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85 + nrows*0.002)    
        
        if not video:
            plt.show()
        else:
            plt.savefig(f"imgs/hidden_{index}.jpg")
            plt.close()

if __name__ == "__main__":
    mnist_data = Mnist()
    xd, xl = mnist_data.get_trainset(True)
    # print(xd[0], xl[0])
    # plt.imshow(mnist_data.vet2mat(xd[0]))
    # plt.show()

    rbm = RBM(28*28, 100)
    rbm.cd(xd, epoch=5, batch_size=10, video=False, verbose=True)
    rbm.show_features()