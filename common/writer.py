import numpy as np
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np


def fig2np(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data


class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_evaluation(self, loss, sample_rate, wav, spec, step):
        
        self.add_scalar('loss', loss, step)

        self.add_audio('wav', wav, step, sample_rate)

        self.add_image('spectrogram', plot_spectrogram_to_numpy(spec), step, dataformats='HWC')
