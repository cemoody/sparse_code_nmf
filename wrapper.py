import chainer
from chainer import cuda
from chainer import training
from chainer.training import extensions
from chainer.datasets import TupleDataset
from chainer.iterators import SerialIterator


class Wrapper(object):
    def __init__(self, model, batchsize=512, n_epochs=100, device=None,
                 resume=True):
        self.model = model
        self.n_epochs = n_epochs
        self.device = device
        self.batchsize = batchsize
        self.resume = resume
        if device:
            self.model.to_gpu(device)
        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self.model)

    def fit(self, X, y):
        train = TupleDataset(X)
        train_iter = SerialIterator(train, self.batchsize)
        updater = training.StandardUpdater(train_iter, self.optimizer,
                                           device=self.device)
        trainer = training.Trainer(updater, (self.n_epochs, 'epoch'),
                                   out='out_' + str(self.device))

        # Setup logging, printing & saving
        keys = self.model.keys
        reports = ['epoch']
        reports += ['main/' + key for key in keys]
        trainer.extend(extensions.snapshot(), trigger=(1000, 'epoch'))
        trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
        trainer.extend(extensions.PrintReport(reports))
        trainer.extend(extensions.ProgressBar(update_interval=10))

        # If previous model detected, resume
        if self.resume:
            print("Loading from {}".format(self.resume))
            chainer.serializers.load_npz(self.resume, trainer)


        # Run the model
        trainer.run()
