from chainer import Chain
from chainer import reporter

import chainer.links as L
import chainer.functions as F


class SCNMF(Chain):
    keys = ['loss', 'reg']

    def __init__(self, n_docs, n_dim, n_atoms, true, lnorm=1.0):
        """ Solve L = ||x_i - D r_i|| + c ||r_i|| ^ lnorm

        Attributes
        ----------
        atom: 
            This is the `D` matrix describing each atom
        docs:
            This is the document representation matrix, where each row
            vector `r_i` is the loading onto each atom. Should be sparse.
            This is stored as the log of the docs, and is always exponentiated
            before being used.
        true:
            Input doc representations. 
        """
        self.n_docs = n_docs
        self.lnorm = lnorm
        super(SCNMF, self).__init__(
                atom=L.EmbedID(n_atoms, n_dim),
                docs=L.EmbedID(n_docs, n_atoms),
                true=L.EmbedID(n_docs, n_dim))
        self.true.W.data[...] = true[...]

    def reg(self):
        reg = F.sum(F.exp(self.docs.W) ** self.lnorm)
        # Note that the regularizer is scaled to be computed once
        # over each example
        return reg * 1.0 / self.n_docs

    def __call__(self, doc_index):
        batchsize = doc_index.shape[0]
        rep = F.exp(self.docs(doc_index))
        rec = F.matmul(rep, self.atom.W)
        tru = self.true.W * 1.0
        tru.unchain_backward()
        # How similar are true signals & reconstriction?
        sim = F.matmul(rec, tru, transb=True)
        loss = F.softmax_cross_entropy(sim, doc_index)
        reg = self.reg() * batchsize
        reporter.report(dict(loss=loss, reg=reg), self)
        return loss + reg
