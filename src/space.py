import numpy as np
import smallutils as u

class Space(object):

    def __init__(self, matrix_, id2row_):

        self.mat = matrix_
        self.id2row = id2row_
        self.create_row2id()

    def create_row2id(self):
        self.row2id = {}
        for idx, word in enumerate(self.id2row):
            if not word in self.row2id:
                #raise ValueError("Found duplicate word: %s" % (word))
                self.row2id[word] = idx


    @classmethod
    def build(cls, fname, lexicon=None, threshold=0, dim=0):
        #if a threshold is provided, we stop reading once it's reached
        #if a lexicon is provided, only words in the lexicon are loaded
        #if dim is provided, all spaces within MWEs are converted into ~
        id2row = []
        def filter_lines(f,ncols):
            for i,line in enumerate(f):
                x = line.split()
                xlen=len(x)
                word = '~'.join(x[0:xlen+1-ncols])
                
                if i != 0 and xlen>=ncols and (lexicon is None or word in lexicon) and (word[0].isalpha()) and (threshold==0 or i<threshold):
                    if u.verbosity>4:
                        print('Word %s has %d fields' % (word,xlen))
                    id2row.append(word)
                    word_length=len(word)
                    if (word_length>0):
                        yield line[word_length+1:]

        #get the number of columns
        if not dim:
            with u.myopen(fname,encoding='utf8') as f:
                f.readline()
                ncols = len(f.readline().split())
        else:
            ncols=dim+1
        with u.myopen(fname,encoding='utf8') as f:
            m = np.matrix(np.loadtxt(filter_lines(f,ncols),
                          comments=None, usecols=range(0,ncols-1)))

        return Space(m, id2row)

    def normalize(self):
        row_norms = np.sqrt(np.multiply(self.mat, self.mat).sum(1))
        row_norms = row_norms.astype(np.double)
        row_norms[row_norms != 0] = np.array(1.0/row_norms[row_norms != 0]).flatten()
        self.mat = np.multiply(self.mat, row_norms)

    def printmat(self):
        print('%d %d' % self.mat.shape)
        for i in range(len(self.id2row)):
            print(self.id2row[i] + ' ' + ' '.join(['%.5g' % x for x in self.mat[i].tolist()[0]]))
