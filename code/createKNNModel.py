#!/usr/bin/env python
import sys
import glob
import os
import ntpath

import numpy as np
import cPickle
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.neighbors import NearestNeighbors


def __usage__():
    print "Usage: ./createKNNModel.py eval_npys_dir top_k"
    print "for example, ./createKNNModel.py /tmp3/bean/Alibaba/feature_npy/ 20"
    return

def readAllFilesWithExtInDir(dir_path, ext):

    # get the list of npys in directory
    file_list = glob.glob(dir_path + "/*." + ext)

    # convert to absolute path
    file_list = [os.path.abspath(x) for x in file_list]
    
    return file_list

def concatNpysIntoOneBigMatrix(npy_list):

    print "Start reading eval images..."

    result = None
    tmpArr = None
    count = 0

    for npy in npy_list:
        if( (count+1) % 1000 == 0):
            sys.stderr.write(str(count+1) + " eval images loaded\n")

        # convert to sparse format
        # csr means row major
        f = csr_matrix(np.load(npy), dtype='float64')
        # concat into one big sparse matrix
        if result is None:
            result = f
        else:
            if(tmpArr == None):
                tmpArr = f
            # merge into main array every 1000 npy loaded (hope it could make the IO less)
            else:
                tmpArr = vstack([tmpArr, f])
            if( count % 1000 == 0 ):
                result = vstack([result, tmpArr])             
                tmpArr = None
        count += 1
    result = vstack([result, tmpArr])
    print result.shape
    return result

def __main__():

    if( len(sys.argv) < 3):
        __usage__()
        return

    # parse arguments
    eval_npys_dir = sys.argv[1]
    top_k = sys.argv[2]

    eval_list = readAllFilesWithExtInDir(eval_npys_dir, 'npy')
    eval_features = concatNpysIntoOneBigMatrix(eval_list)
    
    sys.stderr.write("Constructing KNN tree...\n")
    # knn infrastructure initialization
    nbrs = NearestNeighbors(n_neighbors=int(top_k), algorithm='auto', metric='euclidean').fit(eval_features)

    # save knn tree in cPickle format
    sys.stderr.write("KNN tree construction finished.\n")
    sys.stderr.write("Save to " + os.path.normpath(eval_npys_dir) + ".cPickle ...\n")
    cPickle.dump(nbrs, open(os.path.normpath(eval_npys_dir)+".cPickle", 'w'))
    
if __name__ == '__main__':
    __main__()
