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
    print "Usage: ./query.py query_feature_path eval_npys_dir cPicle_path top_k"
    print "for example, ./query.py xxxx.npy /tmp3/bean/Alibaba/feature_npy/ /tmp3/bean/alibaba/npys/xxx.cPickle 20"
    print "if the given \"query_feature_path\" is a folder, it will use all the .npy in folder to query"
    return

def readAllFilesWithExtInDir(dir_path, ext):

    # get the list of npys in directory
    file_list = glob.glob(dir_path + "/*." + ext)

    # convert to absolute path
    file_list = [os.path.abspath(x) for x in file_list]
    
    return file_list

def __main__():

    if( len(sys.argv) < 5):
        __usage__()
        return

    # parse arguments
    query_image = sys.argv[1]
    eval_npys_dir = sys.argv[2]
    eval_cPickle = sys.argv[3]
    top_k = int(sys.argv[4])

    eval_list = readAllFilesWithExtInDir(eval_npys_dir, 'npy')
    query_list = query_image
    
    if(os.path.isdir(query_image)):
        query_list = readAllFilesWithExtInDir(query_image, 'npy')
    
    sys.stderr.write("Loading " + eval_cPickle + " ...\n")
   
    # load KNN tree object back from cPickle file (ball tree or kd tree)
    nbrs = cPickle.load(open(eval_cPickle,'r'))
    sys.stderr.write("Loading is finished\n")

    count = 0
    for query in query_list:
        if( (count+1) % 10 == 0):
            sys.stderr.write(str(count+1) + ' queries have been processed\n')

        # load query feature ( the shape should be (1, num_dimension) )
        query_npy = np.load(query)
        
        # find knn
        distances, indices = nbrs.kneighbors(query_npy, top_k)
        
        result = ntpath.basename(query).split('.')[0] + ","
        # map indice back to the image name
        for indice in indices[0]:
            result += ntpath.basename(eval_list[indice]).split('.')[0] + ";"
        print result
        count += 1

    sys.stderr.write'All ' + str(len(query_list)) + ' queries have been processed\n')

if __name__ == '__main__':
    __main__()
