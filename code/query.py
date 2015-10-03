#!/usr/bin/env python
import sys
import glob
import os

import numpy as np
from sklearn.neighbors import NearestNeighbors
def __usage__():
    print "Usage: ./query.py query_feature_path eval_npys_dir top_k"
    print "for example, ./query.py xxxx.npy /tmp3/bean/Alibaba/feature_npy/ 20"
    print "if the given \"query_feature_path\" is a folder, it will use all the .npy in folder to query"
    return

def readAllFilesWithExtInDir(dir_path, ext):

    # get the list of npys in directory
    file_list = glob.glob(dir_path + "/*." + ext)

    # convert to absolute path
    file_list = [os.path.abspath(x) for x in file_list]
    
    return file_list

def concatNpysIntoOneBigMatrix(npy_list):

    print "Start reading eval images..."

    result = np.array([])
    count = 0
    for npy in npy_list:
        if( (count+1) % 500 == 0):
            sys.stderr.write(str(count+1) + " eval images loaded\n")
        f = np.load(npy)
        result = np.append(result, f)
        count += 1

    return result

def __main__():

    if( len(sys.argv) < 4):
        __usage__()
        return

    # parse arguments
    query_image = sys.argv[1]
    eval_npys_dir = sys.argv[2]
    top_k = sys.argv[3]

    eval_list = readAllFilesWithExtInDir(eval_npys_dir, 'npy')
    eval_features = concatNpysIntoOneBigMatrix(eval_list)

    query_list = query_image
    if(os.path.isdir(query_image)):
        query_list = readAllFilesWithExtInDir(query_image, 'npy')
    
    sys.stderr.write("Constructing KNN tree...\n")
    # knn infrastructure initialization
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='auto', metric='euclidean').fix(eval_features)

    sys.stderr.write("KNN tree construction finished.\n")
    
    count = 0
    for query in query_list:
        if( (count+1) % 50 == 0):
            sys.stderr.write(str(count) + ' queries have been processed\n')
        query_npy = np.load(query)
        distances, indices = nbrs.kneighbors(query_npy)
        result = query + ","
        for indice in indices:
            result += eval_list[indice] + ";"
        print result
        count += 1

if __name__ == '__main__':
    __main__()
