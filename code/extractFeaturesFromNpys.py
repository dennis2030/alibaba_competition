#!/usr/bin/env python

import sys
import glob
import ntpath
import numpy as np
import os

def __usage__():
    print 'Usage:./extratFeaturesFromNpy.py npy_dir'

def readAllFilesWithExtInDir(dir_path, ext):

    # get the list of npys in directory
    file_list = glob.glob(dir_path + "/*." + ext)

    # convert to absolute path
    file_list = [os.path.abspath(x) for x in file_list]
    
    return file_list
def __main__():
    
    if( len(sys.argv) < 2 ):
        __usage__()
        return

    npy_dir = sys.argv[1]

    # get file list in dir (.npy)
    npy_list = readAllFilesWithExtInDir(npy_dir, 'npy')
    count = 0
    # dump feature in the given format
    for npy in npy_list:
        if( (count+1) % 100 == 0 ):
            sys.stderr.write(str(count+1) + ' image features extracted\n')
        image_id = ntpath.basename(npy).split('.')[0]
        result = image_id + ','
        tmp_arr = np.load(npy)

        for d in tmp_arr[0]:
            result += str(d) + ';'
        print result
        count += 1
    
    sys.stderr.write('All ' + str( len(npy_list)) + ' image features extracted\n')

if __name__ == '__main__':
    __main__()
