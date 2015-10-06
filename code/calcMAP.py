#!/usr/bin/env python

import sys
import os

def __usage__():
    print 'Usage: ./calcMAP.py query answer'

def parseTopK(inputFile):
    top_k_dict = {}

    lines = open(inputFile, 'r').readlines()
    for line in lines:
        tmp_list = line.strip().split(',')
        top_k_dict[tmp_list[0]] = []
        
        tmp_list2 = tmp_list[1].split(';')
        for tmp in tmp_list2:
            top_k_dict[tmp_list[0]].append(tmp)

    return top_k_dict

def calcAP(query, answer):
    hit = 0
    count = 0
    P_sum = 0

    for q in query:
        count += 1
        if q in answer:
            hit += 1
            P = float(hit)/count
            P_sum += P
    num_ground_truth = len(answer)
    if(num_ground_truth > 20):
        num_ground_truth = 20

    # Notice: This is AP
    return P_sum/num_ground_truth

def __main__():
    
    if( len(sys.argv) < 3 ):
        __usage__()
        return
    
    query = sys.argv[1]
    answer = sys.argv[2]

    query_dict = parseTopK(query)
    answer_dict = parseTopK(answer)

    query_num = len(answer_dict)
    ap_sum = 0

    for key,value in query_dict.iteritems():
        answer_list = answer_dict[key]
        ap = calcAP(value, answer_list)
        ap_sum += ap

    print 'MAP is ' + str(ap_sum/len(answer_dict))



if __name__ == '__main__':
    __main__()
