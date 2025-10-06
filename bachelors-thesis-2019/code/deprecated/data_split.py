# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 01:59:37 2018

@author: Ceyx
"""
import sys

#split dataframe into train and test
def split(data, method=2, value=None):
    try:
        #determined by testing set size
        if method == 1:
            if value == None:
                test = data.sample(n=1000)
            else:
                test = data.sample(n=value)
        #determined by split percentage
        elif method == 2:
            if value == None:
                test = data.sample(frac=0.1)
            else:
                test = data.sample(frac=value)
        train = data.drop(test.index)
        return train, test
    except Exception:
        print('Invalid method, incorrect dataset input, \
invalid testing set size or slicing \
percentage. Program will exit to avoid \
further errors.')
        sys.exit()