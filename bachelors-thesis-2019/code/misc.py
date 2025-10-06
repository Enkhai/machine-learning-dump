#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 23:05:53 2019

@author: ceyx
"""

#iteratively prints elements of a queue. end keyword is ".end"
def queue_iter_print(q):
  
    while True:
        prt = q.get()
        if prt == ".end": break
        print(prt)