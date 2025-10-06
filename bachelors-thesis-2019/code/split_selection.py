# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 03:19:57 2018

@author: Ceyx
"""
import os #operating system

#returns valid splitting method selection and proper values
#for splitting
def input_select():
    print('Choose splitting method:')
    print('1. Testing size')
    print('2. Percentage split')
    print('3. Use default (Percentage split of 0.1)')
    method = input('')
    if method == '1':
        value = input_split()
    elif method == '2':
        value = input_mask()
    elif method == '3':
        method = 2
        value = float(0.1)
    else:
        print('Again...')
        method, value = input_select()
    return int(method), value
        
#returns splitting value, fixed testing set size case
def input_split():
    try:
        value = int(input('Insert testing set size:'))
        if value > 0:
            return value
        else:
            print('Not a positive integer! Again...')
            value = input_split()
            return value
    except ValueError:
        print('Not an integer! Again...')
        value = input_split()
        return value
    
#returns splitting value, testing set percentage size case
def input_mask():
    try:
        value = float(input('Insert splitting mask (test\
 set percentage size):'))
        if 0.0 < value < 1.0:
            return value
        else:
            print('Splitting mask has to be between 0 and\
 1! Again...')
            value = input_mask()
            return value
    except ValueError:
        print('Not a floating number! Again...')
        value = input_mask()
        return value

#returns valid number of tasks for distributed parallel model
#training
def input_tasks(node=False):
    print('\n|--------------------------------|')
    print('\nThis program will distributedly parallelize training of the given model')
    tasks = input('\n\nPlease insert number of tasks for the distributed \
training to be split or simply type [d] to use the \
default number of logical processors on your machine \
as the number of tasks:\n\n')
    try:
        if tasks == 'd':
            tasks = os.cpu_count()
        elif int(tasks) > 80:
            print('\nInput number of tasks is potentially hazardous \
for the system. Please insert a value lower than 80.')
            tasks = input_tasks()
        elif int(tasks) <= 0:
            print('\nNumber of tasks cannot be a negative number! Again...')
            tasks = input_tasks()            
    except ValueError:
        print('\nNot an integer! Again...')
        tasks = input_tasks()
    return int(tasks)