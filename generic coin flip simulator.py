# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:56:30 2021
coin flip simulator for those conditional prob coin problems. Inspired by the 
number of coin type problems and how tricky they can be. This is meant to be used 
to check ones work will doing / thinking about these sort of stats problems. This can be 
extended to other "occurance of sequence" type problems by modiyfing the flip 
and check condition methods in coinPath or coinNPaths. For example if we drew
cards until we got a sequence of 3,4,5.

coinPath flips a coin until it finds one of the end conditions. It repeats this
N times and reports back how many times it found each condition.

coinNPaths flips a coin until it finds a specific end conditions, repeats N times
to get the average number of flips to find that endpoint. Then repeats the experiemnt
for all other endpoints.

coin toss problem example.
https://www.ted.com/talks/peter_donnelly_how_juries_are_fooled_by_statistics#t-351987
https://cemc.uwaterloo.ca/events/mathcircles/2011-12/Winter/Intermediate_Mar28-Solns.pdf
the average number of tosses until we get a sequence of HTH 101 is 10, while HTT is 8. 

BUT

if we flip until we get either (stop after the first occurance of either HTH or HTT)
then there's an equal probability of reaching each one. 
The ends_list = [(1, 0, 1), (1, 0, 0)] simulates this condition for both the senarios
outlined above.

could probably speed this up using numba or similar, but it's not like we
really need to do millions for flips.

@author: Ben
"""
import random
import numpy as np
class coinPath():
    
    def __init__(self, end_sequences: set, length:int, trials: int):
        self.path = [] # path is just an empty list
        self.ends = end_sequences # this is a set of tuples
        self.trials = trials
        self.sequence_l = length
        temp = {}
        for sequence in end_sequences:
            temp[sequence] = 0
        
        self.end_counts = temp
        
    def flip(self):
        self.path.append(random.randint(0, 1))

    def check_condition(self):
        if tuple(self.path[self.sequence_l:]) in self.ends:
                self.end_counts[tuple(self.path[self.sequence_l:])] += 1
                self.path = []

    def run_sim(self):
        count = 0
        while count < self.trials:
            self.flip()
            count += 1
            self.check_condition()
            
        return self.end_counts
        
    
class coinNPaths(coinPath): # inherit form coinPath, will reuse __init__ and flip
    
    
    def check_condition(self, count, flip_num, endpoint):
        if tuple(self.path[self.sequence_l:]) == endpoint:
            self.end_counts[endpoint][count] = flip_num
            count += 1
            flip_num = 0
            self.path = []
            
        return count, flip_num
            
    def run_sim(self):
        # this would save memory but take more time if i did a running average
        for endpoint in self.end_counts:
            self.end_counts[endpoint] = np.zeros(self.trials)
            flip_num = 0
            count = 0
            while count < self.trials:
                
                self.flip()
                flip_num += 1
                count, flip_num = self.check_condition(count, flip_num, endpoint)
                    
            self.end_counts[endpoint] = self.end_counts[endpoint].mean()
    
        return self.end_counts
    
ends_list = [(1, 0, 1), (1, 0, 0)]

coin = coinPath(set(ends_list), -3, 100000)
out = coin.run_sim()        
print('probability of sequence 1 being seen first: ')
print(out[ends_list[0]] / (out[ends_list[1]] + out[ends_list[0]]))
        
coin2 = coinNPaths(set(ends_list), -3, 10000)   

out2 = coin2.run_sim()    

print('mean number of flips before each sequence appears is ')
print(out2)
