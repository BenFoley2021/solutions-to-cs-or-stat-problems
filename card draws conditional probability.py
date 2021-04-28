# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 19:59:49 2021

PROBLEM STATEMENT: If we draw from a deck of cards (with replacement) until the value of the cards in 
our hand is at least X, what is average (or expected value) that we will have overshot X by?
Example: we draw from a deck containing only cards 2-10. as long as the sum of the cards we have 
drawn is less than 100, we must draw another card. How much do we expect to overshot 100 by?

The discusion below describes the delvelopment of the algorithm used to solve the problem
DISCUSSION: This can be solved using a path search approach. Each draw of a card opens up new paths.
We need to keep track of each path until it reaches its endpoint, and determine the likelyhood of
reaching each possible endpoint. Since we have no conditions requiring us to keep track of the order
of the cards or which cards have been drawn, a path is defined entirely by the sum of the cards. So
all ways of drawing X number of cards which add up to Y can be considered one path. This allows us to greatly 
simplify the path search, as paths will rejoin each other in addition to branching apart. 

Consider a deck with only 2 types of cards, 2s and 4s. if we draw 2 times, there are 3 possible
values for our hand: 4,6,8. If the order in which we had drawn the cards mattered 
then there would have been 4. This is the difference between n choose k with and 
without order mattering. For this simplifed deck of 2 cards, rejoining paths with the same value
is the same as n choose k when order doesn't matter, and yeilds pascals triangle. Whether we 
draw a 2 or 4 is like a Bernoulli trial.

                                            1
                                           / \ 
                                         1     1
                                        / \   / \ 
                                       1    2    1
                                       
it helps to normalize the triangle, so we are seeing the relative probability of each point occuring

                                            1
                                           / \ 
                                         0.5  0.5
                                        / \   / \ 
                                    0.25   0.5  0.25
                                       
If we keep track of the conditional probability of each path throughout the process of drawing cards
and re-join paths that have same value, we can find the probaility of reaching an arbitary endpoint.
Now let's say that instead of just drawing 2 cards from our deck of 2s and 4s, we draw until our cards
sum to at least 4. What values can we end up with after this is done and how likely are they?
The triangle is drawn to show this senario (the #'s in () are the value of our hand at each point)    
                                            
                                             1 (0)
                                           /  \ 
                                       0.5(2)  0.5(4)
                                       /   \   
                                 0.25(4)  0.25(6)
                                       
Each time we draw a card, the 2 and 4 both have a 0.5 probability, each time we draw the current 
state, or path, forks into two. We just need to keep track of these probabilities throughout the
calculation. The probability of the initial state is 1 (we must be there at the start). Then we split 
into two states each with a probability of 0.5. When we re-join paths, we sum the probabiliies. In the example 
above, two paths end at 4, so we re-join them. After this, the answers to question is that there's a 75%
chance our hand has a value of 4, and 25% it has a value of 6.

We now have an approach which can be used to get the probability of encountering an arbitary endpoint
for an arbitary deck of cards. If we had drawn until we hit a higher number than 4, we would end up
re-joing some paths before hitting the endpoint (drawing 2,2,4 is the same as 2,4,2). This allows the
calcualtion to scale reasonably well instead of blowing up if we had to draw until we hit a large number

METHODS USED TO SOLVE: The path search method described in the section above is used, and a Monte Carlo
integration is used to verify the solution. As long as we have enough computing power to run the path
search, it will return an eact solution. The monte carlo method is conceptually simpler,
but cannot return the exact answer. It also cannot take advantage of re-joining paths, and will 
preform much worse in senarios such as this.

As the endpoints become more complex the path search will scale less favorably. If the end condition 
was to draw a value of at least 21 and have drawn a 7 from a deck the cards valued 2-10, we'd have to
keep track of twice as many paths: each value and whether or not it contained a 7. 
As conditions become complex enough that we must keep track of many paths (for example if order matters)
then the monte carlo solutino is better because it's trivial to parrallelize

@author: Ben
"""
from dataclasses import dataclass
import numpy as np
from numpy.random import randint

class conditionalProbOfEndpoints():
    """An object which holds data related to solving the problem using a path search 
        method, and functions which work with that data. #DR use proper capitalization / punctuation especially in docstring comments (the long triple quoted ones)
        
        Attributes:
            transitions: List, cards which can be drawn.
            target: Int, drawn cards until this number.
            endPointDic: Dicitonary that stores endpoints.
            probablity_dist: Np.ndarry, the probability mass function of endpoints found so far.
            expected_val: Float, expected value.
            stdev: Float, standard deviation.
            
        Methods:
            doPathWalk(self) -> ConditionalProbOfEndpoints: Initiates and continues the pathwalk as 
                long as there are paths which can be contiued.
            takeNextStep(self) -> None: Core of algorithm. Takes a "step" by moving along each 
                possible path, ending paths which have reached and endpoint, and updates the
                dictionaries which store that current paths and endpoints.
    """
    def __init__(self,target: float, transitions: list):
        """ Initializes the solution object used as a container for data.
        The information needed by the functions which perform the path walk is 
        placed here. 
        """
        self.transitions = transitions
        self.target = target 
        self.path_dic = None
        self.end_point_dic = None
        self.probablity_dist = None
        self.expected_val = None
        self.stdev = None
    
    def take_next_step(self) -> None:
        """ Moves each path forward one step, keeping track of conditional probability.

        Each possible transition (draw of a card, coin, etc) is done for each path. The new states generated
        and their probabilities are stored in the in "nextPathDic". After new paths are caluculated
        all that meet the end condition are moved to the "end_point_dic". The new state is then set
        to the current state to complete the step.
        
        Args:
            Self: The pathDic, end_point_dic, target, and transitions attributes are used to calculate
            the state of the pathDic and end_point_dict after performing all tranisitions.
        
        Returns:
            None: The attributes of self are modified within the function
        """
        next_path_dic = {} # temporary var used to keep track of the result of the step
        paths_to_end = set() # temporary var used to keep track of which paths have met the termination criteria
        
        for current_path_val in self.path_dic: # loop through each point, or current state of a path
            for transition in self.transitions:# loop through each transformation (or card draw)
                next_path_val = current_path_val + transition # this is value after a card has been drawn
                
                if next_path_val >= self.target: # if the path has reached an endpoint, add to a set
                # which will be used later to move paths to the endpoint dictionary
                    paths_to_end.add(next_path_val)

                # doing the transformation
                if next_path_val in next_path_dic: #this point has already been found, just need to update its probability
                    next_path_dic[next_path_val] += self.path_dic[current_path_val] \
                        / len(self.transitions)
                else: # this point hasn't been found yet, need to create it
                    next_path_dic[next_path_val] = self.path_dic[current_path_val] / len(self.transitions)
                    
        self.path_dic = next_path_dic # all transformations have been done. The next state is set as the current state
                    
        # now that we've calucated the next steps for all paths, 
        # loop through paths that met the end condition and move them from
        # the path dictionary to the endpoint dictionary
        for point in paths_to_end:
            if point in self.end_point_dic: # if this endpoint has been reached before, add the
            # probability of current path to probablility of endpoint
                self.end_point_dic[point] += self.path_dic.pop(point) #pop from the pathDic becuase this path is ended
                
            else: #havent reached this endpoint before, add it to the dictionary
                self.end_point_dic.update({point: self.path_dic.pop(point)})
                    
    def doPathWalk(self): 
        """ Defines initial state and takes steps until all paths have been terminated.
        
        Sets the initial condition (first point in the path) by defining self.pathDic and self.end_point_dic.
        Then calls takeNextStep as long as there is a path which has not yet reached the termination condition

        Args:
            Self: The container made by __init__
        
        Returns:
            Self: The container with end_point_dic fully specified
        """
        self.path_dic = {0: 1} ### first step is the initial state before we've done anything
        self.end_point_dic = {} # initializing the dict that keeps track of all endpoints and their probabilities
        while len(self.path_dic): #       ## the dict is functioning as a stack in a breadth first search
                                            # as long as there is a path, keep iterating
            self.take_next_step()   #### state of self is updated automatically

        return self     

class monteCarlo(): 
    """An object which holds data related to solving the problem using a monte Carlo integration 
        method, and functions which work with that data.
        
        Attributes:
            transitions: List, the cards which are available to be drawn.
            target: Int, the findEndPoints() method will draw cards until this number.
            end_point_dic: Dicitonary, this stores the endpoints found by findEndPoints()
            probablity_dist: Np.ndarry, the probability mass function of endpoints found so far.
            expected_val: Float, the expected value of probablity_dist.
            stdev: Float, the standard deviation of probablity_dist.
            tol: Float, the tolerance for calculation to be considered converged.
            batch: Int, the number of endpoints to find before checking convergence.
            
        Methods:
            mainMonteCarloIntegrationLoop(self) -> object: Manages the monte carlo interation by
                checking convergence and calling methods to: find endpoints, update dictionary of endpoints.  
            findEndPoints(self) -> np.ndarray: finds endpoints by drawing cards until target, returns
                the endpoints it found as an Nx2 array.
            updateEndPointDic(self, tempEndPoints) -> None: takes the latest endpoints found by 
                findEndPoints and adds them to the end_point_dic.
    """
    def __init__(self,target: int,transitions: list, tol: float, batch: int):
        """Initializes the solution object used as a container for data
        The information needed by the functions which do the monte carlo itegration is 
        placed here. 
        
        Args:
            target: int, point at which to stop drawing cards
            transitions: list, the possible transitions (cards to be drawn)
            tol: float, the tolerance. determines the criteria for when calculation is converged
            batch: the number of endpoints to find before checking if the calculation is converged
        
        returns:
            None, constructs self
        """
        self.transitions = transitions # cards which can be drawn
        self.target = target # drawn until this number
        self.end_point_dic = None # dicitonary that stores endpoints
        self.probablity_dist = None # the probability mass function of endpoints found so far
        self.expected_val = None # expected value
        self.stdev = None # standard deviation
        self.tol = tol # tolerance for integration
        self.batch = batch # number of endpoints to find before checking convergence
        
    def main_monte_carlo_integration_loop(self):
        """ Initializes variables and performs steps in mc integration until the tolerance is met.
        
        Overall structure: a histogram of the frequency of occurance of endpoints is generated by 
        drawing cards until the endpoint, "batch" number of times. This histogram is 
        then combined with the histogram made by previous runs, and the expected value calculated.
        if the expected value of the this updated histogram is close enough to the previously calculated
        expected value (as defined by the tolerance), and has been so for the last 5 iterations, the
        calculation is considered converged solution object is returned.
        
        Args:
            self: The solution object whose attributes define the problem and record the solution and intermediates.

        Returns:
            self: After the integration has converged.
        """
        self.end_point_dic = {}  # dictionary will used to store the number of counts for each endpoint
        self.expected_val = np.inf  # this intial state ensures that integration cannot accidently converge
        below_tolerance_count = 0 # used to keep track of the number of times in a row the tolerance has been met
        
        while below_tolerance_count < 5: #if haven't been been below the tolerance 5 times in a row, keep integrating
            temp_end_points = monteCarlo.find_end_points(self) # one "batch" of integrations returns an array of length "batch" 
                                                           # with the outcomes of each trial. 
            self.update_end_point_dic(temp_end_points) # the results from latest batch are added to
                # the dictionary keeping track of the probability of endpoints
            self.probablity_dist = make_probability_dist(self.end_point_dic) # the dictionary is converted to a numpy array
            cur_expected_val = expect_val(self.probablity_dist) # getting the expected value for probability mass 
            # we have created so far function
            
            if abs(cur_expected_val - self.expected_val) / cur_expected_val < self.tol: # comparing the expected Val of this round to 
            # that of the previous. If it's within the tolerance update the belowToleranceCoun
                below_tolerance_count += 1

            else: # if not within the tolerance the count is reset
            # must be within the tolerance for 5 iterations in a row for the MC interation to be considered converged
                below_tolerance_count = 0

            self.expected_val = cur_expected_val 

        return self # once the tolerance has been met, return the solution container 

    def find_end_points(self) -> np.ndarray:
        """Gererates the endpoints used to build up the probability mass function. 
        
        Cards are drawn until the termination condition is met, and the resulting number put in an
        array. This is repeated "batch" times, and the array of endpoints found is returned.
        
        Args:
            self: Parameters stored in self.batch and self.transitions are used here.
        
        Returns:
            tempEndPoints: Numpy array, an array of all the end points found.
        """
        temp_end_points = np.zeros(self.batch, dtype = int) # declaring array which the endpoints found will be placed in

        for i in range(self.batch):
            cur_path_val = 0 # the initial state is 0, haven't drawn any cards yet 
            while cur_path_val < self.target: # as long as the sum of the cards is below the limit, keep drawing more
                cur_path_val = cur_path_val + transitions[randint(0, len(transitions))] #adding the value of a random card 
            temp_end_points[i] = cur_path_val # once we've reached an endpoint, add place it in the array

        return temp_end_points

    def update_end_point_dic(self, temp_end_points) -> None: 
        """Updates the dictionary containg how many times each endpoint has been found.
        
        Args:
            tempEndPoints: Numpy array, an array of all the end points to be added
            self: This method uses self.end_point_dic, a dictionary which keeps track of how many times 
                each endpoint has been found throughout the MC integration process.
                
        Returns:
            None: Self.end_point_dic is updated to include the endpoints in temp_end_points
        """
        for m in temp_end_points: # loop through 
            if m in self.end_point_dic: # if it's already there, add 1 to reflect it was found again
                self.end_point_dic[m] += 1
            else: # if it's not in the dict, create a key for that end point and note that's its been found once
                self.end_point_dic[m] = 1
                

"""
The following section contains functions for building probablity mass functions and calculating 
expected value and standard deviation, which are used by both the Monte Carlo and path walk solutions.
"""

def make_probability_dist(end_point_dic: dict) -> np.ndarray:
    """Converts the dictionary of endpoints into a probability mass function.
    
        Takes the keys of the dictionary as the x axis, and the probablity (dict values) as 
        the y axis. Normalizes and returns a numpy array
    
        Args:
            end_point_dic: Dictionary, the end points and their likelyhood of occurance, as
            obtained from either the monte carlo or path search solutins.
            
        Returns:
            probMassDist: Nx2 numpy array of float, the probability mass function where [:,0] 
            is the endpoint and [:,1] is the probability of that endpoint.
    """

    #make the probability density function given the out_dic
    def norm(arrayIn: np.ndarray) -> np.ndarray: #### normalize
        """ This is a sub function to normalize the probability mass function. 
        
        Args:
            arrayIn: Np.ndarray, the un-normalized probalility mass function.
        Returns:
            arrayIn: Np.ndarray, the array after normalization.
        """
        normFact = sum(arrayIn[:, 1]) # sum all probabilities 
        arrayIn[: ,1] = arrayIn[:, 1]/normFact # divide by the sum of all probabilities

        return arrayIn

    prob_mass_dist = np.zeros([len(end_point_dic), 2]) # creating empty array to be populated shortly

    for i ,endPoint in enumerate(end_point_dic): #### placing info from dictionary in numpy array
        prob_mass_dist[i][0] = endPoint
        prob_mass_dist[i][1] = end_point_dic[endPoint]

    return norm(prob_mass_dist) # normalizing and returning

def expect_val(arrayIn: np.ndarray) -> float: 
    """Takes a normalized array and returns the expected value, where
        expected value = sum( xi*p(xi)).

    Args:
        arrayIn: Int, a 2xN array containing the value and its relative probability.
        This input must be normalized.

    Returns:
        eVal: Float, the expected value of the probability mass function.
    """
    eVal = 0.0 # declare expected value
    for i in arrayIn:
        eVal += i[0] * i[1]

    return eVal

def stdev(arrayIn: int, expected_val: float) -> float:
    """Takes a normalized array and returns the standard deviation, where
        variance = E((X-u)^2) = Sum [p(xi)*(xi-u)^2] and stdev = var**0.5.
        
    Args:
        arrayIn: Int, a 2xN array containing the value and its relative probability.
        This input must be normalized.
        expected_val: Float, the expected value of the probability mass distribution.
        
    Returns:
        stdev: Float, the standard deviation of the prob. mass function.
    """
    stdev = 0.0 # declare standard deviation
    for i in arrayIn: # loop through a prob mass function and calc stdev
        stdev += i[1] * ((i[0] - expected_val) ** 2) 

    return stdev ** 0.5

if __name__ == "__main__":
    """
    Script which sets up the problem, calls functions to solve problem using the path search and
    Monte Carlo (MC) integration methods, and compares the results of those two methods.
    """
    def callPathWalk(target: int, transitions: list):
        """ Calls the functions used find solution and intermediates using the path walk strategy.
        
            Gets the solution container from the pathwalk algorithm, and converts the dictionary with
            the conditional probability of each endpoint into a probability mass function. The expected value
            and standard deviation are calcuated from the prob. mass func. These are saved as attributes of
            the solution container initialized by ConditionalProbOfEndpoints
        
            Args:
                target: the number at which to stop drawing cards
                transitions: a list ways the state of each path can be modified. For drawing
                cards with replacement, it is composition of the deck
                
            returns:
                pathWalkSolution: a container which stores input variables defining the problem, the solution,
                and intermediates 
        """
        # the solution object has the conditional probabilities of all endpoints 
        pathWalkSolution = conditionalProbOfEndpoints(target, transitions).doPathWalk()
        # calcuating the probability mass function of all endpoints from the end_point_dic
        pathWalkSolution.probablity_dist = make_probability_dist(pathWalkSolution.end_point_dic)
        # converting the x axis of the prob mass function from value of endpoint to the amount of "overshoot"
        pathWalkSolution.probablity_dist[:, 0] = (pathWalkSolution.probablity_dist[:, 0] - target)
        # calculating the expected value from the prob mass function
        pathWalkSolution.expected_val= expect_val(pathWalkSolution.probablity_dist)
        # calculation the standard deviation from the prob mass function and expected value
        pathWalkSolution.stdev= stdev(pathWalkSolution.probablity_dist,\
                                                        pathWalkSolution.expected_val)

        return pathWalkSolution

    def callMonteCarloIntegration(target: int, transitions: list, tol: float, batch: int): 
        ''' calls the functions used find solution and intermediates using the monte carlo integration strategy
        
            This function is similar in structure to callPathWalk. The solution container is returned from 
            monteCarlo, and calculates expected value and standard deviation from .probablity_dist
        
            Args: 
                target: the number at which to stop drawing cards 
                transitions: a list ways the state of each path can be modified. For drawing 
                    cards with replacement, it is composition of the deck 
                tolerance: the difference between the expected value of the current and past steps
                    at which the integration is stopped
                batch: number of paths to calculate (number of times to draw until the endpoint) before 
                    recalcuating expected value and checking if within tolerance
                
            Returns:
                pathWalkSolution: a container which stores input variables defining the problem, the solution,
                and intermediates 
        '''
        # get solution object
        monteCarloSolution = monteCarlo(target, transitions, tol, batch).main_monte_carlo_integration_loop() 
        # converting the x axis of the prob mass function from value of endpoint to the amount of "overshoot"
        monteCarloSolution.probablity_dist[:, 0] = (monteCarloSolution.probablity_dist[:, 0] - target)
        # recalculate expected value for the overshoot. could also just have subtracted targer from expected val
        monteCarloSolution.expected_val = expect_val(monteCarloSolution.probablity_dist)
        # calculation the standard deviation from the prob mass function and expected value
        monteCarloSolution.stdevN = stdev(monteCarloSolution.probablity_dist, \
                                                   monteCarloSolution.expected_val)

        return monteCarloSolution
    
    #variables which define the problem
    transitions = [8,9,10] #list, the possible transitions, or cards in the deck
    target = 30 # int, threshold at which to stop drawing cards. when the summed value of the cards is greater or equal, we have reached and endpoint
    
    #variables used to set behavior of monte carlo integrator
    tol = 0.1 # float, convergence tolerance for integration, setting too small will make the MC take a really long time
    batch = 1000 #int, number of endpoints to find before checking if converged

    #calling functions and getting solutions for path walk and monte carlo 
    pathWalkSolution = callPathWalk(target, transitions)

    monteCarloSolution = callMonteCarloIntegration(target, transitions, tol, batch)
    
    # checking how close the monte carlo is to the path walk answer
    print('the difference between the expected value for pathwalk and monte carlo is ' + \
          str(np.round((pathWalkSolution.expected_val - monteCarloSolution.expected_val) \
                       / pathWalkSolution.expected_val * 100, 3)) + ' %')
    

    
