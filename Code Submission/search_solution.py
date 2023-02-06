from typing import Tuple
from numpy import array
import numpy as np
from registration import registration
import search

# you can want to use the class registration_iasd
# from your solution.py (from previous assignment)
from solution import registration_iasd

# Choose what you think it is the best data structure
# for representing actions.
Action = []

# Choose what you think it is the best data structure
# for representing states.
class State:
    def __init__(self, X = [-180,180] ,Y = [-180,180], Z = [-180,180]):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.value = None

    def __eq__(self, other):
        return (sum(self.X)/2 == sum(other.X)/2) and (sum(self.Y)/2 == sum(other.Y)/2) and (sum(self.Z)/2 == sum(other.Z)/2)
    
    def __hash__(self):
        return hash((sum(self.X)/2,sum(self.Y)/2,sum(self.Z)/2))
    
    def __lt__(self, other):
        return True 
    
    # compute the rotation matrix
    def rotation(self):
        R = np.zeros([3,3])
        rad = np.radians([sum(self.X)/2,sum(self.Y)/2,sum(self.Z)/2])
        ca,cb,cc =  tuple(np.cos(rad))
        sa,sb,sc =  tuple(np.sin(rad))

        R[0,0] = (ca*cb)
        R[0,1] = (ca*sb*sc)-(sa*cc)
        R[0,2] = (ca*sb*cc)+(sa*sc)

        R[1,0] = (sa*cb)
        R[1,1] = (sa*sb*sc) + (ca*cc)
        R[1,2] = (sa*sb*cc) - (ca*sc)

        R[2,0] = -sb
        R[2,1] = cb*sc
        R[2,2] = cb*cc

        return R
    
     #compute the distance
    def dist(self, scan1, scan2):
        if self.value is None:
            R = self.rotation()
            scan1 = np.dot(R,(scan1).T).T
            reg = registration_iasd(scan1, scan2)
            dic = reg.find_closest_points()
            self.value = sum(correspondence['dist2']**2 for correspondence in dic.values())/len(dic.values())
        return self.value
        
     # depending on the action returns the state
    def goal(self, scan1, scan2):
        R = self.rotation()
        scan1 = np.dot(R,(scan1).T).T
        reg = registration_iasd(scan1, scan2)
        R2, T = reg.get_compute()
        scan1 = (np.dot(R2, scan1.T)).T
        reg = registration_iasd(scan1, scan2)
        dic = reg.find_closest_points()
        return sum(correspondence['dist2']**2 for correspondence in dic.values())/len(dic.values())
    
    # define the actions to be performed
    def find_actions(self):
        Act = []
        if np.absolute(self.X[1]-self.X[0]) >45:
            Act.append(-1)
            Act.append(1)

        if np.absolute(self.Y[1]-self.Y[0]) >45:
            Act.append(2)
            Act.insert(0,-2)
            
        if np.absolute(self.Z[1]-self.Z[0]) >45:
            Act.append(3)
            Act.insert(0,-3)

        return Act
    
    def atuar(self, act):
        
        if act > 0:
            if act == 1:
                return State([sum(self.X)/2, self.X[1]], self.Y, self.Z)
            elif act == 2:
                return State(self.X , [sum(self.Y)/2, self.Y[1]] , self.Z)
            elif act == 3:
                return State(self.X , self.Y, [sum(self.Z)/2, self.Z[1]])
        else:
            if act == -1:
                return State( [self.X[0], sum(self.X)/2 ], self.Y, self.Z)
            elif act == -2:
                return State(self.X , [self.Y[0], sum(self.Y)/2] , self.Z)
            elif act == -3:
                return State(self.X , self.Y, [self.Z[0], sum(self.Z)/2])
 
class align_3d_search_problem(search.Problem):

    def __init__(
            self,
            scan1: array((...,3)),
            scan2: array((...,3)),
            ) -> None:
        """Module that instantiate your class.
        You CAN change the content of this __init__ if you want.

        :param scan1: input point cloud from scan 1
        :type scan1: np.array
        :param scan2: input point cloud from scan 2
        :type scan2: np.array
        """

        # Creates an initial state.
        # You may want to change this to something representing
        # your initial state.

        self.initial = State()
        data = np.absolute(scan1)
        ind = np.unravel_index(data.argmax(), data.shape)
        self.scan1 = scan1/data[ind]
        self.scan2 = scan2/data[ind]
       

        self.treshold = 0.01
        self.goalhold = 0.001
        
        return


    def actions(
            self,
            state: State
            ) -> Tuple:
        """Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment

        :param state: Abstract representation of your state
        :type state: State
        :return: Tuple with all possible actions
        :rtype: Tuple
        """

        return state.find_actions()


    def result(
            self,
            state: State,
            action: Action
            ) -> State:
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: Abstract representation of your state
        :type state: [type]
        :param action: An action
        :type action: [type]
        :return: A new state
        :rtype: State
        """
        
        return state.atuar(action)


    def goal_test(
            self,
            state: State
            ) -> bool:
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough.

        :param state: gets as input the state
        :type state: State
        :return: returns true or false, whether it represents a node state or not
        :rtype: bool
        """

        # verify goal node with 4 steps to minimize the computational cost
        
        if state.dist(self.scan1, self.scan2) < self.treshold:
            state.value = None
            if state.dist(self.scan1[0:45,:], self.scan2) < self.treshold:
                if state.goal(self.scan1[0:45,:], self.scan2) < self.goalhold:
                    if state.goal(self.scan1[0:200,:], self.scan2) < self.goalhold:
                        return True

        return False


    def path_cost(
            self,
            c,
            state1: State,
            action: Action,
            state2: State
            ) -> float:
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path.

        :param c: cost to get to the state1
        :type c: [type]
        :param state1: parent node
        :type state1: State
        :param action: action that changes the state from state1 to state2
        :type action: Action
        :param state2: state2
        :type state2: State
        :return: [description]
        :rtype: float
        """

        pass


    def heuristic(
        self,
        node):
        """Returns the heuristic at a specific node.
        note: use node.state to access the state

        :param node: node to include the heuristic
        :return: heuristic value
        :rtype: float
        """
        return node.state.dist(self.scan1[0:20,:],self.scan2)*np.sqrt(node.depth)



def compute_alignment(
        scan1: array((...,3)),
        scan2: array((...,3)),
        ) -> Tuple[bool, array, array, int]:
    """Function that will return the solution.
    You can use any UN-INFORMED SEARCH strategy we study in the
    theoretical classes.

    :param scan1: first scan of size (..., 3)
    :type scan1: array
    :param scan2: second scan of size (..., 3)
    :type scan2: array
    :return: outputs a tuple with: 1) true or false depending on
        whether the method is able to get a solution; 2) rotation parameters
        (numpy array with dimension (3,3)); 3) translation parameters
        (numpy array with dimension (3,)); and 4) the depth of the obtained
        solution in the proposes search tree.
    :rtype: Tuple[bool, array, array, int]
    """
    
    
              
    ## Get the center points
    c1 = np.average(scan1,axis=0)
    c2 = np.average(scan2,axis=0)
    

    problem=align_3d_search_problem(scan1 - c1, scan2 - c2)
    

    sol = search.greedy_best_first_graph_search(problem,problem.heuristic)
    
    if sol is None :
        return (False, np.zeros([3,3]), np.zeros([3]), 0)

    
    # compute the final transformation
    depth = sol.depth
    R = sol.state.rotation()
    t = c2 - np.dot(R, c1)
    TR = np.zeros([4,4])
    TR[0:3,0:3] = R
    TR[0:3,3] = t
    TR[3,3] = 1
    scan_1_ = (np.dot(R, scan1.T) + np.dot(t.reshape((3,1)),np.ones((1,scan1.shape[0])))).T
    reg = registration_iasd(scan_1_, scan2)
    R2, t2 = reg.get_compute()
    TR2 = np.zeros([4,4])
    TR2[0:3,0:3] = R2
    TR2[0:3,3] = t2
    TR2[3,3] = 1
    TR_f = np.dot(TR2, TR)

    return (True, TR_f[0:3,0:3], TR_f[0:3,3], depth)

