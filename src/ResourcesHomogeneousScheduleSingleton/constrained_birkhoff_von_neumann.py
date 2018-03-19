"""
=======================================
constrained_birkhoff_von_neumann.py 
=======================================
Decomposes a matrix into a weighted sum of basis matrices with binary entries satisfying user imposed constraints. 
When the starting matrix is doubly stochastic and the basis elements are required to be permutation matrices, this is the classical Birkhoff von-Neumann decomposition.
Here we implement the algorithm identified in Budish, Che, Kojima, and Milgrom (2013). 
The constraints must form what they call a bihierarchy.
The user will be informed if the proposed constraint structure is not a bihierarchy.

Copyright 2017 Aubrey Clark.
(e.g. the value (1,1) means that the coordinates that this value's key represents sum to exactly one in each of the basis matrices.)

X = np.array([[.5, .2,.3], [.5,.5, 0], [.8, 0, .2], [.2, .3, .5]])
constraint_structure = {frozenset({(0, 0), (0, 1), (0,2)}): (1,1), frozenset({(1, 0), (1, 1), (1,2)}):(1,1), frozenset({(2, 0), (2, 1), (2,2)}):(1,1), frozenset({(3, 0), (3, 1), (3,2)}):(1,1), frozenset({(0, 0), (1, 0), (2,0), (3,0)}):(1,2),  frozenset({(0, 1), (1, 1), (2,1), (3,1)}):(1,1), frozenset({(0, 2), (1, 2), (2,2), (3,2)}):(1,1), frozenset({(0, 0), (1, 0)}):(1,1)}
"""

__author__ = "Aubrey Clark, Sailik Sengupta"

#: The current version of this package.
__version__ = '0.0.1-dev'

import networkx as nx
import numpy as np
import copy
import itertools
import math
import sys
from pprint import pprint

#global things
tolerance = np.finfo(np.float).eps*10e10

#feasibity_test tests whether all entries of the target matrix X are in [0,1].
#Essential since each basis matrix has entries that are either zero or one.
def feasibility_test(X, constraint_structure):
  S = {index for index, x in np.ndenumerate(X)}
  if any(X[i]<0 or X[i]>1 for i in S):
    print("matrix entries must be between zero and one")
  for key, value in constraint_structure.items():
    if sum([X[i] for i in key]) < value[0] or sum([X[i] for i in key]) > value[1]:
      print("matrix entries must respect constraint structure capacities")
 
#bihierarchy_test attempts to decompose the constraint structure into a bihierarchy.
#The user is informed if this is not possible.
#Unfortunately, its success depends on the order of the constraint structure sets.
#So, it must consider all permutations of these sets when the constraint structure is not a bihierarchy.
def bihierarchy_test(constraint_structure):
  constraint_sets = []
  for key, value in constraint_structure.items():
    constraint_sets.append(set(key))
  permutations =  itertools.permutations(constraint_sets)
  for constraint_set_ordering in permutations:
    listofA, listofB = [], []
    for idx, x in enumerate(constraint_set_ordering):
      if all( x < y or y < x or x.isdisjoint(y) for y in [constraint_set_ordering[i] for i in listofA]):
        target = listofA
      elif all(x < y or y < x or x.isdisjoint(y) for y in [constraint_set_ordering[i] for i in listofB]):
        target = listofB
      else:
        break
      target.append(idx)
    if len(listofA) + len(listofB) == len(constraint_sets):
      return [[constraint_set_ordering[i] for i in listofA], [constraint_set_ordering[i] for i in listofB]]
  print("this constraint structure is not a bihierarchy")

#graph_constructor takes a target matrix X and a bihierarchy = [A,B],
#constructs a directed weighted graph G
def graph_constructor(X,bihierarchy,constraint_structure):
  S = {index for index, x in np.ndenumerate(X)}
  A, B = bihierarchy
  A.append(S), B.append(S)
  for x in S:
    A.append({x}), B.append({x})
  for x in S:
    constraint_structure.update({frozenset({x}):(0,1)})
  R1 = nx.DiGraph()
  for x in A:
    for y in A:
      if x < y and not any(x < z < y for z in A):
        R1.add_edge(frozenset(y),frozenset(x),weight=sum([X[i] for i in x]), min_capacity = constraint_structure[frozenset(x)][0], max_capacity = constraint_structure[frozenset(x)][1])
  R2 = nx.DiGraph()
  for x in B:
    for y in B:
      if y < x and not any(y < z < x for z in B):
        R2.add_edge((frozenset(y),'p'),(frozenset(x),'p'),weight = sum( [X[i] for i in y]), min_capacity = constraint_structure[frozenset(y)][0], max_capacity = constraint_structure[frozenset(y)][1])
  G=nx.compose(R1,R2) 
  for index, x in np.ndenumerate(X):
    G.add_edge(frozenset({index}), (frozenset({index}),'p'), weight=x, min_capacity = 0, max_capacity = 1)
  return(G)

#constrained_birkhoff_von_neumann_iterator is the main step.
#After target matrix X and constraint structure have been represented as a weighted directed graph G, 
#this function takes as input a list H = [(G,p)] (where p is a probability, initially one) 
#and decomposes the graph into two graphs, each with an associated probability, 
#and each of which are closer to representing a basis matrix. Seqential iteration, 
#done in the main function constrained_birkhoff_von_neumann_decomposition, leads to the decomposition.
def constrained_birkhoff_von_neumann_iterator(H, X):
  (G, p) = H.pop(0)
  #remove edges with integer weights
  #extracts all edges satisfy the weight threshold:
  eligible_edges = [(from_node,to_node,edge_attributes) for from_node,to_node,edge_attributes in G.edges(data=True) if all(i < edge_attributes['weight'] or edge_attributes['weight'] < i for i in range(0,int(math.floor(sum(sum(X)))+1)))]
  if not eligible_edges:
    return(H)
  else:
    K = nx.DiGraph()
    K.add_edges_from(eligible_edges)
  #find a cycle and compute the push_forward and push_reverse probabilities and graphs
  cycle = nx.find_cycle(K, orientation='ignore')
  forward_weights = [(d['weight'],d['min_capacity'],d['max_capacity']) for (u,v,d) in K.edges(data=True) if (u,v,'forward') in cycle]
  reverse_weights = [(d['weight'],d['min_capacity'],d['max_capacity']) for (u,v,d) in K.edges(data=True) if (u,v,'reverse') in cycle]
  push_forward = min([x[2] - x[0] for x in forward_weights])
  push_reverse = min([x[2] - x[0] for x in reverse_weights])
  pull_forward = min([x[0] - x[1] for x in forward_weights])
  pull_reverse = min([x[0] - x[1] for x in reverse_weights])
  push_forward_pull_reverse = min(push_forward,pull_reverse)
  push_reverse_pull_forward = min(pull_forward,push_reverse)
  #Construct the push_forward_pull_reverse graph
  G1 = copy.deepcopy(G)
  for (u,v,d) in G1.edges(data=True):
    if (u,v,'forward') in cycle:
      d['weight']+=push_forward_pull_reverse
    if (u,v,'reverse') in cycle:
      d['weight']+=-push_forward_pull_reverse
  #Construct the push_reverse_pull_forward graph
  G2 = copy.deepcopy(G)
  for (u,v,d) in G2.edges(data=True):
    if (u,v,'reverse') in cycle:
      d['weight']+=push_reverse_pull_forward
    if (u,v,'forward') in cycle:
      d['weight']+=-push_reverse_pull_forward
  gamma = min([1,max([0,push_reverse_pull_forward/(push_forward_pull_reverse + push_reverse_pull_forward)])])
  return([(G1,p*gamma), (G2,p*(1-gamma))])

#iterate_constrained_birkhoff_von_neumann_iterator iterates constrained_birkhoff_von_neumann_iterator, 
#initially on the weighted directed graph (and probability) [(G,1)] where G is given by graph_constructor, 
#and then on its children, until the terminal nodes of the tree, which each represents a basis matrix (modulo tolerance)
def iterate_constrained_birkhoff_von_neumann_iterator(X, G):
  S = {index for index, x in np.ndenumerate(X)}
  H=[(G,1)]
  solution=[]
  while len(H) > 0:
    if any(tolerance < x < 1-tolerance for x in [d['weight'] for (u,v,d) in H[0][0].edges(data=True) if u in [frozenset({x}) for x in S]]):
      H.extend(constrained_birkhoff_von_neumann_iterator([H.pop(0)], X))
    else:
      solution.append(H.pop(0))
  return(solution)

# solution_cleaner takes the solution, which comes in the form of a collection of weighted and directed graphs and 
# probabilities. The central column of each graph corresponds to a basis matrix and the probability attached to the graph 
# corresponds to the probability assigned to that basis matrix. solution_cleaner rounds the entries of the basis matrices 
# according to tolerance so that each entry is either zero or one, merges duplicate basis matrices, and then converts the 
# solution to a list whose first entry is the distribution over basis matrices, whose second entry is the list of basis 
# matrices, and whose third and fourth entry are checks that the coefficients sum to one and that the average of the basis 
# matrices is indeed the target matrix X 
def solution_cleaner(X, solution):
  S = {index for index, x in np.ndenumerate(X)}
  solution_columns_and_probs = []
  for y in solution:
    solution_columns_and_probs.append([[(u,d['weight']) for (u,v,d) in y[0].edges(data=True) if u in [frozenset({x}) for x in S]],y[1]])
  solution_zeroed = []
  for z in solution_columns_and_probs:
    list = []
    for y in z[0]:
      if y[1] < tolerance:
        list.append((y[0],0))
      elif y[1] > 1-tolerance:
        list.append((y[0],1))
    solution_zeroed.append([list,z[1]])
  list = []
  for idx, x in enumerate(solution_zeroed):
    if all(x[0]!= z[0] for z in [solution_zeroed[i] for i in list]):
      list.append(idx)
  solution_simplified = []
  for i in list:
    solution_simplified.append([solution_zeroed[i][0],sum([x[1] for x in solution_zeroed if x[0]==solution_zeroed[i][0]])])
  assignments = []
  coefficients = []
  for a in solution_simplified:
    Y = np.zeros(X.shape)
    for x in a[0]:
      for y in x[0]:
        Y[y]=x[1]
    assignments.append(Y)
    coefficients.append(a[1])
  return([coefficients, assignments, sum(coefficients), sum(i[1]*i[0] for i in zip(coefficients, assignments))])

#constrained_birkhoff_von_neumann_decomposition puts the pieces together
def constrained_birkhoff_von_neumann_decomposition(X,constraint_structure):
  S = {index for index, x in np.ndenumerate(X)}
  feasibility_test(X,constraint_structure)
  #print(solution_cleaner(X, iterate_constrained_birkhoff_von_neumann_iterator(X, graph_constructor(X, bihierarchy_test(constraint_structure), constraint_structure ) ) ) ) 
  return(solution_cleaner(X, iterate_constrained_birkhoff_von_neumann_iterator(X, graph_constructor(X, bihierarchy_test(constraint_structure), constraint_structure ) ) ) ) 

