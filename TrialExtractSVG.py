import os
from os.path import split

from networkx.classes import neighbors
from svgpathtools import svg2paths, svg2paths2, wsvg, Path, Line
from pprint import pprint
from gurobipy import *
import xml.etree.ElementTree as ET
import numpy as np
import logging
from collections import defaultdict
from itertools import combinations
import re


# Get the path to the drawings
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH= os.path.abspath(os.path.join(dir_path, os.pardir))
PATH_drawings= PATH+"\\Eerste aantekeningen"
#Get the paths and attributes of the third floor of Carré

#IDEA: make a list of nodes, for all of the path starts and ends. After we also know the scale and made the
#Edges have the correct real world weights, we can start at merging nodes in one, if they are closer than X meters in real life.

paths, attributes = svg2paths(PATH_drawings+"\\CARRE 1412\\paths1412.3.svg")

for path, attr in zip(paths, attributes):
    if "inkscape:label" in attr.keys():
        if attr['inkscape:label'] == "CR31mtr":
            print(f"the scale label is: {attr['inkscape:label']}")
            lengthForOneMeter= abs(path.start - path.end)
            break


print(f"length for one meter is {lengthForOneMeter}")

# code to check in what order we loop over dict items, but that is in order of creation.
# Hence, when adding the maxvnum per floor per building, in ascending order, looping over them
# gives the lowest keys first. Use this functionality to find the building a node belongs to based on nodenumber

# for i in range(10,0, -1):
#     maxVnu[i]=100+1
#
# for key, value in maxVnu.items():
#     print(f"key is: {key}, with value: {value}")

nodeToCoordinate={}
coordinateToNode={}
maxVnum={}# keys are maximum numbers, value is a tuple of (building, floor) where this is the maximum vnum of

nextnode=0
building="CARRE 1412"
floor=str(3)# Gurobi uses dictionaries where the keys are pairs of (start, end) having a value
# of the distance between them. To keep the information about the labels, I will create
# a separate dictionary, having keys (start, end) and value the label name I assigned to it
# then its easy to check if special actions are required.

specialPaths={}
hallways={}

for path, attr in zip(paths, attributes):
    for i, line in enumerate(path):
        start = np.round(line.start,3)
        end = np.round(line.end,3)
        edgeweight = abs(start - end)/lengthForOneMeter #edgeweight in meters
        specialPath=False
        if "inkscape:label" in attr.keys():
            if 'mtr' in attr['inkscape:label']:
                continue
            else:
                specialPath=True


        if not (start, building, floor) in coordinateToNode:
            coordinateToNode[(start, building, floor)]=nextnode
            nodeToCoordinate[nextnode]={'Building': building, 'Floor': floor, 'Location':start}
            snode=nextnode
            nextnode+=1
        else:
            snode= coordinateToNode[(start, building, floor)]

        if not (end, building, floor) in coordinateToNode:
            coordinateToNode[(end, building, floor)]=nextnode
            nodeToCoordinate[nextnode]={'Building': building, 'Floor': floor, 'Location':end}
            enode=nextnode
            nextnode+=1
        else:
            enode= coordinateToNode[(end, building, floor)]

        if specialPath:
            specialPaths[attr['inkscape:label'] + str(i)] = {'Start':{'Vnum': snode, "Location": start}, "End":{"Vnum":enode, "Location":end}, 'Building': building, 'Floor': floor}

        hallways[(snode, enode)] = edgeweight


maxVnum[nextnode-1]=(building, floor)

print(f"the special paths are:")
pprint(specialPaths)
print("The hallways are:")
pprint(hallways)

print(f"we had a total of {nextnode} crossings for carre 3rd floor, matches reality of 63: {nextnode==63}?\n In"
      f"before adding dummy vertex and edges")


# Add hallways to the dummy vertex:
vdum= nextnode
nextnode += 1

for i in range(nextnode):
    hallways[(vdum, i)]=0
# define a dictionary where each vertex maps to a set of its neighbours
neighbours={i:{vdum} for i in range(vdum)}
neighbours[vdum]=set(range(vdum))
for v1,v2 in hallways.keys():
    neighbours[v1].add(v2)
    neighbours[v2].add(v1)

print(f"the neighbourhoods are: {neighbours}")

# Now try out the gurobi libary:

def getReachable(neighborhoods, start, reachable=None):
    if reachable==None:
        reachable={start}
    for neighbor in neighborhoods[start]:
        if not neighbor in reachable:
            reachable.add(neighbor)
            reachable= getReachable(neighborhoods,neighbor, reachable)
    return reachable

def vdum_reachable(edges, vdum):
    # Create a mapping from each node to its neighbours in the feasible solution
    node_neighbors = defaultdict(list)
    for i, j in edges:
        node_neighbors[i].append(j)
        node_neighbors[j].append(i)

    assert all(len(neighbors) %2 ==0 for neighbors in node_neighbors.values())#assert even degree of the vertices in the solution
    assert vdum in node_neighbors #assert vdum being a vertex visited in the solution

    reachableFromVdum= getReachable(node_neighbors, vdum)
    notReachable= [key for key in node_neighbors.keys() if key not in reachableFromVdum]
    return reachableFromVdum, notReachable

# I literally copied this TSPCallBack class from :https://docs.gurobi.com/projects/examples/en/current/examples/python/tsp.html#subsubsectiontsp-py
# To see if it works. I have constructed my own subtour elimination constraint, but I wanted to see what the original structure in the example  does first.
# I then started to play around and understand the structure, and adjust it where needed to fit my problem, of finding a longest trail visiting vdum exactly once
class TSPCallback:
    """Callback class implementing lazy constraints for the TSP.  At MIPSOL
    callbacks, solutions are checked for disconnected subtours and then add subtour elimination
    constraints if needed/ the solution contains disconneced subtours."""

    def __init__(self, nodes, x):
        self.nodes = nodes
        self.x = x

    def __call__(self, model, where):
        """Callback entry point: call lazy constraints routine when new
        solutions are found. Stop the optimization if there is an exception in
        user code."""
        if where == GRB.Callback.MIPSOL:
            try:
                self.eliminate_subtours(model)
            except Exception:
                logging.exception("Exception occurred in MIPSOL callback")
                model.terminate()

    def eliminate_subtours(self, model):
        """Extract the current solution, find the vertices reachable starting from vdum in the solution
        and define the edge cut of these vertices in the original graph. then formulate lazy
        constraints to cut off the current solution if disconnected vertices are found.
        Assumes we are at MIPSOL."""
        values = model.cbGetSolution(self.x)
        edges = [(i, j) for (i, j), v in values.items() if v > 0.5]
        reachableVdum, notReachableVdum = vdum_reachable(edges, len(self.nodes)-1)
        print(f"reachable from vdum:{reachableVdum}\n not reachable from vdum:{notReachableVdum}")

        edgesS = [(v, n) for v in reachableVdum for n in neighbours[v] if n in reachableVdum] + [(n, v) for v in reachableVdum for n in neighbours[v] if n in reachableVdum]
        edgesNotS = [(v, n) for v in notReachableVdum for n in neighbours[v] if n in notReachableVdum] + [(n, v) for v in notReachableVdum for n in neighbours[v] if n in notReachableVdum]
        edgeCutS= [edge for edge in hallways if (edge not in edgesNotS) and (edge not in edgesS)]

        for f in edgesS:
            for g in edgesNotS:
                model.cbLazy(
                    quicksum([self.x[edge] for edge in edgeCutS])
                    >= 2*(self.x[f]+self.x[g]-1))

def runModel(halls, nvdum):
    m = Model()

    # Variables: the hallway connecting crossing i and j in the tour?
    varssol = m.addVars(halls.keys(), vtype=GRB.BINARY, name='x')

    # Symmetric direction: use dict.update to alias variable with new key
    varssol.update({(j, i): varssol[i, j] for i, j in varssol.keys()})

    # Add auxiliary variable to help ensure even degree
    varsaux = m.addVars(range(nvdum), vtype=GRB.INTEGER, name="y")

    # Set the objective function for the model
    m.setObjective(sum([halls[e] * varssol[e] for e in halls.keys()]), sense=GRB.MAXIMIZE)

    # Add the even degree constraint for dummyvetex nvdum=2:
    m.addConstr(sum([varssol[(nvdum, e)] for e in neighbours[nvdum]]) == 2, name='evenDegreeVDUM')

    # Add the even degree constraint for the other vertices
    for i in range(nvdum):
        m.addConstr(sum([varssol[(i, e)] for e in neighbours[i]]) == 2 * varsaux[i],
                        name=f'evenDegreeVertex{i}')

    # Set up for the callbacks/ lazy constraints for connectivity
    m.Params.LazyConstraints = 1
    cb = TSPCallback(range(vdum+1), varssol)

    # Call optimize with the callback structure to get a connected solution solution
    m.optimize(cb)
    return m, varssol, varsaux


#Retreive final values for the varshall: hallway variables and print them
def getEdgesResult(model, varssol):
    sol = model.getAttr('X', varssol)
    edges = set()
    for key, value in sol.items():
        if value >= 0.5:
            if key[0] < key[1]:
                edges.add(key)
    return edges

#Define a function returning the building and floor a vertex vnum lies in
def getBuildingFloor(vnum):
   return nodeToCoordinate[vnum]['Building'], nodeToCoordinate[vnum]['Floor']

#Translate a vertex pair representing an edge, back to coordinates:
def getCoordinatesPair(vtuple):
    return nodeToCoordinate[vtuple[0]]["Location"], nodeToCoordinate[vtuple[1]]["Location"]

#Get the building name and number from the a building name containing both of them with a space in between.
def splitNameNumber(buildingNo):
    res = buildingNo.split()
    print(f"buildingnumber:{buildingNo}, res:{res}, elem 0: {res[0]}, elem 1:{res[1]}")
    return res[0], res[1]

def drawEdgesInFloorplans(edges, vdum):
    #first get the start and end of the trail
    startend = []
    for i, j in edges:
        if j == vdum:
            startend.append(i)

    testPath = PATH + ("\\Eerste aantekeningen\\TestLoopsBuildings")

    #Create the dictionary where the information for the floor plans for the result are stored
    figuresResultBuildings  = dict()
    for building in os.listdir(testPath):
        buildingTestPath= testPath+f"\\{building}"
        listOfFiles = os.listdir(buildingTestPath)
        for file in listOfFiles:
            if file.endswith(".svg"):
                floor= file.split('.')[1]
                newFigurePath = buildingTestPath + f"\\{file}"

                tree = ET.parse(newFigurePath)
                root = tree.getroot()
                ET.register_namespace("", "http://www.w3.org/2000/svg")  # Register SVG namespace

                figuresResultBuildings[building]={floor: {'tree': tree, 'root': root}}

    # Add the paths to the roots
    for edge in edges:
        if vdum not in edge:
            building0, floor0= getBuildingFloor(edge[0])
            building1, floor1= getBuildingFloor(edge[1])
            #Now check if the buildings are the same, if not, we use a walking bridge/connection hallway
            if building0==building1:
                #Check if we are on the same floor, if not, we take a staircase or elevator.
                if floor0==floor1:
                    #Now we can draw the lines in the floor plan.
                    if startend[0] in edge:
                        color="red"
                    elif startend[1] in edge:
                        color="green"
                    else:
                        color="purple"
                    startco, endco = getCoordinatesPair(edge)
                    new_path_element = ET.Element("path", attrib={
                        "d": Path(Line(start=startco, end=endco)).d(),
                        "stroke": color,
                        "fill": "none",
                        "stroke-width": "2"
                    })
                    thisRoot= figuresResultBuildings[building0][floor0]['root']
                    thisRoot.append(new_path_element)

                    #Maybe different colorings if we take an elevator, stair up, stair down, another building?
                elif floor0<floor1:
                    print(f"We take stairs up")
                else:
                    print(f"We take stairs down")
            else:
                print(f"We go from {building0} to {building1}")

    # Draw the figures in a new file:
    for building, buildinginfo in figuresResultBuildings.items():
        buildingTestPath= testPath+f"\\{building}"
        for floor, floorinfo in buildinginfo.items():
            buildingName, buildingNumber = splitNameNumber(building)
            floortree= floorinfo['tree']
            testfilename= f"\\testingLoop{buildingNumber}.{floor}.svg"
            floortree.write(buildingTestPath+testfilename)

model, varshall, varsdegree = runModel(hallways, vdum)
lengthLongestTrail=model.getAttr('ObjVal')
print(f"The longest trail is {lengthLongestTrail} meters long")
used_edges= getEdgesResult(model, varshall)
pprint(f"The used edges in the solution are:\n{used_edges}")
drawEdgesInFloorplans(used_edges, vdum)
