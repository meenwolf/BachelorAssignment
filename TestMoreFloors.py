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
PATH_test= PATH+"\\Eerste aantekeningen\\MoreFloors"
PATH_drawings= PATH_test +"\\OriginalPaths"
PATH_empty= PATH_test+"\\Empty Floors"
PATH_result= PATH_test+"\\ResultPaths"

# Initiate some data structures to save information in
nodeToCoordinate={}
coordinateToNode={}
maxVnum={}# keys are maximum numbers, value is a tuple of (building, floor) where this is the maximum vnum of

nextnode=0

specialPaths={}
specialEdges={}
hallways={}

figuresResultBuildings  = dict()

# Loop over the files to save a file where we can draw the resulting longest paths in
for building in os.listdir(PATH_drawings):
    buildingEmpty= PATH_empty+f"\\{building}"
    listOfFiles = os.listdir(buildingEmpty)
    for file in listOfFiles:
        if file.endswith(".svg"):
            floor= file.split('.')[1]
            # if floor=='3':
            #     continue
            newFigurePath = buildingEmpty + f"\\{file}"

            tree = ET.parse(newFigurePath)
            root = tree.getroot()
            ET.register_namespace("", "http://www.w3.org/2000/svg")  # Register SVG namespace
            if building in figuresResultBuildings:
                figuresResultBuildings[building][floor]={'tree':tree, 'root':root}
            else:
                figuresResultBuildings[building]={floor: {'tree': tree, 'root': root}}

            #Now start extracting the path information
            paths, attributes = svg2paths(PATH_drawings + f"\\{building}\\{file}")

            for path, attr in zip(paths, attributes):
                if "inkscape:label" in attr.keys():
                    if "1mtr" in attr['inkscape:label']:
                        print(f"the scale label is: {attr['inkscape:label']}")
                        lengthForOneMeter = abs(path.start - path.end)
                        break

            for path, attr in zip(paths, attributes):
                for i, line in enumerate(path):
                    start = np.round(line.start, 3)
                    end = np.round(line.end, 3)
                    edgeweight = abs(start - end) / lengthForOneMeter  # edgeweight in meters
                    specialPath = False
                    if "inkscape:label" in attr.keys():
                        if 'mtr' in attr['inkscape:label']:
                            continue
                        else:
                            specialPath = True

                    if not (start, building, floor) in coordinateToNode:
                        coordinateToNode[(start, building, floor)] = nextnode
                        nodeToCoordinate[nextnode] = {'Building': building, 'Floor': floor, 'Location': start}
                        snode = nextnode
                        nextnode += 1
                    else:
                        snode = coordinateToNode[(start, building, floor)]

                    if not (end, building, floor) in coordinateToNode:
                        coordinateToNode[(end, building, floor)] = nextnode
                        nodeToCoordinate[nextnode] = {'Building': building, 'Floor': floor, 'Location': end}
                        enode = nextnode
                        nextnode += 1
                    else:
                        enode = coordinateToNode[(end, building, floor)]

                    if specialPath:
                        specialPaths[attr['inkscape:label']] = {'Start': {'Vnum': snode, "Location": start},
                                                                         "End": {"Vnum": enode, "Location": end},
                                                                         'Building': building, 'Floor': floor}

                        specialEdges[(snode, enode)]=attr['inkscape:label']
                        specialEdges[(enode, snode)]=attr['inkscape:label']
                    hallways[(snode, enode)] = edgeweight



print(figuresResultBuildings)
print(f"the special paths are:")
pprint(specialPaths)
print("The hallways are:")
pprint(hallways)

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

neighboursnew=neighbours
#Connect special paths, elevators first:
#CONVENTION CHANGE: use double digits for index elevator, stair, exit, and double digits for the floors! to keep things consistent
# So old version: CRE11 now becomes CRE0101 and CRE25 is now CRE0205, so that in horst, or buildings with more than
# 10 staircases, elevators or exits, the same code can be used
def findSingleEnd(specialedge): #uses the global variables specialPaths and neighbours
    Nstart = len(neighbours[specialPaths[specialedge]["Start"]["Vnum"]])
    Nend = len(neighbours[specialPaths[specialedge]["End"]["Vnum"]])
    if Nstart < Nend:
        end = specialPaths[specialedge]["Start"]["Vnum"]
    else:
        end = specialPaths[specialedge]["End"]["Vnum"]
    return end

connected=[]
for pathname, pathinfo in specialPaths.items():
    if pathname[2]=="E": #We have an edge to an elevator
        startname = pathname[:5]
        if not startname in connected: #If we have not yet connected the floors that can be reached from this elevator, do so
            print(startname)
            elevatorConnects=[key for key in specialPaths.keys() if startname in key]
            for e1, e2 in combinations(elevatorConnects,2):
                print(f"comb {e1} and {e2}")
                end1= findSingleEnd(e1)
                end2= findSingleEnd(e2)
                hallways[(end1, end2)]=1
                neighboursnew[end1].add(end2)
                neighboursnew[end2].add(end1)
            connected.append(startname)
    elif pathname[2] == "S": #We have an edge to a staircase
        if pathname not in connected: # if we have not connected these two floors with this staircase
            end1= findSingleEnd(pathname)
            otherSide= pathname[:5]+pathname[7:]+pathname[5:7]
            if otherSide in specialPaths:
                print(f"original stair:{pathname}, other side:{otherSide}")
                end2=findSingleEnd(otherSide)
                hallways[(end1, end2)] = 7 # 1 stair 7 meter? idkkk
                neighboursnew[end1].add(end2)
                neighboursnew[end2].add(end1)
                connected.append(pathname)
                connected.append(otherSide)
    elif pathname[2] == "C": #connecting buildings? naming conv?
        otherSide = pathname[3:5]+pathname[2]+pathname[0:2]+pathname[5:]
        print(f"connection building from CR:{pathname} to CR:{otherSide}")
    elif pathname[2]=='X':
        print(f"we have an exit to outdoors: {pathname}")
    else:
        print(f"somehting went wrong: {pathname} not a stair, elevator, connection or exit")



print(f"the neighbourhoods are: {neighbours}")
neighbours=neighboursnew
#Now define functions needed to find the longest route.

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

        edgesS = [(v, n) for v in reachableVdum for n in neighbours[v] if n in reachableVdum] + [(n, v) for v in reachableVdum for n in neighbours[v] if n in reachableVdum]
        edgesNotS = [(v, n) for v in notReachableVdum for n in neighbours[v] if n in notReachableVdum] + [(n, v) for v in notReachableVdum for n in neighbours[v] if n in notReachableVdum]
        edgeCutS= [edge for edge in hallways if (edge not in edgesNotS) and (edge not in edgesS)]

        for g in edgesNotS:
            model.cbLazy(
                quicksum([self.x[edge] for edge in edgeCutS])
                >= 2 * (self.x[edgesS[0]] + self.x[g] - 1))
        # if len(edgesNotS) >0:
        #     model.cbLazy(
        #         quicksum([self.x[edge] for edge in edgeCutS])
        #         >= 2)
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
                    elif edge in specialEdges:
                        print(f"We might want to color to which floor we are taking the stairs? or elevator?")
                        color="pink"
                    else:
                        color="purple"

                    startco, endco = getCoordinatesPair(edge)
                    if building0 =="CARRE 1412":
                        if floor0 == '4':
                            startco=startco +126.822+ 494.891j
                            endco=endco +126.822+ 494.891j
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
        buildingResultPath= PATH_result+f"\\{building}"
        for floor, floorinfo in buildinginfo.items():
            buildingName, buildingNumber = splitNameNumber(building)
            floortree= floorinfo['tree']
            print(f"the type of floor tree is: {type(floortree)}\n {floortree}")
            testfilename= f"\\testingMoreFloors3{buildingNumber}.{floor}.svg"
            floortree.write(buildingResultPath+testfilename)


model, varshall, varsdegree = runModel(hallways, vdum)
lengthLongestTrail=model.getAttr('ObjVal')
print(f"The longest trail is {lengthLongestTrail} meters long")
used_edges= getEdgesResult(model, varshall)
pprint(f"The used edges in the solution are:\n{used_edges}")
drawEdgesInFloorplans(used_edges, vdum)