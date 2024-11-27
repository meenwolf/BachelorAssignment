import os

from networkx.classes import neighbors
from svgpathtools import svg2paths, svg2paths2, wsvg, Path, Line
from pprint import pprint
from gurobipy import *
import xml.etree.ElementTree as ET
import numpy as np

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
floor=3# Gurobi uses dictionaries where the keys are pairs of (start, end) having a value
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

# now try out the gurobi libary:

def runModel(halls, nvdum):
    m = Model()

    # Variables: the hallway connecting crossing i and j in the tour?
    varssol = m.addVars(halls.keys(), vtype=GRB.BINARY, name='x')
    for v in m.getVars():
        print(f"v name: {v.VarName} with obj value: {v.X:g} ")
    # Symmetric direction: use dict.update to alias variable with new key
    varssol.update({(j, i): varssol[i, j] for i, j in varssol.keys()})

    # Add variable to help ensure even degree
    varsaux = m.addVars(range(nvdum), vtype=GRB.INTEGER, name="y")

    # Set the objective function for the model
    m.setObjective(sum([halls[e] * varssol[e] for e in halls.keys()]), sense=GRB.MAXIMIZE)
    # Then add constaints, but not yet the connectivity constraint.

    # Add the even degree constraint for dummyvetex nvdum=2:
    m.addConstr(sum([varssol[(nvdum, e)] for e in neighbours[nvdum]]) == 2, name='evenDegreeVDUM')

    for i in range(nvdum):
        m.addConstr(sum([varssol[(i, e)] for e in neighbours[i]]) == 2 * varsaux[i],
                        name=f'evenDegreeVertex{i}')
    # Call optimize to get the solution
    m.optimize()
    return m, varssol, varsaux
model, varshall, varsdegree = runModel(hallways, vdum)
#Retreive final values for the varshall: hallway variables and print them

def getEdgesResult(model, varssol):
    sol = model.getAttr('X', varssol)
    edges = set()
    for key, value in sol.items():
        if value >= 0.5:
            if key[0] < key[1]:
                edges.add(key)
    return edges

used_edges= getEdgesResult(model, varshall)
print("The used edges in the solution are:")
pprint(used_edges)

def getBuildingFloor(vnum):
   return nodeToCoordinate[vnum]['Building'], nodeToCoordinate[vnum]['Floor']

#Translate vertex pairs back to coordinates:

def getCoordinatesPair(vtuple):
    return nodeToCoordinate[vtuple[0]]["Location"], nodeToCoordinate[vtuple[1]]["Location"]

testPath= PATH+("\\Eerste aantekeningen\\CARRE 1412")
newFigurePath = testPath+"\\empty1412.3.svg"

tree = ET.parse(newFigurePath)
tree.write(testPath+"\\backupempty1412.3.svg")

root = tree.getroot()
print(f"the root of the empty svg file with only the floorplan image: {root}")


ET.register_namespace("", "http://www.w3.org/2000/svg")  # Register SVG namespace
for edge in used_edges:
    if vdum not in edge:
        startco, endco= getCoordinatesPair(edge)
        new_path_element = ET.Element("path", attrib={
            "d": Path(Line(start=startco, end=endco)).d(),
            "stroke": "purple",
            "fill": "none",
            "stroke-width": "2"
        })
        root.append(new_path_element)

tree.write(testPath+"\\longestCyclesWithVdumDisconnectedModelFunc1412.3.svg")
print(f"New result is in newly created file{testPath+'\\longestCyclesWithVdumDisconnectedModelFunc1412.3.svg'}")
pprint(f"We have  {vdum} nodes without the dummy vertex")
