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
#
# for path, attr in zip(paths, attributes):
#     if "inkscape:label" in attr.keys():
#         if 'path' not in attr['inkscape:label']:
#             start= path.start
#             end= path.end
#             edgeweight= abs(start-end)
#         else:
#             print(f"We renamed this path to {attr['inkscape:label']}, because once a name\n was given, but it was wrong and forgot about the orignal path number")


# code to check in what order we loop over dict items, but that is in order of creation.
# Hence, when adding the maxvnum per floor per building, in ascending order, looping over them
# gives the lowest keys first. Use this functionality to find the building a node belongs to based on nodenumber

# for i in range(10,0, -1):
#     maxVnu[i]=100+1
#
# for key, value in maxVnu.items():
#     print(f"key is: {key}, with value: {value}")

newNodeToCoordinate={}
newCoordinateToNode={}
maxVnum={}# keys are maximum numbers, value is a tuple of (building, floor) where this is the maximum vnum of

nextnode=0
building="CARRE 1412"
floor=3# Gurobi uses dictionaries where the keys are pairs of (start, end) having a value
# of the distance between them. To keep the information about the labels, I will create
# a separate dictionary, having keys (start, end) and value the label name I assigned to it
# then its easy to check if special actions are required.

specialPaths={building:{floor:{}}}
hallways={}
hallwaysnew={}

coordinateToNode={building:{floor:{}}}
nodeToCoordinate={building:{floor:{}}}

coordinateToNode[building]['minvnum']=nextnode
coordinateToNode[building][floor]['minvnum']=nextnode

for path, attr in zip(paths, attributes):
    for i, line in enumerate(path):
        start = np.round(line.start,3)
        end = np.round(line.end,3)
        edgeweight = abs(start - end)/lengthForOneMeter #edgeweight in meters
        if "inkscape:label" in attr.keys():
            if 'mtr' in attr['inkscape:label']:
                continue
            else:
                specialPaths[building][floor][(start, end)]=attr['inkscape:label']+str(i)


        if not start in coordinateToNode[building][floor]:
            coordinateToNode[building][floor][start]=nextnode
            nodeToCoordinate[building][floor][nextnode]=start
            newCoordinateToNode[(start, building, floor)]=nextnode
            newNodeToCoordinate[nextnode]={'Building': building, 'Floor': floor, 'Location':start}
            snode=nextnode
            snodenew=nextnode
            nextnode+=1
        else:
            snode=coordinateToNode[building][floor][start]
            snodenew= newCoordinateToNode[(start, building, floor)]

        if not end in coordinateToNode[building][floor]:
            coordinateToNode[building][floor][end]=nextnode
            nodeToCoordinate[building][floor][nextnode]=end
            newCoordinateToNode[(end, building, floor)]=nextnode
            newNodeToCoordinate[nextnode]={'Building': building, 'Floor': floor, 'Location':end}
            enode=nextnode
            enodenew=nextnode
            nextnode+=1
        else:
            enode=coordinateToNode[building][floor][end]
            enodenew= newCoordinateToNode[(end, building, floor)]
        hallways[(snode, enode)] = edgeweight
        hallwaysnew[(snodenew, enodenew)]=edgeweight

coordinateToNode[building]['maxvnum']= nextnode-1
nodeToCoordinate[building]['maxvnum']= nextnode-1

coordinateToNode[building][floor]['maxvnum']= nextnode-1
nodeToCoordinate[building][floor]['maxvnum']= nextnode-1

maxVnum[nextnode-1]=(building, floor)

print(f"the special paths are:")
pprint(specialPaths)
print(f"And the original hallways are equal to the new hallways: {hallways==hallwaysnew}")
# pprint(hallways)

print(f"we had a total of {nextnode} crossings for carre 3rd floor. Does this match reality of 63??\n In"
      f"other words, did snapping work correctly?")


# Add hallways to the dummy vertex:
vdum= nextnode
nextnode += 1

for i in range(nextnode):
    hallways[(vdum, i)]=0
    hallwaysnew[(vdum, i)]=0
# define a dictionary where each vertex maps to a set of its neighbours
neighbours={i:{vdum} for i in range(vdum)}
neighbours[vdum]=set(range(vdum))
for v1,v2 in hallways.keys():
    neighbours[v1].add(v2)
    neighbours[v2].add(v1)

neighboursnew = {i: {vdum} for i in range(vdum)}
neighboursnew[vdum] = set(range(vdum))
for v1, v2 in hallwaysnew.keys():
    neighboursnew[v1].add(v2)
    neighboursnew[v2].add(v1)

print(f"the old and new neighbourhoods are equal: {neighbours==neighboursnew}")

# now try out the gurobi libary:
m = Model()

# Variables: the hallway connecting crossing i and j in the tour?
varshall = m.addVars(hallways.keys(),vtype=GRB.BINARY, name='x')
for v in m.getVars():
    print(f"v name: {v.VarName} with obj value: {v.X:g} ")
# Symmetric direction: use dict.update to alias variable with new key
varshall.update({(j,i):varshall[i,j] for i,j in varshall.keys()})

# Add variable to help ensure even degree
varsdegree=m.addVars(range(vdum),vtype=GRB.INTEGER, name="y")

#Set the objective function for the model
m.setObjective(sum([hallways[e]*varshall[e] for e in hallways.keys()]),sense=GRB.MAXIMIZE)
# Then add constaints, but not yet the connectivity constraint.

#Add the even degree constraint for vdum=2:

for i in range(nextnode):
    if i == vdum:
        m.addConstr(sum([varshall[(i,e)] for e in neighbours[i]])==2, name='evenDegreeVDUM')
    else:
        m.addConstr(sum([varshall[(i,e)] for e in neighbours[i]])==2*varsdegree[i], name=f'evenDegreeVertex{i}')
#Call optimize to get the solution
m.optimize()

# now try out the gurobi libary:
mnew = Model()

# Variables: the hallway connecting crossing i and j in the tour?
varshallnew = mnew.addVars(hallwaysnew.keys(),vtype=GRB.BINARY, name='x')
for v in mnew.getVars():
    print(f"v name: {v.VarName} with obj value: {v.X:g} ")
# Symmetric direction: use dict.update to alias variable with new key
varshallnew.update({(j,i):varshallnew[i,j] for i,j in varshallnew.keys()})

# Add variable to help ensure even degree
varsdegreenew=mnew.addVars(range(vdum),vtype=GRB.INTEGER, name="y")

#Set the objective function for the model
mnew.setObjective(sum([hallwaysnew[e]*varshallnew[e] for e in hallwaysnew.keys()]),sense=GRB.MAXIMIZE)
# Then add constaints, but not yet the connectivity constraint.

#Add the even degree constraint for vdum=2:

for i in range(nextnode):
    if i == vdum:
        mnew.addConstr(sum([varshallnew[(i,e)] for e in neighboursnew[i]])==2, name='evenDegreeVDUM')
    else:
        mnew.addConstr(sum([varshallnew[(i,e)] for e in neighboursnew[i]])==2*varsdegreenew[i], name=f'evenDegreeVertex{i}')

#Call optimize to get the solution
mnew.optimize()

#Retreive final values for the varshall: hallway variables and print them
solution = m.getAttr('X', varshall)
# pprint(solution)
used_edges=set()
for key, value in solution.items():
    if value >= 0.5:
        if key[0] < key[1]:
            used_edges.add(key)

def getEdgesResult(model):
    sol = m.getAttr('X', varshall)
    edges = set()
    for key, value in sol.items():
        if value >= 0.5:
            if key[0] < key[1]:
                edges.add(key)
    return edges

newSolution= getEdgesResult(mnew)
print(f"The old solution equals the new solution: {newSolution==used_edges}")
pprint(newSolution)
pprint(used_edges)
def getDrawingInfoNew(vnum):
    for key, value in maxVnum.items():
        if key >= vnum:
            return maxVnum[key]


# Get the building and floor for which the vertex belongs to
def GetDrawingInfo(vnumber):
    for building, buildinginfo in coordinateToNode.items():
        if buildinginfo['minvnum']<= vnumber:
            if buildinginfo['maxvnum']>=vnumber:
                for floor, floorinfo in buildinginfo.items():
                    if floor not in ['minvnum','maxvnum']:
                        if floorinfo['minvnum']<=vnumber:
                            if floorinfo['maxvnum']>=vnumber:
                                return (building, floor)

print(f"The old way of getting drawing building and floor equals the new one: {getDrawingInfoNew(3)==GetDrawingInfo(3)} : with new: {getDrawingInfoNew(3)} and old:{GetDrawingInfo(3)}")

#Translate vertex pairs back to coordinates:
def GetCoordinatesPair(vtuple):
    # print(f"we get coordinates for{vtuple}")
    b0, f0 = GetDrawingInfo(vtuple[0])
    b1, f1 = GetDrawingInfo(vtuple[1])
    # print(f"we have b0:{b0}, f0:{f0}, b1:{b1}, f1:{f1}")
    return (nodeToCoordinate[b0][f0][np.round(vtuple[0],3)], nodeToCoordinate[b1][f1][np.round(vtuple[1],3)])

def getCoordinatesPairNew(vtuple):
    return (newNodeToCoordinate[vtuple[0]]["Location"], newNodeToCoordinate[vtuple[1]]["Location"])

print(f"The old and new way of getting the coordinates pairs are equal:{GetCoordinatesPair((0,1))==getCoordinatesPairNew((0,1))}")

testPath= PATH+("\\Eerste aantekeningen\\CARRE 1412")
newFigurePath = testPath+"\\empty1412.3.svg"

tree = ET.parse(newFigurePath)
tree.write(testPath+"\\backupempty1412.3.svg")

root = tree.getroot()
print(f"the root of the empty svg file with only the floorplan image: {root}")


ET.register_namespace("", "http://www.w3.org/2000/svg")  # Register SVG namespace
for edge in newSolution:
    if vdum not in edge:
        startco, endco= getCoordinatesPairNew(edge)
        new_path_element = ET.Element("path", attrib={
            "d": Path(Line(start=startco, end=endco)).d(),
            "stroke": "green",
            "fill": "none",
            "stroke-width": "2"
        })
        root.append(new_path_element)

tree.write(testPath+"\\newtestresults1412.3.svg")
print(f"New result is in newly created file{testPath+'\\newtestresults1412.3.svg'}")
pprint(f"We have  {vdum} nodes without the dummy vertex")
#WE MUST HAVE 63 excluding or 65 nodes including the scale
# print(np.sort_complex([coordinate for coordinate in coordinateToNode[building][floor].keys() if coordinate not in ['minvnum','maxvnum']]))