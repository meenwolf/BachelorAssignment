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

# Gurobi uses dictionaries where the keys are pairs of (start, end) having a value
# of the distance between them. To keep the information about the labels, I will create
# a separate dictionary, having keys (start, end) and value the label name I assigned to it
# then its easy to check if special actions are required.

specialPaths={"CR":{3:{}}}
hallways={}
coordinateToNode={"CR":{3:{}}}
nodeToCoordinate={"CR":{3:{}}}
nextnode=0
coordinateToNode['CR']['minvnum']=nextnode
coordinateToNode['CR'][3]['minvnum']=nextnode
for path, attr in zip(paths, attributes):
    for i, line in enumerate(path):
        start = np.round(line.start,3)
        end = np.round(line.end,3)
        edgeweight = abs(start - end)/lengthForOneMeter #edgeweight in meters
        if "inkscape:label" in attr.keys():
            if 'mtr' in attr['inkscape:label']:
                continue
            else:
                specialPaths["CR"][3][(start, end)]=attr['inkscape:label']+str(i)


        if not start in coordinateToNode["CR"][3]:
            coordinateToNode["CR"][3][start]=nextnode
            nodeToCoordinate["CR"][3][nextnode]=start
            snode=nextnode
            nextnode+=1
        else:
            snode=coordinateToNode["CR"][3][start]

        if not end in coordinateToNode["CR"][3]:
            coordinateToNode["CR"][3][end]=nextnode
            nodeToCoordinate["CR"][3][nextnode]=end
            enode=nextnode
            nextnode+=1
        else:
            enode=coordinateToNode["CR"][3][end]
        hallways[(snode, enode)] = edgeweight
coordinateToNode['CR']['maxvnum']= nextnode-1
nodeToCoordinate['CR']['maxvnum']= nextnode-1

coordinateToNode['CR'][3]['maxvnum']= nextnode-1
nodeToCoordinate['CR'][3]['maxvnum']= nextnode-1
print(f"the special paths are:")
pprint(specialPaths)
print("And all the original hallways are:")
pprint(hallways)
print(f"we had a total of {nextnode} crossings for carre 3rd floor. Does this match reality?\n In"
      f"other words, did snapping work correctly?")
#Add hallways to the dummy vertex:
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

#Add the even degree constraint for vdum:

for i in range(nextnode):
    if i == vdum:
        m.addConstr(sum([varshall[(i,e)] for e in neighbours[i]])==2, name='evenDegreeVDUM')
    else:
        m.addConstr(sum([varshall[(i,e)] for e in neighbours[i]])==2*varsdegree[i], name=f'evenDegreeVertex{i}')

#Call optimize to get the solution
m.optimize()
#Retreive final values for the varshall: hallway variables and print them
solution = m.getAttr('X', varshall)
# pprint(solution)
used_edges=[]
for key, value in solution.items():
    if value ==1:
        if (key[0], key[1]) not in used_edges:
            if (key[1],key[0]) not in used_edges:
                used_edges.append(key)
        elif key not in used_edges:
            used_edges.append(key)

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

#Translate vertex pairs back to coordinates:
def GetCoordinatesPair(vtuple):
    # print(f"we get coordinates for{vtuple}")
    b0, f0 = GetDrawingInfo(vtuple[0])
    b1, f1 = GetDrawingInfo(vtuple[1])
    # print(f"we have b0:{b0}, f0:{f0}, b1:{b1}, f1:{f1}")
    return (nodeToCoordinate[b0][f0][np.round(vtuple[0],3)], nodeToCoordinate[b1][f1][np.round(vtuple[1],3)])

testPath= PATH+("\\Eerste aantekeningen\\CARRE 1412")
newFigurePath = testPath+"\\empty1412.3.svg"

tree = ET.parse(newFigurePath)
tree.write(testPath+"\\backupempty1412.3.svg")

root = tree.getroot()
print(f"the root of the empty svg file with only the floorplan image: {root}")


ET.register_namespace("", "http://www.w3.org/2000/svg")  # Register SVG namespace
for edge in used_edges:
    if vdum not in edge:
        startco, endco= GetCoordinatesPair(edge)
        new_path_element = ET.Element("path", attrib={
            "d": Path(Line(start=startco, end=endco)).d(),
            "stroke": "purple",
            "fill": "none",
            "stroke-width": "2"
        })
        root.append(new_path_element)

tree.write(testPath+"\\testresults1412.3.svg")
print(f"Result is in newly created file{testPath+'\\testresults1412.3.svg'}")
pprint(f"We have  {vdum} nodes without the dummy vertex")
#WE MUST HAVE 63 excluding or 65 nodes including the scale
print(np.sort_complex([coordinate for coordinate in coordinateToNode['CR'][3].keys() if coordinate not in ['minvnum','maxvnum']]))