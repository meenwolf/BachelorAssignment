import os

from networkx.classes import neighbors
from svgpathtools import svg2paths, svg2paths2, wsvg
from pprint import pprint
from gurobipy import *
# Get the path to the drawings
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH= os.path.abspath(os.path.join(dir_path, os.pardir))
PATH_drawings= PATH+"\\FloorplanAnnotated"
#Get the paths and attributes of the third floor of Carré

#IDEA: make a list of nodes, for all of the path starts and ends. After we also know the scale and made the
#Edges have the correct real world weights, we can start at merging nodes in one, if they are closer than X meters in real life.

paths, attributes = svg2paths(PATH_drawings+"\\CARRE 1412\\withPNG1412.3.svg")

for path, attr in zip(paths, attributes):
    if "inkscape:label" in attr.keys():
        if attr['inkscape:label'] == "CR31mtr":
            print(f"the scale label is: {attr['inkscape:label']}")
            lengthForOneMeter= abs(path.start - path.end)
            break

print(f"length for one meter is {lengthForOneMeter}")

for path, attr in zip(paths, attributes):
    if "inkscape:label" in attr.keys():
        print(f"inkscape:label is in attributes: {attr['inkscape:label']}")
        print(f"and the path is: {path}")
        start= path.start
        end= path.end
        edgeweight= abs(start-end)
        print(f"edge has length: {edgeweight}")

# Gurobi uses dictionaries where the keys are pairs of (start, end) having a value
# of the distance between them. To keep the information about the labels, I will create
# a separate dictionary, having keys (start, end) and value the label name I assigned to it
# then its easy to check if special actions are required.

specialPaths={"CR":{3:{}}}
hallways={}
coordinateToNode={"CR":{3:{}}}
nextnode=0
for path, attr in zip(paths, attributes):
    for i, line in enumerate(path):
        start = line.start
        end = line.end
        edgeweight = abs(start - end)/lengthForOneMeter #edgeweight in meters
        if "inkscape:label" in attr.keys():
            if 'mtr' in attr['inkscape:label']:
                continue
            else:
                specialPaths["CR"][3][(start, end)]=attr['inkscape:label']+str(i)


        if not start in coordinateToNode["CR"][3]:
            coordinateToNode["CR"][3][start]=nextnode
            snode=nextnode
            print(f" snode before increment: {snode}")
            nextnode+=1
            print(f" snode after increment: {snode}")
        else:
            snode=coordinateToNode["CR"][3][start]

        if not end in coordinateToNode["CR"][3]:
            coordinateToNode["CR"][3][end]=nextnode
            enode=nextnode
            print(f" enode before increment: {enode}")
            nextnode+=1
            print(f" enode after increment: {enode}")
        else:
            enode=coordinateToNode["CR"][3][end]
        hallways[(snode, enode)] = edgeweight


pprint(specialPaths)
print("And all the original hallways are:")
pprint(hallways)

#Add hallways to the dummy vertex:
vdum= nextnode
nextnode += 1

for i in range(nextnode):
    hallways[(vdum, i)]=0
# define a dictionary where each vertex maps to a set of its neighbours
neighbours={i:{vdum} for i in range(vdum)}
neighbours[vdum]=set(range(vdum))
for v1,v2 in hallways.keys():
    print(f"v1:{v1}, v2:{v2}")
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
        print(f'i: {i} when vdum: {vdum}, and nextnode: {nextnode}')
        m.addConstr(sum([varshall[(i,e)] for e in neighbours[i]])==2*varsdegree[i], name=f'evenDegreeVertex{i}')

#Call optimize to get the solution
m.optimize()
#Retreive final values for the varshall: hallway variables and print them
solution = m.getAttr('X', varshall)
pprint(solution)

