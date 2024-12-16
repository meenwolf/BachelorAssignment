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
import math
from copy import deepcopy
#Define colors for the different floors you can go to in the result
colorFloors=["pink","palevioletred",'deeppink','firebrick','orangered','orange','gold', 'lawngreen','green','darkcyan','cyan'
             'steelblue', 'rebeccapurple', 'purple', 'fuchsia']

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
            # Get or set width and height
            # Ensure viewBox is present
            if "viewBox" not in root.attrib:
                # root.attrib["viewBox"] = f"0 0 {width.replace('mm', '')} {height.replace('mm', '')}"
                print(f"in file: {file} there is no viewbox")
            else:
                print(f"in file: {file} there is a viewbox")
            root.attrib["width"] = "100%"
            root.attrib["height"] = "100%"
            # Save the fixed file
            # tree.write(output_file)
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

neighboursnew=deepcopy(neighbours)
#Connect special paths, elevators first:
#CONVENTION CHANGE: use double digits for index elevator, stair, exit, and double digits for the floors! to keep things consistent
# So old version: CRE11 now becomes CRE0101 and CRE25 is now CRE0205, so that in horst, or buildings with more than
# 10 staircases, elevators or exits, the same code can be used
def findSingleEnd(specialedge,neighbourhood): #uses the global variables specialPaths
    Nstart = len(neighbourhood[specialPaths[specialedge]["Start"]["Vnum"]])
    Nend = len(neighbourhood[specialPaths[specialedge]["End"]["Vnum"]])
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
                end1= findSingleEnd(e1, neighbours)
                end2= findSingleEnd(e2, neighbours)
                hallways[(end1, end2)]=1
                neighboursnew[end1].add(end2)
                neighboursnew[end2].add(end1)
            connected.append(startname)
    elif pathname[2] == "S": #We have an edge to a staircase
        if pathname not in connected: # if we have not connected these two floors with this staircase
            end1= findSingleEnd(pathname, neighbours)
            otherSide= pathname[:5]+pathname[7:]+pathname[5:7]
            if otherSide in specialPaths:
                print(f"original stair:{pathname}, other side:{otherSide}")
                end2=findSingleEnd(otherSide, neighbours)
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


print(f"Before reassigning: og and new the same? {neighbours==neighboursnew}")
print(f"the neighbourhoods are: {neighbours}")
#Now define functions needed to find the longest route.
neighboursold= neighbours
neighbours=neighboursnew
print(f"neighbours old is new? {neighboursold==neighbours}")
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
def runModel(halls, nvdum, maxtime=None, maxgap=None, printtime=None):

    m = Model()
    if maxtime != None:
        m.Params.TimeLimit = maxtime
    if maxgap!=None:
        m.Params.MIPGap= maxgap
    if printtime!=None:
        m.Params.DisplayInterval= printtime
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

# def getSpecialUsedVertices(used_edges):
#     specialVertices={}
#     for edge in used_edges:
#         if edge in specialEdges:
#             specialVertices[edge[0]]=
#             specialVertices.append(edge[1])
#     return specialVertices

def getRainbowColors(NwantedColors):
    startred= 150 #to take bordeaux/dark red into account
    maxgreen=220 #to prevent light yellow
    minblue= 100 #for magenta
    maxblue= 200 #for light pink

    colors=[]
    red=startred
    green=0
    blue=0
    while red < 255:
        colors.append((red,green,blue))
        red+=1

    while green < maxgreen:
        colors.append((red,green,blue))
        green+=1

    while red>0:
        colors.append((red,green,blue))
        red-=1

    while green< 255:
        colors.append((red,green,blue))
        green+=1

    while blue<255:
        colors.append((red,green,blue))
        blue+=1

    while green >0:
        colors.append((red,green,blue))
        green -= 1


    while red < 255:
        colors.append((red,green,blue))
        red +=1

    while blue > 100:
        colors.append((red,green,blue))
        blue -=1

    while blue <= maxblue:
        colors.append((red,round(green),blue))
        blue +=1
        green += 1.64

    #toRed: startred,0,0 to 255,0,0
    Red= 255- startred # possible reds, increase red value by 1 for each color
    Orange = maxgreen  # to go from red to orange, increase green value by 1 this many times
    Yellow= 255 # yellow tones, remove red by 1 for each color
    Green=  255-maxgreen # green tones, add 1 green for each color
    Lightblue= 255 # to light green/blue, add 1 blue for each color
    Blue= 255 # to blue, remove 1 green for each color
    Purple= 255 # to purple, add 1 red for each color
    Magenta= 255-minblue #To magenta, remove 1 blue for each color
    Lightpink= maxblue-minblue # to lighter pink, add 1 blue and 1.64 green for each color.
    NpossibleColors= Red+Orange+Yellow+Green+Lightblue+Blue+Purple+Magenta+Lightpink

    colors=[]
    #Calculate how many of these colors need to be added to the colorgrid
    nred= math.ceil(Red/NpossibleColors*NwantedColors)
    norange= math.ceil(Orange/NpossibleColors*NwantedColors)
    nyellow= math.ceil(Yellow/NpossibleColors*NwantedColors)
    ngreen= math.ceil(Green/NpossibleColors*NwantedColors)
    nlightblue=math.ceil(Lightblue/NpossibleColors*NwantedColors)
    nblue= math.ceil(Blue/NpossibleColors*NwantedColors)
    npurple= math.ceil(Purple/NpossibleColors*NwantedColors)
    nmagenta= math.ceil(Magenta/NpossibleColors*NwantedColors)
    nlightpink= math.ceil(Lightpink/NpossibleColors*NwantedColors)

    #first add the red colors:
    cval= np.linspace(startred, 255,nred)
    for c in cval:
        colors.append((round(c),0,0))

    # Add the orange colors:
    cval= np.linspace(0, maxgreen, norange)
    for c in cval:
        colors.append((255,round(c),0))

    # Add the yellow colors
    cval= np.linspace(0, 255, nyellow)
    for c in cval:
        colors.append((255-round(c), maxgreen, 0))

    # Add the green colors
    cval= np.linspace(maxgreen, 255, ngreen)
    for c in cval:
        colors.append((0,round(c),0))

    # Add the lightblue colors
    cval= np.linspace(0, 255, nlightblue)
    for c in cval:
        colors.append((0,255,round(c)))

    # Add the blue colors
    cval= np.linspace(0, 255, nblue)
    for c in cval:
        colors.append((0,255-round(c),255))

    # Add the purple colors
    cval = np.linspace(0, 255, npurple)
    for c in cval:
        colors.append((round(c),0,255))

    # Add the magenta colors
    cval= np.linspace(0, 255-minblue, nmagenta)
    for c in cval:
        colors.append((255, 0, 255-round(c)))

    # Add the light pink colors
    cval= np.linspace(0, maxblue-minblue, nlightpink)
    for c in cval:
        colors.append((255, round(c*1.64), minblue+round(c)))

    return colors

def rgb_to_string(rgb_tuple):
    return f"rgb({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]})"

def drawEdgesInFloorplans(edges):
    #first get the start and end of the trail
    # startend = []
    # for i, j in edges:
    #     if j == vdum:
    #         startend.append(i)
    # Add the paths to the roots
    # specialVertices= getSpecialUsedVertices(edges)
    rainbowColors= getRainbowColors(len(edges))
    startedge= edges[0]
    endedge= edges[-1]

    for i,edge in enumerate(edges):
        # if vdum not in edge:
        building0, floor0= getBuildingFloor(edge[0])
        building1, floor1= getBuildingFloor(edge[1])
        #Now check if the buildings are the same, if not, we use a walking bridge/connection hallway
        if building0==building1:
            #Check if we are on the same floor, if not, we take a staircase or elevator.
            if floor0==floor1:
                #Now we can draw the lines in the floor plan.
                if startedge == edge:
                    color="saddlebrown"
                elif endedge == edge:
                    color="saddlebrown"
                # elif edge in specialEdges:
                #     edgename= specialEdges[edge]
                #     if edgename[2] == "S":
                #         toFloor= int(edgename[7:])
                #         color=colorFloors[toFloor]
                #     elif edgename[2] == "E":
                #         otherEdgesElev=[key for key in specialEdges.keys() if specialEdges[key][:5]==edgename[:5]]
                #         print(f"we have this many times that we take elevator:{edgename[:5]}:{len(otherEdgesElev)}")
                #         for  otherEdge in otherEdgesElev:
                #             if not otherEdge == edge:
                #                 toFloor= int(specialEdges[otherEdge][5:])
                #                 print(f"we take elevator from floor:{int(edgename[5:])}:{toFloor}")
                #                 color=colorFloors[toFloor]
                #     else:
                #         color=rainbowColors[i]
                # elif (edge[0] in specialVertices) or (edge[1] in specialVertices):
                #     print(f"We might want to color to which floor we are taking the stairs? or elevator?")
                #     if specialEdges[edge][2]=="S":
                #         #Convention, the last two numbers indicate the floor you go to
                #     color="pink"
                else:
                    color=rgb_to_string(rainbowColors[i])

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
                if "viewBox" not in thisRoot.attrib:
                    # root.attrib["viewBox"] = f"0 0 {width.replace('mm', '')} {height.replace('mm', '')}"
                    print(f"in floor: {floor0} there is no viewbox")
                else:
                    print(f"in floor: {floor0} there is a viewbox")
                thisRoot.append(new_path_element)

                #Maybe different colorings if we take an elevator, stair up, stair down, another building?
            # elif int(floor0)<int(floor1):
            #     if int(floor0)+1 == int(floor1):
            #         print(f"We take stairs up")
            #     else:
            #         print(f"We take elevator up to floor{floor1}")
            #     # Create a new <text> element
            #     toFloor=nodeToCoordinate[edge[1]]['Floor']
            #     drawCoord= nodeToCoordinate[edge[0]]['Location']
            #     text_element = ET.Element("text", attrib={
            #         "x": drawCoord.real,
            #         "y": drawCoord.imag,
            #         "font-size": "16",  # Font size in pixels
            #         "fill": "dimgray"  # Text color
            #     })
            #     text_element.text = floor1
            else: # Print the number of the floor we draw to on the coordinate of the point we start in
                # else:
                #     print(f"We take elevator down to floor{floor1}")
                color = rgb_to_string(rainbowColors[i])
                toFloor = nodeToCoordinate[edge[1]]['Floor']
                drawCoord = nodeToCoordinate[edge[0]]['Location']
                if building0 =="CARRE 1412":
                    if floor0 == '4':
                        drawCoord=drawCoord +126.822+ 494.891j
                text_element = ET.Element("text", attrib={
                    "x": str(drawCoord.real),
                    "y": str(drawCoord.imag),
                    "font-size": "16",  # Font size in pixels
                    "fill": color  # Text color
                })
                text_element.text = toFloor
                thisRoot = figuresResultBuildings[building0][floor0]['root']
                if "viewBox" not in thisRoot.attrib:
                    # root.attrib["viewBox"] = f"0 0 {width.replace('mm', '')} {height.replace('mm', '')}"
                    print(f"in floor: {floor0} there is no viewbox")
                else:
                    print(f"in floor: {floor0} there is a viewbox")
                thisRoot.append(text_element)
        else:
            print(f"We go from {building0} to {building1}")
            color = rgb_to_string(rainbowColors[i])
            toBuild = nodeToCoordinate[edge[1]]['Building']
            drawCoord = nodeToCoordinate[edge[0]]['Location']
            if building0 == "CARRE 1412":
                if floor0 == '4':
                    drawCoord = drawCoord + 126.822 + 494.891j
            text_element = ET.Element("text", attrib={
                "x": str(drawCoord.real),
                "y": str(drawCoord.imag),
                "font-size": "16",  # Font size in pixels
                "fill": color  # Text color
            })
            text_element.text = toBuild
            thisRoot = figuresResultBuildings[building0][floor0]['root']
            if "viewBox" not in thisRoot.attrib:
                # root.attrib["viewBox"] = f"0 0 {width.replace('mm', '')} {height.replace('mm', '')}"
                print(f"BRIDGE: in floor: {floor0} there is no viewbox")
            else:
                print(f"BRIDGE: in floor: {floor0} there is a viewbox")
            thisRoot.append(text_element)

    # Draw the figures in a new file:
    for building, buildinginfo in figuresResultBuildings.items():
        buildingResultPath= PATH_result+f"\\{building}"
        for floor, floorinfo in buildinginfo.items():
            buildingName, buildingNumber = splitNameNumber(building)
            floortree= floorinfo['tree']
            print(f"the type of floor tree is: {type(floortree)}\n {floortree}")
            testfilename= f"\\testingMoreFloors3{buildingNumber}.{floor}.svg"
            floortree.write(buildingResultPath+testfilename)

def constructTrail(edges,vdum):
    print(f"{len(edges)} edges:{edges}")
    nedges= len(edges)
    trail=[]
    node_neighbors = defaultdict(list)
    #Get the neighbourhoods of the induced graph by the edges, but not considering the dummy vertex and dummy edges
    dummyEdges=[]
    for i, j in edges:
        if i != vdum and j != vdum:
            node_neighbors[i].append(j)
            node_neighbors[j].append(i)
        else:
            dummyEdges.append((i,j))
            dummyEdges.append((j,i))
    # Remove the dummy edges from the used edges
    for edge in dummyEdges:
        if edge in edges:
            edges.remove(edge)


    for node, neighbs in node_neighbors.items():
        if len(neighbs)==1:
            # We found a start!
            currentNode= node
            break

    while len(trail)<nedges-2:
        neighbs= node_neighbors[currentNode]
        print(f"The {len(neighbs)} neihgbours of  vertex {currentNode} are: {neighbs}\n"
              f"for neighbourhoodkeys:{node_neighbors.keys()}")
        if len(neighbs)==1:
            vertex= neighbs[0]
            trail.append((currentNode,vertex))
            print(f"neighborhood of currrent node: {node_neighbors[currentNode] if currentNode in node_neighbors.keys() else FALSE}\n")
            print(f"now we want to remove {vertex} from {node_neighbors[currentNode]}")
            print(f"neighborhood of next node: {node_neighbors[vertex] if vertex in node_neighbors.keys() else FALSE}\n")

            node_neighbors[currentNode].remove(vertex)
            print(f"neighbourhoodkeys are now: {node_neighbors.keys()}\n")
            node_neighbors[vertex].remove(currentNode)
            print(f"neighborhood of next node: {node_neighbors[vertex] if vertex in node_neighbors.keys() else FALSE}\n")
            currentNode= vertex
        elif len(neighbs)==0:
            print(f"zero neighbours , len trail: {len(trail)} and len edges:{len(edges)}")
            break
        else:
            for vertex in neighbs:
                if (currentNode, vertex) in edges:
                    edgeToConsider= (currentNode, vertex)
                elif (vertex, currentNode) in edges:
                    edgeToConsider= (vertex, currentNode)
                else:
                    print(f"ERROR: vertex {vertex} is a neighbour of current node {currentNode} but no edge is in the edge set")
                print(f"check if this edge is a bridge")
                nReachableBefore= getReachable(node_neighbors, currentNode)
                node_neighbors[edgeToConsider[0]].remove(edgeToConsider[1])
                node_neighbors[edgeToConsider[1]].remove(edgeToConsider[0])
                nReachableAfter= getReachable(node_neighbors, currentNode)
                if nReachableAfter< nReachableBefore:
                    print(f"edge {edgeToConsider} is a bridge so look for the next after adding the vertices back to nodeneighbours")
                    node_neighbors[edgeToConsider[0]].append(edgeToConsider[1])
                    node_neighbors[edgeToConsider[1]].append(edgeToConsider[0])

                else:
                    print(f"edge{edgeToConsider} is not a bridge, so we can take this one")
                    print(f"we found a non bridge, which is saved in edgetoconsider:{edgeToConsider}, with "
                          f"current node:{currentNode} and next node:{vertex}\n"
                          f" and already removed from node_neighbors, so take that bridge and continue")
                    edges.remove(edgeToConsider)
                    trail.append((currentNode, vertex))
                    currentNode=vertex
                    break
            if not currentNode == vertex:
                print(f"ERROOOORRRRR ???? length trail: {len(trail)} we did not find a nonbridge edge? is that even possible?? for currentNode {currentNode}")
                break
    return trail



            #loop over the neighbours that can still be visited, and check if it is a bridge, if one is not a bridge
            # take that one. else take the bridge, and continue until the trail covered all edges. Should work


model, varshall, varsdegree = runModel(hallways, vdum, maxgap=0.1, printtime= 5)
lengthLongestTrail=model.getAttr('ObjVal')
print(f"The longest trail is {lengthLongestTrail} meters long")
used_edges= getEdgesResult(model, varshall)
print(f"we have {vdum} as dummy vertex")
print(f"edges used that connected here: {[edge for edge in used_edges if vdum in edge]}")
pprint(f"The used edges in the solution are:\n{used_edges}")
trailresult= constructTrail(used_edges, vdum)
drawEdgesInFloorplans(trailresult)