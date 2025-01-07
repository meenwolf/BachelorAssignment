import os
from os.path import split

from bokeh.colors.named import saddlebrown
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

import pandas as pd
import gurobi_logtools as glt
import plotly.graph_objects as go

# from TrialExtractSVG import getBuildingFloor
def getBuildingFloor(nodeToCoordinate, vnum):
   return nodeToCoordinate[vnum]['Building'], nodeToCoordinate[vnum]['Floor']


# from TrialExtractSVG import getReachable
def getReachable(neighborhoods, start, reachable=None):
    if reachable==None:
        reachable={start}
    for neighbor in neighborhoods[start]:
        if not neighbor in reachable:
            reachable.add(neighbor)
            reachable= getReachable(neighborhoods,neighbor, reachable)
    return reachable
# Get the path to the drawings
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH= os.path.abspath(os.path.join(dir_path, os.pardir))

PATH_drawings= PATH +"\\TestMoreBuildings\\OriginalPaths"
PATH_empty= PATH+"\\TestMoreBuildings\\EmptyFloorplans"
PATH_result= PATH+"\\TestMoreBuildings\\ResultFloorplans"

# Initiate some data structures to save information in
nodeToCoordinate={}
coordinateToNode={}

nextnode=0

specialPaths={}
specialEdges={}
hallways={}

figuresResultBuildings  = dict()

# Measured, but made up for now, bridge lengths:
bridgeLengths={"C01CRWA": 10, "C01WACR": 10, "C01CRNL": 10, "C01NLCR": 10, "C01CRWH": 10, "C01WHCR": 10, "C01CIRA": 10, "C01RACI": 10,
               "C01ZHNL": 10, "C01NLZH": 10, "C01RAZI": 10, "C01ZIRA": 10, }

# Loop over the files to save a file where we can draw the resulting longest paths in
# distfloorci=dict()
# edgesRAfive=[]
for building in os.listdir(PATH_drawings):
    if 'WAAIER' in building:
        continue
    buildingEmpty= PATH_empty+f"\\{building}"
    listOfFiles = os.listdir(buildingEmpty)
    for file in listOfFiles:
        if file.endswith(".svg"):
            floor= file.split('.')[1]

            newFigurePath = buildingEmpty + f"\\{file}"

            tree = ET.parse(newFigurePath)
            root = tree.getroot()

            # Set the width and height to prevent the result file to be of A4 size or in any case not showing the final result in microsoft edge(in inkscape it still showed the correct file)
            root.attrib["width"] = "100%"
            root.attrib["height"] = "100%"
            ET.register_namespace("", "http://www.w3.org/2000/svg")  # Register SVG namespace

            if building in figuresResultBuildings:
                figuresResultBuildings[building][floor]={'tree':tree, 'root':root}
            else:
                figuresResultBuildings[building]={floor: {'tree': tree, 'root': root}}

            #Now start extracting the path information
            paths, attributes = svg2paths(PATH_drawings + f"\\{building}\\{file}")
            lengthForOneMeter= 1
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
                        elif not 'path' in attr['inkscape:label']: #meaning that we did not delete the name of a path in inkscape (e.g. chaning it on accident, and delete the wrong name)
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
print(f"special edges are: {specialEdges}")
print("The hallways are:")
pprint(hallways)
print(f"next node is dummy vertex number: {nextnode}")
# Define a dictionary where each vertex maps to a set of its neighbours
neighboursold= {i:set() for i in range(nextnode)} #to store the neighbours that we draw in the floor plans

for v1,v2 in hallways.keys():
    neighboursold[v1].add(v2)
    neighboursold[v2].add(v1)

neighbours=deepcopy(neighboursold) # deep copy the neighbourhood, to which we can add the
# Connections regarding staircases, elevators and connections between buildings

#Connect special paths, elevators first:
#CONVENTION: use double digits for index elevator, stair, exit, and double digits for the floors! to keep things consistent

def findSingleEnd(specialedge,neighbourhood): #uses the global variables specialPaths
    Nstart = len(neighbourhood[specialPaths[specialedge]["Start"]["Vnum"]])
    Nend = len(neighbourhood[specialPaths[specialedge]["End"]["Vnum"]])
    if Nstart < Nend:
        end = specialPaths[specialedge]["Start"]["Vnum"]
    else:
        end = specialPaths[specialedge]["End"]["Vnum"]
    return end

elevatorVertices=dict() # Keep a dictionary storing the vertices for elevators, which need to have degree <=2
elevatorEdges=set()
connected=[]
for pathname, pathinfo in specialPaths.items():
    if pathname[2]=="E": #We have an edge to an elevator
        startname = pathname[:5]
        if not startname in connected: #If we have not yet connected the floors that can be reached from this elevator, do so
            # print(startname)
            elevatorConnects=[key for key in specialPaths.keys() if startname in key]
            velevator= nextnode
            elevatorVertices[startname]=velevator
            nextnode+=1
            neighbours[velevator]=set()
            for edge in elevatorConnects:
                end= findSingleEnd(edge, neighboursold)
                hallways[(end, velevator)]=1
                elevatorEdges.add((end, velevator))
                elevatorEdges.add((velevator,end))
                neighbours[end].add(velevator)
                neighbours[velevator].add(end)
            connected.append(startname)
    elif pathname[2] == "S": #We have an edge to a staircase
        if pathname not in connected: # if we have not connected these two floors with this staircase
            end1= findSingleEnd(pathname, neighboursold)
            otherSide= pathname[:5]+pathname[7:]+pathname[5:7]
            if otherSide in specialPaths:
                # print(f"original stair:{pathname}, other side:{otherSide}")
                end2=findSingleEnd(otherSide, neighboursold)
                hallways[(end1, end2)] = 7 # 1 stair 7 meter? idkkk
                neighbours[end1].add(end2)
                neighbours[end2].add(end1)
                connected.append(pathname)
                connected.append(otherSide)
    elif pathname[1] == "0": #connecting buildings? naming conv?
        # otherSide = pathname[3:5]+pathname[2]+pathname[0:2]+pathname[5:]
        print(f"connection building from CR:{pathname} but we deal with this when we find the other end this bridge is connected to.")
    elif pathname[2] == "C":
        buildingOfEdge= pathname[0:2]
        buildingToEdge= pathname[5:7]
        connection= pathname[2:5]
        #Since this edge represents a hallway drawn in buildingOfEdge and connects to the bridge connection between this building en buildingToedge,
        #The bridge itself is not drawn into the floorplan, but let's check for that anyway:
        bridgeName= connection+buildingOfEdge+buildingToEdge
        otherSide = buildingToEdge+connection+buildingOfEdge
        otherBridgeName = connection + buildingToEdge + buildingOfEdge
        # print(f"This hallway {pathname} ends at a walking bridge{bridgeName} or {otherBridgeName} and the other side is: {otherSide}")

        if  bridgeName in specialPaths.keys():
            if bridgeName not in connected:
                if otherSide in specialPaths:
                    print(f"????THE???? walking bridge is connected to edge {pathname} in the same building. I overdid the naming and will connect this hallway to the otherSide")
                    end1= findSingleEnd(bridgeName, neighboursold)
                    end2=findSingleEnd(otherSide, neighboursold)
                    hallways[(end1, end2)] = 0  # Connect the endpoint of a bridge to the corresponding entry of the other building
                    neighbours[end1].add(end2)
                    neighbours[end2].add(end1)
                    connected.append(pathname)
                    connected.append(bridgeName)
                    connected.append(otherSide)
                else:
                    print(f"the other building is not drawn into so we cannot connect the walking bridge for which we overdid the naming on the other end")
            else:
                print(f"overdid the naming, but already connected this hallway, so just continue")
        elif pathname not in connected:
            print(f"We did not connect this endpoint yet")
            #Check if the bridge is drawn into the other buildings floorplan
            if otherBridgeName in specialPaths.keys():
                print(f"The bridge is drawn fully on the floorplan of the other building, so we just connect those two ends with an edge of weight zero")
                #We find the end of edge pathname and of edge otherBridgeName and connect them with an edge of weight zero
                end1= findSingleEnd(pathname, neighboursold)
                end2= findSingleEnd(otherBridgeName, neighboursold)
                hallways[(end1, end2)] = 0  # Connect the endpoint of a bridge to the corresponding entry of the other building
                neighbours[end1].add(end2)
                neighbours[end2].add(end1)
                connected.append(pathname)
                connected.append(otherBridgeName)
                #to be sure also append the other side of this connection (for if I overdid the naming of the drawn in edges)
                connected.append(otherSide)
            elif otherSide not in specialPaths:
                print(f"we have not yet drawn in the floorplans of the other building so we cannot connect edge: {pathname}")
            else:
                print(f"We did draw the paths in the other building of the bridge between {pathname} and {otherSide}. Have to measure this bridge by hand and connects the two ends of bridge with an edge of this length.")
                lenbridge= bridgeLengths[bridgeName]
                end1 = findSingleEnd(pathname, neighboursold)
                print(f"end1: {end1}: {nodeToCoordinate[end1]}")
                end2 = findSingleEnd(otherSide, neighboursold)
                print(f"end2: {end2}: {nodeToCoordinate[end2]}")

                hallways[(end1, end2)] = lenbridge  # Connect the endpoint of a bridge to the corresponding entry of the other building
                neighbours[end1].add(end2)
                neighbours[end2].add(end1)
                connected.append(pathname)
                connected.append(otherSide)
        else:
            print(f"We already connected this bridge.")

    elif pathname[2]=='X':
        print(f"we have an exit to outdoors: {pathname}")
    else:
        print(f"somehting went wrong: {pathname} not a stair, elevator, connection or exit")

# Add hallways to the dummy vertex:
vdum= nextnode
nextnode += 1
print(f"neighbourhoods are: {neighbours}")
print(f"are all vertices reachable from vertex 1 before adding dummy vertex?? {len(getReachable(neighbours, 1))==vdum}")



neighbours[vdum]= set(range(vdum))

for i in range(nextnode):
    hallways[(vdum, i)]=0
    neighbours[i].add(vdum)


print(f"Neighbours old is new? {neighboursold==neighbours}")


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

def runModel(halls, nvdum, maxtime=None, maxgap=None, printtime=None, logfile=None):

    m = Model()
    if maxtime != None:
        m.Params.TimeLimit = maxtime
    if maxgap!=None:
        m.Params.MIPGap= maxgap
    if printtime!=None:
        m.Params.DisplayInterval= printtime
    if logfile != None:
        m.Params.LogFile= PATH+logfile
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

    # Add the constraint that forbids you to use an elevator multiple times:
    for name, vertex in elevatorVertices.items():
        m.addConstr(varsaux[vertex]<= 1, name=f'visit{name}AtMostOnce')

    # Set up for the callbacks/ lazy constraints for connectivity
    m.Params.LazyConstraints = 1
    cb = TSPCallback(range(vdum+1), varssol)

    # Call optimize with the callback structure to get a connected solution solution
    m.optimize(cb)
    return m, varssol, varsaux


#Retreive final values for the varshall: hallway variables and print them
def getEdgesResult(model, varssol):
    #Only get the edges that have a value x_e >0.5, namely about 1.
    sol = model.getAttr('X', varssol)
    edges = set()
    for key, value in sol.items():
        if value >= 0.5:
            if key[0] < key[1]:
                edges.add(key)
    return edges

# #Define a function returning the building and floor a vertex vnum lies in

#Translate a vertex pair representing an edge, back to coordinates:
def getCoordinatesPair(vtuple):
    return nodeToCoordinate[vtuple[0]]["Location"], nodeToCoordinate[vtuple[1]]["Location"]

#Get the building name and number from the building name containing both of them with a space in between.
def splitNameNumber(buildingNo):
    res = buildingNo.split()
    # print(f"buildingnumber:{buildingNo}, res:{res}, elem 0: {res[0]}, elem 1:{res[1]}")
    return res[0], res[1]


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
    rainbowColors= getRainbowColors(len(edges))
    startedge= edges[0]
    endedge= edges[-1]

    for i,edge in enumerate(edges):
        if edge in elevatorEdges:
            # We can not draw this line in a floor plan
            continue
        building0, floor0= getBuildingFloor(nodeToCoordinate,edge[0])
        building1, floor1= getBuildingFloor(nodeToCoordinate,edge[1])

        #Now check if the buildings are the same, if not, we use a walking bridge/connection hallway
        if building0==building1:
            #Check if we are on the same floor, if not, we take a staircase or elevator.
            if floor0==floor1:
                #Now we can draw the lines in the floor plan.
                if startedge == edge:
                    #To indicate an endpoint of the trail
                    color="saddlebrown"
                elif endedge == edge:
                    #To indicate an endpoint of the trail
                    color="saddlebrown"

                elif edge in specialEdges:
                    # Otherwise, we color this edge according to the fading rainbow
                    color = rgb_to_string(rainbowColors[i])
                    if specialEdges[edge][2]=="E":
                        # We found elevator connection for which we might need to print a number
                        if edges[i+1] in elevatorEdges:
                            # We step into the elevator so we need to print a number
                            toFloor= nodeToCoordinate[edges[i+3][0]]['Floor']
                            drawCoord = nodeToCoordinate[edge[1]]['Location']

                            if building0 == "CARRE 1412":
                                if floor0 == '4':# since this floor had a weird shift in it
                                    drawCoord = drawCoord + 126.822 + 494.891j

                            # Add the number
                            text_element = ET.Element("text", attrib={
                                "x": str(drawCoord.real),
                                "y": str(drawCoord.imag),
                                "font-size": "24",  # Font size in pixels
                                "fill": "saddlebrown",  # Text color
                                "stroke": "saddlebrown"
                            })
                            text_element.text = str(toFloor)
                            thisRoot = figuresResultBuildings[building0][floor0]['root']
                            thisRoot.append(text_element)

                    elif specialEdges[edge][2]=="C":
                        #We go to a walking bridge, so we print the building on the end
                        drawCoord = nodeToCoordinate[edge[0]]['Location']

                        if building0 == "CARRE 1412":
                            if floor0 == '4':  # since this floor had a weird shift in it
                                drawCoord = drawCoord + 126.822 + 494.891j

                        # Add the number
                        text_element = ET.Element("text", attrib={
                            "x": str(drawCoord.real),
                            "y": str(drawCoord.imag),
                            "font-size": "24",  # Font size in pixels
                            "fill": "saddlebrown",  # Text color
                            "stroke": "saddlebrown"
                        })
                        text_element.text = specialEdges[edge][5:]
                        thisRoot = figuresResultBuildings[building0][floor0]['root']
                        thisRoot.append(text_element)

                else:
                    # Normal edges are also colored according to the fading rainbow.
                    color=rgb_to_string(rainbowColors[i])
                # Get the coordinates to draw a line connecting them in the correct floor plan
                startco, endco = getCoordinatesPair(edge)
                if building0 =="CARRE 1412":
                    if floor0 == '4': # since this floor had a weird shift in it
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


            else: # We have a staircase edge, so we include the number of the floor we go to in the floor plan
                # Print the number of the floor we draw to on the coordinate of the point we start in
                toFloor = nodeToCoordinate[edge[1]]['Floor']
                drawCoord = nodeToCoordinate[edge[0]]['Location']
                if building0 =="CARRE 1412":
                    if floor0 == '4': # since this floor had a weird shift in it
                        drawCoord=drawCoord +126.822+ 494.891j

                text_element = ET.Element("text", attrib={
                    "x": str(drawCoord.real),
                    "y": str(drawCoord.imag),
                    "font-size": "24",  # Font size in pixels
                    "fill": "saddlebrown",  # Text color
                    "stroke": "saddlebrown"
                })
                text_element.text = str(toFloor)
                thisRoot = figuresResultBuildings[building0][floor0]['root']
                thisRoot.append(text_element)

    # Draw the figures in a new file:
    for building, buildinginfo in figuresResultBuildings.items():
        buildingResultPath= PATH_result+f"\\{building}"
        for floor, floorinfo in buildinginfo.items():
            buildingName, buildingNumber = splitNameNumber(building)
            floortree= floorinfo['tree']
            testfilename= f"\\testingMoreBuildings{buildingNumber}.{floor}.svg"
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

    print(node_neighbors)

    for node, neighbs in node_neighbors.items():
        if len(neighbs)==1:
            # We found a start!
            print(f"potential start: {node}")
            currentNode=node

    while len(trail)<nedges-2:
        neighbs= node_neighbors[currentNode]
        print(f"we are at: {currentNode} with neighbours {neighbs}\n")

        if len(neighbs)==1: # We go to this neighbour
            vertex= neighbs[0]
            print(f'we only have one neighbour to visit so we go to {vertex}')
            trail.append((currentNode,vertex))
            if (vertex, currentNode) in edges:
                edges.remove((vertex, currentNode))
            elif (currentNode, vertex) in edges:
                edges.remove((currentNode, vertex))
            else:
                print(f"this edge was not in the edgeset, but a neighbour?? ")
            node_neighbors[currentNode].remove(vertex)
            node_neighbors[vertex].remove(currentNode)
            currentNode= vertex

        elif len(neighbs)==0: # No more places to go, we used all the edges
            print(f"No more places to go, we used all the edges and break")
            break

        else: # We have to loop over the vertices in the neighourhood and check if the edge to there is a bridge or not
            for vertex in neighbs:
                if (currentNode, vertex) in edges:
                    edgeToConsider= (currentNode, vertex)
                elif (vertex, currentNode) in edges:
                    edgeToConsider= (vertex, currentNode)
                else:
                    print(f"ERROR: vertex {vertex} is a neighbour of current node {currentNode} but no edge is in the edge set")

                #check if this edge is a bridge
                nReachableBefore= getReachable(node_neighbors, currentNode)
                print(f"to start with, reachable from {currentNode} are {len(nReachableBefore)} nodes: {nReachableBefore}")
                node_neighbors[edgeToConsider[0]].remove(edgeToConsider[1])
                node_neighbors[edgeToConsider[1]].remove(edgeToConsider[0])
                nReachableAfter= getReachable(node_neighbors, currentNode)
                print(f"after deleting the edge to {vertex} there are {len(nReachableAfter)} nodes reachable: {nReachableAfter}")

                if nReachableAfter< nReachableBefore: # It is a bridge
                    print(f"So this edge is a bridge, don't go to {vertex} now, but continue searching.")
                    #edge edgeToConsider is a bridge so look for the next after adding the vertices back to nodeneighbours
                    node_neighbors[edgeToConsider[0]].append(edgeToConsider[1])
                    node_neighbors[edgeToConsider[1]].append(edgeToConsider[0])

                else:
                    print(f"edge  to {vertex} is not a bridge, so we can take this one")
                    edges.remove(edgeToConsider)
                    trail.append((currentNode, vertex))
                    currentNode=vertex
                    print(f"we went to vertex {currentNode} and break the forloop")
                    break
            if not currentNode == vertex:
                print(f"ERROOOORRRRR ???? length trail: {len(trail)} we did not find a nonbridge edge? is that even possible?? for currentNode {currentNode}")
                break
    return trail




model, varshall, varsdegree = runModel(hallways, vdum, maxtime=120, printtime= 15, logfile= "\\log0701try1.log")
lengthLongestTrail=model.getAttr('ObjVal')
print(f"The longest trail is {lengthLongestTrail} meters long")
used_edges= getEdgesResult(model, varshall)
print(f"we have {vdum} as dummy vertex")
print(f"edges used that connected here: {[edge for edge in used_edges if vdum in edge]}")
pprint(f"The {len(used_edges)} used edges in the solution are:\n{used_edges}")
trailresult= constructTrail(used_edges, vdum)
print(f"trail result gives {len(trailresult)} edges in order:{trailresult}")
drawEdgesInFloorplans(trailresult)
results = glt.parse(PATH+"\\log1812try1.log")
nodelogs = results.progress("nodelog")
pd.set_option("display.max_columns", None)
print(f"type of nodelogs: {nodelogs}, and has columns: {[i for i in nodelogs]}")
print(nodelogs.head(10))
fig = go.Figure()
fig.add_trace(go.Scatter(x=nodelogs["Time"], y=nodelogs["Incumbent"], mode='markers',name="Primal Bound"))
fig.add_trace(go.Scatter(x=nodelogs["Time"], y=nodelogs["BestBd"], mode='markers',name="Dual Bound"))
fig.update_xaxes(title_text="Runtime in seconds")
fig.update_yaxes(title_text="Objective value function (in meters)")
fig.update_layout(title_text="The bounds on the length of the longest trail on Carré floor 1,2,3 and 4 together,<br> at each moment in time when running the gurobi solver")
fig.show()
