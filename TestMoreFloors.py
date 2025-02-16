import os
from os.path import split

import kaleido
from holoviews.plotting.bokeh.styles import font_size
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
from datetime import datetime

from MoreBuildings import constructTrailCheckComponents


#Connect special paths, elevators first:
#CONVENTION: use double digits for index elevator, stair, exit, and double digits for the floors! to keep things consistent

def getNeighbourhood(edges): #THis function is not used in this file, but will be in another file.
    neighbourhood = dict()
    for v1, v2 in edges:
        if v1 in neighbourhood:
            neighbourhood[v1].add(v2)
        else:
            neighbourhood[v1]={v2}
        if v2 in neighbourhood:
            neighbourhood[v2].add(v1)
        else:
            neighbourhood[v2]={v1}
    return neighbourhood


def findSingleEnd(specialedge,neighbourhood, specialPaths): #uses the global variables specialPaths
    Nstart = len(neighbourhood[specialPaths[specialedge]["Start"]["Vnum"]])
    Nend = len(neighbourhood[specialPaths[specialedge]["End"]["Vnum"]])
    if Nstart < Nend:
        end = specialPaths[specialedge]["Start"]["Vnum"]
    else:
        end = specialPaths[specialedge]["End"]["Vnum"]
    return end


def getReachable(neighborhoods, start, reachable=None):
    if reachable==None:
        reachable={start}
    for neighbor in neighborhoods[start]:
        if not neighbor in reachable:
            reachable.add(neighbor)
            reachable= getReachable(neighborhoods,neighbor, reachable)
    return reachable

# Add hallways to the dummy vertex:
def addDummy(neighbours, weigthededges, vdum=None):
    if vdum ==None:
        vdum= max(list(neighbours.keys()))+1
    neighbours[vdum] = {v for v in neighbours.keys()}

    for i in neighbours.keys():
        weigthededges[(vdum, i)] = 0
        neighbours[i].add(vdum)
    return neighbours, weigthededges, vdum

def vdum_reachable(edges, vdum):
    # Create a mapping from each node to its neighbours in the feasible solution
    node_neighbors = defaultdict(list)
    for i, j in edges:
        # print(f"i:{i}, j:{j}")
        node_neighbors[i].append(j)
        node_neighbors[j].append(i)
    # print(f"neihgbours in vdumreachable:\n{node_neighbors}")
    assert all(len(neighbors) %2 ==0 for neighbors in node_neighbors.values())#assert even degree of the vertices in the solution
    assert vdum in node_neighbors #assert vdum being a vertex visited in the solution

    reachableFromVdum= getReachable(node_neighbors, vdum)
    notReachable= [key for key in node_neighbors.keys() if key not in reachableFromVdum]
    return reachableFromVdum, notReachable

class TSPCallback:
    """Callback class implementing lazy constraints for the TSP.  At MIPSOL
    callbacks, solutions are checked for disconnected subtours and then add subtour elimination
    constraints if needed/ the solution contains disconneced subtours."""

    def __init__(self, nodes, x, neighbours, weightedhalls, vdum):
        self.nodes = nodes
        self.x = x
        self.neighbs=neighbours
        self.halls=weightedhalls
        self.vdum=vdum

    def __call__(self, model, where):
        """Callback entry point: call lazy constraints routine when new
        solutions are found. Stop the optimization if there is an exception in
        user code."""

        if where == GRB.Callback.MIPSOL:
            # if model.getAttr(GRB.Attr.MIPGap)<30:
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
        # trail= constructTrailCheckComponents(,edges, self.vdum,)
        # bestbound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        # longestTrail = 0
        # for edge in trail:
        #     if edge in self.halls:
        #         longestTrail += self.halls[edge]
        #     else:
        #         longestTrail += self.halls[(edge[1], edge[0])]
        # gap = abs(bestbound - longestTrail) / abs(longestTrail)
        # print(f"gap:{gap} bestbound:{bestbound}, lngest trail: {longestTrail}, gap:{gap}")
        # if gap < 30:
        #     model.terminate()
        reachableVdum, notReachableVdum = vdum_reachable(edges, self.vdum)

        edgesS = [(v, n) for v in reachableVdum for n in self.neighbs[v] if n in reachableVdum] + [(n, v) for v in reachableVdum for n in self.neighbs[v] if n in reachableVdum]
        edgesNotS = [(v, n) for v in notReachableVdum for n in self.neighbs[v] if n in notReachableVdum] + [(n, v) for v in notReachableVdum for n in self.neighbs[v] if n in notReachableVdum]
        edgeCut=[]
        for edge in self.halls.keys():
            if edge not in edgesNotS:
                if edge not in edgesS:
                    edgeCut.append(edge)

        edgeCutS= [edge for edge in self.halls.keys() if (edge not in edgesNotS) and (edge not in edgesS)]
        # print(f"are two methods to find edge cut equal? set:{set(edgeCutS)==set(edgeCut)} lists:{edgeCutS==edgeCut}")
        # for i in range(len(edgesS)):
        # if len(edgesNotS) ==0:
        #     print(f"NO CALLBACK CONSTRAINTS ADDED")
        # else:
        #     print(f"CALLBACK CONSTRAINTS ADDED")
            # model.cbLazy(
            #     quicksum([self.x[edge] for edge in edgeCutS])
            #     >= 2 )


        for g in edgesNotS:
            #Mayyybee instead of adding them for any combination, I should use that if there is an edge in E^*[S] and one in E^*[V\S], that
            # The number of edges from the cut that must be present in the solution is >=2
            # lhs= sum([model.cbGetSolution(self.x[edge]) for edge in edgeCutS])
            # rhs= 2* (model.cbGetSolution(self.x[edgesS[0]])+ model.cbGetSolution(self.x[g])-1)
            # print(f" lhs={lhs}  >= rhs={rhs}")
            model.cbLazy(
                quicksum([self.x[edge] for edge in edgeCutS])
                >= 2 * self.x[g])


def runModel(logfolder, halls, neighbours,nvdum=None, maxtime=None, maxgap=None, printtime=None, log=False, elevatorVertices=[]):
    if nvdum== None:
        nvdum=max(list(neighbours.keys()))

    # print(f"IN MODELRUN VDUM:{nvdum}")
    m = Model()
    if maxtime != None:
        m.Params.TimeLimit = maxtime
    if maxgap!=None:
        m.Params.MIPGap= maxgap
    if printtime!=None:
        m.Params.DisplayInterval= printtime
    if log:
        if type(log)== str:
            print(f"date: {log} will be the logfile")
            if ".log" in log:
                logfile=log
            else:
                logfile=f"\\{log}.log"
        else:
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            datenew = date.replace(':', '-')
            logfile = "\\log" + datenew + ".log"
        m.Params.LogFile= logfolder+logfile
    # Variables: the hallway connecting crossing i and j in the tour?
    varssol = m.addVars(halls.keys(), vtype=GRB.BINARY, name='x')

    # Symmetric direction: use dict.update to alias variable with new key
    varssol.update({(j, i): varssol[i, j] for i, j in varssol.keys()})

    # Add auxiliary variable to help ensure even degree
    varsaux = m.addVars(range(nvdum+1), vtype=GRB.INTEGER, name="y")

    # Set the objective function for the model
    m.setObjective(sum([halls[e] * varssol[e] for e in halls.keys()]), sense=GRB.MAXIMIZE)

    # Add the even degree constraint for dummyvetex nvdum=2:
    # m.addConstr(sum([varssol[(nvdum, e)] for e in neighbours[nvdum]]) == 2, name='evenDegreeVDUM')
    m.addConstr(varsaux[nvdum]==1, name='evenDegreeVDUM')
    # Add the even degree constraint for the other vertices
    for i in range(nvdum+1):
        m.addConstr(sum([varssol[(i, e)] for e in neighbours[i]]) == 2 * varsaux[i],
                        name=f'evenDegreeVertex{i}')

    # Add the constraint that forbids you to use an elevator multiple times:
    for name, vertex in elevatorVertices.items():
        m.addConstr(varsaux[vertex]<= 1, name=f'visit{name}AtMostOnce')

    # Set up for the callbacks/ lazy constraints for connectivity
    m.Params.LazyConstraints = 1
    cb = TSPCallback(range(nvdum+1), varssol, neighbours,halls, nvdum)

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

#Define a function returning the building and floor a vertex vnum lies in
def getBuildingFloor(vnum, nodeToCoordinate):
   return nodeToCoordinate[vnum]['Building'], nodeToCoordinate[vnum]['Floor']

#Translate a vertex pair representing an edge, back to coordinates:
def getCoordinatesPair(vtuple, nodeToCoordinate):
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


def constructTrail(edges,vdum):
    nedges= len(edges)
    print(f"{len(edges)} edges to construct trail from:{edges}")
    trail=[]
    node_neighbors = defaultdict(list)
    #Get the neighbourhoods of the induced graph by the edges, but not considering the dummy vertex and dummy edges
    dummyEdges=[]
    for i, j in edges:
        if i != vdum and j != vdum:
            if i in node_neighbors:
                node_neighbors[i].append(j)
            else:
                node_neighbors[i]=[j]
            if j in node_neighbors:
                node_neighbors[j].append(i)
            else:
                node_neighbors[j]=[i]
        else:
            dummyEdges.append((i,j))
            dummyEdges.append((j,i))

    # Remove the dummy edges from the used edges
    for edge in dummyEdges:
        if edge in edges:
            edges.remove(edge)
    print(f"after removing dummy edges we heve :{len(edges)} edges")
    for node, neighbs in node_neighbors.items():
        if len(neighbs)==1:
            # We found a start!
            currentNode= node
            break

    while len(trail)<nedges-2:
        neighbs= node_neighbors[currentNode]

        if len(neighbs)==1: # We go to this neighbour
            vertex= neighbs[0]
            trail.append((currentNode,vertex))
            if (currentNode, vertex) in edges:
                edges.remove((currentNode, vertex))
            else:
                edges.remove((vertex, currentNode))
            node_neighbors[currentNode].remove(vertex)
            node_neighbors[vertex].remove(currentNode)
            currentNode= vertex

        elif len(neighbs)==0: # No more places to go, we used all the edges
            print(f"WE QUIT THE WIHILE LOOP SINCE NEIGHBS ARE:{neighbs} for len trail:{len(trail)} and nedges to sort{nedges}")
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
                node_neighbors[edgeToConsider[0]].remove(edgeToConsider[1])
                node_neighbors[edgeToConsider[1]].remove(edgeToConsider[0])
                nReachableAfter= getReachable(node_neighbors, currentNode)

                if nReachableAfter< nReachableBefore: # It is a bridge
                    #edge edgeToConsider is a bridge so look for the next after adding the vertices back to nodeneighbours
                    node_neighbors[edgeToConsider[0]].append(edgeToConsider[1])
                    node_neighbors[edgeToConsider[1]].append(edgeToConsider[0])

                else:
                    # edge edgeToConsider is not a bridge, so we can take this one
                    edges.remove(edgeToConsider)
                    trail.append((currentNode, vertex))
                    currentNode=vertex
                    break
            if not currentNode == vertex:
                print(f"ERROOOORRRRR ???? length trail: {len(trail)} we did not find a nonbridge edge? is that even possible?? for currentNode {currentNode}")
                break
    print(f"WE CONSTRUCTED TRAIL of {len(trail)} edges: {trail}")
    return trail

def drawEdgesInFloorplans(edges, nodeToCoordinate,elevatorEdges,specialEdges, figuresResultBuildings,resultfolder, prefixfilename):
    rainbowColors= getRainbowColors(len(edges))
    startedge= edges[0]
    endedge= edges[-1]

    for i,edge in enumerate(edges):
        if edge in elevatorEdges:
            # We can not draw this line in a floor plan
            continue
        building0, floor0= getBuildingFloor(edge[0], nodeToCoordinate)
        building1, floor1= getBuildingFloor(edge[1], nodeToCoordinate)

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
                        if edges[i + 1] in elevatorEdges:
                            # We step into the elevator so we need to print a number
                            toFloor = nodeToCoordinate[edges[i + 3][0]]['Floor']
                            drawCoord = nodeToCoordinate[edge[1]]['Location']
                        elif edges[i - 1] in elevatorEdges:
                            # we step out of the elevator, but still print a number from which floor we came for if you walk in the other order
                            toFloor = nodeToCoordinate[edges[i - 3][1]]['Floor']
                            drawCoord = nodeToCoordinate[edge[0]]['Location']
                        else:
                            print(f"EDGE {edge} with name{specialEdges[edge]} is not in or out of an elevator!")

                        if building0 == "CARRE 1412":
                            if floor0 == '4':  # since this floor had a weird shift in it
                                drawCoord = drawCoord + 126.822 + 494.891j

                        # Add the number
                        text_element = ET.Element("text", attrib={
                            "x": str(drawCoord.real),
                            "y": str(drawCoord.imag),
                            "font-size": "50",  # Font size in pixels
                            "fill": "saddlebrown",  # Text color
                            "stroke": "saddlebrown"
                        })
                        text_element.text = str(toFloor)
                        thisRoot = figuresResultBuildings[building0][floor0]['root']
                        thisRoot.append(text_element)


                else:
                    # Normal edges are also colored according to the fading rainbow.
                    color=rgb_to_string(rainbowColors[i])
                # Get the coordinates to draw a line connecting them in the correct floor plan
                startco, endco = getCoordinatesPair(edge, nodeToCoordinate)
                if building0 =="CARRE 1412":
                    if floor0 == '4': # since this floor had a weird shift in it
                        startco=startco +126.822+ 494.891j
                        endco=endco +126.822+ 494.891j

                new_path_element = ET.Element("path", attrib={
                    "d": Path(Line(start=startco, end=endco)).d(),
                    "stroke": color,
                    "fill": "none",
                    "stroke-width": "5"
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
                    "font-size": "50",  # Font size in pixels
                    "fill": "saddlebrown",  # Text color
                    "stroke": "saddlebrown"
                })
                text_element.text = str(toFloor)
                thisRoot = figuresResultBuildings[building0][floor0]['root']
                thisRoot.append(text_element)

                #Also add the other way
                toFloor = nodeToCoordinate[edge[0]]['Floor']
                drawCoord = nodeToCoordinate[edge[1]]['Location']
                if building0 == "CARRE 1412":
                    if floor1 == '4':  # since this floor had a weird shift in it
                        drawCoord = drawCoord + 126.822 + 494.891j

                text_element = ET.Element("text", attrib={
                    "x": str(drawCoord.real),
                    "y": str(drawCoord.imag),
                    "font-size": "50",  # Font size in pixels
                    "fill": "saddlebrown",  # Text color
                    "stroke": "saddlebrown"
                })
                text_element.text = str(toFloor)
                thisRoot = figuresResultBuildings[building0][floor1]['root']
                thisRoot.append(text_element)
        else:
            # We go from building0 to building1
            toBuild = nodeToCoordinate[edge[1]]['Building']
            drawCoord = nodeToCoordinate[edge[0]]['Location']
            if building0 == "CARRE 1412":
                if floor0 == '4': # since this floor had a weird shift in it
                    drawCoord = drawCoord + 126.822 + 494.891j

            text_element = ET.Element("text", attrib={
                "x": str(drawCoord.real),
                "y": str(drawCoord.imag),
                "font-size": "50",  # Font size in pixels
                "fill": "saddlebrown",  # Text color
                "stroke": "saddlebrown"
            })
            text_element.text = toBuild
            thisRoot = figuresResultBuildings[building0][floor0]['root']
            thisRoot.append(text_element)

    # Draw the figures in a new file:
    for building, buildinginfo in figuresResultBuildings.items():
        buildingResultPath= resultfolder+f"\\{building}"
        for floor, floorinfo in buildinginfo.items():
            buildingName, buildingNumber = splitNameNumber(building)
            floortree= floorinfo['tree']
            testfilename= f"\\{prefixfilename}{buildingNumber}.{floor}.svg"
            floortree.write(buildingResultPath+testfilename)

def plotBounds(logfolder, logfile, title, showresult=False, savename=False):
    logpath = logfolder + logfile
    print(f"path to parse: {logpath}")
    results = glt.parse(logpath)
    nodelogs = results.progress("nodelog")
    pd.set_option("display.max_columns", None)
    # print(f"type of nodelogs: {nodelogs}, and has columns: {[i for i in nodelogs]}")
    print(nodelogs.head(10))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nodelogs["Time"], y=nodelogs["Incumbent"], mode='markers', name="Primal Bound"))
    fig.add_trace(go.Scatter(x=nodelogs["Time"], y=nodelogs["BestBd"], mode='markers', name="Dual Bound"))
    fig.update_xaxes(title_text="Runtime in seconds")
    fig.update_yaxes(title_text="Objective value function (in meters)")
    fig.update_layout(title_text=title, font_size=24)
    # if showresult:
    #     fig.show()

    if savename:
        PATHplot = logfolder + f"\\boundsOverTime"
        if not os.path.exists(PATHplot):
            os.mkdir(PATHplot)
        print(f"pathplot: {PATHplot}")
        print(f"file will be saved to :{PATHplot + "\\" + savename}")
        fig.write_html(PATHplot+"\\"+savename)

if __name__ == "__main__":

    # Get the path to the drawings
    dir_path = os.path.dirname(os.path.realpath(__file__))
    PATH= os.path.abspath(os.path.join(dir_path, os.pardir))
    PATH_test= PATH+"\\Eerste aantekeningen\\MoreFloors"
    PATH_drawings= PATH_test +"\\OriginalPaths"
    PATH_empty= PATH_test+"\\Empty Floors"
    PATH_result= PATH_test+"\\ResultPaths"
    PATH_log=PATH_test+"\\Logs"

    # Initiate some data structures to save information in
    nodeToCoordinate={}
    coordinateToNode={}

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

    # Define a dictionary where each vertex maps to a set of its neighbours
    neighboursold= {i:set() for i in range(nextnode)} #to store the neighbours that we draw in the floor plans

    for v1,v2 in hallways.keys():
        neighboursold[v1].add(v2)
        neighboursold[v2].add(v1)

    neighbours=deepcopy(neighboursold) # deep copy the neighbourhood, to which we can add the
    # Connections regarding staircases, elevators and connections between buildings

    elevatorVertices=dict() # Keep a dictionary storing the vertices for elevators, which need to have degree <=2
    elevatorEdges=set()
    connected=[]
    for pathname, pathinfo in specialPaths.items():
        if pathname[2]=="E": #We have an edge to an elevator
            startname = pathname[:5]
            if not startname in connected: #If we have not yet connected the floors that can be reached from this elevator, do so
                print(startname)
                elevatorConnects=[key for key in specialPaths.keys() if startname in key]
                velevator= nextnode
                elevatorVertices[startname]=velevator
                nextnode+=1
                neighbours[velevator]=set()
                for edge in elevatorConnects:
                    end= findSingleEnd(edge, neighboursold, specialPaths)
                    hallways[(end, velevator)]=1
                    elevatorEdges.add((end, velevator))
                    elevatorEdges.add((velevator,end))
                    neighbours[end].add(velevator)
                    neighbours[velevator].add(end)
                connected.append(startname)
        elif pathname[2] == "S": #We have an edge to a staircase
            if pathname not in connected: # if we have not connected these two floors with this staircase
                end1= findSingleEnd(pathname, neighboursold,specialPaths)
                otherSide= pathname[:5]+pathname[7:]+pathname[5:7]
                if otherSide in specialPaths:
                    print(f"original stair:{pathname}, other side:{otherSide}")
                    end2=findSingleEnd(otherSide, neighboursold,specialPaths)
                    hallways[(end1, end2)] = 7 # 1 stair 7 meter? idkkk
                    neighbours[end1].add(end2)
                    neighbours[end2].add(end1)
                    connected.append(pathname)
                    connected.append(otherSide)
        elif pathname[1] in ["1","2","3"]: #connecting buildings? naming conv?
            # otherSide = pathname[3:5]+pathname[2]+pathname[0:2]+pathname[5:]
            print(f"connection building from CR:{pathname}")
        elif pathname[2] == "C":
            otherSide = pathname[4:6]+pathname[2:4]+pathname[0:2]
            print(f"This hallway {pathname} ends at a walking bridge and the other side is: {otherSide}")
        elif pathname[2]=='X':
            print(f"we have an exit to outdoors: {pathname}")
        else:
            print(f"somehting went wrong: {pathname} not a stair, elevator, connection or exit")


    neighboursnew, hallwaysnew, vdummy = addDummy(neighbours, hallways)
    #
    # vdum= nextnode
    # nextnode += 1
    #
    # neighbours[vdum]= set(range(vdum))
    #
    # for i in range(nextnode):
    #     hallways[(vdum, i)]=0
    #     neighbours[i].add(vdum)


    print(f"Neighbours old is new? {neighboursold==neighbours}")
    print(f"two ways of adding the dummy neighbourhoood the same:{neighboursnew==neighbours}\n and for the hallways? {hallways==hallwaysnew}")
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    datenew = date.replace(':', '-')
    logfile = "\\log" + datenew + ".log"

    model, varshall, varsdegree = runModel(PATH_log, hallwaysnew, neighboursnew, nvdum=vdummy,maxtime=600, printtime= 5, log= logfile, elevatorVertices=elevatorVertices)

    lengthLongestTrail=model.getAttr('ObjVal')
    print(f"The longest trail is {lengthLongestTrail} meters long")
    used_edges= getEdgesResult(model, varshall)
    # drawEdgesInFloorplans([edge for edge in used_edges if vdummy not in edge], nodeToCoordinate,elevatorEdges,specialEdges, figuresResultBuildings,PATH_result, prefixfilename='isconnectedresult')

    # vdummy=max(list(neighbours.keys()))
    print(f"we have {vdummy} as dummy vertex")
    print(f"edges used that are connected to the dummy vertex: {[edge for edge in used_edges if vdummy in edge]}")
    pprint(f"The used edges in the solution are:\n{used_edges}")
    trailresult= constructTrail(used_edges, vdummy)
    drawEdgesInFloorplans(trailresult, nodeToCoordinate,elevatorEdges,specialEdges, figuresResultBuildings,PATH_result, prefixfilename='reorderedMoreFloors')

    titleplot="The bounds on the length of the longest trail on Carré floor 1,2,3 and 4 together,<br> at each moment in time when running the gurobi solver"
    plotBounds(PATH_log, logfile, titleplot, showresult=True, savename=f'{datenew}.svg')
