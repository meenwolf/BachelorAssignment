import os
from os.path import split

import kaleido
from holoviews.plotting.bokeh.styles import font_size, validate
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
import sys
import json


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

# Start exporting hallways: dict with keys: (v1, v2) representing an edge between vertex v1 and vertex v2, values are the weights of the hallway in real life
# nodeToCoordinate: dict with keys: v1, representing the vertex v1, and values a dict containing the building, floor and coordinate on that floor of vertex v1, example: v1:{"Building": "CARRE 1412", "Floor": "3", "Location": 100.9347+221.876j}
# buildingScales: dict with keys the building names, and the values a dict that contains for each floor the scale x. (that is: 1meter in real life: distance x in the floor plan)
# Convert and write JSON object to file

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

def plotBounds(logfolder, logfile, title, savename=False):
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

    if savename:
        PATHplot = logfolder + f"\\boundsOverTime"
        if not os.path.exists(PATHplot):
            os.mkdir(PATHplot)
        print(f"pathplot: {PATHplot}")
        print(f"file will be saved to :{PATHplot + "\\" + savename}")
        fig.write_html(PATHplot+"\\"+savename, auto_open=False, validate=False)


def exportGraphinfo(halls, nodeToCoordinate, scales,trail, prefix=""):
    weightedEdges = [{'key': key, 'value': value} for key, value in halls.items()]
    with open(prefix+"weigthedEdges.json", "w") as outfile:
        json.dump(weightedEdges, outfile)

    nodeToCoordinates = {vertex: {'Building': info["Building"], "Floor": info["Floor"], "x": np.real(info["Location"]),
                                  "y": np.imag(info["Location"])} for vertex, info in nodeToCoordinate.items()}
    with open(prefix+"nodeToCoordinates.json", "w") as outfile:
        json.dump(nodeToCoordinates, outfile)

    with open(prefix+"buildingScales.json", "w") as outfile:
        json.dump(scales, outfile)

    todraw=[]
    for edge in trail:
        if (edge[0], edge[1]) in halls:
            todraw.append({"key": edge, "value": halls[edge]})
        else:
            todraw.append({"key": edge, "value": halls[(edge[1],edge[0])]})

    with open(prefix+"trail.json", "w") as outfile:
        json.dump(todraw, outfile)

def runModelends(logfolder, halls, neighbours,ends=[],nvdum=None, maxtime=None, maxgap=None, printtime=None, log=False, elevatorVertices=[]):

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
    varsaux = m.addVars(neighbours.keys(), vtype=GRB.INTEGER, name="y")

    # Set the objective function for the model
    m.setObjective(sum([halls[e] * varssol[e] for e in halls.keys()]), sense=GRB.MAXIMIZE)

    # Add the even degree constraint for dummyvetex nvdum=2:
    m.addConstr(sum([varssol[(nvdum, e)] for e in neighbours[nvdum]]) == 2, name='evenDegreeVDUM')

    # Add the even degree constraint for the other vertices
    for i in neighbours.keys():
        m.addConstr(sum([varssol[(i, e)] for e in neighbours[i]]) == 2 * varsaux[i],
                        name=f'evenDegreeVertex{i}')

    # Add the constraint that forbids you to use an elevator multiple times:
    for name, vertex in elevatorVertices.items():
        if vertex in neighbours.keys():
            m.addConstr(varsaux[vertex]<= 1, name=f'visit{name}AtMostOnce')

    # Add the constraint that the vertices in ends must be connected to vdum in the solution, that is, x(vdum,end)=1 for end in end
    assert len(ends) in [0,1,2]
    if len(ends)>0:
        for end in ends:
            # print(f"ENDS:{ends}:end{end} in runmodelends, {(nvdum, end)}")
            m.addConstr(varssol[(nvdum, end)]==1, name=f'mustStartorEndin{end}')
            # m.addConstr(varssol[(end, nvdum)]==1, name=f'mustStartorEndin{end}')



    # Set up for the callbacks/ lazy constraints for connectivity
    m.Params.LazyConstraints = 1
    cb = TSPCallback(list(neighbours.keys()), varssol, neighbours,halls, nvdum)

    # Call optimize with the callback structure to get a connected solution solution
    m.optimize(cb)

    return m, varssol, varsaux

def isCutEdge(node_neighbors, edgeToConsider,currentNode=None, getComponent=False):
    if currentNode== None:
        currentNode=edgeToConsider[0]
    reachableBefore= getReachable(node_neighbors, currentNode)
    # print(f"to start with, reachable from {currentNode} are {len(nReachableBefore)} nodes: {nReachableBefore}")
    # print(f"we call is cut edge on {node_neighbors}\n with edge to consider{edgeToConsider}, and currentnode={currentNode}")
    node_neighbors[edgeToConsider[0]].remove(edgeToConsider[1])
    node_neighbors[edgeToConsider[1]].remove(edgeToConsider[0])
    reachableAfter= getReachable(node_neighbors, currentNode)
    node_neighbors[edgeToConsider[0]].add(edgeToConsider[1])
    node_neighbors[edgeToConsider[1]].add(edgeToConsider[0])
    if len(reachableAfter)< len(reachableBefore): # It is a bridge
        if getComponent:
            otherVertices= [vertex for vertex in node_neighbors.keys() if vertex not in reachableAfter]
            return True, reachableAfter, otherVertices
        else:
            return True
    else:
        return False

def isTwoCut(node_neighbors, edge1, edge2, getComponent=False):
    currentNode= edge1[0]
    reachableBefore= getReachable(node_neighbors, currentNode)
    # print(f"to start with, reachable from {currentNode} are {len(nReachableBefore)} nodes: {nReachableBefore}")
    node_neighbors[edge1[0]].remove(edge1[1])
    node_neighbors[edge1[1]].remove(edge1[0])
    node_neighbors[edge2[0]].remove(edge2[1])
    node_neighbors[edge2[1]].remove(edge2[0])

    reachableAfter= getReachable(node_neighbors, currentNode)
    node_neighbors[edge1[0]].add(edge1[1])
    node_neighbors[edge1[1]].add(edge1[0])
    node_neighbors[edge2[0]].add(edge2[1])
    node_neighbors[edge2[1]].add(edge2[0])
    if len(reachableAfter)< len(reachableBefore): # It is a 2 edge cut
        if getComponent:
            otherVertices= [vertex for vertex in node_neighbors.keys() if vertex not in reachableAfter]
            return True, reachableAfter, otherVertices
        else:
            return True
    else:
        return False


def findCuts(edges, specialPaths, neighbourhood=None):
    onecuts = dict()
    twocuts = dict(dict())
    if neighbourhood ==None:
        neighbourhood = getNeighbourhood(edges)
    # bridges = dict()
    potentialtwocuts = dict()
    for path, info in specialPaths.items():
        if path[1] == "0" and path not in ['C01HBWA', 'C01CRWH', 'C02CRNL']:  # meaning that it is a bridge
            # print(f"specialPaths: {specialPaths}")
            # print(f"path is of type {type(specialPaths[path])}: {specialPaths[path]}")
            # print(f"start: {specialPaths[path]['Start']}")
            # print(f"vnum: {specialPaths[path]['Start']['Vnum']}")


            pathedge = (specialPaths[path]['Start']['Vnum'], specialPaths[path]['End']['Vnum'])
            if pathedge in edges: # meaning that this bridge is indeed in the component we are looking for
                cutedge, component1, component2 = isCutEdge(neighbourhood, pathedge, getComponent=True)
                if cutedge:
                    onecuts[pathedge] = {'Name': path, 'Component1vertices': component1, 'Component2vertices': component2}
                    return onecuts
                else:  # check for each other combination if it forms a 2 cut.
                    potentialtwocuts[pathedge] = path


    # for bridge, name in bridges.items():
    #     cutedge, component1, component2 = isCutEdge(neighbourhood, bridge, getComponent=True)
    #     if cutedge:
    #         onecuts[bridge] = {'Name': name, 'Component1vertices': component1, 'Component2vertices': component2}
    #     else:  # check for each other combination if it forms a 2 cut.
    #         potentialtwocuts[bridge]=name

    for path1, path2 in combinations(potentialtwocuts.keys(), 2):
        twocut, component1, component2 = isTwoCut(neighbourhood, path1, path2, getComponent=True)
        if twocut:
            twocuts[(path1,path2)] = {'Name':potentialtwocuts[path1]+'and'+potentialtwocuts[path2] ,'Component1vertices': component1, 'Component2vertices': component2}
            return twocuts
    return False


def getBuildings(vertices, nodeToCoordinate):
    buildings=dict()
    for vertex in vertices:
        if vertex in nodeToCoordinate: #meaning that the vertex is not representing an elevator.
            building= nodeToCoordinate[vertex]['Building']
            floor= nodeToCoordinate[vertex]['Floor']
            if building in buildings:
                buildings[building].add(floor)
            else:
                buildings[building] = {floor}
    return buildings


def getEdgesComponent(weightedEdges, vertices): # neighbourhood=None):
    if type(weightedEdges) != dict:
        edges= dict()
        for edge in weightedEdges:
            edges[edge]=0
    else:
        edges = weightedEdges

    edgesComponent= dict()
    for edge, weight in edges.items():
        if edge[0] in vertices:
            if edge[1] in vertices:
                edgesComponent[(min(edge[0], edge[1]), max(edge[0], edge[1]))]=weight

    # print(f"length of edgetotalbased inn number of edges: {len(edgesComponent.keys())}")
    return edgesComponent

#This version used the iscutedge, but since for only carre it finds a decent option pretty quick, lets stick with that for now.
def constructTrailCheckComponents(edgeswithdum,vdum):
    # print(f"construct trail with {len(edgeswithdum)} edges")
    nedges= len(edgeswithdum)
    trail=[]
    # node_neighbors = defaultdict(list)
    #Get the neighbourhoods of the induced graph by the edges, but not considering the dummy vertex and dummy edges
    edges = [ edge for edge in edgeswithdum if vdum not in edge]
    # for edge in edgeswithdum:
    #     if vdum in edge:
    #         edges.remove(edge)
    node_neighbors = getNeighbourhood(edges)

    #
    #     else:
    #         dummyEdges.append((i,j))
    #         dummyEdges.append((j,i))
    #
    # # Remove the dummy edges from the used edges
    # for edge in dummyEdges:
    #     if edge in edges:
    #         edges.remove(edge)

    print(node_neighbors)
    currentNode=False
    for node, neighbs in node_neighbors.items():
        neighbs= list(neighbs)
        if len(neighbs)%2 ==1:
            # We found a start!
            # print(f"potential start: {node}")
            currentNode=node
    if not currentNode:
        # print(f"all vertices have even degree, so just take the first one {edges[0][0]}")
        currentNode = edges[0][0]

    while len(trail)<nedges-2:
        neighbs= list(node_neighbors[currentNode])
        # print(f"we are at: {currentNode} with neighbours {neighbs}\n")

        if len(neighbs)==1: # We go to this neighbour
            vertex= neighbs[0]
            # print(f'we only have one neighbour to visit so we go to {vertex}')
            trail.append((currentNode,vertex))
            if (vertex, currentNode) in edges:
                edges.remove((vertex, currentNode))
            if (currentNode, vertex) in edges:
                edges.remove((currentNode, vertex))
            node_neighbors[currentNode].remove(vertex)
            node_neighbors[vertex].remove(currentNode)
            currentNode= vertex

        elif len(neighbs)==0: # No more places to go, we used all the edges
            # print(f"No more places to go, we used all the edges and break")
            break

        else: # We have to loop over the vertices in the neighourhood and check if the edge to there is a bridge or not
            for vertex in neighbs:
                if (currentNode, vertex) in edges:
                    edgeToConsider= (currentNode, vertex)
                if (vertex, currentNode) in edges:
                    edgeToConsider= (vertex, currentNode)
                # else:
                #     print(f"ERROR: vertex {vertex} is a neighbour of current node {currentNode} but no edge is in the edge set")

                #check if this edge is a bridge
                isBridge= isCutEdge(node_neighbors= node_neighbors, edgeToConsider=edgeToConsider, currentNode=currentNode, getComponent=False)
                # nReachableBefore= getReachable(node_neighbors, currentNode)
                # print(f"to start with, reachable from {currentNode} are {len(nReachableBefore)} nodes: {nReachableBefore}")
                # node_neighbors[edgeToConsider[0]].remove(edgeToConsider[1])
                # node_neighbors[edgeToConsider[1]].remove(edgeToConsider[0])
                # nReachableAfter= getReachable(node_neighbors, currentNode)
                # print(f"after deleting the edge to {vertex} there are {len(nReachableAfter)} nodes reachable: {nReachableAfter}")

                # if isBridge: # It is a bridge
                #     print(f"So this edge is a bridge, don't go to {vertex} now, but continue searching.")
                #     #edge edgeToConsider is a bridge so look for the next after adding the vertices back to nodeneighbours
                # else: # not a bridge so we take this edge
                if not isBridge:
                    node_neighbors[edgeToConsider[0]].remove(edgeToConsider[1])
                    node_neighbors[edgeToConsider[1]].remove(edgeToConsider[0])
                    # print(f"edge  to {vertex} is not a bridge, so we can take this one")
                    edges.remove(edgeToConsider)
                    if (edgeToConsider[1], edgeToConsider[0]) in edges:
                        edges.remove((edgeToConsider[1], edgeToConsider[0]))
                    trail.append((currentNode, vertex))
                    currentNode=vertex
                    # print(f"we went to vertex {currentNode} and break the forloop")
                    break
            if not currentNode == vertex:
                print(f"ERROOOORRRRR ???? length trail: {len(trail)} we did not find a nonbridge edge? is that even possible?? for currentNode {currentNode}")
                break
    return trail

def findTrailComponent(logfolder, resultfolder, edges, specialEdges, figuresResultBuildings, nodeToCoordinate, elevatorEdges,vdummy,neighbours=None,
                       ends=[],maxtime=None, maxgap=None, printtime=None, logfile=False, elevatorVertices=[],
                       prefixdrawcomp=False, plotboundsname=False, showboundplot=False, saveboundplotname=False):
    if neighbours == None:
        neighbours = getNeighbourhood(edges)
    neighboursnew, edgesnew,vdummy = addDummy(neighbours, edges, vdummy)
    if len(ends) in [0,1,2]:
        print(f"run new model")
        model, varshall, varsdegree = runModelends(ends=ends,logfolder=logfolder, halls=edgesnew, nvdum=vdummy,
                                               neighbours=neighboursnew,
                                               maxtime=maxtime, maxgap=maxgap, printtime=printtime, log=logfile,
                                               elevatorVertices=elevatorVertices)
        # lengthLongestTrail = model.getAttr('ObjVal')
        # print(f"The longest trail is {lengthLongestTrail} meters long")
        used_edges = getEdgesResult(model, varshall)
        # drawEdgesInFloorplans(edges=[edge for edge in used_edges if vdummy not in edge], nodeToCoordinate=nodeToCoordinate, elevatorEdges=elevatorEdges,
        #                       specialEdges=specialEdges, figuresResultBuildings=figuresResultBuildings,
        #                       resultfolder=PATH_result, prefixfilename='edgesResult')
        # print(f"{len(getReachable(getNeighbourhood(used_edges),vdummy))} vertices of the {len(getNeighbourhood(used_edges))} vertices are reachable from vdum{vdummy}")
        # print(f"we have {vdummy} as dummy vertex for this component")
        # print(f"and the {len(used_edges)} used edges are:{used_edges}")
        trailresult = constructTrailCheckComponents(used_edges, vdummy)
        lengthtrail=0
        for edge in trailresult:
            if edge in edgesnew:
                lengthtrail+=edgesnew[edge]
            else:
                lengthtrail+=edgesnew[(edge[1],edge[0])]
    # else:
    #     print(f"run old model")
    #     model, varshall, varsdegree = runModel(logfolder=logfolder, halls=edgesnew, nvdum=vdummy, neighbours=neighboursnew,
    #                                            maxtime=maxtime, maxgap=maxgap, printtime=printtime, log=logfile,
    #                                            elevatorVertices=elevatorVertices)
    #
    #     # lengthLongestTrail = model.getAttr('ObjVal')
    #     # print(f"The longest trail is {lengthLongestTrail} meters long")
    #     used_edges = getEdgesResult(model, varshall)
    #     # vdummy = max(list(neighbours.keys()))
    #     # print(f"we have {vdummy} as dummy vertex for this component")
    #     # print(f"and the {len(used_edges)} used edges are:")
    #     trailresult = constructTrail(used_edges, vdummy)
    #     lengthtrail= sum([edgesnew[edge] for edge in trailresult])

    print(f"findTrailComponent result contains {len(trailresult)} edges and is {lengthtrail} meters long")
    # if type(prefixdrawcomp) == str:
    #     drawEdgesInFloorplans(edges=trailresult, nodeToCoordinate=nodeToCoordinate, elevatorEdges=elevatorEdges,
    #                           specialEdges=specialEdges, figuresResultBuildings=figuresResultBuildings,
    #                           resultfolder=resultfolder, prefixfilename=prefixdrawcomp)
    # if type(plotboundsname) == str:
    #     plotBounds(logfolder=logfolder, logfile=logfile, title=plotboundsname, showresult=showboundplot,
    #                savename=saveboundplotname)

    return trailresult, lengthtrail

def reduceGraphBridges(specialPaths, logfolder, resultfolder, edges, specialEdges, figuresResultBuildings, nodeToCoordinate, vdummy,elevatorEdges=[], ends=[],neighbours=None,maxtime=None, maxgap=None, printtime=None, logfile=False, elevatorVertices=[],prefixdrawcomp=False, plotboundsname=False, showboundplot=False, saveboundplotname=False):
    if neighbours ==None:
        neighbours= getNeighbourhood(edges.keys())
    bridges=dict()
    for path, info in specialPaths.items():
        if path[1] == "0" and path not in ['C01HBWA', 'C01CRWH', 'C02CRNL']:  # meaning that it is a bridge
            # print(f"EdgeCutspecialPaths: {specialPaths}")
            # print(f"path is of type {type(specialPaths[path])}: {specialPaths[path]}")
            # print(f"start: {specialPaths[path]['Start']}")
            # print(f"vnum: {specialPaths[path]['Start']['Vnum']}")

            pathedge = (specialPaths[path]['Start']['Vnum'], specialPaths[path]['End']['Vnum'])


            bridges[pathedge] = path
    cutMade = findCuts(edges=edges.keys(), specialPaths=specialPaths, neighbourhood=neighbours)
    # onecuts=[]
    if not cutMade:
        print(f"reduce graph bridges found no 1 or 2 edgecut, so we have to just call findtrail component")
        trailcomp, lengthtrail = findTrailComponent(ends=ends, logfolder=logfolder, resultfolder=resultfolder,
                                                    edges=edges, specialEdges=specialEdges,
                                                    figuresResultBuildings=figuresResultBuildings,
                                                    nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,elevatorEdges=elevatorEdges,
                                                    neighbours=neighbours, maxtime=maxtime,
                                                    maxgap=maxgap, printtime=printtime, logfile=logfile,
                                                    elevatorVertices=elevatorVertices,
                                                    prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                                    showboundplot=showboundplot,
                                                    saveboundplotname=saveboundplotname)

        return trailcomp, lengthtrail


    else: #meaning that there are 1 or 2 cuts
        print(f"reduceGraphbridges found a cut edge so lets branch off to the correct version, we now have {len(edges)} edges in our component!")
        for cut, info in cutMade.items():
            if 'and' in info['Name']:
                print(f"We have a 2 cut")
            else:
                print(f"We have a 1 cut")
                print(f"one cuts are: {cutMade}\n with keys: {cutMade.keys()}")
                #just take the first onecut and repeat the process.
                # print(f"onecutedge:{cut} with: {nodeToCoordinate[cut[0]]} and \n {nodeToCoordinate[cut[1]]}")
                # print(f"onecutinfo: {info}")
                # print(f"neighbourhood of the cut edge:{neighbours[cut[0]]} and {neighbours[cut[1]]}")
                # # Sort the components left after removing one 1cut such that the buildings and component of onecut[0] are in c0 and building0
                # and the ones of onecut[1] are in c1 and building 1.
                v0= cut[0]
                v1= cut[1]
                if v0 in info['Component1vertices']:
                    c0= info['Component1vertices']
                    c1= info['Component2vertices']
                else:
                    c0 = info['Component2vertices']
                    c1 = info['Component1vertices']

                building0 = nodeToCoordinate[v0]['Building']
                building1 = nodeToCoordinate[v1]['Building']
                edgesC0= getEdgesComponent(edges, c0) #weighted edges
                edgesC1= getEdgesComponent(edges, c1) #weighted edges

                print(f"one cut edge separating hallways in {getBuildings(c0, nodeToCoordinate)} \n from hallways in {getBuildings(c1, nodeToCoordinate)}")
                # Now find the longest trail in c0, called Tc0, in c1, called Tc1
                # Also find the longest trail in c0 starting in v0, called Tv0
                # Also find the longest trail in c1 starting in v1, called Tv1,
                # Then return the longest of Tc0, Tc1, Tv0+Tv1+w((v0,v1))

                #check if we needed to start in a certain vertex and in which component that lies in.
                assert len(ends) in [0,1,2] # Since the only options are for a component to have a mandatory start, end, both or none
                if len(ends)==0: #meaning no restrictions on where to start or end
                    print(f"in reduce Graph bridges, we are free to find a longest trail anywhere in this component with c0:nvertices:{len(c0)}nedges:{len(edgesC0)} and then c1:nvertices:{len(c1)} nedges:{len(edgesC1)}.In total {max(list(neighbours.keys()))+1} vertices and {len(edges)} edges")
                    Tc0, LTc0= reduceGraphBridges(logfolder=logfolder, resultfolder=resultfolder, specialPaths=specialPaths, edges=edgesC0, specialEdges=specialEdges,
                           figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                               maxtime=maxtime, maxgap=None, printtime=5, elevatorVertices=elevatorVertices) # longest trail in c0
                    print(f"length Tc0: {LTc0}")
                    Tc1, LTc1 = reduceGraphBridges(logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC1, specialEdges=specialEdges,
                           figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                               maxtime=maxtime, maxgap=None, printtime=5, elevatorVertices=elevatorVertices) # longest trail in c1
                    print(f"length Tc1: {LTc1}")
                    # print(f"ends will be v0:{v0}")
                    Tv0, LTv0= reduceGraphBridges(logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC0, specialEdges=specialEdges,
                           figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                               maxtime=maxtime, maxgap=None, printtime=5, elevatorVertices=elevatorVertices, ends=[v0]) # longest trail in c0 that starts or ends in v0
                    print(f"length in c0 starting from v0: {LTv0}")
                    # print(f"ends will be v1:{v1}")
                    Tv1, LTv1= reduceGraphBridges(logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC1, specialEdges=specialEdges,
                           figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                               maxtime=maxtime, maxgap=None, printtime=5, elevatorVertices=elevatorVertices, ends=[v1]) # longest trail in c1 that starts or ends in v1
                    print(f"length in c1 starting from v1: {LTv1}")
                    # now calculate the length, so first see if v0v1 is in the edges, or the other way around:
                    LTc0c1= LTv0+edges[cut]+LTv1
                    print(f"so length c0c1:{LTc0c1}")

                    T=max(LTc0,LTc0c1 ,LTc1)
                    if T == LTc0:
                        return  Tc0, LTc0
                    elif T== LTc1:
                        return Tc1, LTc1
                    else:
                        # Construct the combined trail, starting in c0, going to v0, then walk the edge to v1, and walk a longest trail in c1
                        # check the order of the edges in Tv0,
                        print(f"edgesTv0:{Tv0}")
                        if v0 in Tv0[0]:
                            print(f"Reverse the order of the list Tv0")
                            Tv0.reverse()
                        elif v0 in Tv0[-1]:
                            print(f"the edges are in the correct order Tv0")

                        else:
                            print(f"ERRORRR the vertex v0 is neither in the start nor end edge of the trail in Tv0??")
                        # check the order of the edges in Tv1,
                        print(f"edges tv1: {Tv1}")
                        if v1 in Tv1[0]:  # meaning that the edges are already in the correct order
                            print(f"order Tv1 correct")
                        elif v1 in Tv1[-1]:
                            print(f"we have to reverse the order of the edges in Tv1")
                            # reverse the order of the list
                            Tv1.reverse()
                            print(f"type of Tv1 reversed:{type(Tv1)},{Tv1}")
                        else:
                            print(f"ERRORRR the vertex v1 is neither in the start nor end edge of the trail in Tv1??")

                        Tc0c1 = Tv0 + [(v0, v1)] + Tv1
                        return Tc0c1, LTc0c1



                    print(f"as we had")
                elif len(ends)==1: #meaning we have 1 mandatory vertex to visit in a longest trail in this component.
                    print(f"in reduce graph bridges we must find a trail that starts or ends in{ends[0]}.")
                    # We have two possible trails to find, one staying in component 0, starting in ends[0], or the longest trail that starts in end[0], and goes through both components by means of taking the cutedge
                    if ends[0] in c0: #meaning we take the longest trail in c0 starting at ends[0] to anywhere in c0
                        #and also find the longest trail in c0 that starts and ends in ends[0], v0.
                        # also find the longest trail in c1 that starts in v1 and ends anywhere in c1.
                        print(f"we're good to go")
                    else:
                        print(f"we have to swap c0 and c1 in order to have end[0] in the same component as c0")
                        tempedges= edgesC0
                        edgesC0= edgesC1
                        edgesC1= tempedges

                        tempvertices= c0
                        c0=c1
                        c1=tempvertices

                        tempv= v0
                        v0=v1
                        v1=tempv

                        tempbuilding= building0
                        building0= building1
                        building1= tempbuilding


                    Tc0, LTc0 = reduceGraphBridges(logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC0,
                                                        specialEdges=specialEdges,
                                                        figuresResultBuildings=figuresResultBuildings, vdummy=vdummy,
                                                        elevatorEdges=elevatorEdges, nodeToCoordinate=nodeToCoordinate,
                                                        maxtime=maxtime, maxgap=None, printtime=5,
                                                        elevatorVertices=elevatorVertices,
                                                        ends=ends)  # longest trail in c1 that starts or ends in v1
                    # print(f"ends will be [ends[0],v0]:{[ends[0],v0]}")
                    Tc0v0, LTc0v0 = reduceGraphBridges(logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC0,
                                                        specialEdges=specialEdges,
                                                        figuresResultBuildings=figuresResultBuildings, vdummy=vdummy,
                                                        elevatorEdges=elevatorEdges, nodeToCoordinate=nodeToCoordinate,
                                                        maxtime=maxtime, maxgap=None, printtime=5,
                                                        elevatorVertices=elevatorVertices,
                                                        ends=[ends[0],v0])  # longest trail in c0 that starts or ends in v0
                    # print(f"ends will be v1:{v1}")
                    Tv1, LTv1 = reduceGraphBridges(logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC1,
                                                        specialEdges=specialEdges,
                                                        figuresResultBuildings=figuresResultBuildings, vdummy=vdummy,
                                                        elevatorEdges=elevatorEdges, nodeToCoordinate=nodeToCoordinate,
                                                        maxtime=maxtime, maxgap=None, printtime=5,
                                                        elevatorVertices=elevatorVertices,
                                                        ends=[v1])  # longest trail in c1 that starts or ends in v1
                    LTc0c1= LTc0v0+edges[cut]+LTv1

                    T = max(LTc0, LTc0c1)
                    if T == LTc0:
                        if ends[0] in Tc0v0[0]:
                            print(f"the edges are in the correct order")
                            # Tc0 = edgesTc0
                        elif ends[0] in Tc0[-1]:
                            # Reverse the order of the list
                            Tc0.reverse()
                        else:
                            print(f"ERRORRR the vertex end0 is neither in the start nor end edge of the trail in Tc0??")
                        # check the order of the edges in Tv1,
                        return Tc0, LTc0
                    else:

                        # Construct the combined trail, starting in end[0] in c0, going to v0, then walk the edge to v1, and walk a longest trail in c1
                        # check the order of the edges in Tv0,
                        if ends[0] in Tc0v0[0]:
                            print(f"the edges are in the correct order")
                        elif ends[0] in Tc0v0[-1]:
                            # Reverse the order of the list
                            Tc0v0.reverse()
                        else:
                            print(f"ERRORRR the vertex v0 is neither in the start nor end edge of the trail in Tv0??")
                        # check the order of the edges in Tv1,

                        if v1 in Tv1[0]:  # meaning that the edges are already in the correct order
                            print(f'the edges are in correct order')
                        elif v1 in Tv1[-1]:
                            # reverse the order of the list
                            Tv1.reverse()
                        else:
                            print(f"ERRORRR the vertex v1 is neither in the start nor end edge of the trail in Tv1??")

                        Tc0c1 = Tc0v0 + [(v0,v1)] + Tv1
                        return Tc0c1, LTc0c1
                else: #we have a mandatory start and a mandatory end, so two options, they either are in the same component or not
                    print(f"in reduce graph bridges we must find a trail that starts and ends in{ends}.")

                    if ends[0] in c0 and ends[1] in c0:
                        print(f"we only have to find a longest trail in c0 with ends {ends}")
                        # print(f"ends was:{ends} and will be the same: {ends}")
                        Tc0, LTc0 = reduceGraphBridges(logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC0,
                                                                specialEdges=specialEdges,
                                                                figuresResultBuildings=figuresResultBuildings,
                                                                elevatorEdges=elevatorEdges,
                                                                nodeToCoordinate=nodeToCoordinate,  vdummy=vdummy,
                                                                maxtime=maxtime, maxgap=None, printtime=5,
                                                                elevatorVertices=elevatorVertices,
                                                                ends=ends)  # longest trail in c0 that starts and ends in vertices in ends
                        if ends[0] in Tc0[0]:#they are in the correct order
                            print(f"edges are in correct order")
                        elif ends[0] in Tc0[-1]: #they are in reverse order
                            Tc0.reverse()
                        else:
                            print(f"ERRORRR the vertex ends0 is neither in the start nor end edge of the trail in Tc0??")
                        return Tc0, LTc0

                    elif ends[0] in c1 and ends[1] in c1:
                        print(f"we only have to find a longest trail in c1 with ends {ends}")
                        Tc1, LTc1 = reduceGraphBridges(logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC1,
                                                            specialEdges=specialEdges,
                                                            figuresResultBuildings=figuresResultBuildings,
                                                            elevatorEdges=elevatorEdges,
                                                            nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                                                            maxtime=maxtime, maxgap=None, printtime=5,
                                                            elevatorVertices=elevatorVertices,
                                                            ends=ends)  # longest trail in c0 that starts and ends in vertices in ends
                        if ends[0] in Tc1[0]:  # they are in the correct order
                            print(f"edges are in correct order")
                        elif ends[0] in Tc1[-1]:  # they are in reverse order
                            Tc1.reverse()
                        else:
                            print(f"ERRORRR the vertex ends0 is neither in the start nor end edge of the trail in Tc1??")
                        return Tc1, LTc1
                    else:
                        print(f"They are in a different component, so we must find a trail from ends[0] to v0 to v1 to ends[0], if ends[0] in c0")
                        if ends[0] in c0:  # meaning we take the longest trail in c0 starting at ends[0] to anywhere in c0
                            # and also find the longest trail in c0 that starts and ends in ends[0], v0.
                            # also find the longest trail in c1 that starts in v1 and ends anywhere in c1.
                            print(f"we're good to go")
                        else:
                            print(f"we have to swap c0 and c1 in order to have end[0] in the same component as c0")
                            tempedges = edgesC0
                            edgesC0 = edgesC1
                            edgesC1 = tempedges

                            tempvertices = c0
                            c0 = c1
                            c1 = tempvertices

                            tempv = v0
                            v0 = v1
                            v1 = tempv

                            tempbuilding = building0
                            building0 = building1
                            building1 = tempbuilding
                        #Now find a longest trail in co from ends0 to v0 and a longest trail in c1 from v1 to ends1
                        # print(f"ends will be: [ends[0],v0]: {[ends[0],v0]}")
                        Tc0v0, LTc0v0 = reduceGraphBridges(logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC0,
                                                                specialEdges=specialEdges,
                                                                figuresResultBuildings=figuresResultBuildings,
                                                                elevatorEdges=elevatorEdges,
                                                                nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                                                                maxtime=maxtime, maxgap=None, printtime=5,
                                                                elevatorVertices=elevatorVertices,
                                                                ends=[ends[0],v0])  # longest trail in c0 that starts or ends in v0
                        # print(f"ends will be: [v1,ends[1]]: {[v1,ends[1]]}")
                        Tc1v1, LTc1v1 = reduceGraphBridges(logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC1,
                                                                specialEdges=specialEdges,
                                                                figuresResultBuildings=figuresResultBuildings,
                                                                elevatorEdges=elevatorEdges,
                                                                nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                                                                maxtime=maxtime, maxgap=None, printtime=5,
                                                                elevatorVertices=elevatorVertices,
                                                                ends=[v1,ends[1]])  # longest trail in c0 that starts or ends in v0
                        LTc0c1= LTc0v0 + edges[cut] + LTc1v1
                        # Construct the combined trail, starting in ends[0] in c0, going to v0, then walk the edge to v1, and walk a longest trail in c1 to ends1
                        # check the order of the edges in Tv0,
                        if ends[0] in Tc0v0[0]:
                            print(f"the edges are in the correct order")
                        elif ends[0] in Tc0v0[-1]:
                            # Reverse the order of the list
                            Tc0v0.reverse()
                        else:
                            print(f"ERRORRR the vertex ends0 is neither in the start nor end edge of the trail in Tc0v0??")
                        # check the order of the edges in Tv1,

                        if v1 in Tc1v1[0]:  # meaning that the edges are already in the correct order
                            print(f"edges are in correct order")
                        elif v1 in Tc1v1[-1]:
                            # reverse the order of the list
                            Tc1v1.reverse()
                        else:
                            print(f"ERRORRR the vertex ends1 is neither in the start nor end edge of the trail in Tc1v1??")

                        Tc0c1 = Tc0v0 + [(v0, v1)] + Tc1v1
                        return Tc0c1, LTc0c1

    # elif True:
    #     print(f"reduce graph bridges found no cut edge, so we have to just call findtrail component")
    #     trailcomp, lengthtrail = findTrailComponent(ends=ends,logfolder=logfolder, resultfolder=resultfolder, edges=edges, specialEdges=specialEdges,
    #                        figuresResultBuildings=figuresResultBuildings, nodeToCoordinate=nodeToCoordinate,elevatorEdges=elevatorEdges,neighbours=neighbours, maxtime=maxtime,
    #                        maxgap=maxgap, printtime=printtime, logfile=logfile, elevatorVertices=elevatorVertices,
    #                        prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname, showboundplot=showboundplot,
    #                        saveboundplotname=saveboundplotname)
    #     return trailcomp, lengthtrail

    # else: # we connect the dummy vertex to all the points in the graph with no walking bridges, and find the longest trail
    #     #For now it is left in an unattained part, since I have to use the function of the other file to do this.
    #     vdum= max(list(neighbourhood.keys()))+1
    #     neighbourhood[vdum] = set(range(vdum))
    #     print(f"WE SET VDUM={vdum} AND ADDED THE KEY TO NEIGHBOURHOOD")
    #     for i in range(vdum):
    #         weigthededges[(vdum, i)]=0
    #         neighbourhood[i].add(vdum)
    #
    #     model, varshall, varsdegree = runModel(weigthededges,neighbourhood, elevatorVertices, vdum, maxtime=600, printtime=15,
    #                                            logfile="\\log0801try2.log", ends=ends)
    #     lengthLongestTrail=model.getAttr('ObjVal')
    #     print(f"The longest trail is {lengthLongestTrail} meters long")
    #     used_edges= getEdgesResult(model, varshall)
    #     print(f"we have {vdum} as dummy vertex")
    #     print(f"edges used that connected here: {[edge for edge in used_edges if vdum in edge]}")
    #     pprint(f"The {len(used_edges)} used edges in the solution are:\n{used_edges}")
    #     trailresult= constructTrail(used_edges, vdum)
    #     return trailresult
    #     # print(f"trail result gives {len(trailresult)} edges in order:{trailresult}")
    #     # drawEdgesInFloorplans(trailresult)
    #     # results = glt.parse(PATH+"\\log0801try1.log")
    #     # nodelogs = results.progress("nodelog")
    #     # pd.set_option("display.max_columns", None)
    #     # print(f"type of nodelogs: {nodelogs}, and has columns: {[i for i in nodelogs]}")
    #     # print(nodelogs.head(10))
    #     # fig = go.Figure()
    #     # fig.add_trace(go.Scatter(x=nodelogs["Time"], y=nodelogs["Incumbent"], mode='markers',name="Primal Bound"))
    #     # fig.add_trace(go.Scatter(x=nodelogs["Time"], y=nodelogs["BestBd"], mode='markers',name="Dual Bound"))
    #     # fig.update_xaxes(title_text="Runtime in seconds")
    #     # fig.update_yaxes(title_text="Objective value function (in meters)")
    #     # fig.update_layout(title_text="The bounds on the length of the longest trail through CI, RA, ZI and CR, <br> at each moment in time when running the gurobi solver")
    #     # fig.show()
