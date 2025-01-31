import os

from svgpathtools import svg2paths, svg2paths2, wsvg, Path, Line
from gurobipy import *
import xml.etree.ElementTree as ET
import numpy as np
import logging
from collections import defaultdict
from itertools import combinations
import math
from copy import deepcopy

import pandas as pd
import gurobi_logtools as glt
import plotly.graph_objects as go
from datetime import datetime
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
    nedges=len(edges)
    rainbowColors= getRainbowColors(nedges)
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
                        if i+1 in range(nedges): #check that edge i is not the last one, because then we don't take an elevator anymore and don't print a number
                            if edges[i + 1] in elevatorEdges:
                                # We step into the elevator so we need to print a number
                                toFloor = nodeToCoordinate[edges[i + 2][1]]['Floor']
                                drawCoord = nodeToCoordinate[edge[1]]['Location']
                            elif i-1 in range(nedges): #check if we are not at the first edge
                                if edges[i - 1] in elevatorEdges:# we stepped just out of the elevator
                                    # we step out of the elevator, but still print a number from which floor we came for if you walk in the other order
                                    toFloor = nodeToCoordinate[edges[i - 2][0]]['Floor']
                                    drawCoord = nodeToCoordinate[edge[0]]['Location']
                                else:
                                    print(f"we did not enter not leave an elevator so we draw nothing")
                                    continue
                            else:
                                print(f"We start the trail by walking away from an elevator")
                                continue
                        elif edges[i-1] in elevatorEdges:#meaning that we just stepped out of an elevator
                            # we step out of the elevator, but still print a number from which floor we came for if you walk in the other order
                            toFloor = nodeToCoordinate[edges[i - 2][0]]['Floor']
                            drawCoord = nodeToCoordinate[edge[0]]['Location']
                        else:
                            print(f"we end our trail here because we already took the elevator")
                            continue

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
        buildingResultPath= resultfolder+f"/{building}"
        for floor, floorinfo in buildinginfo.items():
            buildingName, buildingNumber = splitNameNumber(building)
            floortree= floorinfo['tree']
            testfilename= f"/{prefixfilename}{buildingNumber}.{floor}.svg"
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
        PATHplot = logfolder + f"/boundsOverTime"
        if not os.path.exists(PATHplot):
            os.mkdir(PATHplot)
        print(f"pathplot: {PATHplot}")
        print(f"file will be saved to :{PATHplot + "/" + savename}")
        fig.write_html(PATHplot+"/"+savename, auto_open=False, validate=False)


def exportGraphinfo(Path, halls, nodeToCoordinate, scales,trail, prefix=""):
    if scales:
        PATHd = Path + "/dataruns"
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        datenew = date.replace(':', '-')
        PATH_data = PATHd+f"/log" + datenew + ".log"
    else:
        PATH_data=Path+"visualizeWeirdCases"

    if not os.path.exists(PATH_data):
        os.mkdir(PATH_data)
    weightedEdges = [{'key': key, 'value': value} for key, value in halls.items()]

    with open(os.path.join(PATH_data, prefix+"weigthedEdges.json"), "w") as outfile:
        json.dump(weightedEdges, outfile)

    nodeToCoordinates = {vertex: {'Building': info["Building"], "Floor": info["Floor"], "x": np.real(info["Location"]),
                                  "y": np.imag(info["Location"])} for vertex, info in nodeToCoordinate.items()}
    with open(os.path.join(PATH_data, prefix+"nodeToCoordinates.json"), "w") as outfile:
        json.dump(nodeToCoordinates, outfile)

    if scales:
        with open(os.path.join(PATH_data, prefix+"buildingScales.json"), "w") as outfile:
            json.dump(scales, outfile)

    todraw=[]
    for edge in trail:
        if (edge[0], edge[1]) in halls:
            todraw.append({"key": edge, "value": halls[edge]})
        else:
            todraw.append({"key": edge, "value": halls[(edge[1],edge[0])]})

    with open(os.path.join(PATH_data, prefix+"trail.json"), "w") as outfile:
        json.dump(todraw, outfile)

def runModelends(logfolder, halls, neighbours,ends=[],auxedge=False,nvdum=None, maxtime=None, maxgap=None, printtime=None, log=False, elevatorVertices=[]):
    # print(f"IN MODELRUN VDUM:{nvdum}")
    m = Model()
    m.Params.MIPFocus=1

    # if maxtime != None:
    #     m.Params.TimeLimit = maxtime
    if {729, 577} == set(ends): # Otherwise I got a memory error. at 65 seconds.
        m.Params.TimeLimit =60
    elif maxtime == 7200:
        print("Set the specific timelimit for a component of horst containing the tower")
        m.Params.TimeLimit = 5000
    else:
        m.Params.TimeLimit = 300
    m.Params.MIPGap = 0.05

    # if maxgap!=None:
    #     m.Params.MIPGap= maxgap
    if printtime!=None:
        m.Params.DisplayInterval= printtime
    if log:
        if type(log)== str:
            print(f"date: {log} will be the logfile")
            if ".log" in log:
                logfile=log
            else:
                logfile=f"/{log}.log"
        else:
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            datenew = date.replace(':', '-')
            logfile = "/log" + datenew + ".log"
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
    if auxedge:
        for manedge in auxedge:
            if manedge in varssol.keys():
                m.addConstr(varssol[manedge]==1, name=f"auxiliaryedgeforcut{auxedge}")
            else:
                print(f"run model ends had mandatory edge:{manedge} that is not in varssol")

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
        if getComponent:
            return False, [], []
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
        if getComponent:
            return False, [], []
        else:
            return False


def findCuts(edges, specialPaths, neighbourhood=None, inBuildings=[]):
    onecuts = dict()
    twocuts = dict(dict())
    inBN= getBNfromBuildingName(inBuildings)
    if neighbourhood ==None:
        neighbourhood = getNeighbourhood(edges)
    # bridges = dict()
    potentialtwocuts = dict()
    for path, info in specialPaths.items():
        if path[1] == "0":  # meaning that it is a bridge
            # print(f"path:{path},")
            b1 = path[3:5]
            b2 = path[5:7]
            # print(f"path:{path}, b1:{b1}, b2:{b2}, inBN:{inBN}")

            # print(f"constructed list: {[b1 in inBN, b2 in inBN]} and all in in buildings:{all([b1 in inBN, b2 in inBN])}")
            if all([b1 in inBN, b2 in inBN]):
                pathedge = (specialPaths[path]['Start']['Vnum'], specialPaths[path]['End']['Vnum'])
                pathedgeswap=(pathedge[1],pathedge[0])
                # print(f"pathedge:{pathedge} for path:{path}")
                if pathedge in edges: # meaning that this bridge is indeed in the component we are looking for
                    edgecheckcut= pathedge
                else:
                    edgecheckcut = pathedgeswap

                cutedge, component1, component2 = isCutEdge(neighbourhood, edgecheckcut, getComponent=True)
                # print(f"cutedge:{cutedge}")
                if cutedge:
                    onecuts[edgecheckcut] = {'Name': path, 'Component1vertices': component1, 'Component2vertices': component2}
                    return onecuts
                elif 'WH' in path:  # check for each other combination if it forms a 2 cut.
                    potentialtwocuts[edgecheckcut] = path
                elif 'ZH' in path:
                    potentialtwocuts[edgecheckcut] = path
                # elif "WA" in path:
                #     potentialtwocuts[edgecheckcut] = path

    # for bridge, name in bridges.items():
    #     cutedge, component1, component2 = isCutEdge(neighbourhood, bridge, getComponent=True)
    #     if cutedge:
    #         onecuts[bridge] = {'Name': name, 'Component1vertices': component1, 'Component2vertices': component2}
    #     else:  # check for each other combination if it forms a 2 cut.
    #         potentialtwocuts[bridge]=name

    if len(potentialtwocuts.keys()) > 1:
        print(f"potential twocuts{potentialtwocuts.keys()}")
        for path1, path2 in combinations(potentialtwocuts.keys(), 2):
            twocut, component1, component2 = isTwoCut(neighbourhood, path1, path2, getComponent=True)
            if twocut:
                print(f"we found twocut:{twocut}")
                twocuts[(path1,path2)] = {'Name':potentialtwocuts[path1]+'and'+potentialtwocuts[path2] ,'Component1vertices': component1, 'Component2vertices': component2}
                return twocuts
        return False


def getBuildings(vertices, nodeToCoordinate, returnNonsinglePoints=False):
    buildings=dict()
    nonsingleBuildingsfloors=dict()
    for vertex in vertices:
        if vertex in nodeToCoordinate: #meaning that the vertex is not representing an elevator.
            building= nodeToCoordinate[vertex]['Building']
            floor= nodeToCoordinate[vertex]['Floor']
            if building in buildings:
                if floor in buildings[building]:
                    if building in nonsingleBuildingsfloors:
                        nonsingleBuildingsfloors[building].add(floor)
                    else:
                        nonsingleBuildingsfloors[building]={floor}
                else:
                    buildings[building].add(floor)
            else:
                buildings[building] = {floor}
    if returnNonsinglePoints:
        return nonsingleBuildingsfloors
    else:
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

    # print(node_neighbors)
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
                       ends=[],auxedge=False,maxtime=None, maxgap=5, printtime=None, logfile=False, elevatorVertices=[],
                       prefixdrawcomp=False, plotboundsname=False, showboundplot=False, saveboundplotname=False):
    if neighbours == None:
        neighbours = getNeighbourhood(edges)
    neighboursnew, edgesnew,vdummy = addDummy(neighbours, edges, vdummy)
    if len(ends) in [0,1,2]:
        existingvertices = set()
        for v0, v1 in edges:
            existingvertices.add(v0)
            existingvertices.add(v1)
        buildingsvisited = getBuildings(existingvertices, nodeToCoordinate, returnNonsinglePoints=True)
        print(f"WE WILL FIND A LONGEST TRAIL THROUGH BUILDINGS:{buildingsvisited}\n"
              f"WITH {len(ends)} MANDATORY ENDS: {ends}")
        for end in ends:
            if end in nodeToCoordinate:
                print(f"end{end}: {nodeToCoordinate[end]}")
            else:
                print(f"end {end} not in node to coordinate")
        model, varshall, varsdegree = runModelends(auxedge=auxedge,ends=ends,logfolder=logfolder, halls=edgesnew, nvdum=vdummy,
                                               neighbours=neighboursnew,
                                               maxtime=maxtime, maxgap=maxgap, printtime=printtime, log=logfile,
                                               elevatorVertices=elevatorVertices)
        # lengthLongestTrail = model.getAttr('ObjVal')
        # print(f"The longest trail is {lengthLongestTrail} meters long")
        try:
            used_edges = getEdgesResult(model, varshall)
        except:
            todraw = []
            for edge in edges:
                todraw.append({"key": edge, "value": edges[edge]})
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            datenew = date.replace(':', '-')
            pathcases=logfolder+f"/componentscheck/{datenew}"
            if not os.path.exists(pathcases):
                os.makedirs(pathcases)

            prefix=""
            for building in buildingsvisited.keys():
                if "NANO" in building:
                    prefix+="NL "
                elif "CARRE" in building:
                    prefix+="CR "
                else:
                    prefix+=f"{building[:2]} "
            prefix+=f"{ends} and {len(edges)} edges"

            with open(os.path.join(pathcases,  prefix +".json"), "w") as outfile:
                json.dump(todraw, outfile)
            return [], 0


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

def getBNfromBuildingName(listofbuildings):
    listofbuildings=[building.split()[0] for building in listofbuildings]
    abbreviations=[]
    if 'CITADEL' in listofbuildings:
        abbreviations.append('CI')
    if 'RAVELIJN' in listofbuildings:
        abbreviations.append('RA')
    if 'ZILVERLING' in listofbuildings:
        abbreviations.append('ZI')
    if 'CARRE' in listofbuildings:
        abbreviations.extend(['CR', 'HB'])
    if 'WAAIER' in listofbuildings:
        abbreviations.append('WA')
    if 'NANOLAB' in listofbuildings:
        abbreviations.append('NL')
    if 'HORSTCOMPLEX' in listofbuildings or 'HORST' in listofbuildings:
        abbreviations.extend(['NH','HN','WH','HC','OH','HZ','ME', 'HW','ZH','BH'])
    return abbreviations

def swapTrailIfNeeded(start, trail):
    if start in trail[0]:
        print(f"the edges are in the correct order")
        # Tc0 = edgesTc0
    elif start in trail[-1]:
        # Reverse the order of the list
        trail.reverse()
    else:
        print(f"ERRORRR the vertex end0 is neither in the start nor end edge of the trail in Tc0??")

    return trail

def dealWith3cut(custom3cut, trailsComponents ,specialPaths, logfolder, resultfolder, edges, specialEdges, figuresResultBuildings, nodeToCoordinate, vdummy,elevatorEdges=[],auxedge=False, ends=[],neighbours=None,maxtime=None, maxgap=None, printtime=None, logfile=False, elevatorVertices=[],prefixdrawcomp=False, plotboundsname=False, showboundplot=False, saveboundplotname=False):
    # First start by defining the components c0 and c1, u0u1, v0v1, w0,w1.
    u=custom3cut[0]
    v= custom3cut[1]
    w=custom3cut[2]
    if u not in edges:
        u=(u[1],u[0])
    if v not in edges:
        v= (v[1],v[0])
    if w not in edges:
        w=(w[1],w[0])

    Lu=edges.pop(u)
    Lv=edges.pop(v)
    Lw=edges.pop(w)

    neighbours= getNeighbourhood(edges)
    c0= getReachable(neighbours, ends[0])
    c1=[vertex for vertex in neighbours.keys() if vertex not in c0]

    edgesC0= getEdgesComponent(edges, c0)
    edgesC1= getEdgesComponent(edges, c1)

    if u[0] in c0:
        u0= u[0]
        u1=u[1]
    else:
        u0=u[1]
        u1=u[0]

    if v[0] in c0:
        v0= v[0]
        v1=v[1]
    else:
        v0=v[1]
        v1=v[0]

    if w[0] in c0:
        w0= w[0]
        w1=w[1]
    else:
        w0=w[1]
        w1=w[0]

    # find longest trail in c0, from end[0] to u0 and from v0 to w0
    sumedgesC0=sum(edgesC0.values())
    edgesC0[(u0,v0)]= 0
    if auxedge:
        ae=auxedge+[(u0,v0)]
    else:
        ae=[(u0,v0)]
    horsttime=1200
    Tc0u0, LTc0u0 = findTrailComponent(ends=[ends[0],w0], auxedge=ae,logfolder=logfolder, resultfolder=resultfolder,
                                                    edges=edgesC0, specialEdges=specialEdges,
                                                    figuresResultBuildings=figuresResultBuildings,
                                                    nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,elevatorEdges=elevatorEdges,
                                                    maxtime=horsttime,
                                                    maxgap=maxgap, printtime=printtime, logfile=logfile,
                                                    elevatorVertices=elevatorVertices,
                                                    prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                                    showboundplot=showboundplot,
                                                    saveboundplotname=saveboundplotname)
    # LTc0u0-=edgesC0[(u0,v0)]
    edgesC0.pop((u0,v0))

    # find longest trail in c0 from end[0] to v0, and from u0 to w0
    edgesC0[(v0,w0)]= 0
    if auxedge:
        ae=auxedge+[(v0,w0)]
    else:
        ae=[(v0,w0)]
    Tc0v0, LTc0v0 = findTrailComponent(ends=[ends[0], u0],auxedge=ae, logfolder=logfolder, resultfolder=resultfolder,
                                       edges=edgesC0, specialEdges=specialEdges,
                                       figuresResultBuildings=figuresResultBuildings,
                                       nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, elevatorEdges=elevatorEdges,
                                       maxtime=horsttime,
                                       maxgap=maxgap, printtime=printtime, logfile=logfile,
                                       elevatorVertices=elevatorVertices,
                                       prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                       showboundplot=showboundplot,
                                       saveboundplotname=saveboundplotname)
    # LTc0v0-=edgesC0[(v0,w0)]
    edgesC0.pop((v0,w0))

    # find longest trail in c0 from end[0] to w0, and from u0 to v0
    edgesC0[(w0, u0)] = 0
    if auxedge:
        ae = auxedge + [(w0, u0)]
    else:
        ae = [(w0, u0)]
    Tc0w0, LTc0w0 = findTrailComponent(ends=[ends[0], v0], auxedge=ae,logfolder=logfolder, resultfolder=resultfolder,
                                       edges=edgesC0, specialEdges=specialEdges,
                                       figuresResultBuildings=figuresResultBuildings,
                                       nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, elevatorEdges=elevatorEdges,
                                       maxtime=horsttime,
                                       maxgap=maxgap, printtime=printtime, logfile=logfile,
                                       elevatorVertices=elevatorVertices,
                                       prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                       showboundplot=showboundplot,
                                       saveboundplotname=saveboundplotname)
    # LTc0w0 -= edgesC0[(w0, u0)]
    edgesC0.pop((w0, u0))

    # find longest trail in c1, from end[1] to u1 and from v1 to w1
    sumedgesC1=sum(edgesC1.values())
    edgesC1[(u1,v1)]= 0
    if auxedge:
        ae = auxedge + [(u1,v1)]
    else:
        ae = [(u1,v1)]
    Tc1u1, LTc1u1 = findTrailComponent(ends=[ends[1],w1],auxedge=ae, logfolder=logfolder, resultfolder=resultfolder,
                                                    edges=edgesC1, specialEdges=specialEdges,
                                                    figuresResultBuildings=figuresResultBuildings,
                                                    nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,elevatorEdges=elevatorEdges,
                                                    maxtime=horsttime,
                                                    maxgap=maxgap, printtime=printtime, logfile=logfile,
                                                    elevatorVertices=elevatorVertices,
                                                    prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                                    showboundplot=showboundplot,
                                                    saveboundplotname=saveboundplotname)
    # LTc1u1-=edgesC1[(u1,v1)]
    edgesC1.pop((u1,v1))

    # find longest trail in c1 from end[1] to v1, and from u1 to w1
    edgesC1[(v1,w1)]= 0
    if auxedge:
        ae = auxedge + [(v1,w1)]
    else:
        ae = [(v1,w1)]
    Tc1v1, LTc1v1 = findTrailComponent(ends=[ends[1], u1], auxedge=ae, logfolder=logfolder, resultfolder=resultfolder,
                                       edges=edgesC1, specialEdges=specialEdges,
                                       figuresResultBuildings=figuresResultBuildings,
                                       nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, elevatorEdges=elevatorEdges,
                                       maxtime=horsttime,
                                       maxgap=maxgap, printtime=printtime, logfile=logfile,
                                       elevatorVertices=elevatorVertices,
                                       prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                       showboundplot=showboundplot,
                                       saveboundplotname=saveboundplotname)
    # LTc1v1-=edgesC1[(v1,w1)]
    edgesC1.pop((v1,w1))

    # find longest trail in c1 from end[1] to w1, and from u1 to v1
    edgesC1[(w1, u1)] = 0
    if auxedge:
        ae = auxedge + [(w1, u1)]
    else:
        ae = [(w1, u1)]
    Tc1w1, LTc1w1 = findTrailComponent(ends=[ends[1], v1], auxedge=ae, logfolder=logfolder, resultfolder=resultfolder,
                                       edges=edgesC1, specialEdges=specialEdges,
                                       figuresResultBuildings=figuresResultBuildings,
                                       nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, elevatorEdges=elevatorEdges,
                                       maxtime=horsttime,
                                       maxgap=maxgap, printtime=printtime, logfile=logfile,
                                       elevatorVertices=elevatorVertices,
                                       prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                       showboundplot=showboundplot,
                                       saveboundplotname=saveboundplotname)
    # LTc1w1 -= edgesC1[(w1, u1)]
    edgesC1.pop((w1, u1))

    # First remove all the auxiliary edges between ends of uvw
    if (u0,v0) in Tc0u0:
        Tc0u0.remove((u0,v0))
    elif Tc0u0!= []:
        Tc0u0.remove((v0,u0))

    if (v0,w0) in Tc0v0:
        Tc0v0.remove((v0,w0))
    elif Tc0v0 != []:
        Tc0v0.remove((w0,v0))

    if (w0,u0) in Tc0w0:
        Tc0w0.remove((w0,u0))
    elif Tc0w0 != []:
        Tc0w0.remove((u0,w0))

    if (u1, v1) in Tc1u1:
        Tc1u1.remove((u1, v1))
    elif Tc1u1 != []:
        Tc1u1.remove((v1, u1))

    if (v1, w1) in Tc1v1:
        Tc1v1.remove((v1, w1))
    elif Tc1v1!= []:
        Tc1v1.remove((w1, v1))

    if (w1, u1) in Tc1w1:
        Tc1w1.remove((w1, u1))
    elif Tc1w1 != []:
        Tc1w1.remove((u1, w1))

    # now its left to find the maximum of these combined
    LT=LTc0u0+LTc1v1
    T= Tc0u0+Tc1v1
    if LT< LTc0u0+LTc1w1:
        LT=LTc0u0+LTc1w1
        T=Tc0u0+Tc1w1
    if LT< LTc0v0+LTc1u1:
        LT=LTc0v0+LTc1u1
        T= Tc0v0+Tc1u1
    if LT< LTc0v0+LTc1w1:
        LT=LTc0v0+LTc1w1
        T=Tc0v0+Tc1w1
    if LT< LTc0w0+LTc1u1:
        LT= LTc0w0+LTc1u1
        T= Tc0w0+Tc1u1
    if LT< LTc0w0+LTc1v1:
        LT= LTc0w0+LTc1v1
        T= Tc0w0+Tc1v1
    LT= LT+Lu+Lv+Lw
    T= T+[u,v,w]

    # Now check the cases where we only take one walking bridge
    #only taking u
    Tc0u, LTc0u = findTrailComponent(ends=[ends[0], u0],auxedge=auxedge, logfolder=logfolder, resultfolder=resultfolder,
                                       edges=edgesC0, specialEdges=specialEdges,
                                       figuresResultBuildings=figuresResultBuildings,
                                       nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, elevatorEdges=elevatorEdges,
                                       maxtime=horsttime,
                                       maxgap=maxgap, printtime=printtime, logfile=logfile,
                                       elevatorVertices=elevatorVertices,
                                       prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                       showboundplot=showboundplot,
                                       saveboundplotname=saveboundplotname)
    if LTc0u+ sumedgesC1 + Lu > LT: # meaning that it could be possible to walk here longer, only taking u
        Tc1u, LTc1u = findTrailComponent(ends=[ends[1], u1], auxedge=auxedge, logfolder=logfolder, resultfolder=resultfolder,
                                           edges=edgesC1, specialEdges=specialEdges,
                                           figuresResultBuildings=figuresResultBuildings,
                                           nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                                           elevatorEdges=elevatorEdges,
                                           maxtime=horsttime,
                                           maxgap=maxgap, printtime=printtime, logfile=logfile,
                                           elevatorVertices=elevatorVertices,
                                           prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                           showboundplot=showboundplot,
                                           saveboundplotname=saveboundplotname)
        if LTc0u + LTc1u +Lu > LT:
            LT=LTc0u + LTc1u + Lu
            T= Tc0u + Tc1u + [u]

    # Now check if only taking v could be longer
    Tc0v, LTc0v = findTrailComponent(ends=[ends[0], v0], auxedge=auxedge, logfolder=logfolder, resultfolder=resultfolder,
                                     edges=edgesC0, specialEdges=specialEdges,
                                     figuresResultBuildings=figuresResultBuildings,
                                     nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, elevatorEdges=elevatorEdges,
                                     maxtime=7200,
                                     maxgap=maxgap, printtime=printtime, logfile=logfile,
                                     elevatorVertices=elevatorVertices,
                                     prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                     showboundplot=showboundplot,
                                     saveboundplotname=saveboundplotname)
    if LTc0v+ sumedgesC1 + Lv > LT: # meaning that it could be possible to walk here longer, only taking u
        Tc1v, LTc1v = findTrailComponent(ends=[ends[1], v1],  auxedge=auxedge,logfolder=logfolder, resultfolder=resultfolder,
                                           edges=edgesC1, specialEdges=specialEdges,
                                           figuresResultBuildings=figuresResultBuildings,
                                           nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                                           elevatorEdges=elevatorEdges,
                                           maxtime=horsttime,
                                           maxgap=maxgap, printtime=printtime, logfile=logfile,
                                           elevatorVertices=elevatorVertices,
                                           prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                           showboundplot=showboundplot,
                                           saveboundplotname=saveboundplotname)
        if LTc0v + LTc1v +Lv > LT:
            LT=LTc0v + LTc1v +Lv
            T= Tc0v + Tc1v + [v]

    # Now check if only taking w could be longer
    Tc0w, LTc0w = findTrailComponent(ends=[ends[0], w0], auxedge=auxedge, logfolder=logfolder, resultfolder=resultfolder,
                                     edges=edgesC0, specialEdges=specialEdges,
                                     figuresResultBuildings=figuresResultBuildings,
                                     nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, elevatorEdges=elevatorEdges,
                                     maxtime=horsttime,
                                     maxgap=maxgap, printtime=printtime, logfile=logfile,
                                     elevatorVertices=elevatorVertices,
                                     prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                     showboundplot=showboundplot,
                                     saveboundplotname=saveboundplotname)
    if LTc0w+ sumedgesC1 +Lw > LT: # meaning that it could be possible to walk here longer, only taking u
        Tc1w, LTc1w = findTrailComponent(ends=[ends[1], w1],  auxedge=auxedge,logfolder=logfolder, resultfolder=resultfolder,
                                           edges=edgesC1, specialEdges=specialEdges,
                                           figuresResultBuildings=figuresResultBuildings,
                                           nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                                           elevatorEdges=elevatorEdges,
                                           maxtime=horsttime,
                                           maxgap=maxgap, printtime=printtime, logfile=logfile,
                                           elevatorVertices=elevatorVertices,
                                           prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                           showboundplot=showboundplot,
                                           saveboundplotname=saveboundplotname)
        if LTc0w + LTc1w + Lw> LT:
            LT=LTc0w + LTc1w + Lw
            T= Tc0w + Tc1w + [w]

    return T,LT



def reduceGraphBridges(custom3cut,trailsComponents ,specialPaths, logfolder, resultfolder, edges, specialEdges, figuresResultBuildings, nodeToCoordinate, vdummy,elevatorEdges=[], ends=[],auxedge=False, neighbours=None,maxtime=None, maxgap=5, printtime=None, logfile=False, elevatorVertices=[],prefixdrawcomp=False, plotboundsname=False, showboundplot=False, saveboundplotname=False):
    if neighbours ==None:
        neighbours= getNeighbourhood(edges.keys())

    buildingscomponent = getBuildings(vertices=neighbours.keys(), nodeToCoordinate=nodeToCoordinate, returnNonsinglePoints=True)
    componentkey= tuple(sorted(list(set([(min(edge), max(edge)) for edge in edges if vdummy not in edge])))+ sorted(ends))
    if componentkey in trailsComponents:
        print(f"YES trailscomponents with {len(list(trailsComponents.keys()))} keys,\n has componentkey for {len(edges)} edges in {buildingscomponent}")
        return trailsComponents[componentkey]['trail'], trailsComponents[componentkey]['length'], trailsComponents
    else:
        print(f"NO trailscomponents with {len(list(trailsComponents.keys()))} keys,\n does not have componentkey for {len(edges)} edges in {buildingscomponent}")
    cutMade = findCuts(edges=edges.keys(), specialPaths=specialPaths, neighbourhood=neighbours, inBuildings=list(buildingscomponent.keys()))
    if not cutMade:
        if len(buildingscomponent.keys())==1:
            if "HORSTCOMPLEX 1410" in buildingscomponent:
                if len(ends)==2:
                    hasedges=[edgecut in edges.keys() for edgecut in custom3cut]
                    hasreverse=[(edgecut[1],edgecut[0]) in edges.keys() for edgecut in custom3cut]
                    cutedgeexist = [a or b for a, b in zip(hasedges, hasreverse)]
                    if all(cutedgeexist):
                        #meaning that we are in the case where the gurobi solver did not find any feasible solution in 2 minutes and
                        #resulted in a memeory error when running it for 3minutes (with other stuff open on my laptop)
                        trailcomp, lengthtrail = dealWith3cut(auxedge=auxedge, custom3cut=custom3cut, trailsComponents=trailsComponents, logfolder=logfolder, resultfolder=resultfolder,
                                                                             specialPaths=specialPaths, edges=edges,
                                                                             specialEdges=specialEdges,
                                                                             figuresResultBuildings=figuresResultBuildings,
                                                                             elevatorEdges=elevatorEdges,
                                                                             nodeToCoordinate=nodeToCoordinate,
                                                                             vdummy=vdummy,
                                                                             maxtime=maxtime, maxgap=None, printtime=5,
                                                                             elevatorVertices=elevatorVertices,
                                                                             ends=ends)
                        trailsComponents[componentkey] = {'trail': trailcomp, 'length': lengthtrail}
                        return trailcomp, lengthtrail, trailsComponents
            #     maxtime= 1200
        print(f"reduce graph bridges found no 1 or 2 edgecut, so we have to just call findtrail component")
        trailcomp, lengthtrail = findTrailComponent(auxedge=auxedge, ends=ends, logfolder=logfolder, resultfolder=resultfolder,
                                                    edges=edges, specialEdges=specialEdges,
                                                    figuresResultBuildings=figuresResultBuildings,
                                                    nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,elevatorEdges=elevatorEdges,
                                                    neighbours=neighbours, maxtime=maxtime,
                                                    maxgap=maxgap, printtime=printtime, logfile=logfile,
                                                    elevatorVertices=elevatorVertices,
                                                    prefixdrawcomp=prefixdrawcomp, plotboundsname=plotboundsname,
                                                    showboundplot=showboundplot,
                                                    saveboundplotname=saveboundplotname)
        trailsComponents[componentkey]={'trail': trailcomp, 'length': lengthtrail}
        return trailcomp, lengthtrail, trailsComponents


    else: #meaning that there are 1 or 2 cuts
        print(f"reduceGraphbridges found a cut edge so lets branch off to the correct version, we now have {len(edges)} edges in our component!")
        for cut, info in cutMade.items():
            if 'and' in info['Name']:
                print(f"We have a 2 cut with edges:{cut} and info:{info}.")
                for end in ends:
                    if end in nodeToCoordinate:
                        print(f"We have mandatory end {end}:{nodeToCoordinate[end]}")
                    else:
                        print(f"We have mandatory end{end} that is not in node to coordinate")

                if len(ends)>0:
                    vinc0=ends[0]
                else:
                    vinc0= cut[0][0]

                if vinc0 in info['Component1vertices']:
                    v0 = cut[0][0]
                    v1 = cut[0][1]
                    c0 = info['Component1vertices']
                    c1 = info['Component2vertices']
                    print(f"vinc0:{vinc0} also in v1?: {vinc0 in c1}")
                    if cut[1][0] in c0:
                        u0= cut[1][0]
                        u1=cut[1][1]
                    else:
                        u0=cut[1][1]
                        u1=cut[1][0]
                else:
                    v1 = cut[0][0]
                    v0 = cut[0][1]
                    c1 = info['Component1vertices']
                    c0 = info['Component2vertices']
                    if cut[1][0] in c0:
                        u0= cut[1][0]
                        u1=cut[1][1]
                    else:
                        u0=cut[1][1]
                        u1=cut[1][0]

                print(
                    f"sanity check: v0 in c0:{v0 in c0}, u0 in c0:{u0 in c0}, v1 in c1:{v1 in c1}, u1 in c1:{u1 in c1}\n"
                    f"v0 in c1:{v0 in c1}, u0 in c1:{u0 in c1}, v1 in c0:{v1 in c0}, u1 in c0:{u1 in c0}")
                for end in ends:
                    print(f"sanity check on end {end} in c0: {end in c0}, and in c1?:{end in c1}")
                buildings0 = getBuildings(c0, nodeToCoordinate)
                buildings1 = getBuildings(c1, nodeToCoordinate)
                edgesC0 = getEdgesComponent(edges, c0)  # weighted edges
                edgesC1 = getEdgesComponent(edges, c1)  # weighted edges

                print(
                    f"one cut edge separating hallways in {getBuildings(c0, nodeToCoordinate)} \n from hallways in {getBuildings(c1, nodeToCoordinate)}")

                assert len(ends) in [0, 1, 2]  # Since the only options are for a component to have a mandatory start, end, both or none
                if len(ends) == 2:
                    print(f"we have to start in {[ends[0]]} and end in {[ends[1]]}")
                    if ends[1] in c0:
                        print(f"we only have to find a longest trail in c0 with ends {ends}, or going through both components by visiting either{cut[0]} or{cut[1]} first")
                        # Find a longest trail only through c0
                        print(f"2-cut 2 ends in c0, not visiting c1. Now redgb is called on c0:{buildings0}, and ends:{ends}")
                        # Find a longest trail from one end over v0v1, in c1, taking u1u0, and then ending in the other end. The order does not matter
                        # We connect v0 and u0 with an edge of a weight that is the sum of all the eges in c0. Then finding a longest trail that may start
                        # and end anywhere. That edge is guaranteed to be in the solution, deleting that edge afterwards, gives the longest 2 trails
                        # that can be walked from v0 and u0, that do not use the same hallways.
                        edgesC0[(v0,u0)]= 0
                        if auxedge:
                            ae = auxedge + [(v0,u0)]
                        else:
                            ae = [(v0,u0)]
                        print(f"2-cut 2 ends in same component, visit c1, part in c0.Now redgb is called on c0:{buildings0}, and ends:{ends}")
                        Tv0u0, LTv0u0, trailsComponents = reduceGraphBridges(auxedge=ae, custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder, resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC0,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=ends)

                        print(f"2-cut 2 ends in same component, visit c1, part in c1. Now redgb is called on c1:{buildings1}, and ends:{[v1,u1]}")
                        Tc1, LTc1, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder, resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC1,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=[v1,u1])
                        LT= LTv0u0+LTc1+ edges[cut[0]]+edges[cut[1]]

                        edgesC0.pop((v0,u0),None)
                        if sum(edgesC0.values())<= LT: #meaning that we can never walk longer in only c0 than what we already found
                            T= Tv0u0+ [cut[0],cut[1]]+Tc1
                            trailsComponents[componentkey] = {'trail': T, 'length': LT}
                            return T, LT, trailsComponents
                        Tc0, LTc0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder, resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC0,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=ends)  # longest trail in c0 that starts and ends in vertices in ends

                        if LTc0> LT:
                            # Tc0 = swapTrailIfNeeded(ends[0], Tc0)
                            trailsComponents[componentkey] = {'trail': Tc0, 'length': LTc0}
                            return Tc0, LTc0, trailsComponents
                        else:
                            T = Tv0u0+[cut[0], cut[1]]+ Tc1
                            trailsComponents[componentkey] = {'trail': T, 'length': LT}
                            return T, LT, trailsComponents

                            # if ((v0,u0)) in Tv0u0:
                            #     pos= Tv0u0.index((v0,u0))
                            # else:
                            #     pos=Tv0u0.index((u0,v0))
                            # T0= Tv0u0[0:pos]
                            # T1=Tv0u0[pos+1:]
                            # Tc0c1=[]
                            # if T0[-1][1]==v0:
                            #     Tc0c1.extend(T0)
                            #     Tc0c1.append((v0,v1))
                            #     Tc1=swapTrailIfNeeded(v1, Tc1)
                            #     Tc0c1.extend(Tc1)
                            #     Tc0c1.append((u1,u0))
                            #     Tc0c1.extend(T1)
                            #
                            # else:
                            #     Tc0c1.extend(T0)
                            #     Tc0c1.append((u0, u1))
                            #     Tc1 = swapTrailIfNeeded(u1, Tc1)
                            #     Tc0c1.extend(Tc1)
                            #     Tc0c1.append((v1, v0))
                            #     Tc0c1.extend(T1)
                            # trailsComponents[componentkey] = {'trail': Tc0c1, 'length': LTc0c1}
                            # return Tc0c1, LTc0c1, trailsComponents

                    else:
                        print(f"The ends lie in different components, so we must find a trail from ends[0] to v0 to v1 to ends[1]")
                        # We can either take v0v1 or u0u1.
                        # first case: we take v0v1:
                        print(f"2-cut ends in diff comp, part for c0 to v0. Now redgb is called on c0:{buildings0}, and ends:{[ends[0], v0]}")

                        Tc0v0, LTc0v0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                             logfolder=logfolder,
                                                                             resultfolder=resultfolder,
                                                                             specialPaths=specialPaths, edges=edgesC0,
                                                                             specialEdges=specialEdges,
                                                                             figuresResultBuildings=figuresResultBuildings,
                                                                             elevatorEdges=elevatorEdges,
                                                                             nodeToCoordinate=nodeToCoordinate,
                                                                             vdummy=vdummy,
                                                                             maxtime=maxtime, maxgap=None, printtime=5,
                                                                             elevatorVertices=elevatorVertices,
                                                                             ends=[ends[0], v0])
                        print(f" 2-cut 2 ends in diff comp, part c1 from v1. Now redgb is called on c1:{buildings1}, and ends:{[v1, ends[1]]}")
                        Tc1v1, LTc1v1, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                             logfolder=logfolder,
                                                                             resultfolder=resultfolder,
                                                                             specialPaths=specialPaths, edges=edgesC1,
                                                                             specialEdges=specialEdges,
                                                                             figuresResultBuildings=figuresResultBuildings,
                                                                             elevatorEdges=elevatorEdges,
                                                                             nodeToCoordinate=nodeToCoordinate,
                                                                             vdummy=vdummy,
                                                                             maxtime=maxtime, maxgap=None, printtime=5,
                                                                             elevatorVertices=elevatorVertices,
                                                                             ends=[v1, ends[
                                                                                 1]])
                        LTv = LTc0v0 + edges[cut[0]] + LTc1v1

                        # Second case, we take u0u1:
                        print(f"2-cut ends in diff comp. part c0 to u0. Now redgb is called on c0:{buildings0}, and ends:{[ends[0],
                                                                                   u0]}")
                        Tc0u0, LTc0u0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                             logfolder=logfolder,
                                                                             resultfolder=resultfolder,
                                                                             specialPaths=specialPaths, edges=edgesC0,
                                                                             specialEdges=specialEdges,
                                                                             figuresResultBuildings=figuresResultBuildings,
                                                                             elevatorEdges=elevatorEdges,
                                                                             nodeToCoordinate=nodeToCoordinate,
                                                                             vdummy=vdummy,
                                                                             maxtime=maxtime, maxgap=None, printtime=5,
                                                                             elevatorVertices=elevatorVertices,
                                                                             ends=[ends[0],
                                                                                   u0])
                        print(f"2-cuts ends in diff comp, part c1 from u1. Now redgb is called on c1:{buildings1}, and ends:{[u1, ends[1]]}")
                        if LTc0u0 +edges[cut[1]]+ sum(edgesC1.values())<= LTv: # no need to check c1u1, since we can never walk longer there.
                            T = Tc0v0+[cut[0]]+Tc1v1
                            trailsComponents[componentkey] = {'trail': T, 'length': LTv}
                            return T, LTv, trailsComponents

                        Tc1u1, LTc1u1, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                             logfolder=logfolder,
                                                                             resultfolder=resultfolder,
                                                                             specialPaths=specialPaths, edges=edgesC1,
                                                                             specialEdges=specialEdges,
                                                                             figuresResultBuildings=figuresResultBuildings,
                                                                             elevatorEdges=elevatorEdges,
                                                                             nodeToCoordinate=nodeToCoordinate,
                                                                             vdummy=vdummy,
                                                                             maxtime=maxtime, maxgap=None, printtime=5,
                                                                             elevatorVertices=elevatorVertices,
                                                                             ends=[u1, ends[
                                                                                 1]])
                        LTu = LTc0u0 + edges[cut[1]] + LTc1u1

                        if LTv>LTu:
                            Tv=  Tc0v0+[cut[0]]+Tc1v1
                            # Tv=swapTrailIfNeeded(ends[0], Tc0v0)
                            # Tv.append((v0,v1))
                            # Tc1v1=swapTrailIfNeeded(v1, Tc1v1)
                            # Tv.extend(Tc1v1)
                            trailsComponents[componentkey] = {'trail': Tv, 'length': LTv}
                            return Tv, LTv, trailsComponents
                        else:
                            Tu= Tc0u+[cut[1]]+Tc1u1
                            # Tu = swapTrailIfNeeded(ends[0], Tc0u0)
                            # Tu.append((u0, u1))
                            # Tc1u1 = swapTrailIfNeeded(u1, Tc1u1)
                            # Tu.extend(Tc1u1)
                            trailsComponents[componentkey] = {'trail': Tu, 'length': LTu}
                            return Tu, LTu, trailsComponents

                        # # Construct the combined trail, starting in ends[0] in c0, going to v0, then walk the edge to v1, and walk a longest trail in c1 to ends1
                        # # check the order of the edges in Tv0,
                        # Tc0v0 = swapTrailIfNeeded(ends[0], Tc0v0)
                        #
                        # # check the order of the edges in Tv1,
                        # Tc1v1 = swapTrailIfNeeded(v1, Tc1v1)
                        #
                        # Tc0c1 = Tc0v0 + [(v0, v1)] + Tc1v1
                        # trailsComponents[componentkey] = {'trail': Tc0c1, 'length': LTc0c1}
                        # return Tc0c1, LTc0c1, trailsComponents

                elif len(ends)==1:
                    # We have one mandatory starting vertex, that lies in c0. So we either:
                    # Stay in c0
                    # go to c1 via v0v1
                    # go to c1 via u0u1
                    # through both c0 and c1 using both bridges
                    # we start by considering taking v and u
                    LT=0
                    T=[]
                    edgesC0[(v0, u0)] =0
                    print(
                        f"2-cut 1 end in c0, taking u and v, part in c0. (u0v0). Now redgb is called on c0:{buildings0}, and ends:{ends}")
                    if auxedge:
                        ae = auxedge + [(v0, u0)]
                    else:
                        ae = [(v0, u0)]
                    Tv0u0, LTv0u0, trailsComponents = reduceGraphBridges(auxedge=ae, custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder, resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC0,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=ends)

                    print(
                        f"2-cut 1 end in c0, taking u and v, part c1. Now redgb is called on c1:{buildings1}, and ends:{v1} and {u1}")
                    Tc1, LTc1, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                     logfolder=logfolder, resultfolder=resultfolder,
                                                                     specialPaths=specialPaths, edges=edgesC1,
                                                                     specialEdges=specialEdges,
                                                                     figuresResultBuildings=figuresResultBuildings,
                                                                     elevatorEdges=elevatorEdges,
                                                                     nodeToCoordinate=nodeToCoordinate,
                                                                     vdummy=vdummy,
                                                                     maxtime=maxtime, maxgap=None, printtime=5,
                                                                     elevatorVertices=elevatorVertices,
                                                                     ends=[v1, u1])
                    LTuv = LTv0u0 + LTc1 + edges[cut[0]] + edges[cut[1]]
                    LT= LTuv
                    if (u0,v0) in Tv0u0:
                        print(f"can remove {u0,v0} since it found a trail")
                        Tv0u0.remove((u0,v0))
                    elif (v0,u0) in Tv0u0:
                        print(f"can remove {v0,u0} since it found a trail")
                        Tv0u0.remove((v0,u0))
                    elif Tv0u0 != []:
                        print(f"cant remove v0u0{v0,u0} because it dit not find any trail")
                    else:
                        print(f"ERRORRR it did find a trail, but not with {v0,u0} in it...")
                    T=Tv0u0+[cut[0],cut[1]]+Tc1
                    edgesC0.pop((v0,u0),None)
                    if sum(edgesC0.values())>LT: # It could be possible to walk longer in C0 than when taking both bridges
                        print(
                            f"2-cut 1 end in c0, staying in c0. Now redgb is called on c0:{buildings0}, and ends:{ends}")
                        Tc0, LTc0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder, resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC0,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=ends)
                        if LT< LTc0:
                            LT=LTc0
                            T=Tc0
                    # if we take v find c0v0:
                    print(
                        f"2-cut 1end in c0, taking v0v1. part c0v0. Now redgb is called on c0:{buildings0}, and ends:{[ends[0], v0]}")
                    Tc0v0, LTc0v0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder,
                                                                         resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC0,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=[ends[0],
                                                                               v0])
                    if sum(edgesC1.values())+LTc0v0+edges[cut[0]]> LT: # it could be possible to walk longer when only taking v
                        print(
                            f"2-cut 1 end in c0, taking v, part c1v1. Now redgb is called on c1:{buildings1}, and ends:{[v1]}")
                        Tc1v1, LTc1v1, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                             logfolder=logfolder,
                                                                             resultfolder=resultfolder,
                                                                             specialPaths=specialPaths, edges=edgesC1,
                                                                             specialEdges=specialEdges,
                                                                             figuresResultBuildings=figuresResultBuildings,
                                                                             elevatorEdges=elevatorEdges,
                                                                             nodeToCoordinate=nodeToCoordinate,
                                                                             vdummy=vdummy,
                                                                             maxtime=maxtime, maxgap=None, printtime=5,
                                                                             elevatorVertices=elevatorVertices,
                                                                             ends=[v1])
                        LTv = LTc0v0 + edges[cut[0]] + LTc1v1
                        if LT< LTv:
                            LT=LTv
                            T=Tc0v0+[cut[0]]+ Tc1v1

                    # if we take u, find c0cu
                    print(
                        f"2-cut 1 end in c0, taking u, c0u0. Now redgb is called on c0:{buildings0}, and ends:{[ends[0], u0]}")
                    Tc0u0, LTc0u0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder,
                                                                         resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC0,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=[ends[0],
                                                                               u0])
                    if sum(edgesC1.values())+edges[cut[1]]+LTc0u0>LT: # it could be possible to walk longer when only taking u
                        print(
                            f"2-cut 1 end in c0, taking u, c1u1. Now redgb is called on c1:{buildings1}, and ends:{[u1]}")
                        Tc1u1, LTc1u1, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                             logfolder=logfolder,
                                                                             resultfolder=resultfolder,
                                                                             specialPaths=specialPaths, edges=edgesC1,
                                                                             specialEdges=specialEdges,
                                                                             figuresResultBuildings=figuresResultBuildings,
                                                                             elevatorEdges=elevatorEdges,
                                                                             nodeToCoordinate=nodeToCoordinate,
                                                                             vdummy=vdummy,
                                                                             maxtime=maxtime, maxgap=None, printtime=5,
                                                                             elevatorVertices=elevatorVertices,
                                                                             ends=[u1])
                        LTu = LTc0u0 + edges[cut[1]] + LTc1u1
                        if LT<LTu:
                            LT=LTu
                            T= Tc0u0+[cut[1]]+ Tc1u1



                    # # if we stay in c0:
                    # print(f"2-cut 1 end in c0, staying in c0. Now redgb is called on c0:{buildings0}, and ends:{ends}")
                    # Tc0, LTc0, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                  logfolder=logfolder, resultfolder=resultfolder,
                    #                                                  specialPaths=specialPaths, edges=edgesC0,
                    #                                                  specialEdges=specialEdges,
                    #                                                  figuresResultBuildings=figuresResultBuildings,
                    #                                                  elevatorEdges=elevatorEdges,
                    #                                                  nodeToCoordinate=nodeToCoordinate,
                    #                                                  vdummy=vdummy,
                    #                                                  maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                  elevatorVertices=elevatorVertices,
                    #                                                  ends=ends)
                    # # if we take v0v1
                    # print(f"2-cut 1end in c0, taking v0v1. part c0v0. Now redgb is called on c0:{buildings0}, and ends:{[ends[0], v0]}")
                    # Tc0v0, LTc0v0, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                      logfolder=logfolder,
                    #                                                      resultfolder=resultfolder,
                    #                                                      specialPaths=specialPaths, edges=edgesC0,
                    #                                                      specialEdges=specialEdges,
                    #                                                      figuresResultBuildings=figuresResultBuildings,
                    #                                                      elevatorEdges=elevatorEdges,
                    #                                                      nodeToCoordinate=nodeToCoordinate,
                    #                                                      vdummy=vdummy,
                    #                                                      maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                      elevatorVertices=elevatorVertices,
                    #                                                      ends=[ends[0],
                    #                                                            v0])
                    # print(f"2-cut 1 end in c0, taking v, part c1v1. Now redgb is called on c1:{buildings1}, and ends:{[v1]}")
                    #
                    # Tc1v1, LTc1v1, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                      logfolder=logfolder,
                    #                                                      resultfolder=resultfolder,
                    #                                                      specialPaths=specialPaths, edges=edgesC1,
                    #                                                      specialEdges=specialEdges,
                    #                                                      figuresResultBuildings=figuresResultBuildings,
                    #                                                      elevatorEdges=elevatorEdges,
                    #                                                      nodeToCoordinate=nodeToCoordinate,
                    #                                                      vdummy=vdummy,
                    #                                                      maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                      elevatorVertices=elevatorVertices,
                    #                                                      ends=[v1])
                    # LTv = LTc0v0 + edges[cut[0]] + LTc1v1
                    #
                    # # If we take u0u1:
                    # print(f"2-cut 1 end in c0, taking u, c0u0. Now redgb is called on c0:{buildings0}, and ends:{[ends[0], u0]}")
                    # Tc0u0, LTc0u0, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                      logfolder=logfolder,
                    #                                                      resultfolder=resultfolder,
                    #                                                      specialPaths=specialPaths, edges=edgesC0,
                    #                                                      specialEdges=specialEdges,
                    #                                                      figuresResultBuildings=figuresResultBuildings,
                    #                                                      elevatorEdges=elevatorEdges,
                    #                                                      nodeToCoordinate=nodeToCoordinate,
                    #                                                      vdummy=vdummy,
                    #                                                      maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                      elevatorVertices=elevatorVertices,
                    #                                                      ends=[ends[0],
                    #                                                            u0])
                    # print(f"2-cut 1 end in c0, taking u, c1u1. Now redgb is called on c1:{buildings1}, and ends:{[u1]}")
                    # Tc1u1, LTc1u1, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                      logfolder=logfolder,
                    #                                                      resultfolder=resultfolder,
                    #                                                      specialPaths=specialPaths, edges=edgesC1,
                    #                                                      specialEdges=specialEdges,
                    #                                                      figuresResultBuildings=figuresResultBuildings,
                    #                                                      elevatorEdges=elevatorEdges,
                    #                                                      nodeToCoordinate=nodeToCoordinate,
                    #                                                      vdummy=vdummy,
                    #                                                      maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                      elevatorVertices=elevatorVertices,
                    #                                                      ends=[u1])
                    # LTu = LTc0u0 + edges[cut[1]] + LTc1u1
                    #
                    # # We take both v0v1 and u0u1:
                    # edgesC0[(v0, u0)] = sum(list(edgesC0.values()))
                    # print(f"2-cut 1 end in c0, taking u and v, part in c0. (u0v0). Now redgb is called on c0:{buildings0}, and ends:{ends}")
                    # Tv0u0, LTv0u0, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                      logfolder=logfolder, resultfolder=resultfolder,
                    #                                                      specialPaths=specialPaths, edges=edgesC0,
                    #                                                      specialEdges=specialEdges,
                    #                                                      figuresResultBuildings=figuresResultBuildings,
                    #                                                      elevatorEdges=elevatorEdges,
                    #                                                      nodeToCoordinate=nodeToCoordinate,
                    #                                                      vdummy=vdummy,
                    #                                                      maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                      elevatorVertices=elevatorVertices,
                    #                                                      ends=ends)
                    #
                    # print(f"2-cut 1 end in c0, taking u and v, part c1. Now redgb is called on c1:{buildings1}, and ends:{v1} and {u1}")
                    # Tc1, LTc1, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                  logfolder=logfolder, resultfolder=resultfolder,
                    #                                                  specialPaths=specialPaths, edges=edgesC1,
                    #                                                  specialEdges=specialEdges,
                    #                                                  figuresResultBuildings=figuresResultBuildings,
                    #                                                  elevatorEdges=elevatorEdges,
                    #                                                  nodeToCoordinate=nodeToCoordinate,
                    #                                                  vdummy=vdummy,
                    #                                                  maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                  elevatorVertices=elevatorVertices,
                    #                                                  ends=[v1, u1])
                    # LTuv = LTv0u0 - edgesC0[(v0, u0)] + LTc1 + edges[cut[0]] + edges[cut[1]]
                    #
                    # LT= max(LTc0, LTv, LTu, LTuv)
                    # if LT == LTc0:
                    #     T=LTc0
                    # elif LT == LTv:
                    #     T=swapTrailIfNeeded(ends[0], Tc0v0)
                    #     T.append((v0,v1))
                    #     T.extend(swapTrailIfNeeded(v1, Tc1v1))
                    # elif LT == LTu:
                    #     T = swapTrailIfNeeded(ends[0], Tc0u0)
                    #     T.append((u0, u1))
                    #     T.extend(swapTrailIfNeeded(u1, Tc1u1))
                    # else:
                    #     if ((v0, u0)) in Tv0u0:
                    #         pos = Tv0u0.index((v0, u0))
                    #     else:
                    #         pos = Tv0u0.index((u0, v0))
                    #     T = Tv0u0[0:pos]
                    #     T1 = Tv0u0[pos + 1:]
                    #     if T[-1][1] == v0:
                    #         T.append((v0, v1))
                    #         Tc1 = swapTrailIfNeeded(v1, Tc1)
                    #         T.extend(Tc1)
                    #         T.append((u1, u0))
                    #         T.extend(T1)
                    #     else:
                    #         T.append((u0, u1))
                    #         Tc1 = swapTrailIfNeeded(u1, Tc1)
                    #         T.extend(Tc1)
                    #         T.append((v1, v0))
                    #         T.extend(T1)
                    trailsComponents[componentkey] = {'trail': T, 'length': LT}
                    return T, LT, trailsComponents

                else: # the length of mandatory ends is 0 so we are free to start whereever.We can either have a longest trail in
                    # c0
                    # c1
                    # c0 to c1 over v0v1
                    # c0 to c1 over u0u1
                    # c0 to c1 to c0 using both bridges
                    #c1 to c0 to c1 using both bridges
                    edgesC0[(v0, u0)] =0
                    print(
                        f"2-cut no ends, taking both u and v starting in c0. part c0. Now redgb is called on c0:{buildings0}, and ends:{[]}")
                    if auxedge:
                        ae = auxedge + [(v0, u0)]
                    else:
                        ae = [(v0, u0)]
                    Tv0u0, LTv0u0, trailsComponents = reduceGraphBridges(auxedge=ae ,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder,
                                                                         resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC0,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=[])
                    print(
                        f"2-cut 0ends, taking u and v, starting in c0. part c1. Now redgb is called on c1:{buildings1}, and ends:{[v1, u1]}")

                    Tc1uv, LTc1uv, trailsComponents = reduceGraphBridges(auxedge=auxedge, custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder,
                                                                         resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC1,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=[v1, u1])
                    LT = LTv0u0 + LTc1uv + edges[cut[0]] + edges[cut[1]]
                    if (u0, v0) in Tv0u0:
                        print(f"can remove {u0,v0} since it found a trail")
                        Tv0u0.remove((u0,v0))
                    elif (v0,u0) in Tv0u0:
                        print(f"can remove {v0,u0} since it found a trail")
                        Tv0u0.remove((v0,u0))
                    elif Tv0u0!=[]:
                        print(f"cant remove v0u0{v0,u0} because it dit not find any trail")
                    else:
                        print(f"ERRORRR it did find a trail, but not with {v0,u0} in it...")
                    # Tv0u0.remove((v0,u0))
                    T=Tv0u0 + [cut[0], cut[1]] + Tc1uv
                    edgesC0.pop((v0,u0),None)
                    print(
                        f"2-cut 0 ends taking u and v, starting in c1. part c0. Now redgb is called on c0:{buildings0}, and ends:{[v0, u0]}")
                    # now consider a longest trail starting and ending in c1, taking both edges.
                    Tc0uv, LTc0uv, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder,
                                                                         resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC0,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=[v0, u0])
                    edgesC1[(v1, u1)] = 0
                    if LTc0uv+ edges[cut[0]]+edges[cut[1]]+edgesC1[(v1,u1)] > LT: #it could be possible to walk a longest trail here.
                        print(
                            f"2-cut 0 ends taking u and v, starting in c1. part c1Now redgb is called on c1:{buildings0}, and ends:{ends}")
                        if auxedge:
                            ae = auxedge + [(v1, u1)]
                        else:
                            ae = [(v1, u1)]
                        Tv1u1, LTv1u1, trailsComponents = reduceGraphBridges(auxedge=ae,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                             logfolder=logfolder,
                                                                             resultfolder=resultfolder,
                                                                             specialPaths=specialPaths, edges=edgesC1,
                                                                             specialEdges=specialEdges,
                                                                             figuresResultBuildings=figuresResultBuildings,
                                                                             elevatorEdges=elevatorEdges,
                                                                             nodeToCoordinate=nodeToCoordinate,
                                                                             vdummy=vdummy,
                                                                             maxtime=maxtime, maxgap=None, printtime=5,
                                                                             elevatorVertices=elevatorVertices,
                                                                             ends=ends)

                        LTc1c0c1 = LTv1u1 + LTc0uv + edges[cut[0]] + edges[cut[1]]
                        if LT< LTc1c0c1:
                            LT=LTc1c0c1
                            if Tv1u1 == []:
                                print(f"cant remove v1u1{v1, u1} because it dit not find any trail")
                            elif (v1, u1) in Tv1u1:
                                print(f"can remove v1u1 {v1, u1} since it found a trail")
                                Tv1u1.remove((v1, u1))
                            elif (u1,v1) in Tv1u1:
                                print(f"can remove u1v1 {u1, v1} since it found a trail")
                                Tv1u1.remove((u1, v1))
                            else:
                                print(f"ERRORRR it did find a trail, but not with v1u1 {v1, u1} in it...")
                            # Tv1u1.remove((v1,u1))
                            T= Tv1u1+[cut[0], cut[1]]+ Tc0uv
                    edgesC1.pop((v1,u1),None)

                    if sum(edgesC0.values())>LT: # it could be possible to wlk longer in only c0 than taking any other bridge.
                        print(f"2-cut 0 ends, staying in c0. Now redgb is called on c0:{buildings0}, and ends:{ends}")
                        Tc0, LTc0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder, resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC0,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=ends)
                        if LT< LTc0:
                            LT=LTc0
                            T=Tc0

                    if sum(edgesC1.values()) > LT: # it could be possible to walk longer here than before.
                        # second case: longest trail in c1:
                        print(f"2-cut 0 ends staying in c1. Now redgb is called on c1:{buildings1}, and ends:{ends}")
                        Tc1, LTc1, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder, resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC1,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=ends)
                        if LT< LTc1:
                            LT=LTc1
                            T=Tc1

                    # only taking v find c0v0
                    print(f"2-cut 0 ends, taking v, part c0v0. Now redgb is called on c0:{buildings0}, and ends:{[v0]}")
                    Tc0v0, LTc0v0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder, resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC0,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=[v0])
                    if sum(edgesC1.values())+edges[cut[0]]+LTc0v0 >LT: #it could be possible to walk here longer
                        print(
                            f"2-cut 0 ends, taking v, part c1v1. Now redgb is called on c1:{buildings1}, and ends:{[v1]}")
                        Tc1v1, LTc1v1, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                             logfolder=logfolder,
                                                                             resultfolder=resultfolder,
                                                                             specialPaths=specialPaths, edges=edgesC1,
                                                                             specialEdges=specialEdges,
                                                                             figuresResultBuildings=figuresResultBuildings,
                                                                             elevatorEdges=elevatorEdges,
                                                                             nodeToCoordinate=nodeToCoordinate,
                                                                             vdummy=vdummy,
                                                                             maxtime=maxtime, maxgap=None, printtime=5,
                                                                             elevatorVertices=elevatorVertices,
                                                                             ends=[v1])
                        LTv = LTc0v0 + LTc1v1 + edges[cut[0]]
                        if LT< LTv:
                            LT=LTv
                            T=Tc0v0+[cut[0]]+Tc1v1

                    # only taking u: find c0u0
                    print(
                        f"2-cut 0 ends, taking u, part c0u0. Now redgb is called on c0:{buildings0}, and ends:{[u0]}")
                    Tc0u0, LTc0u0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder,
                                                                         resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC0,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         vdummy=vdummy,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=[u0])
                    if sum(edgesC0.values())+edges[cut[1]]+LTc0u0 >LT: # it could be possible to walk here longer
                        print(
                            f"2-cut 0 ends, taking u, part c1u1. Now redgb is called on c1:{buildings1}, and ends:{[u1]}")
                        Tc1u1, LTc1u1, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                             logfolder=logfolder,
                                                                             resultfolder=resultfolder,
                                                                             specialPaths=specialPaths, edges=edgesC1,
                                                                             specialEdges=specialEdges,
                                                                             figuresResultBuildings=figuresResultBuildings,
                                                                             elevatorEdges=elevatorEdges,
                                                                             nodeToCoordinate=nodeToCoordinate,
                                                                             vdummy=vdummy,
                                                                             maxtime=maxtime, maxgap=None, printtime=5,
                                                                             elevatorVertices=elevatorVertices,
                                                                             ends=[u1])
                        LTu = LTc0u0 + LTc1u1 + edges[cut[1]]
                        if LT< LTu:
                            LT=LTu
                            T= Tc0u0+[cut[1]]+Tc1u1
                    trailsComponents[componentkey] = {'trail': T, 'length': LT}
                    return T, LT, trailsComponents
                    # #first case: longest trail in c0:
                    # print(f"2-cut 0 ends, staying in c0. Now redgb is called on c0:{buildings0}, and ends:{ends}")
                    # Tc0, LTc0, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                      logfolder=logfolder, resultfolder=resultfolder,
                    #                                                      specialPaths=specialPaths, edges=edgesC0,
                    #                                                      specialEdges=specialEdges,
                    #                                                      figuresResultBuildings=figuresResultBuildings,
                    #                                                      elevatorEdges=elevatorEdges,
                    #                                                      nodeToCoordinate=nodeToCoordinate,
                    #                                                      vdummy=vdummy,
                    #                                                      maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                      elevatorVertices=elevatorVertices,
                    #                                                      ends=ends)
                    #     # second case: longest trail in c1:
                    # print(f"2-cut 0 ends staying in c1. Now redgb is called on c1:{buildings1}, and ends:{ends}")
                    # Tc1, LTc1, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                      logfolder=logfolder, resultfolder=resultfolder,
                    #                                                      specialPaths=specialPaths, edges=edgesC1,
                    #                                                      specialEdges=specialEdges,
                    #                                                      figuresResultBuildings=figuresResultBuildings,
                    #                                                      elevatorEdges=elevatorEdges,
                    #                                                      nodeToCoordinate=nodeToCoordinate,
                    #                                                      vdummy=vdummy,
                    #                                                      maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                      elevatorVertices=elevatorVertices,
                    #                                                      ends=ends)
                    #
                    #     # third case: only using v0v1:
                    # print(f"2-cut 0 ends, taking v, part c0v0. Now redgb is called on c0:{buildings0}, and ends:{[v0]}")
                    # Tc0v0, LTc0v0, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                      logfolder=logfolder, resultfolder=resultfolder,
                    #                                                      specialPaths=specialPaths, edges=edgesC0,
                    #                                                      specialEdges=specialEdges,
                    #                                                      figuresResultBuildings=figuresResultBuildings,
                    #                                                      elevatorEdges=elevatorEdges,
                    #                                                      nodeToCoordinate=nodeToCoordinate,
                    #                                                      vdummy=vdummy,
                    #                                                      maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                      elevatorVertices=elevatorVertices,
                    #                                                      ends=[v0])
                    # print(f"2-cut 0 ends, taking v, part c1v1. Now redgb is called on c1:{buildings1}, and ends:{[v1]}")
                    # Tc1v1, LTc1v1, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                      logfolder=logfolder, resultfolder=resultfolder,
                    #                                                      specialPaths=specialPaths, edges=edgesC1,
                    #                                                      specialEdges=specialEdges,
                    #                                                      figuresResultBuildings=figuresResultBuildings,
                    #                                                      elevatorEdges=elevatorEdges,
                    #                                                      nodeToCoordinate=nodeToCoordinate,
                    #                                                      vdummy=vdummy,
                    #                                                      maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                      elevatorVertices=elevatorVertices,
                    #                                                      ends=[v1])
                    # LTv= LTc0v0+LTc1v1+edges[cut[0]]
                    #
                    # # Fourth case: only using u0u1
                    # print(f"2-cut 0 ends, taking u, part c0u0. Now redgb is called on c0:{buildings0}, and ends:{[u0]}")
                    # Tc0u0, LTc0u0, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                      logfolder=logfolder, resultfolder=resultfolder,
                    #                                                      specialPaths=specialPaths, edges=edgesC0,
                    #                                                      specialEdges=specialEdges,
                    #                                                      figuresResultBuildings=figuresResultBuildings,
                    #                                                      elevatorEdges=elevatorEdges,
                    #                                                      nodeToCoordinate=nodeToCoordinate,
                    #                                                      vdummy=vdummy,
                    #                                                      maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                      elevatorVertices=elevatorVertices,
                    #                                                          ends=[u0])

                    # print(f"2-cut 0 ends, taking u, part c1u1. Now redgb is called on c1:{buildings1}, and ends:{[u1]}")
                    # Tc1u1, LTc1u1, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                          logfolder=logfolder, resultfolder=resultfolder,
                    #                                                          specialPaths=specialPaths, edges=edgesC1,
                    #                                                          specialEdges=specialEdges,
                    #                                                          figuresResultBuildings=figuresResultBuildings,
                    #                                                          elevatorEdges=elevatorEdges,
                    #                                                          nodeToCoordinate=nodeToCoordinate,
                    #                                                          vdummy=vdummy,
                    #                                                          maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                          elevatorVertices=elevatorVertices,
                    #                                                          ends=[u1])
                    # LTu = LTc0u0 + LTc1u1 + edges[cut[1]]

                    # # Fifth case, using both bridges, starting in C0
                    # edgesC0[(v0, u0)] = sum(list(edgesC0.values()))
                    # print(f"2-cut no ends, taking both u and v starting in c0. part c0. Now redgb is called on c0:{buildings0}, and ends:{[]}")
                    #
                    # Tv0u0, LTv0u0, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                          logfolder=logfolder, resultfolder=resultfolder,
                    #                                                          specialPaths=specialPaths, edges=edgesC0,
                    #                                                          specialEdges=specialEdges,
                    #                                                          figuresResultBuildings=figuresResultBuildings,
                    #                                                          elevatorEdges=elevatorEdges,
                    #                                                          nodeToCoordinate=nodeToCoordinate,
                    #                                                          vdummy=vdummy,
                    #                                                          maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                          elevatorVertices=elevatorVertices,
                    #                                                          ends=[])
                    # print(f"2-cut 0ends, taking u and v, starting in c0. part c1. Now redgb is called on c1:{buildings1}, and ends:{[v1,u1]}")
                    #
                    # Tc1uv, LTc1uv, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                      logfolder=logfolder, resultfolder=resultfolder,
                    #                                                      specialPaths=specialPaths, edges=edgesC1,
                    #                                                      specialEdges=specialEdges,
                    #                                                      figuresResultBuildings=figuresResultBuildings,
                    #                                                      elevatorEdges=elevatorEdges,
                    #                                                      nodeToCoordinate=nodeToCoordinate,
                    #                                                      vdummy=vdummy,
                    #                                                      maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                      elevatorVertices=elevatorVertices,
                    #                                                      ends=[v1, u1])
                    # LTc0c1c0 = LTv0u0 - edgesC0[(v0, u0)] + LTc1uv + edges[cut[0]] + edges[cut[1]]
                    #
                    # # sixth case, using both bridges, starting in C1
                    # edgesC1[(v1, u1)] = sum(list(edgesC1.values()))
                    # print(f"2-cut 0 ends taking u and v, starting in c1. part c1Now redgb is called on c1:{buildings0}, and ends:{ends}")
                    # Tv1u1, LTv1u1, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                      logfolder=logfolder, resultfolder=resultfolder,
                    #                                                      specialPaths=specialPaths, edges=edgesC1,
                    #                                                      specialEdges=specialEdges,
                    #                                                      figuresResultBuildings=figuresResultBuildings,
                    #                                                      elevatorEdges=elevatorEdges,
                    #                                                      nodeToCoordinate=nodeToCoordinate,
                    #                                                      vdummy=vdummy,
                    #                                                      maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                      elevatorVertices=elevatorVertices,
                    #                                                      ends=ends)
                    # print(f"2-cut 0 ends taking u and v, starting in c1. part c0. Now redgb is called on c0:{buildings0}, and ends:{[v0,u0]}")
                    # Tc0uv, LTc0uv, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents,
                    #                                                  logfolder=logfolder, resultfolder=resultfolder,
                    #                                                  specialPaths=specialPaths, edges=edgesC0,
                    #                                                  specialEdges=specialEdges,
                    #                                                  figuresResultBuildings=figuresResultBuildings,
                    #                                                  elevatorEdges=elevatorEdges,
                    #                                                  nodeToCoordinate=nodeToCoordinate,
                    #                                                  vdummy=vdummy,
                    #                                                  maxtime=maxtime, maxgap=None, printtime=5,
                    #                                                  elevatorVertices=elevatorVertices,
                    #                                                  ends=[v0,u0])
                    #
                    # LTc1c0c1 = LTv1u1 - edgesC1[(v1, u1)] + LTc0uv + edges[cut[0]] + edges[cut[1]]
                    #
                    # LT= max(LTc0, LTc1, LTv, LTu, LTc0c1c0, LTc1c0c1)
                    # if LT == LTc0:
                    #     T=Tc0
                    # elif LT == LTc1:
                    #     T=Tc1
                    # elif LT== LTv:
                    #     T=swapTrailIfNeeded(v0, Tc0v0)
                    #     T.reverse()
                    #     T.append((v0,v1))
                    #     T.extend(swapTrailIfNeeded(v1, Tc1v1))
                    # elif LT == LTu:
                    #     T = swapTrailIfNeeded(u0, Tc0u0)
                    #     T.reverse()
                    #     T.append((u0, u1))
                    #     T.extend(swapTrailIfNeeded(u1, Tc1u1))
                    # elif LT== LTc0c1c0:
                    #     if ((v0, u0)) in Tv0u0:
                    #         pos = Tv0u0.index((v0, u0))
                    #     else:
                    #         pos = Tv0u0.index((u0, v0))
                    #     T = Tv0u0[0:pos]
                    #     T1 = Tv0u0[pos + 1:]
                    #     if T[-1][1] == v0:
                    #         T.append((v0, v1))
                    #         Tc1uv = swapTrailIfNeeded(v1, Tc1uv)
                    #         T.extend(Tc1uv)
                    #         T.append((u1, u0))
                    #         T.extend(T1)
                    #     else:
                    #         T.append((u0, u1))
                    #         Tc1uv = swapTrailIfNeeded(u1, Tc1uv)
                    #         T.extend(Tc1uv)
                    #         T.append((v1, v0))
                    #         T.extend(T1)
                    #
                    # else:
                    #     if ((v1, u1)) in Tv1u1:
                    #         pos = Tv1u1.index((v1, u1))
                    #     else:
                    #         pos = Tv1u1.index((u1, v1))
                    #     T = Tv1u1[0:pos]
                    #     T1 = Tv1u1[pos + 1:]
                    #     if T[-1][1] == v1:
                    #         T.append((v1, v0))
                    #         Tc0uv = swapTrailIfNeeded(v0, Tc0uv)
                    #         T.extend(Tc0uv)
                    #         T.append((u0, u1))
                    #         T.extend(T1)
                    #     else:
                    #         T.append((u0, u1))
                    #         Tc0uv = swapTrailIfNeeded(u1, Tc0uv)
                    #         T.extend(Tc0uv)
                    #         T.append((v0, v1))
                    #         T.extend(T1)
                    # trailsComponents[componentkey] = {'trail': T, 'length': LT}
                    # return T, LT, trailsComponents

            else:
                print(f"We have a 1 cut")
                print(f"1-edge cut is: {cutMade}\n with keys: {cutMade.keys()}")
                if len(ends)>0:
                    vinc0=ends[0]
                else:
                    vinc0= cut[0]

                if vinc0 in info['Component1vertices']:
                    v0= cut[0]
                    v1 = cut[1]
                    c0 = info['Component1vertices']
                    c1 = info['Component2vertices']
                else:
                    v1 = cut[0]
                    v0 = cut[1]
                    c1 = info['Component1vertices']
                    c0 = info['Component2vertices']

                print(
                    f"sanity check: v0 in c0:{v0 in c0},  v1 in c1:{v1 in c1},\n"
                    f"v0 in c1:{v0 in c1}, v1 in c0:{v1 in c0}")
                for end in ends:
                    print(f"sanity check on end {end} in c0: {end in c0}, and in c1?:{end in c1}")
                buildings0 = getBuildings(c0, nodeToCoordinate)
                buildings1 = getBuildings(c1, nodeToCoordinate)
                edgesC0 = getEdgesComponent(edges, c0)  # weighted edges
                edgesC1 = getEdgesComponent(edges, c1)  # weighted edges
                if buildings1 =={'NANOLAB 1411': {'2'}, 'ZILVERLING 1313': {'4', '3', '5', '2', '1'}}:
                    exportGraphinfo(Path=resultfolder, halls=edges, nodeToCoordinate=nodeToCoordinate,scales=False,trail=edgesC1,prefix="NLandZI1comp")

                print(f"one cut edge separating hallways in {getBuildings(c0, nodeToCoordinate)} \n from hallways in {getBuildings(c1, nodeToCoordinate)}")
                # Now find the longest trail in c0, called Tc0, in c1, called Tc1
                # Also find the longest trail in c0 starting in v0, called Tv0
                # Also find the longest trail in c1 starting in v1, called Tv1,
                # Then return the longest of Tc0, Tc1, Tv0+Tv1+w((v0,v1))

                #check if we needed to start in a certain vertex and in which component that lies in.
                assert len(ends) in [0,1,2] # Since the only options are for a component to have a mandatory start, end, both or none
                if len(ends)==0: #meaning no restrictions on where to start or end
                    print(f"in reduce Graph bridges, we are free to find a longest trail anywhere in this component with c0:nvertices:{len(c0)}nedges:{len(edgesC0)} and then c1:nvertices:{len(c1)} nedges:{len(edgesC1)}.In total {max(list(neighbours.keys()))+1} vertices and {len(edges)} edges")
                    print(f"1-cut 0ends taking v, c0v0. Calling redgb on c0:{buildings0} ending in v0:{v0}")
                    Tv0, LTv0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                     logfolder=logfolder, resultfolder=resultfolder,
                                                                     specialPaths=specialPaths, edges=edgesC0,
                                                                     specialEdges=specialEdges,
                                                                     figuresResultBuildings=figuresResultBuildings,
                                                                     elevatorEdges=elevatorEdges,
                                                                     nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                                                                     maxtime=maxtime, maxgap=None, printtime=5,
                                                                     elevatorVertices=elevatorVertices, ends=[
                            v0])  # longest trail in c0 that starts or ends in v0
                    print(f"length in c0 starting from v0: {LTv0}")
                    # print(f"ends will be v1:{v1}")
                    print(f"1-cut 0 ends taking v, part c1v1. Calling redgb on c1:{buildings1}, ends:v1{v1} ")
                    Tv1, LTv1, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                     logfolder=logfolder, resultfolder=resultfolder,
                                                                     specialPaths=specialPaths, edges=edgesC1,
                                                                     specialEdges=specialEdges,
                                                                     figuresResultBuildings=figuresResultBuildings,
                                                                     elevatorEdges=elevatorEdges,
                                                                     nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                                                                     maxtime=maxtime, maxgap=None, printtime=5,
                                                                     elevatorVertices=elevatorVertices, ends=[
                            v1])  # longest trail in c1 that starts or ends in v1
                    # print(f"length in c1 starting from v1: {LTv1}")
                    LT= LTv0+edges[cut]+LTv1
                    T= Tv0+[cut]+Tv1
                    if sum(edgesC0.values())> LT: #it is possible to walk here longer
                        print(f"1-cut 0 ends, staying in c0. Calling redgb for c0:{buildings0}, and no ends param specified")
                        Tc0, LTc0, trailsComponents= reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents ,logfolder=logfolder, resultfolder=resultfolder, specialPaths=specialPaths, edges=edgesC0, specialEdges=specialEdges,
                               figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                                   maxtime=maxtime, maxgap=None, printtime=5, elevatorVertices=elevatorVertices) # longest trail in c0
                        print(f"length Tc0: {LTc0}")
                        if LT< LTc0:
                            LT=LTc0
                            T=Tc0
                    if sum(edgesC1.values())> LT: # it could be possible to walk here logner
                        print(f"1-cut 0ends staying in c1. Calling redgb for c1:{buildings1}, and no ends param specified")
                        Tc1, LTc1, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents ,logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC1, specialEdges=specialEdges,
                               figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                                   maxtime=maxtime, maxgap=None, printtime=5, elevatorVertices=elevatorVertices) # longest trail in c1
                        print(f"length Tc1: {LTc1}")
                        if LT< LTc1:
                            LT=LTc1
                            T=Tc1
                    trailsComponents[componentkey] = {'trail': T, 'length': LT}
                    return T, LT, trailsComponents
                    # print(f"ends will be v0:{v0}")

                    # now calculate the length, so first see if v0v1 is in the edges, or the other way around:
                    # LTc0c1= LTv0+edges[cut]+LTv1
                    # print(f"so length c0c1:{LTc0c1}")

                    # T=max(LTc0,LTc0c1 ,LTc1)
                    # if T == LTc0:
                    #     trailsComponents[componentkey] = {'trail': Tc0, 'length': LTc0}
                    #     return  Tc0, LTc0, trailsComponents
                    # elif T== LTc1:
                    #     trailsComponents[componentkey] = {'trail': Tc1, 'length': LTc1}
                    #     return Tc1, LTc1, trailsComponents
                    # else:
                    #     # Construct the combined trail, starting in c0, going to v0, then walk the edge to v1, and walk a longest trail in c1
                    #     # check the order of the edges in Tv0,
                    #     Tv0=swapTrailIfNeeded(v0, Tv0)
                    #
                    #     # check the order of the edges in Tv1,
                    #     Tv1=swapTrailIfNeeded(v1, Tv1)
                    #
                    #     Tc0c1 = Tv0 + [(v0, v1)] + Tv1
                    #     trailsComponents[componentkey] = {'trail': Tc0c1, 'length': LTc0c1}
                    #     return Tc0c1, LTc0c1, trailsComponents

                    print(f"as we had")
                elif len(ends)==1: #meaning we have 1 mandatory vertex to visit in a longest trail in this component.
                    print(f"in reduce graph bridges we must find a trail that starts or ends in{ends[0]}.")
                    # We have two possible trails to find, one staying in component 0, starting in ends[0], or the longest trail that starts in end[0], and goes through both components by means of taking the cutedge
                    print(
                        f"1-cut 1 end, taking v, part c0v0. Calling redgb on c0 {buildings0}, and ends:{[ends[0], v0]}")
                    Tc0v0, LTc0v0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                         logfolder=logfolder, resultfolder=resultfolder,
                                                                         specialPaths=specialPaths, edges=edgesC0,
                                                                         specialEdges=specialEdges,
                                                                         figuresResultBuildings=figuresResultBuildings,
                                                                         vdummy=vdummy,
                                                                         elevatorEdges=elevatorEdges,
                                                                         nodeToCoordinate=nodeToCoordinate,
                                                                         maxtime=maxtime, maxgap=None, printtime=5,
                                                                         elevatorVertices=elevatorVertices,
                                                                         ends=[ends[0],
                                                                               v0])  # longest trail in c0 that starts or ends in v0
                    # print(f"ends will be v1:{v1}")
                    print(f"1-cut 1 end taking v part c1v1. Calling redgb on c1:{buildings1}, ends v1:{v1}")
                    Tv1, LTv1, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents,
                                                                     logfolder=logfolder, resultfolder=resultfolder,
                                                                     specialPaths=specialPaths, edges=edgesC1,
                                                                     specialEdges=specialEdges,
                                                                     figuresResultBuildings=figuresResultBuildings,
                                                                     vdummy=vdummy,
                                                                     elevatorEdges=elevatorEdges,
                                                                     nodeToCoordinate=nodeToCoordinate,
                                                                     maxtime=maxtime, maxgap=None, printtime=5,
                                                                     elevatorVertices=elevatorVertices,
                                                                     ends=[
                                                                         v1])  # longest trail in c1 that starts or ends in v1
                    LT = LTc0v0 + edges[cut] + LTv1
                    T= Tc0v0+[cut]+Tv1

                    if sum(edgesC0.values())>LT:# it could be possible to walk here longer
                        print(f"1-cut 1-end staying in c0. Calling redgb on c0:{buildings0}, ends:{ends}")
                        Tc0, LTc0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents ,logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC0,
                                                            specialEdges=specialEdges,
                                                            figuresResultBuildings=figuresResultBuildings, vdummy=vdummy,
                                                            elevatorEdges=elevatorEdges, nodeToCoordinate=nodeToCoordinate,
                                                            maxtime=maxtime, maxgap=None, printtime=5,
                                                            elevatorVertices=elevatorVertices,
                                                            ends=ends)  # longest trail in c1 that starts or ends in v1
                        if LT< LTc0:
                            LT=LTc0
                            T=Tc0
                    trailsComponents[componentkey] = {'trail': T, 'length': LT}
                    return T, LT, trailsComponents

                    # T = max(LTc0, LTc0c1)
                    # if T == LTc0:
                    #     Tc0=swapTrailIfNeeded(ends[0], Tc0)
                    #     trailsComponents[componentkey] = {'trail': Tc0, 'length': LTc0}
                    #     return Tc0, LTc0, trailsComponents
                    #
                    # else:
                    #     # Construct the combined trail, starting in end[0] in c0, going to v0, then walk the edge to v1, and walk a longest trail in c1
                    #     # check the order of the edges in Tv0,
                    #     Tc0v0=swapTrailIfNeeded(ends[0], Tc0v0)
                    #
                    #     # check the order of the edges in Tv1,
                    #     Tv1=swapTrailIfNeeded(v1, Tv1)
                    #
                    #     Tc0c1 = Tc0v0 + [(v0,v1)] + Tv1
                    #     trailsComponents[componentkey] = {'trail': Tc0c1, 'length': LTc0c1}
                    #     return Tc0c1, LTc0c1, trailsComponents

                else: #we have a mandatory start and a mandatory end, so two options, they either are in the same component or not
                    print(f"in reduce graph bridges we must find a trail that starts and ends in{ends}.")

                    if ends[1] in c0:
                        print(f"we only have to find a longest trail in c0 since both ends lay here:{buildings0} with ends {ends}. so redgb is called on this")
                        # print(f"ends was:{ends} and will be the same: {ends}")
                        Tc0, LTc0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents ,logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC0,
                                                                specialEdges=specialEdges,
                                                                figuresResultBuildings=figuresResultBuildings,
                                                                elevatorEdges=elevatorEdges,
                                                                nodeToCoordinate=nodeToCoordinate,  vdummy=vdummy,
                                                                maxtime=maxtime, maxgap=None, printtime=5,
                                                                elevatorVertices=elevatorVertices,
                                                                ends=ends)  # longest trail in c0 that starts and ends in vertices in ends
                        # Tc0=swapTrailIfNeeded(ends[0], Tc0)

                        trailsComponents[componentkey] = {'trail': Tc0, 'length': LTc0}
                        return Tc0, LTc0, trailsComponents

                    # elif ends[0] in c1 and ends[1] in c1:
                    #     print(f"we only have to find a longest trail in c1 with ends {ends}")
                    #     Tc1, LTc1, trailsComponents = reduceGraphBridges(trailsComponents=trailsComponents ,logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC1,
                    #                                         specialEdges=specialEdges,
                    #                                         figuresResultBuildings=figuresResultBuildings,
                    #                                         elevatorEdges=elevatorEdges,
                    #                                         nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                    #                                         maxtime=maxtime, maxgap=None, printtime=5,
                    #                                         elevatorVertices=elevatorVertices,
                    #                                         ends=ends)  # longest trail in c0 that starts and ends in vertices in ends
                    #     if ends[0] in Tc1[0]:  # they are in the correct order
                    #         print(f"edges are in correct order")
                    #     elif ends[0] in Tc1[-1]:  # they are in reverse order
                    #         Tc1.reverse()
                    #     else:
                    #         print(f"ERRORRR the vertex ends0 is neither in the start nor end edge of the trail in Tc1??")
                    #     trailsComponents[componentkey] = {'trail': Tc1, 'length': LTc1}
                    #     return Tc1, LTc1, trailsComponents
                    else:
                        print(f"They are in a different component, so we must find a trail from ends[0] to v0 to v1 to ends[1]")
                        #Now find a longest trail in co from ends0 to v0 and a longest trail in c1 from v1 to ends1
                        # print(f"ends will be: [ends[0],v0]: {[ends[0],v0]}")
                        print(f"1-cut 2 ends taking v, part c0v0. Calling redgb on building{buildings0}, and ends v0:{[ends[0],v0]}")
                        Tc0v0, LTc0v0, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents ,logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC0,
                                                                specialEdges=specialEdges,
                                                                figuresResultBuildings=figuresResultBuildings,
                                                                elevatorEdges=elevatorEdges,
                                                                nodeToCoordinate=nodeToCoordinate, vdummy=vdummy,
                                                                maxtime=maxtime, maxgap=None, printtime=5,
                                                                elevatorVertices=elevatorVertices,
                                                                ends=[ends[0],v0])  # longest trail in c0 that starts or ends in v0
                        # print(f"ends will be: [v1,ends[1]]: {[v1,ends[1]]}")
                        print(f"1-cut 2 end diff comp taking v, part c1v1. Calling redgb on c1:{buildings1}, and ends:{[v1,ends[1]]}")
                        Tc1v1, LTc1v1, trailsComponents = reduceGraphBridges(auxedge=auxedge,custom3cut=custom3cut,trailsComponents=trailsComponents ,logfolder=logfolder, resultfolder=resultfolder,specialPaths=specialPaths, edges=edgesC1,
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
                        # Tc0v0=swapTrailIfNeeded(ends[0], Tc0v0)

                        # check the order of the edges in Tv1,
                        # Tc1v1= swapTrailIfNeeded(v1, Tc1v1)

                        Tc0c1 = Tc0v0 + [cut] + Tc1v1
                        trailsComponents[componentkey] = {'trail': Tc0c1, 'length': LTc0c1}
                        return Tc0c1, LTc0c1, trailsComponents

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


def getGraph(PATH_drawings, PATH_empty,bridgeLengths, buildingsToSkip, floorsToSkip):
    # Initiate some data structures to save information in
    nodeToCoordinate={}
    coordinateToNode={}
    custom3cut=[]
    nextnode=0

    specialPaths={}
    specialEdges={}
    hallways={}

    figuresResultBuildings  = dict()



    # Loop over the files to save a file where we can draw the resulting longest paths in
    buildingScales=dict()
    for building in os.listdir(PATH_drawings):
        if any([skipname in building for skipname in buildingsToSkip]):
            continue

        buildingEmpty= PATH_empty+f"/{building}"
        listOfFiles = os.listdir(buildingEmpty)
        for file in listOfFiles:
            if file.endswith(".svg"):
                floor= file.split('.')[1]
                if floor in floorsToSkip:
                    continue
                newFigurePath = buildingEmpty + f"/{file}"

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
                paths, attributes = svg2paths(PATH_drawings + f"/{building}/{file}")
                lengthForOneMeter= 1
                if "RAVELIJN" in building:
                    if floor == '3': # Since the scale was missing, I measured myself what the length of 1 meter would be: 8
                        if building in buildingScales:
                            buildingScales[building][floor] = 8
                        else:
                            buildingScales[building] = {floor: 8}

                for path, attr in zip(paths, attributes):
                    if "inkscape:label" in attr.keys():
                        if "1mtr" in attr['inkscape:label']:
                            print(f"the scale label is: {attr['inkscape:label']}")
                            lengthForOneMeter = abs(path.start - path.end)
                            if building in buildingScales:
                                buildingScales[building][floor]=lengthForOneMeter
                            else:
                                buildingScales[building]={floor : lengthForOneMeter}
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

                            if "customCut" in attr['inkscape:label']:
                                custom3cut.append((snode,enode))

                        hallways[(snode, enode)] = edgeweight

    print(f"We extracted special paths{specialPaths} from the floor plans")

    # Define a dictionary where each vertex maps to a set of its neighbours
    neighboursold = getNeighbourhood(hallways.keys())


    neighbours=deepcopy(neighboursold) # deep copy the neighbourhood, to which we can add the
    # Connections regarding staircases, elevators and connections between buildings

    #Connect special paths, elevators first:
    #CONVENTION: use double digits for index elevator, stair, exit, and double digits for the floors! to keep things consistent

    elevatorVertices=dict() # Keep a dictionary storing the vertices for elevators, which need to have degree <=2
    elevatorEdges=set()
    connected=[]
    addedBridges=dict()
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
                    end= findSingleEnd(edge, neighboursold, specialPaths)
                    hallways[(end, velevator)]=0.5
                    elevatorEdges.add((end, velevator))
                    elevatorEdges.add((velevator,end))
                    neighbours[end].add(velevator)
                    neighbours[velevator].add(end)
                connected.append(startname)
        elif pathname[2] == "S": #We have an edge to a staircase
            if pathname not in connected: # if we have not connected these two floors with this staircase
                end1= findSingleEnd(pathname, neighboursold, specialPaths)
                otherSide= pathname[:5]+pathname[7:]+pathname[5:7]
                if otherSide in specialPaths:
                    # print(f"original stair:{pathname}, other side:{otherSide}")
                    end2=findSingleEnd(otherSide, neighboursold, specialPaths)
                    hallways[(end1, end2)] = 7 # 1 stair 7 meter? idkkk
                    neighbours[end1].add(end2)
                    neighbours[end2].add(end1)
                    connected.append(pathname)
                    connected.append(otherSide)
        elif pathname[1] == "0": #connecting buildings? naming conv?
            # otherSide = pathname[3:5]+pathname[2]+pathname[0:2]+pathname[5:]
            print(f"connection building from CR:{pathname} but we deal with this when we find the other end this bridge is connected to.")
        elif pathname[2] == "C":
            print(f"CONNECTION TO A WALKING BRIDGE:name: {pathname}")
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
                # if 'and' not in specialPaths[bridgeName]['Building']: #check that we did not add this bridge in the code but in inkscape
                if bridgeName not in connected:
                    if otherSide in specialPaths:
                        print(f"????THE???? walking bridge is connected to edge {pathname} in the same building. I overdid the naming and will connect this hallway to the otherSide")
                        end1= findSingleEnd(bridgeName, neighboursold, specialPaths)
                        end2=findSingleEnd(otherSide, neighboursold, specialPaths)
                        hallways[(end1, end2)] = 0  # Connect the endpoint of a bridge to the corresponding entry of the other building
                        neighbours[end1].add(end2)
                        neighbours[end2].add(end1)
                        connected.append(pathname)
                        connected.append(bridgeName)
                        connected.append(otherSide)
                    else:
                        print(f"the other building for Connection to BRIDGE:name: {pathname} is not drawn into so we cannot connect the walking bridge for which we overdid the naming on the other end")
                else:
                    print(f"overdid the naming, but already connected this hallway:{pathname}, so just continue")
            elif pathname not in connected:
                print(f"We did not connect this endpoint:{pathname} yet")
                #Check if the bridge is drawn into the other buildings floorplan
                if otherBridgeName in specialPaths.keys():

                    print(f"The bridge {otherBridgeName} is drawn fully on the floorplan of the other building, so we just connect those two ends with an edge of weight zero")
                    #We find the end of edge pathname and of edge otherBridgeName and connect them with an edge of weight zero
                    end1= findSingleEnd(pathname, neighboursold, specialPaths)
                    end2= findSingleEnd(otherBridgeName, neighboursold, specialPaths)
                    hallways[(end1, end2)] = 0  # Connect the endpoint of a bridge to the corresponding entry of the other building
                    neighbours[end1].add(end2)
                    neighbours[end2].add(end1)
                    connected.append(pathname)
                    connected.append(otherBridgeName)
                    #to be sure also append the other side of this connection (for if I overdid the naming of the drawn in edges)
                    connected.append(otherSide)
                    print(f"those two ends are:{end1}, {end2}")
                elif otherSide not in specialPaths:
                    print(f"we have not yet drawn in the floorplans of the other building so we cannot connect edge: {pathname}")
                else:
                    print(f"We did draw the paths in the other building of the bridge between {pathname} and {otherSide}. Have to measure this bridge by hand and connects the two ends of bridge with an edge of this length.")
                    lenbridge= bridgeLengths[bridgeName]
                    end1 = findSingleEnd(pathname, neighboursold, specialPaths)
                    print(f"end1: {end1}: {nodeToCoordinate[end1]}")
                    end2 = findSingleEnd(otherSide, neighboursold, specialPaths)
                    print(f"end2: {end2}: {nodeToCoordinate[end2]}")

                    hallways[(end1, end2)] = lenbridge  # Connect the endpoint of a bridge to the corresponding entry of the other building
                    addedBridges[bridgeName]= {'Building': buildingOfEdge+' and '+buildingToEdge,
                 'End': {'Location': nodeToCoordinate[end2]['Location'], 'Vnum': end2},
                 'Floor': nodeToCoordinate[end2]['Floor'],
                 'Start': {'Location': nodeToCoordinate[end1]['Location'], 'Vnum': end1}}
                    neighbours[end1].add(end2)
                    neighbours[end2].add(end1)
                    connected.append(pathname)
                    connected.append(otherSide)
            else:
                print(f"We already connected this bridge {pathname} leads to.")

        elif pathname[2]=='X':
            print(f"we have an exit to outdoors: {pathname}")
        else:
            print(f"somehting went wrong: {pathname} not a stair, elevator, connection or exit")

    specialPaths.update(addedBridges)
    vdummy= max(list(neighbours.keys()))+1
    print(f"all bridges that we added:{addedBridges}")
    return custom3cut, figuresResultBuildings, buildingScales, nodeToCoordinate, specialPaths, specialEdges, hallways, elevatorVertices, elevatorEdges, vdummy, neighbours

