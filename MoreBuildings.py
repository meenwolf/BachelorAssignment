# from OneBuilding import *
from TestMoreFloors import *

import sys
import json

# Start exporting hallways: dict with keys: (v1, v2) representing an edge between vertex v1 and vertex v2, values are the weights of the hallway in real life
# nodeToCoordinate: dict with keys: v1, representing the vertex v1, and values a dict containing the building, floor and coordinate on that floor of vertex v1, example: v1:{"Building": "CARRE 1412", "Floor": "3", "Location": 100.9347+221.876j}
# buildingScales: dict with keys the building names, and the values a dict that contains for each floor the scale x. (that is: 1meter in real life: distance x in the floor plan)
# Convert and write JSON object to file

def exportGraphinfo(halls, nodeToCoordinate, scales, prefix=""):
    weightedEdges = [{'key': key, 'value': value} for key, value in halls.items()]
    with open(prefix+"weigthedEdges.json", "w") as outfile:
        json.dump(weightedEdges, outfile)

    nodeToCoordinates = {vertex: {'Building': info["Building"], "Floor": info["Floor"], "x": np.real(info["Location"]),
                                  "y": np.imag(info["Location"])} for vertex, info in nodeToCoordinate.items()}
    with open(prefix+"nodeToCoordinates.json", "w") as outfile:
        json.dump(nodeToCoordinates, outfile)

    with open(prefix+"buildingScales.json", "w") as outfile:
        json.dump(scales, outfile)

def runModelends(logfolder, halls, neighbours,ends=[],nvdum=None, maxtime=None, maxgap=None, printtime=None, log=False, elevatorVertices=[]):
    # if nvdum== None:
    #     nvdum=max(list(neighbours.keys()))

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
    # m.addConstr(varsaux[nvdum]==1, name='evenDegreeVDUM')
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
    # all_vars = m.getVars()
    # values = m.getAttr("X", all_vars)
    # names = m.getAttr("VarName", all_vars)
    # nedges=0
    # for name, val in zip(names, values):
    #     print(f"{name} = {val}")
    #     if 'x' in name:
    #         if val > 0.5:
    #             nedges += 1
    # print(f"WE HAVE {nedges} IN THE RESULT OF THE MODEL")
    return m, varssol, varsaux
#
# def runModel3OGidea(halls,neighbourhood, elevatorVertices,nvdum, maxtime=None, maxgap=None, printtime=None, logfile=None, ends=[]):
#     print(f"keys of neighbourhood:{neighbourhood.keys()}")
#     m = Model()
#     # m.Params.Aggregate=2
#     # m.Params.Cuts=3
#     # m.Params.Method=1
#     # m.Params.MIPFocus=3
#     # m.Params.BranchDir=1
#     # m.Params.VarBranch=3
#     # m.Params.ImproveStartTime= 10
#     print(f"we run with improvestarttime of 10 seconds!")
#     if maxtime != None:
#         m.Params.TimeLimit = maxtime
#     if maxgap!=None:
#         m.Params.MIPGap= maxgap
#     if printtime!=None:
#         m.Params.DisplayInterval= printtime
#     if logfile != None:
#         m.Params.LogFile= PATH+logfile
#     # Variables: the hallway connecting crossing i and j in the tour?
#     varssol = m.addVars(halls.keys(), vtype=GRB.BINARY, name='x')
#
#     # Symmetric direction: use dict.update to alias variable with new key
#     varssol.update({(j, i): varssol[i, j] for i, j in varssol.keys()})
#
#     # Add auxiliary variable to help ensure even degree
#     varsaux = m.addVars(range(nvdum), vtype=GRB.INTEGER, name="y")
#
#     # Set the objective function for the model
#     m.setObjective(sum([halls[e] * varssol[e] for e in halls.keys()]), sense=GRB.MAXIMIZE)
#
#     # Add the even degree constraint for dummyvetex nvdum=2:
#     m.addConstr(sum([varssol[(nvdum, e)] for e in neighbourhood[nvdum]]) == 2, name='evenDegreeVDUM')
#
#     # Add the even degree constraint for the other vertices
#     for i in range(nvdum):
#         m.addConstr(sum([varssol[(i, e)] for e in neighbourhood[i]]) == 2 * varsaux[i],
#                         name=f'evenDegreeVertex{i}')
#
#     # Add the constraint that forbids you to use an elevator multiple times:
#     for name, vertex in elevatorVertices.items():
#         m.addConstr(varsaux[vertex]<= 1, name=f'visit{name}AtMostOnce')
#
#
#     # Set up for the callbacks/ lazy constraints for connectivity
#     m.Params.LazyConstraints = 1
#     cb = TSPCallback(range(nvdum+1), varssol)
#
#     # Call optimize with the callback structure to get a connected solution solution
#     m.optimize(cb)
#     return m, varssol, varsaux

# Constraint to maybe add to runModel??
# For the case where we simplified the graph with onecuts, we can have a mandatory start or end point
#     So add those constraints underneath.
    #
    # assert len(ends) in [0,1,2] # Since we can only have a mandatory start, end, both or none,
    #
    # if len(ends)>0: #meaning that we have a mandatory start or endpoint or both for our trail
    #     for end in ends:
    #         m.addConstr(varssol[(nvdum, end)]==1,  name=f'muststartorendin{end}') #so we must end(or start) in these vertices since that is the only vertex connected to vdum, and we
    #     # we have our even degree, with degree of vdum=2, and connected graph. Those make sure we end in end.

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

#Redefine draw edges in floorplans, and name it draw AllEdges to prevent confusion with the one drawing in only one component?
def drawAllEdges(edges):
    rainbowColors= getRainbowColors(len(edges))
    startedge= edges[0]
    endedge= edges[-1]

    for i,edge in enumerate(edges):
        if edge in elevatorEdges:
            print(f"We can not draw edge {edge}: since it is an elevator")
            continue
        if edge[0] not in nodeToCoordinate:
            print(f"vertex{edge[0]} of edge {edge} is not in node to coordinate, is a special vertex?")
            continue
        if edge[1] not in nodeToCoordinate:
            print(f"vertex{edge[1]} of edge {edge} is not in node to coordinate, is a special vertex?")
            continue
        # print(f"node to coordinate edge[0]: {nodeToCoordinate[edge[0]]}")
        building0, floor0= getBuildingFloor(vnum=edge[0], nodeToCoordinate=nodeToCoordinate)
        # print(f"node to coordinate edge[1]: {nodeToCoordinate[edge[1]]}")
        building1, floor1= getBuildingFloor(vnum=edge[1], nodeToCoordinate=nodeToCoordinate)

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
                            toFloor= nodeToCoordinate[edges[i+2][0]]['Floor']
                            drawCoord = nodeToCoordinate[edge[1]]['Location']
                        elif edges[i-1] in elevatorEdges:
                            # we step out of the elevator, but still print a number from which floor we came for if you walk in the other order
                            toFloor = nodeToCoordinate[edges[i -2][0]]['Floor']
                            drawCoord = nodeToCoordinate[edge[0]]['Location']
                        else:
                            print(f"EDGE {edge} with name{specialEdges[edge]} is not in or out of an elevator!")

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
                startco, endco = getCoordinatesPair(vtuple=edge, nodeToCoordinate=nodeToCoordinate)
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
            testfilename= f"\\newdrawing{buildingNumber}.{floor}.svg"
            floortree.write(buildingResultPath+testfilename)

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
if __name__ == "__main__":

    # Get the path to the drawings
    dir_path = os.path.dirname(os.path.realpath(__file__))
    PATH= os.path.abspath(os.path.join(dir_path, os.pardir))

    PATH_drawings= PATH +"\\TestMoreBuildings\\OriginalPaths"
    PATH_empty= PATH+"\\TestMoreBuildings\\EmptyFloorplans"
    PATH_result= PATH+"\\TestMoreBuildings\\ResultFloorplans"
    PATH_logs= PATH+"\\TestMoreBuildings\\Logs"
    if not os.path.exists(PATH_logs):
        os.mkdir(PATH_logs)

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
    buildingScales=dict()
    for building in os.listdir(PATH_drawings):
        if 'WAAIER' in building:
            continue
        # if 'CITADEL' in building:
        #     continue
        if 'NANO' in building:
            continue
        if 'HORST' in building:
            continue
        # if 'RAVELIJN' in building:
        #     continue
        if 'ZILVERLING' in building:
            continue
        if 'CARRE' in building:
            continue
        buildingEmpty= PATH_empty+f"\\{building}"
        listOfFiles = os.listdir(buildingEmpty)
        for file in listOfFiles:
            if file.endswith(".svg"):
                floor= file.split('.')[1]
                if floor in ['0','3','4','5','6']:
                    continue
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

                        hallways[(snode, enode)] = edgeweight

    # print(figuresResultBuildings)
    # print(f"the special paths of size {sys.getsizeof(specialPaths)} bytes are:")
    # pprint(specialPaths)
    # print(f"special edges of size {sys.getsizeof(specialEdges)} bytes are \n {specialEdges}:")
    # print("The hallways of size {sys.getsizeof(hallways)} bytes are:")
    # pprint(hallways)
    # print(f"we have this many vertices drawn intothe floor plans: {nextnode}")
    # Define a dictionary where each vertex maps to a set of its neighbours

    neighboursold = getNeighbourhood(hallways.keys())
    # print(f" neighboursold is of size: {sys.getsizeof(neighboursold)} bytes")


    neighbours=deepcopy(neighboursold) # deep copy the neighbourhood, to which we can add the
    # Connections regarding staircases, elevators and connections between buildings
    # print(f" neighbours is of size: {sys.getsizeof(neighbours)} bytes")

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
                        print(f"the other building is not drawn into so we cannot connect the walking bridge for which we overdid the naming on the other end")
                else:
                    print(f"overdid the naming, but already connected this hallway, so just continue")
            elif pathname not in connected:
                print(f"We did not connect this endpoint yet")
                #Check if the bridge is drawn into the other buildings floorplan
                if otherBridgeName in specialPaths.keys():

                    print(f"The bridge is drawn fully on the floorplan of the other building, so we just connect those two ends with an edge of weight zero")
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
                print(f"We already connected this bridge.")

        elif pathname[2]=='X':
            print(f"we have an exit to outdoors: {pathname}")
        else:
            print(f"somehting went wrong: {pathname} not a stair, elevator, connection or exit")

    specialPaths.update(addedBridges)
    vdummy= max(list(neighbours.keys()))+1

    # oneCuts, twoCuts = findCuts(edges=hallways.keys(), specialPaths=specialPaths)
    #
    # print(f"the following bridges are onecuts: {oneCuts}")
    # print(f"the following bridges are twocuts: {twoCuts}")

    def reduceGraphBridges(specialPaths, logfolder, resultfolder, edges, specialEdges, figuresResultBuildings,
                           nodeToCoordinate, vdummy,elevatorEdges=[], ends=[],neighbours=None,
                           maxtime=None, maxgap=None, printtime=None, logfile=False, elevatorVertices=[],
                           prefixdrawcomp=False, plotboundsname=False, showboundplot=False, saveboundplotname=False):
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
                            if ends[0] in edgesTc0v0[0]:
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

    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    datenew = date.replace(':', '-')
    logfile = "\\log" + datenew + ".log"

    # titleplot = "The bounds on the length of a longest trail in Ravelijn,<br> at each moment in time when running the gurobi solver for 2 hours."
    # titleplot = "The bounds on the length of a longest trail in through CI,RA,ZI and CR,<br> at each moment in time when running the gurobi solver for 90 seconds per component."

    # boundplotname = f'RA{datenew}.svg'
    trail, length= reduceGraphBridges(specialPaths=specialPaths, logfolder=PATH_logs, resultfolder=PATH_result, edges=hallways, specialEdges=specialEdges,
                       figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, neighbours=neighbours,
                           maxtime=900, maxgap=None, logfile=logfile, elevatorVertices=elevatorVertices)
                           # prefixdrawcomp='RunRA', plotboundsname=titleplot, showboundplot=True, saveboundplotname=boundplotname)
    print(f"the longest trail found is {length} meters long, visiting {len(trail)}edges\n {trail}")
    # print(f"nodeToCoordinate:{nodeToCoordinate}")
    # buildingsvisited=dict()
    # weightcheck=0
    # for edge in trail:
    #     if edge in hallways:
    #         if hallways[edge]<0:
    #             print(f"length of edge{edge} is{hallways[edge]}")
    #         weightcheck+= hallways[edge]
    #     else:
    #         if hallways[(edge[1],edge[0])]<0:
    #             print(f"length of edge{(edge[1],edge[0])} is{hallways[(edge[1],edge[0])]}")
    #         weightcheck+= hallways[(edge[1],edge[0])]
    #     if edge[0] in nodeToCoordinate:
    #         building, floor= getBuildingFloor(edge[0], nodeToCoordinate)
    #         if building in buildingsvisited:
    #             buildingsvisited[building].add(floor)
    #         else:
    #             buildingsvisited[building]={floor}
    #     else:
    #         print(f"vertex {edge[0]} is not in node to coordinate")
    #     if edge[1] in nodeToCoordinate:
    #         building, floor= getBuildingFloor(edge[1], nodeToCoordinate)
    #         if building in buildingsvisited:
    #             buildingsvisited[building].add(floor)
    #         else:
    #             buildingsvisited[building]={floor}
    #     else:
    #         print(f"vertex {edge[1]} is not in node to coordinate")
    # print(f"trail with {len(trail)} edges of sanity check:{weightcheck} meters long goes through:{buildingsvisited}")

    # # drawAllEdges(edges=trail)#, nodeToCoordinate=nodeToCoordinate, elevatorEdges=elevatorEdges, specialEdges=specialEdges, figuresResultBuildings=figuresResultBuildings, resultfolder= PATH_result, prefixfilename='TestCRZI')
    # exportGraphinfo(halls=hallways, nodeToCoordinate=nodeToCoordinate, scales=buildingScales, prefix="RA")
    drawEdgesInFloorplans(edges=trail, nodeToCoordinate=nodeToCoordinate, elevatorEdges=elevatorEdges, specialEdges=specialEdges, figuresResultBuildings=figuresResultBuildings,resultfolder=PATH_result, prefixfilename='RA')
    # todraw=[]
    # for edge in trail:
    #     if (edge[0], edge[1]) in hallways:
    #         todraw.append({"key": edge, "value": hallways[edge]})
    #     else:
    #         todraw.append({"key": edge, "value": hallways[(edge[1],edge[0])]})
    #
    # with open("trailRA1hours.json", "w") as outfile:
    #     json.dump(todraw, outfile)
    # exportGraphinfo(trail,nodeToCoordinate, buildingScales)
    # print(f"export simple figure")


    # print(f"we are about to plot the bounds")


    # plotBounds(logfolder=PATH_logs, logfile=logfile, title=titleplot, showresult=True, savename='boundsOnRA.html')


