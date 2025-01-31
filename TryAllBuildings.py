import os

from allFunctions import *

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

    # # Measured, but made up for now, bridge lengths:
    bridgeLengths = {"C01CRWA": 12.5, "C01WACR": 12.5, "C01CRNL": 13, "C01NLCR": 13, "C01CRWH": 10, "C01WHCR": 10,
                     "C01CIRA": 8, "C01RACI": 8,
                     "C01ZHNL": 16.5, "C01NLZH": 16.5, "C01RAZI": 13, "C01ZIRA": 13, }

    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    datenew = date.replace(':', '-')
    logfile = "\\log" + datenew + ".log"


# For a longest trail though RA floor 1 and 2, for the poster:
    # figuresResultBuildings, buildingScales, nodeToCoordinate, specialPaths, specialEdges, hallways, elevatorVertices, elevatorEdges, vdummy, neighbours = getGraph(PATH_drawings=PATH_drawings,PATH_empty=PATH_empty, bridgeLengths=bridgeLengths, buildingsToSkip=["HORST","CITADEL","ZILVER","CARRE","WAAIER","NANO"], floorsToSkip=['0','3','4','5','6'])
    # titleplot = "The bounds on the length of a longest trail in Ravelijn floor 1 and 2,<br> at each moment in time when running the gurobi solver for 15 seconds."
    # boundplotname = f'RA{datenew}.html'
    # trail, length= reduceGraphBridges(specialPaths=specialPaths, logfolder=PATH_logs, resultfolder=PATH_result, edges=hallways, specialEdges=specialEdges,
    #                    figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, neighbours=neighbours,
    #                        maxtime=10, maxgap=None, logfile=logfile, elevatorVertices=elevatorVertices)
    #                        # prefixdrawcomp='RunRA', plotboundsname=titleplot, showboundplot=True, saveboundplotname=boundplotname)
    #
    # exportGraphinfo(Path=PATH, halls=hallways, nodeToCoordinate=nodeToCoordinate, scales=buildingScales, trail=trail,
    #                 prefix="RA1and2")
    #
    # drawEdgesInFloorplans(edges=trail, nodeToCoordinate=nodeToCoordinate, elevatorEdges=elevatorEdges,
    #                       specialEdges=specialEdges, figuresResultBuildings=figuresResultBuildings,
    #                       resultfolder=PATH_result, prefixfilename='RA15sec')
    #
    # plotBounds(logfolder=PATH_logs, logfile=logfile, title=titleplot, savename=boundplotname)


# For the longest trail through CI to CR 90 and testing 270 seconds per component. They gave the same result
    # figuresResultBuildings, buildingScales, nodeToCoordinate, specialPaths, specialEdges, hallways, elevatorVertices, elevatorEdges, vdummy, neighbours = getGraph(PATH_drawings=PATH_drawings,PATH_empty=PATH_empty, bridgeLengths=bridgeLengths, buildingsToSkip=["HORST","WAAIER","NANO"], floorsToSkip=[])
    # trailsComponents=dict() # to save the longest found trail for a component in
    # trail, length, trailsComponents= reduceGraphBridges(trailsComponents=trailsComponents ,specialPaths=specialPaths, logfolder=PATH_logs, resultfolder=PATH_result, edges=hallways, specialEdges=specialEdges,
    #                    figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, neighbours=neighbours,
    #                        maxtime=270, maxgap=None, logfile=logfile, elevatorVertices=elevatorVertices)
    #                        # prefixdrawcomp='RunRA', plotboundsname=titleplot, showboundplot=True, saveboundplotname=boundplotname)    #
    # exportGraphinfo(Path=PATH, halls=hallways, nodeToCoordinate=nodeToCoordinate, scales=buildingScales, trail=trail,
    #                 prefix="CItoCR270sec")


# Extract information back from the json for trail CI to CR
#     "C:\Users\meenw\OneDrive - University of Twente\Bureaublad\backup\AM\Year 3\Bachelor assignment\TestMoreBuildings\Logs\
#     "C:\Users\meenw\OneDrive - University of Twente\Bureaublad\backup\AM\Year 3\Bachelor assignment\
#     PATHtrail = PATH_logs + "\\componentscheck\\2025-01-26 21-11-26\\HO [729, 931] and 680 edges.json"
#     PATH_lengths= PATH+"dataruns\\log2025-01-26 21-33-49.log\\try90sec3cutweigthedEdges.json"
#     # PATH_lengths= os.path.join(PATH,"CItoCRtestrunweigthedEdges.json")
#     with open(PATH_lengths, 'r') as f:
#         halls = json.load(f)
#     hallways=dict()
#     for hall in halls:
#         hallways[(hall['key'][0],hall['key'][1])]= hall['value']
#
#     with open(PATHtrail, 'r') as f:
#         data = json.load(f)
#     comp = dict()
#     length = 0
#     print(data)
#     for hall in data:
#         edge=(hall['key'][0], hall['key'][1])
#         edgerev= (hall['key'][1], hall['key'][0])
#         # comp.append((hall['key'][0], hall['key'][1]))
#         if edge in hallways:
#             length = hallways[edge]
#             comp[edge]=length
#         elif edgerev in hallways:
#             length = hallways[edgerev]
#             comp[edgerev]=length
#         else:
#             print(f"LENGTH OF EDGE:{edge} unknown, since it is not in the hallways dict")
#
#     print(f"the component is {length} meters long, visiting {len(comp)}edges\n {comp}")
#     neighbours=getNeighbourhood(comp)
#     vdummy=max(neighbours.keys())+1
#     neighboursnew, edgesnew,vdummy = addDummy(neighbours, comp, vdummy)
#     model, varshall, varsdegree = runModelends(auxedge=auxedge, ends=ends, logfolder=logfolder, halls=edgesnew,
#                                                nvdum=vdummy,
#                                                neighbours=neighboursnew,
#                                                maxtime=maxtime, maxgap=maxgap, printtime=printtime, log=logfile,
#                                                elevatorVertices=elevatorVertices)
#     # lengthLongestTrail = model.getAttr('ObjVal')
#     # print(f"The longest trail is {lengthLongestTrail} meters long")
#     try:
#         used_edges = getEdgesResult(model, varshall)
#     except:
#         todraw = []
#         for edge in edges:
#             todraw.append({"key": edge, "value": edges[edge]})
#         date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         datenew = date.replace(':', '-')
#         pathcases = logfolder + f"\\componentscheck\\{datenew}"
#         if not os.path.exists(pathcases):
#             os.makedirs(pathcases)
#
#         prefix = ""
#         for building in buildingsvisited.keys():
#             if "NANO" in building:
#                 prefix += "NL "
#             elif "CARRE" in building:
#                 prefix += "CR "
#             else:
#                 prefix += f"{building[:2]} "
#         prefix += f"{ends} and {len(edges)} edges"
#
#         with open(os.path.join(pathcases, prefix + ".json"), "w") as outfile:
#             json.dump(todraw, outfile)
#         return [], 0
#
#     # drawEdgesInFloorplans(edges=[edge for edge in used_edges if vdummy not in edge], nodeToCoordinate=nodeToCoordinate, elevatorEdges=elevatorEdges,
#     #                       specialEdges=specialEdges, figuresResultBuildings=figuresResultBuildings,
#     #                       resultfolder=PATH_result, prefixfilename='edgesResult')
#     # print(f"{len(getReachable(getNeighbourhood(used_edges),vdummy))} vertices of the {len(getNeighbourhood(used_edges))} vertices are reachable from vdum{vdummy}")
#     # print(f"we have {vdummy} as dummy vertex for this component")
#     # print(f"and the {len(used_edges)} used edges are:{used_edges}")
#     trailresult = constructTrailCheckComponents(used_edges, vdummy)
#     lengthtrail = 0
#     for edge in trailresult:
#         if edge in edgesnew:
#             lengthtrail += edgesnew[edge]
#         else:
#             lengthtrail += edgesnew[(edge[1], edge[0])]


# Bounds on Ravelijn entire building running for the night
    # figuresResultBuildings, buildingScales, nodeToCoordinate, specialPaths, specialEdges, hallways, elevatorVertices, elevatorEdges, vdummy, neighbours = getGraph(PATH_drawings=PATH_drawings,PATH_empty=PATH_empty, bridgeLengths=bridgeLengths, buildingsToSkip=["HORST","CITADEL","ZILVER","CARRE","WAAIER","NANO"], floorsToSkip=[])
    # titleplot = "The bounds on the length of a longest trail in Ravelijn,<br> at each moment in time when running the gurobi solver for 8 hours."
    # boundplotname = f'RAnight{datenew}.html'
    # trail, length= reduceGraphBridges(specialPaths=specialPaths, logfolder=PATH_logs, resultfolder=PATH_result, edges=hallways, specialEdges=specialEdges,
    #                    figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, neighbours=neighbours,
    #                        maxtime=28800, maxgap=None, logfile=logfile, elevatorVertices=elevatorVertices)
    #                        # prefixdrawcomp='RunRA', plotboundsname=titleplot, showboundplot=True, saveboundplotname=boundplotname)
    # plotBounds(logfolder=PATH_logs, logfile=logfile, title=titleplot, savename=boundplotname)


# Code to check what the total length of all the hallways in CI, RA, ZI and CR is:
    # figuresResultBuildings, buildingScales, nodeToCoordinate, specialPaths, specialEdges, hallways, elevatorVertices, elevatorEdges, vdummy, neighbours = getGraph(PATH_drawings=PATH_drawings,PATH_empty=PATH_empty, bridgeLengths=bridgeLengths, buildingsToSkip=["NANO","WAAIER","HORST"], floorsToSkip=[])
    # totalLength= sum(hallways.values())
    # print(f"total length of hallways in CI, RA, ZI, CR: {totalLength} meters")


# Code to check position of buildings
    custom3cut, figuresResultBuildings, buildingScales, nodeToCoordinate, specialPaths, specialEdges, hallways, elevatorVertices, elevatorEdges, vdummy, neighbours = getGraph(PATH_drawings=PATH_drawings,PATH_empty=PATH_empty, bridgeLengths=bridgeLengths, buildingsToSkip=[], floorsToSkip=[])
    existingvertices = set()
    for v0, v1 in hallways:
        existingvertices.add(v0)
        existingvertices.add(v1)
    buildingsvisited = getBuildings(existingvertices, nodeToCoordinate)

    print(f"We have {len(list(existingvertices))} nodes, in: {buildingsvisited}")
    # exportGraphinfo(Path=PATH, halls=hallways, nodeToCoordinate=nodeToCoordinate, scales=buildingScales, trail=[],
    #                 prefix="all30Ssec")
    print(f"we have {vdummy} vertices, and get reachable for vertex 0:{len(getReachable(neighbours, 0))}")
    # reachablefromzero= getReachable(neighbours, 0)
    # notReachable=[key for key in neighbours.keys() if key not in reachablefromzero]
    # edgesDisconnected= getEdgesComponent(hallways, notReachable)
    #
    # for vertex in notReachable:
    #     print(f"vertex:{vertex}, with neighbours:{neighbours[vertex]} are not reachable from zero")
    #
    #     for edge,info in specialEdges.items():
    #         if vertex in edge:
    #             print(f"edge {edge} is {info}")
    #             break
    #
    #     if vertex in nodeToCoordinate:
    #         print(f"nodeToCoordinate: {nodeToCoordinate[vertex]}")
    #     elif vertex in elevatorVertices.keys():
    #         print(f"its an elevator vertex:{[key for key,value in elevatorVertices.items() if value == vertex]}")
    #     else:
    #         print(f"this vertex {vertex} is not an elevator vertex{elevatorVertices} nor a regular vertex")
    #
    # print(f"elevator edges:{elevatorEdges}\n elevator vertices:{elevatorVertices}")
    # print(f"SpecialPaths:{specialPaths}")
    # velev= elevatorVertices["HCE01"]
    # connectedelev=[edge for edge in hallways if velev in edge]
    # vbasement=[]
    # for vertex in neighbours.keys():
    #     if vertex in nodeToCoordinate:
    #         building, floor = getBuildingFloor(vertex, nodeToCoordinate)
    #         if "HORST" in building:
    #             if floor =="0":
    #                 vbasement.append(vertex)
    # print(f"vertices basement:{vbasement}")
    # edgesbasement=getEdgesComponent(hallways, vbasement)
    # print(f"edges basement:{edgesbasement}")
    # exportGraphinfo(Path=PATH, halls=hallways, nodeToCoordinate=nodeToCoordinate, scales=buildingScales,
    #                 trail=edgesbasement, prefix="basementHC")
    # for edge in connectedelev:
    #     if edge[0] == velev:
    #         vertex=edge[1]
    #     else:
    #         vertex=edge[0]
    #     if vertex in nodeToCoordinate:
    #         print(f"vertex:{vertex}, with info:{nodeToCoordinate[vertex]}")
    #     else:
    #         print(f"vertex:{vertex}, not in nodetocoordinate")
    exportGraphinfo(Path=PATH, halls=hallways, nodeToCoordinate=nodeToCoordinate, scales=buildingScales, trail=[],
                    prefix="try300sec3cut")
    print(f"custom3cut:{custom3cut}")
    print(f"total sum of all the edges: {sum(hallways.values())}")
    trailsComponents=dict() # to save the longest found trail for a component in
    trail, length, trailsComponents= reduceGraphBridges(custom3cut=custom3cut, trailsComponents=trailsComponents ,specialPaths=specialPaths, logfolder=PATH_logs, resultfolder=PATH_result, edges=hallways, specialEdges=specialEdges,
                       figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, neighbours=neighbours,
                           maxtime=300, maxgap=5, logfile=logfile, elevatorVertices=elevatorVertices)
                           # prefixdrawcomp='RunRA', plotboundsname=titleplot, showboundplot=True, saveboundplotname=boundplotname)
    trailresult=  constructTrailCheckComponents(trail, vdummy)
    print(f"the longest trail found is {length} meters long, visiting {len(trail)}edges\n {trailresult}")

    exportGraphinfo(Path=PATH, halls=hallways, nodeToCoordinate=nodeToCoordinate, scales=buildingScales, trail=trailresult,
                    prefix="try300sec3cut")
    visitedvertices = set()
    for v0, v1 in trail:
        visitedvertices.add(v0)
        visitedvertices.add(v1)
    buildingsvisited = getBuildings(visitedvertices, nodeToCoordinate)
    print(f"and goes through: {buildingsvisited}")


    # print(f"save the edges of the {len(list(trailsComponents.keys()))} components")
    # for key, value in trailsComponents.items():
    #     visited = []
    #     currentBuilding = ''
    #     trail= constructTrailCheckComponents(value['trail'], vdummy)
    #     if type(key[-1])== tuple: #meaning no ends for this component
    #         buildingsvisited= getBuildings(getNeighbourhood(list(key)), nodeToCoordinate)
    #         print(f"for the component of {len(key)} edges in buildings: {buildingsvisited}\n with no mandatory ends")
    #         endvertices=""
    #         edges=key
    #     elif type(key[-2])== tuple: # we have one mandatory end
    #         buildingsvisited = getBuildings(getNeighbourhood(list(key[:-1])), nodeToCoordinate)
    #         print(f"for the component of {len(key)-1} edges in buildings: {buildingsvisited}\n with 1 mandatory end {key[-1]}")
    #         endvertices=f"{key[-1]}"
    #         edges=key[:-1]
    #     else: # we have 2 mandatory ends
    #         buildingsvisited = getBuildings(getNeighbourhood(list(key[:-2])), nodeToCoordinate)
    #         print(f"for the component of {len(key)-2} edges in buildings: {buildingsvisited}\n with 2 mandatory ends: {key[-2:]}")
    #         endvertices=f"{key[-2]} {key[-1]}"
    #         edges=key[:-2]
    #
    #     todraw=[]
    #     for edge in edges:
    #         if edge in hallways:
    #             todraw.append({"key": edge, "value": hallways[edge]})
    #         elif (edge[1], edge[0]) in hallways:
    #             todraw.append({"key": edge, "value": hallways[(edge[1], edge[0])]})
    #
    #
    #     pathcases=PATH_logs+f"\\componentscheck\\{datenew}"
    #     if not os.path.exists(pathcases):
    #         os.makedirs(pathcases)
    #
    #     prefix=""
    #     for building in buildingsvisited.keys():
    #         if "NANO" in building:
    #             prefix+="NL "
    #         elif "CARRE" in building:
    #             prefix+="CR "
    #         else:
    #             prefix+=f"{building[:2]} "
    #     prefix+=f"{endvertices} and {len(edges)} edges"
    #
    #     with open(os.path.join(pathcases,  prefix +".json"), "w") as outfile:
    #         json.dump(todraw, outfile)
        #     if edge[0] in nodeToCoordinate:
        #         building, floor= getBuildingFloor(edge[0],nodeToCoordinate)
        #         if building != currentBuilding:
        #             visited.append(building)
        #             currentBuilding=building
        #     if edge[1] in nodeToCoordinate:
        #         building, floor= getBuildingFloor(edge[1],nodeToCoordinate)
        #         if building != currentBuilding:
        #             visited.append(building)
        #             currentBuilding=building
        # print(f"the {value['length']} meter long trail visits these buildings in order: \n{visited}")

# Code for sanity check on new cases

    #     existingvertices = set()
    #     for v0, v1 in hallways:
    #         existingvertices.add(v0)
    #         existingvertices.add(v1)
    #     buildingsvisited = getBuildings(existingvertices, nodeToCoordinate)
    #
    #     print(f"We have {len(list(existingvertices))} nodes, in: {buildingsvisited}")

    # visitedvertices=set()
    # for v0,v1 in trail:
    #     visitedvertices.add(v0)
    #     visitedvertices.add(v1)
    # buildingsvisited= getBuildings(visitedvertices, nodeToCoordinate)
    # print(f"and goes through: {buildingsvisited}")




