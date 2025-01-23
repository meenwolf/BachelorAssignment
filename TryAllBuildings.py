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

    # Measured, but made up for now, bridge lengths:
    bridgeLengths = {"C01CRWA": 12.5, "C01WACR": 12.5, "C01CRNL": 13, "C01NLCR": 13, "C01CRWH": 10, "C01WHCR": 10,
                     "C01CIRA": 8, "C01RACI": 8,
                     "C01ZHNL": 16.5, "C01NLZH": 16.5, "C01RAZI": 13, "C01ZIRA": 13, }

    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    datenew = date.replace(':', '-')
    logfile = "\\log" + datenew + ".log"

    ### For a longest trail though RA floor 1 and 2, for the poster:
    # figuresResultBuildings, buildingScales, nodeToCoordinate, specialPaths, specialEdges, hallways, elevatorVertices, elevatorEdges, vdummy, neighbours = getGraph(PATH_drawings=PATH_drawings,PATH_empty=PATH_empty, bridgeLengths=bridgeLengths, buildingsToSkip=["HORST","CITADEL","ZILVER","CARRE","WAAIER","NANO"], floorsToSkip=['0','3','4','5','6'])
    # titleplot = "The bounds on the length of a longest trail in Ravelijn floor 1 and 2,<br> at each moment in time when running the gurobi solver for 15 seconds."
    # #
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
    figuresResultBuildings, buildingScales, nodeToCoordinate, specialPaths, specialEdges, hallways, elevatorVertices, elevatorEdges, vdummy, neighbours = getGraph(PATH_drawings=PATH_drawings,PATH_empty=PATH_empty, bridgeLengths=bridgeLengths, buildingsToSkip=["HORST","WAAIER","NANO"], floorsToSkip=[])

    existingvertices = set()
    for v0, v1 in hallways:
        existingvertices.add(v0)
        existingvertices.add(v1)
    buildingsvisited = getBuildings(existingvertices, nodeToCoordinate)

    print(f"We have {len(list(existingvertices))} nodes, in: {buildingsvisited}")
    trailsComponents=dict() # to save the longest found trail for a component in
    trail, length, trailsComponents= reduceGraphBridges(trailsComponents=trailsComponents ,specialPaths=specialPaths, logfolder=PATH_logs, resultfolder=PATH_result, edges=hallways, specialEdges=specialEdges,
                       figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, neighbours=neighbours,
                           maxtime=270, maxgap=None, logfile=logfile, elevatorVertices=elevatorVertices)
                           # prefixdrawcomp='RunRA', plotboundsname=titleplot, showboundplot=True, saveboundplotname=boundplotname)
    print(f"the longest trail found is {length} meters long, visiting {len(trail)}edges\n {trail}")

    exportGraphinfo(Path=PATH, halls=hallways, nodeToCoordinate=nodeToCoordinate, scales=buildingScales, trail=trail,
                    prefix="CItoCR270sec")
    visitedvertices = set()
    for v0, v1 in trail:
        visitedvertices.add(v0)
        visitedvertices.add(v1)
    buildingsvisited = getBuildings(visitedvertices, nodeToCoordinate)
    print(f"and goes through: {buildingsvisited}")
    # Extract information back from the json for trail CI to CR
    # PATHd = PATH + "\\dataruns\\log2025-01-22 17-41-44.log"
    # PATH_data= os.path.join(PATHd,"CItoCRtestruntrail.json")
    # PATH_lengths= os.path.join(PATHd,"CItoCRtestrunweigthedEdges.json")
    # with open(PATH_lengths, 'r') as f:
    #     halls = json.load(f)
    # hallways=dict()
    # for hall in halls:
    #     hallways[(hall['key'][0],hall['key'][1])]= hall['value']
    #
    # with open(PATH_data, 'r') as f:
    #     data = json.load(f)
    # trail = []
    # length = 0
    # print(data)
    # for hall in data:
    #     edge=(hall['key'][0], hall['key'][1])
    #     edgerev= (hall['key'][1], hall['key'][0])
    #     trail.append((hall['key'][0], hall['key'][1]))
    #     if edge in hallways:
    #         length += hallways[(hall['key'][0], hall['key'][1])]
    #     elif edgerev in hallways:
    #         length += hallways[edgerev]
    #     else:
    #         print(f"LENGTH OF EDGE:{edge} unknown, since it is not in the hallways dict")

    # print(f"the longest trail found is {length} meters long, visiting {len(trail)}edges\n {trail}")


    # Bounds on Ravelijn entire building running for the night
    # figuresResultBuildings, buildingScales, nodeToCoordinate, specialPaths, specialEdges, hallways, elevatorVertices, elevatorEdges, vdummy, neighbours = getGraph(PATH_drawings=PATH_drawings,PATH_empty=PATH_empty, bridgeLengths=bridgeLengths, buildingsToSkip=["HORST","CITADEL","ZILVER","CARRE","WAAIER","NANO"], floorsToSkip=[])
    #
    # titleplot = "The bounds on the length of a longest trail in Ravelijn,<br> at each moment in time when running the gurobi solver for 8 hours."
    # boundplotname = f'RAnight{datenew}.html'
    #
    # trail, length= reduceGraphBridges(specialPaths=specialPaths, logfolder=PATH_logs, resultfolder=PATH_result, edges=hallways, specialEdges=specialEdges,
    #                    figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, neighbours=neighbours,
    #                        maxtime=28800, maxgap=None, logfile=logfile, elevatorVertices=elevatorVertices)
    #                        # prefixdrawcomp='RunRA', plotboundsname=titleplot, showboundplot=True, saveboundplotname=boundplotname)
    #
    # plotBounds(logfolder=PATH_logs, logfile=logfile, title=titleplot, savename=boundplotname)

    # Code to check what the total length of all the hallways in CI, RA, ZI and CR is:
    # figuresResultBuildings, buildingScales, nodeToCoordinate, specialPaths, specialEdges, hallways, elevatorVertices, elevatorEdges, vdummy, neighbours = getGraph(PATH_drawings=PATH_drawings,PATH_empty=PATH_empty, bridgeLengths=bridgeLengths, buildingsToSkip=["NANO","WAAIER","HORST"], floorsToSkip=[])
    # totalLength= sum(hallways.values())
    # print(f"total length of hallways in CI, RA, ZI, CR: {totalLength} meters")

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




