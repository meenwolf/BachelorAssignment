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
    bridgeLengths = {"C01CRWA": 10, "C01WACR": 10, "C01CRNL": 10, "C01NLCR": 10, "C01CRWH": 10, "C01WHCR": 10,
                     "C01CIRA": 10, "C01RACI": 10,
                     "C01ZHNL": 10, "C01NLZH": 10, "C01RAZI": 10, "C01ZIRA": 10, }

    figuresResultBuildings, buildingScales, nodeToCoordinate, specialPaths, specialEdges, hallways, elevatorVertices, elevatorEdges, vdummy, neighbours = getGraph(PATH_drawings=PATH_drawings,PATH_empty=PATH_empty, bridgeLengths=bridgeLengths, buildingsToSkip=["HORST"], floorsToSkip=[])
    existingvertices = set()
    for v0, v1 in hallways:
        existingvertices.add(v0)
        existingvertices.add(v1)
    buildingsvisited = getBuildings(existingvertices, nodeToCoordinate)
    print(f"We have {len(list(existingvertices))} nodes, in: {buildingsvisited}")
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    datenew = date.replace(':', '-')
    logfile = "\\log" + datenew + ".log"

    # titleplot = "The bounds on the length of a longest trail in Ravelijn floor 1 and 2,<br> at each moment in time when running the gurobi solver for 1 minute."
    #
    # boundplotname = f'RA{datenew}.html'
    trail, length= reduceGraphBridges(specialPaths=specialPaths, logfolder=PATH_logs, resultfolder=PATH_result, edges=hallways, specialEdges=specialEdges,
                       figuresResultBuildings=figuresResultBuildings,elevatorEdges=elevatorEdges ,nodeToCoordinate=nodeToCoordinate, vdummy=vdummy, neighbours=neighbours,
                           maxtime=10, maxgap=None, logfile=logfile, elevatorVertices=elevatorVertices)
                           # prefixdrawcomp='RunRA', plotboundsname=titleplot, showboundplot=True, saveboundplotname=boundplotname)
    print(f"the longest trail found is {length} meters long, visiting {len(trail)}edges\n {trail}")
    visitedvertices=set()
    for v0,v1 in trail:
        visitedvertices.add(v0)
        visitedvertices.add(v1)
    buildingsvisited= getBuildings(visitedvertices, nodeToCoordinate)
    print(f"and goes through: {buildingsvisited}")
    exportGraphinfo(halls=hallways, nodeToCoordinate=nodeToCoordinate, scales=buildingScales, trail=trail,prefix="minushorst")


    # drawEdgesInFloorplans(edges=trail, nodeToCoordinate=nodeToCoordinate, elevatorEdges=elevatorEdges, specialEdges=specialEdges, figuresResultBuildings=figuresResultBuildings,resultfolder=PATH_result, prefixfilename='RAshort')


    # plotBounds(logfolder=PATH_logs, logfile='\\log2025-01-20 20-51-11.log', title=titleplot, savename=boundplotname)

