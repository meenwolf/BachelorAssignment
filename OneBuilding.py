from TestMoreFloors import *

def findTrailComponent(logfolder, resultfolder, edges, specialEdges, figuresResultBuildings, neighbours=None,
                       maxtime=None, maxgap=None, printtime=None, logfile=False, elevatorVertices=[],
                       prefixdrawcomp=False, plotboundsname=False, showboundplot=False, saveboundplotname=False):
    if neighbours == None:
        neighbours = getNeighbourhood(edges)
    neighboursnew, edgesnew, vdummy = addDummy(neighbours, edges)
    model, varshall, varsdegree = runModel(logfolder=logfolder, halls=edgesnew, nvdum=vdummy, neighbours=neighboursnew,
                                           maxtime=maxtime, maxgap=maxgap, printtime=printtime, log=logfile,
                                           elevatorVertices=elevatorVertices)

    lengthLongestTrail = model.getAttr('ObjVal')
    print(f"The longest trail is {lengthLongestTrail} meters long")
    used_edges = getEdgesResult(model, varshall)
    # vdummy = max(list(neighbours.keys()))
    print(f"we have {vdummy} as dummy vertex for this component")
    trailresult = constructTrail(used_edges, vdummy)
    if type(prefixdrawcomp) == str:
        drawEdgesInFloorplans(edges=trailresult, nodeToCoordinate=nodeToCoordinate, elevatorEdges=elevatorEdges,
                              specialEdges=specialEdges, figuresResultBuildings=figuresResultBuildings,
                              resultfolder=resultfolder, prefixfilename=prefixdrawcomp)
    if type(plotboundsname) == str:
        plotBounds(logfolder=logfolder, logfile=logfile, title=plotboundsname, showresult=showboundplot,
                   savename=saveboundplotname)

    return trailresult

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


    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    datenew = date.replace(':', '-')
    logfile = "\\log" + datenew + ".log"

    titleplot = "The bounds on the length of the longest trail on Carré floor 1,2,3 and 4 together,<br> at each moment in time when running the gurobi solver"
    boundplotname= f'{datenew}.svg'
    findTrailComponent(logfolder=PATH_log, resultfolder=PATH_result, edges=hallways, specialEdges=specialEdges, figuresResultBuildings=figuresResultBuildings, neighbours=neighbours, maxtime=60, maxgap=None, printtime=5,logfile=logfile, elevatorVertices=elevatorVertices, prefixdrawcomp='testing1building', plotboundsname=titleplot, showboundplot=True, saveboundplotname=boundplotname)