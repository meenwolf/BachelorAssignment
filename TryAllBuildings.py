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
    def getGraph(bridgeLengths, buildingsToSkip, floorsToSkip):
        # Initiate some data structures to save information in
        nodeToCoordinate={}
        coordinateToNode={}

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

            buildingEmpty= PATH_empty+f"\\{building}"
            listOfFiles = os.listdir(buildingEmpty)
            for file in listOfFiles:
                if file.endswith(".svg"):
                    floor= file.split('.')[1]
                    if floor in floorsToSkip:
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
        return figuresResultBuildings, buildingScales, nodeToCoordinate, specialPaths, specialEdges, hallways, elevatorVertices, elevatorEdges, vdummy, neighbours


    # Measured, but made up for now, bridge lengths:
    bridgeLengths = {"C01CRWA": 10, "C01WACR": 10, "C01CRNL": 10, "C01NLCR": 10, "C01CRWH": 10, "C01WHCR": 10,
                     "C01CIRA": 10, "C01RACI": 10,
                     "C01ZHNL": 10, "C01NLZH": 10, "C01RAZI": 10, "C01ZIRA": 10, }

    figuresResultBuildings, buildingScales, nodeToCoordinate, specialPaths, specialEdges, hallways, elevatorVertices, elevatorEdges, vdummy, neighbours = getGraph(bridgeLengths=bridgeLengths, buildingsToSkip=["HORST"], floorsToSkip=[])

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
    # exportGraphinfo(halls=hallways, nodeToCoordinate=nodeToCoordinate, scales=buildingScales, trail=trail,prefix="RA")


    # drawEdgesInFloorplans(edges=trail, nodeToCoordinate=nodeToCoordinate, elevatorEdges=elevatorEdges, specialEdges=specialEdges, figuresResultBuildings=figuresResultBuildings,resultfolder=PATH_result, prefixfilename='RAshort')


    # plotBounds(logfolder=PATH_logs, logfile='\\log2025-01-20 20-51-11.log', title=titleplot, savename=boundplotname)

