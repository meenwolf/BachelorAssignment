import os
from svgpathtools import svg2paths, svg2paths2, wsvg
from pprint import pprint
# Get the path to the drawings
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH= os.path.abspath(os.path.join(dir_path, os.pardir))
PATH_drawings= PATH+"\\FloorplanAnnotated"
#Get the paths and attributes of the third floor of Carré

#IDEA: make a list of nodes, for all of the path starts and ends. After we also know the scale and made the
#Edges have the correct real world weights, we can start at merging nodes in one, if they are closer than X meters in real life.

paths, attributes = svg2paths(PATH_drawings+"\\CARRE 1412\\withPNG1412.3.svg")

for path, attr in zip(paths, attributes):
    if "inkscape:label" in attr.keys():
        if attr['inkscape:label'] in {"CR31Mtr", "CR31mtr"}:
            lengthForOneMeter= abs(path.start - path.end)
            break

print(f"length for one meter is {lengthForOneMeter}")

for path, attr in zip(paths, attributes):
    if "inkscape:label" in attr.keys():
        print(f"inkscape:label is in attributes: {attr['inkscape:label']}")
        print(f"and the path is: {path}")
        start= path.start
        end= path.end
        edgeweight= abs(start-end)
        print(f"edge has length: {edgeweight}")
