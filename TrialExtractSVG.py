import os
from svgpathtools import svg2paths, svg2paths2, wsvg
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH= os.path.abspath(os.path.join(dir_path, os.pardir))
print(f"the path is {PATH}")
PATH_drawings= PATH+"\\FloorplanAnnotated"
print(f"the path to the drawings is {PATH_drawings}")
paths, attributes, svg_attributes = svg2paths2(PATH_drawings+"\\CARRE 1412\\withPNG1412.3.svg", return_svg_attributes=True)
print(f"paths: {paths} \n attributes: {attributes} \n svg attributes{svg_attributes}\n")
for path in paths:
    print(f"the id is: {path}")

