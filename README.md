# BachelorAssignment
In this read me I will explain the most important different files in this repo.

## PDFtoPNGConverter.py
I have received the floorplans of the relevant buildings of the UT as PDFs. I will draw
a path over the floorplans in inkscape. Loading the PDF's in there, makes the entire floorplan
a vector image, which causes some crashes of inkscape. To prevent these crashes, and accicental
changes in the floorplan, I will convert the PDF's to PNG's and load them as an image in inkscape.
The code in this file will mirror the structure and naming convention of the floorplans as to how I received
them. 

## TrialExtractSVG.py
To test extracting information from the vector images, which will be saved in another directory at my local
computer, I have made this file.

The files that work on all the buildings, are:
## allFunctions.py
This file contains all the functions that I have made to get to the final result

## TryAllBuildings.py
This file contains the code that I used to get the final results.

All the other files were in between files, to extend the code for working for one floor, to one building,
To a few buildings, to all buildings. The allFunctions.py file contains all the functions that are created
here, but adjusted so that they work for all the buildings.

## Data and Results
In this folder I uploaded all the log files and output json files that were needed/used to get the results in my thesis/poster/pitch. The floor_position file is a file where for each building, for each floor, you can provide an x,y offset so that it places it correctly in 3D space. With the angel, the rotation around the z-axis is meant. 
For each of the results, I have created a folder where those files are stored in. I also included the visualizer tool that I used. When uploading the correct files, one can click on load to load them. With right mouse click and drag, one can change the angle of viewpoint. When also pressing ctrl, right click and drag, one can shift the viewpoint perspective and move around/ in the buildings. In the log screen under the file selection area, one can see what happens in the viewer, and the total length of the uploaded trail.

In the tab ``Render'' one can switch between light and dark mode, have the option to show or hide the xyz-axes and the grid. One can also enable or disable anti-aliasing for smoother rendering on your screen. One can adjust the line thickness, and see an animation of a little circle moving over the trail. One can also adjust it's speed.

In the tab ``Debug'' one can enter a vertex number and adjust the size of the marker that will be placed on this vertex. One can read the information, in which building and on which floor this vertex is located, and see it's neighbours. When clicking on a neighbour, the marker will be moved there, and you see it's info. One can manually walk through the graph this way.

In the tab ``Help'' a short overview of the operation keys is given.
With F11, you go full screen
With tab you can show and hide the input window.
With ctrl drag one can shift the viewpoint 
With drag one can rotate the viewpoint
With the mouse wheel one can zoom in and out.

This was a very usefull visualizer tool and all credits for making it go to my boyfriend Pieter de Goeje.
A 30 second video of the 3D result using this tool for the longest found trail through all the buildings is uploaded in this folder.
When a new version of the longest trail viewer, or other data comes available, I will update this readme, and include a newer version of the longest trail viewer too.

