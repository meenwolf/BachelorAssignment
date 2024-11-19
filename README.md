# BachelorAssignment
In this read me I will explain the different files in this repo.

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