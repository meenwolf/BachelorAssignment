# import module
from pdf2image import convert_from_path
import os
from pathlib import Path
# Get the path for the directory where my bachelor assignment files are stored

dir_path = os.path.dirname(os.path.realpath(__file__))
PATH= os.path.abspath(os.path.join(dir_path, os.pardir))

# Save the path where all the floorplans are saved
floorplanPath = PATH+"\\Gebouwplattegronden"
# Create a path where we want the JPG images to be placed:
imagePath= PATH+"\\PNGimages"

# And create the directory if it does not exist yet
if not os.path.exists(imagePath):
    os.makedirs(imagePath)

# Loop over the items in the directory
for building in os.listdir(floorplanPath):
    # Save the path where the images for this building must come
    buildingImagesPath = imagePath + "\\" + building

    if not os.path.exists(buildingImagesPath): # Check if the folder already exists
        os.makedirs(buildingImagesPath)

    # Save the path where the floorplans of this building are stored
    buildingFloorPath= floorplanPath + "\\" + building

    # List all the files in this directory to loop over them
    listOfFiles= os.listdir(buildingFloorPath)
    for file in listOfFiles:
        if file.endswith(".pdf"):#checks if it is a pdf we want to process
            # Create the filepath where this floor plan is located
            filePath= buildingFloorPath + "\\" + file
            # Convert the pdf to an image, providing the path where the poppler \bin is on my computer
            images = convert_from_path(filePath, dpi=300 ,poppler_path=r"C:\Users\meenw\OneDrive - University of Twente\Bureaublad\backup\AM\Year 3\Bachelor assignment\poppler-24.08.0\Library\bin")

            if len(images)>1:
                # We would create the path for this file, but have an extra directory for the page number in here
                # imagePathtemp= buildingImagesPath+ "\\" +file[:-4]
                # Create this directory if it does not exist yet
                # if not os.path.exists(imagePathtemp):
                #     os.makedirs(imagePathtemp)
                # Then loop over all the pages and save them individually, with
                # # for i in range(len(images)):
                #     # Save pages as images in the pdf
                #     # images[i].save(imagePathtemp+ "\\page" + str(i) + '.jpg', 'JPEG')
                print(f" EDGECASE! Check this file: {filePath} because it has  {len(images)} pages/images")

            else:
                # Create a temporary path indicating where we want the image to be stored
                imagePathtemp= buildingImagesPath+ "\\" +file[:-4]
                # Store the image as a PNG
                images[0].save(imagePathtemp+ ".png", 'PNG')

        else:
            print(f"We don't want to work with this file: {file} ")
