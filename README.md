# CS497FinalProject
This repository contains code pertaining to CS 497 Final Project:

TO RUN CODE ON PYTHON 3:
1. Launch command prompt and navigate to master directory
2. Enter: "python face_rec.py"
3. Program will execute and follow prompt on screen

NOTES:
- Images are scanned to match files stored within local database (stored in 'faces')
- In order to 'train' image dataset, add frontal portrait image of personal figure you wish to add as '.jpg' file into '/faces'
- The larger the local database becomes (more images stored), the slower the program takes to scan
- Program cannot scan images taken from side profile; this is bc it will not recognize the image as a face thus cannot find a match
