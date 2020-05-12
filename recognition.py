#============================================
#Name: Ivan Inandan & Max Bourrie
#Professor: Gheni Abla
#Class: CS 497-7 (Intro to Deep Learning)
#Assignment: Final Project
#Project: Facial Recognition w/ Name Display
#============================================


#Import Libraries
import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep


#   Scans folder containing local database of figures encoding facial features
#   Returns dictionary of: (name, face)
def get_encoded_faces():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


#   Encode face from user file name (img)
def unknown_image_encoded(img):
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


#   Scans image to detect if any faces are present
#   If face is found, cross-references it to local database (scans detected faces)
#   Labels face in image if identical facial structure is found in local database
#   Accepts: image file path
#   Returns: list of face names
def classify_face(im):
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)

    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []

    #   Checks to see if face has known match in local database
    for face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"    # If no match is found, set label to 'unknown'

        #   Otherwise if match is found --> Assign label with most common similarities
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            #   Draws border around scanned face in image
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            #   Assigns label to border drawn around scanned face in image
            cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_ITALIC
            cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    # Display results
    while True:
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return face_names


print("\n(**NOTE** --> Format of input MUST BE '[filename].jpg')")
fileName = input("Please enter file you wish to scan: ")

print("\n[...scanning image]")
print("(process may take a few minutes --> ctrl+c to EXIT)")

print(classify_face(fileName))
