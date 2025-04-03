#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 14:35:08 2025

@author: justine
"""

import numpy as np
import cv2
import os
import glob

Path = "/Volumes/Expansion/Reseau/Tests configurations cellules"
files = glob.glob(Path+"/*.MOV")

fps = 0.2

def getFrame(vidcap0, sec, path, count):
    vidcap0.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)  # position en secondes
    hasFrames, image = vidcap0.read()
    if hasFrames:
        cv2.imwrite(os.path.join(path, f"image{count}.tif"), image)
    return hasFrames

for file in files:
    vidcap0 = cv2.VideoCapture(file)
    if not vidcap0.isOpened():
        print(f"Erreur lors de l'ouverture du fichier {file}")
        continue

    savePath = file[:-4] + f"_{fps}fps/"
    print(f"Enregistrement dans : {savePath}")
    
    try:
        os.mkdir(savePath)
    except FileExistsError:
        print(f"Le dossier {savePath} existe déjà.")
    except Exception as e:
        print(f"Erreur lors de la création du dossier {savePath}: {e}")
        continue

    sec = 0
    frameRate = 1 / fps
    count = 0
    success = getFrame(vidcap0, sec, savePath, count)

    while success:
        sec += frameRate
        count += 1
        success = getFrame(vidcap0, sec, savePath, count)
