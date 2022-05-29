# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:20:45 2022

@author: samar
"""

import os

folder = "PlateImages"
new_folder = "Renamed_PlateImages"
for count, filename in enumerate(os.listdir(folder)):
    dst = f"Image_{str(count)}.jpg"
    src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
    dst =f"{new_folder}/{dst}"
    os.rename(src, dst)     
        # rename() function will
        # rename all the files
        

