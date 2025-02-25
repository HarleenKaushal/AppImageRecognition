# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:58:01 2025

@author: Harleen
"""

import cv2
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.drawing.image import Image
from PIL import Image as PILImage
import xlwings as xw
from PIL import ImageGrab
import os
import time
import streamlit as st

# Define file paths
EXCEL_FILE = "img1.xlsm"
IMAGE_FOLDER = "extracted_images"
MATCH_RESULTS_PATH = "match_results.xlsx"

# Streamlit UI
st.title("🔍 Image Matching App")
st.write("Upload an image to find the best match from stored data.")

uploaded_file = st.file_uploader("Upload Target Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    
    file_path = os.path.join(IMAGE_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    target_img = PILImage.open(file_path)
    target_img = np.array(target_img)
    st.image(target_img, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Find Best Matches"):
        # Load Excel File and Extract Images
        app = xw.App(visible=False)
        wb = xw.Book(EXCEL_FILE)
        sheet = wb.sheets.active
        image_shapes = [(shape, shape.top) for shape in sheet.shapes if shape.api.Type == 13]
        image_shapes.sort(key=lambda x: x[1])

        for index, (shape, _) in enumerate(image_shapes, start=0):
            shape.api.Copy()
            time.sleep(1)
            img = ImageGrab.grabclipboard()
            if img:
                img_path = os.path.join(IMAGE_FOLDER, f"image_{index}.png")
                img.save(img_path, "PNG")

        wb.close()
        app.quit()

        # Match Target Image with Extracted Images
        df = pd.read_excel(EXCEL_FILE)
        sift = cv2.SIFT_create()
        target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        target_kp, target_des = sift.detectAndCompute(target_img_gray, None)
        match_scores = {}

        for img_file in os.listdir(IMAGE_FOLDER):
            img_path = os.path.join(IMAGE_FOLDER, img_file)
            img_color = cv2.imread(img_path)
            if img_color is None:
                continue
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(img_gray, None)
            if des is None or target_des is None:
                continue
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(target_des, des)
            match_scores[img_file] = len(matches)

        top_matches = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        match_results = []

        for rank, (img_name, match_count) in enumerate(top_matches, start=1):
            matched_row = int(img_name.split("_")[1])
            if 0 <= matched_row < len(df):
                matched_data = df.iloc[matched_row]
                match_results.append({"Rank": rank, "Image Name": img_name, "Good Matches": match_count, **matched_data.to_dict()})

        if match_results:
            match_df = pd.DataFrame(match_results)
            st.write("### 🎯 Top Matches Found:")
            st.dataframe(match_df)
            match_df.to_excel(MATCH_RESULTS_PATH, index=False)
            with open(MATCH_RESULTS_PATH, "rb") as f:
                st.download_button("Download Match Results (Excel)", f, file_name="match_results.xlsx")
        else:
            st.write("⚠️ No matches found.")
