# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:37:57 2025

@author: Harleen
"""

import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import openpyxl
import xlwings as xw
import time
from PIL import Image, ImageGrab
import matplotlib.pyplot as plt

# **Hardcoded Paths**
EXCEL_FILE = "C:\\Users\\Harleen\\Downloads\\New Folder\\img1.xlsx"
IMAGE_FOLDER = "C:\\Users\\Harleen\\Downloads\\New Folder\\extracted_images"
MATCH_RESULTS_PATH = "C:\\Users\\Harleen\\Downloads\\New Folder\\match_results.xlsx"

# **Load Excel File and Extract Images**
def extract_images():
    wb = xw.Book(EXCEL_FILE)
    sheet = wb.sheets.active
    # Delete the folder if it exists
    if os.path.exists(IMAGE_FOLDER):
        for file in os.listdir(IMAGE_FOLDER):
            os.remove(os.path.join(IMAGE_FOLDER, file))
    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    image_shapes = [(shape, shape.top) for shape in sheet.shapes if shape.api.Type == 13]
    image_shapes.sort(key=lambda x: x[1])

    for index, (shape, _) in enumerate(image_shapes):
        shape.api.Copy()
        time.sleep(1)  # Allow clipboard processing
        img = ImageGrab.grabclipboard()
        if img:
            img.save(os.path.join(IMAGE_FOLDER, f"image_{index}_2.png"), "PNG")
    wb.close()

# **Remove Background Function**
def remove_background(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        img = img[y:y+h, x:x+w]
    return img

# **Match Target Image with Extracted Images**
def match_images(target_img):
    df = pd.read_excel(EXCEL_FILE)
    #target_img_path = "C:\\Users\\Harleen\\Downloads\\img1.jpg"
    #target_img = cv2.imread(target_img_path)
    target_img = remove_background(target_img)
    target_img = cv2.resize(target_img, (300, 300))
    target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    target_kp, target_des = sift.detectAndCompute(target_img_gray, None)

    match_scores = {}
    for img_file in os.listdir(IMAGE_FOLDER):
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        img_color = cv2.imread(img_path)
        if img_color is None:
            continue
        img_color = remove_background(img_color)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        kp, des = sift.detectAndCompute(img_gray, None)
        if des is None or target_des is None:
            continue

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(target_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        match_scores[img_file] = (len(matches), img_gray, img_color, kp, matches)

    top_matches = sorted(match_scores.items(), key=lambda x: x[1][0], reverse=True)[:2]
    match_results = []

    for rank, (img_name, (match_count, img_gray, img_color, kp, matches)) in enumerate(top_matches, start=1):
        matched_row = int(img_name.split("_")[1])  
        if 0 <= matched_row < len(df):
            matched_data = df.iloc[matched_row]
            match_results.append({
                "Rank": rank,
                "Image Name": img_name,
                "Good Matches": match_count,
                "Excel Row": matched_row,
                **matched_data.to_dict()
            })

    if match_results:
        match_df = pd.DataFrame(match_results)
        match_df.to_excel(MATCH_RESULTS_PATH, index=False)

    return top_matches, match_results



# **Streamlit UI**
st.title("🔍 Image Matching App")
st.write("Upload an image to find the best match from stored data.")

uploaded_file = st.file_uploader("Upload Target Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    target_img = Image.open(uploaded_file)
    target_img = np.array(target_img)
    st.image(target_img, caption="Uploaded Image", use_column_width=True)

    if st.button("Find Best Matches"):
        extract_images()
        top_matches, match_results = match_images(target_img)

        if match_results:
            st.write("### 🎯 Top Matches Found:")
            for result in match_results:
                st.write(f"**Rank {result['Rank']}:** {result['Image Name']} ({result['Good Matches']} good matches)")
                st.write("**Matched Data from Excel:**")
                st.dataframe(pd.DataFrame([result]))

            st.write("### 📂 Download Results:")
            with open(MATCH_RESULTS_PATH, "rb") as f:
                st.download_button("Download Match Results (Excel)", f, file_name="match_results.xlsx")
        else:
            st.write("⚠️ No matches found.")