# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:58:01 2025

@author: Harleen
"""
import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
from PIL import Image
import time
import openpyxl
from matplotlib import pyplot as plt

# Define constants
EXCEL_FILE = "img1.xlsm"
IMAGE_FOLDER = "extracted_images"
MATCH_RESULTS_PATH = "match_results.xlsx"

# Streamlit UI
def main():
    st.title("Image Recognition and Matching")
    st.write("Upload an image to find the best matches in the dataset.")
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        target_img = Image.open(uploaded_file)
        target_img = np.array(target_img)
        
        # Convert image to OpenCV format
        target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
        st.image(target_img, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Find Best Matches"):
            process_images(target_img)


def process_images(target_img):
    # Load Excel data
    df = pd.read_excel(EXCEL_FILE, engine='openpyxl')
    
    # Ensure image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        st.error("Image folder not found! Make sure images are extracted and placed in the folder.")
        return
    
    # Remove background from target image
    target_img = remove_background(target_img)
    target_img = cv2.resize(target_img, (300, 300))
    target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT feature detector
    sift = cv2.SIFT_create()
    target_kp, target_des = sift.detectAndCompute(target_img_gray, None)
    
    match_scores = {}
    
    # Compare with extracted images
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
    
    # Get top matches
    top_matches = sorted(match_scores.items(), key=lambda x: x[1][0], reverse=True)[:2]
    
    if top_matches:
        st.write("### Top 2 Best Matches")
        
        match_results = []
        
        for rank, (img_name, (match_count, img_gray, img_color, kp, matches)) in enumerate(top_matches, start=1):
            st.write(f"**Rank {rank}: {img_name} ({match_count} good matches)**")
            
            matched_row = int(img_name.split("_")[1])  # Extract row number
            
            if 0 <= matched_row < len(df):
                matched_data = df.iloc[matched_row]
                st.write(matched_data)
                
                match_results.append({
                    "Rank": rank,
                    "Image Name": img_name,
                    "Good Matches": match_count,
                    "Excel Row": matched_row,
                    **matched_data.to_dict()
                })
            
            # Display matched images
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            
            img_target_kp = cv2.drawKeypoints(target_img_gray, target_kp, None, color=(0, 255, 0))
            ax[0].imshow(img_target_kp, cmap='gray')
            ax[0].set_title("Target Image")
            
            img_match_kp = cv2.drawKeypoints(img_gray, kp, None, color=(255, 0, 0))
            ax[1].imshow(img_match_kp, cmap='gray')
            ax[1].set_title(f"Matched Image {rank}")
            
            st.pyplot(fig)
        
        # Save results to Excel
        match_df = pd.DataFrame(match_results)
        match_df.to_excel(MATCH_RESULTS_PATH, index=False)
        st.success("Match results saved to match_results.xlsx")
    else:
        st.write("No matches found.")


def remove_background(img):
    """Remove white background and keep only the object."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        img = img[y:y+h, x:x+w]
    return img

if __name__ == "__main__":
    main()
