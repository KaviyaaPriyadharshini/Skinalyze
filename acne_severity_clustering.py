import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Function to load images and preprocess them
def load_and_preprocess_images(image_folder):
    images = []
    image_files = []
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add any image formats you need
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))  # Resize images to a fixed size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            images.append(img)
            image_files.append(filename)
    return np.array(images), image_files

# Main function for K-Means clustering
def predict_severity_levels():
    image_folder = "/Users/daksha/Desktop/kavs/JPEGImages"  # Change to your image folder
    images, image_files = load_and_preprocess_images(image_folder)

    # Step 1: Flatten the images for clustering
    num_images = images.shape[0]
    flat_images = images.reshape(num_images, -1)  # Flatten to 2D array

    # Step 2: Apply K-Means clustering
    num_clusters = 4  # Change number of clusters to 4 for severity levels
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(flat_images)

    # Step 3: Optional - Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    reduced_images = pca.fit_transform(flat_images)

    # Step 4: Visualize clusters (optional)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_images[:, 0], reduced_images[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
    plt.title('K-Means Clustering of Acne Images')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.show()

    # Step 5: Map clusters to severity levels
    cluster_severity_mapping = {
        0: 'Mild',
        1: 'Moderate',
        2: 'Severe',
        3: 'Very Severe'
    }

    # Step 6: Create a dictionary to hold the results
    image_severity_results = {}

    # Step 7: Assign severity levels to each image based on its cluster
    for i in range(len(labels)):
        image_severity_results[image_files[i]] = cluster_severity_mapping[labels[i]]

    # Step 8: Save results to a CSV file
    results_df = pd.DataFrame(list(image_severity_results.items()), columns=['Image', 'Severity Level'])
    results_df.to_csv('acne_severity_results.csv', index=False)

# Run the function to generate the CSV
predict_severity_levels()
