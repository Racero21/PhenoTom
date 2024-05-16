import cv2
import numpy as np

# Load the image
image = cv2.imread("C:\\Users\\Ralph\\try\\PhenoTom\\SP\\2024 Tomato Pictures for Ralph\\20240326_093016.jpg")

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the green color range in HSV
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# Create a binary mask where the green areas are white
mask = cv2.inRange(hsv, lower_green, upper_green)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate centroids of the contours
centroids = []
valid_contours = []
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids.append((cX, cY))
        valid_contours.append(contour)

# Convert centroids list to a NumPy array
centroids = np.array(centroids)

# Check if we have centroids to process
if len(centroids) > 0:
    # Calculate the mean and standard deviation of the centroids
    mean_centroid = np.mean(centroids, axis=0)
    std_dev_centroid = np.std(centroids, axis=0)

    # Define a threshold for outlier detection (e.g., 2 standard deviations)
    threshold = 2

    # Filter out contours whose centroids are far from the mean centroid
    filtered_contours = []
    for i, contour in enumerate(valid_contours):
        distance = np.linalg.norm(centroids[i] - mean_centroid)
        if distance < threshold * np.linalg.norm(std_dev_centroid):
            filtered_contours.append(contour)

    # Draw filtered contours on the original image
    output_image = image.copy()
    cv2.drawContours(output_image, filtered_contours, -1, (0, 255, 0), 2)
    image1 = cv2.resize(output_image, (0, 0), fx = 0.3, fy = 0.3)
    image2 = cv2.resize(output_image, (0, 0), fx = 0.3, fy = 0.3)
    # Display the original and output images
    cv2.imshow('Original Image', image1)
    cv2.imshow('Filtered Contours', image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the output image
    # cv2.imwrite('filtered_contours_image.jpg', output_image)
else:
    print("No valid centroids found to process.")