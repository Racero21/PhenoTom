import cv2
import numpy as np

# Read the image
image = cv2.imread("C:\\Users\\edwin\\Downloads\\SP\\Image Physio Phenotype\\Jul 31\\nicap2.jpg")
original = image.copy()

# Convert the image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the green color in HSV
lower_green = np.array([30, 30, 30])
upper_green = np.array([80, 255, 255])

# Create a mask to extract the green regions
green_mask = cv2.inRange(hsv, lower_green, upper_green)

# Convert the single-channel mask back to 3 channels
green_mask_colored = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)

# Set the overlay color to blue
overlay_color = (255, 0, 0)  # Blue

# Overlay the blue mask on the original image
result = cv2.addWeighted(original, 1, green_mask_colored, 0.5, 0)

# Display the images
cv2.imshow("HSV", hsv)
cv2.imshow("Original Image", original)
cv2.imshow("Green Mask", green_mask_colored)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
# import cv2
# import numpy as np
# # import os

# # def extract_parameters(image_path, output_folder):
# # image = cv2.imread("C:\\Users\\edwin\\Downloads\\sp_dataset\\jp2_cropped.jpg")
# image = cv2.imread("C:\\Users\\edwin\\Documents\\Image Physio Phenotype\\Jul 31\\nicap2.jpg")
# original = image.copy()

# # Convert the image to HSV
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# cv2.imshow("hsv", hsv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # Define the lower and upper bounds for the green color in HSV
# # lower_green = np.array([40, 40, 40])
# lower_green = np.array([30, 30, 30])
# upper_green = np.array([80, 255, 255])
# yellows = np.array([255,255,0])

# # Create a mask to extract the green regions
# green_mask = cv2.inRange(hsv, lower_green, upper_green, yellows)

# # Find contours in the binary green mask
# contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Calculate and sum up the area for each contour
# total_projected_area = 0
# for contour in contours:
#     area = cv2.contourArea(contour)
#     total_projected_area += area

# # Draw contours on the original image
# cv2.drawContours(original, contours, -1, (0, 255, 0), 2)

# # Calculate bounding rectangle for the combined contours
# x, y, w, h = cv2.boundingRect(np.concatenate(contours))
# cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)

#     # Display the original image with contours and bounding rectangle
# # cv2.imshow('Contours and Bounding Rectangle', original)
# # Calculate object extent, eccentricity, and convex hull area for the combined contour
# extent_x = w
# extent_y = h

# if len(contours) >= 5:
#     ellipse = cv2.fitEllipse(np.concatenate(contours))
#     eccentricity = float(np.sqrt(1 - (ellipse[1][0] ** 2) / (ellipse[1][1] ** 2)))
#     # Draw the fitted ellipse
#     cv2.ellipse(original, ellipse, (0, 0, 255), 2)
# else:
#     eccentricity = 0

# hull = cv2.convexHull(np.concatenate(contours))
# hull_area = cv2.contourArea(hull)
# # color = (0,255,255)
# # Draw the convex hull on the original image
# cv2.drawContours(original, [hull], 0, (255, 0, 0), 2)
# # cv2.line(original, (0,0), (0,364), color, 5)
# # Display the original image with contours, fitted ellipse, and convex hull
# # cv2.imshow('Contours and Parameters', original)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # output_file_params = os.path.join(output_folder, f"contours_{os.path.basename(image_path)}")
# # cv2.imwrite(output_file_params, original)

# print(f"Total Projected Area: {total_projected_area}, Extent X: {extent_x}, Extent Y: {extent_y}, Eccentricity: {eccentricity}, Hull Area: {hull_area}")
# # return total_projected_area, extent_x, extent_y, eccentricity, hull_area