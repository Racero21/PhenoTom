import cv2
import numpy as np

image = cv2.imread("C:\\Users\\Ralph\\try\\PhenoTom\\SP\\2024 Tomato Pictures for Ralph\\20240326_093016.jpg")
height, width = image.shape[:2]

# Calculate the amount to remove from each side (5% of the width and height)
remove_height = int(height * 0.05)
remove_width = int(width * 0.05)

# Define the new boundaries
start_x = remove_width
start_y = remove_height
end_x = width - remove_width
end_y = height - remove_height

# Crop the image using array slicing
cropped_image = image[start_y:end_y, start_x:end_x]
# image = cv2.imread("C:\\Users\\Ralph\\try\\PhenoTom\\SP\\Image Physio Phenotype\\Jul 31\\Jp1.jpg")
original = cropped_image.copy()
# Convert the image to HSV color space
hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

# Define the green color range in HSV
# lower_green = np.array([35, 100, 100])
# upper_green = np.array([85, 255, 255])
lower_green = np.array([30, 30, 30])
upper_green = np.array([80, 255, 255])
yellows = np.array([255,255,0])

# Create a binary mask where the green areas are white
mask = cv2.inRange(hsv, lower_green, upper_green, yellows)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

all_contours = np.vstack(contours)

# Calculate the area and perimeter of the concatenated contour
area = cv2.contourArea(all_contours)
perimeter = cv2.arcLength(all_contours, True)

# Avoid division by zero
if perimeter == 0:
    circularity = 0
else:
    # Calculate circularity
    circularity = 4 * np.pi * (area / (perimeter * perimeter))

# Print the circularity
print(f'Combined Circularity: {circularity}')

# Optionally, draw the combined contour and circularity on the image
cv2.drawContours(cropped_image, [all_contours], -1, (0, 255, 0), 2)
cv2.putText(image, f'{circularity:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

image2 = cv2.resize(cropped_image, (0, 0), fx = 0.3, fy = 0.3)

# Display the original image with the combined contour and circularity value
cv2.imshow('Image with Combined Circularity', image2)

green_mask = cv2.inRange(hsv, lower_green, upper_green, yellows)

    # Find contours in the binary green mask
contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate and sum up the area for each contour
total_projected_area = 0
for contour in contours:
    area = cv2.contourArea(contour)
    total_projected_area += area

# Draw contours on the original image
cv2.drawContours(original, contours, -1, (0, 255, 0), 2)

# Calculate bounding rectangle for the combined contours
x, y, w, h = cv2.boundingRect(np.concatenate(contours))
cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the original image with contours and bounding rectangle
# cv2.imshow('Contours and Bounding Rectangle', original)
# Calculate object extent, eccentricity, and convex hull area for the combined contour
extent_x = w
extent_y = h

if len(contours) >= 5:
    ellipse = cv2.fitEllipse(np.concatenate(contours))
    eccentricity = float(np.sqrt(1 - (ellipse[1][0] ** 2) / (ellipse[1][1] ** 2)))
    # Draw the fitted ellipse
    cv2.ellipse(original, ellipse, (0, 0, 255), 2)
else:
    eccentricity = 0

hull = cv2.convexHull(np.concatenate(contours))
hull_area = cv2.contourArea(hull)
# color = (0,255,255)
# Draw the convex hull on the original image
cv2.drawContours(original, [hull], 0, (255, 0, 0), 2)
# cv2.line(original, (0,0), (0,364), color, 5)
# Display the original image with contours, fitted ellipse, and convex hull
image3 = cv2.resize(original, (0, 0), fx = 0.3, fy = 0.3)
cv2.imshow('Contours and Parameters', image3)

cv2.waitKey(0)
cv2.destroyAllWindows()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # output_file_params = os.path.join(output_folder, f"contours_{os.path.basename(image_path)}")
    # cv2.imwrite(output_file_params, original)

    # coin = getCoinScale(image_path)

    # print(f"Total Projected Area: {total_projected_area}, Extent X: {extent_x}, Extent Y: {extent_y}, Eccentricity: {eccentricity}, Hull Area: {hull_area}")


# # Iterate through each contour to calculate circularity
# for contour in contours:
#     # Calculate the area and perimeter of the contour
#     area = cv2.contourArea(contour)
#     perimeter = cv2.arcLength(contour, True)
    
#     # Avoid division by zero
#     if perimeter == 0:
#         continue
    
#     # Calculate circularity
#     circularity = 4 * np.pi * (area / (perimeter * perimeter))
    
#     # Print the circularity
#     print(f'Circularity: {circularity}')

#     # Optionally, draw the contour and circularity on the image
#     cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
#     cv2.putText(image, f'{circularity:.2f}', tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#     image2 = cv2.resize(image, (0, 0), fx = 0.3, fy = 0.3)

# # Display the original image with contours and circularity values
# cv2.imshow('Image with Circularity', image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply thresholding
# _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# # Find contours
# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Loop through each contour to calculate the circularity
# for contour in contours:
#     # Calculate area
#     area = cv2.contourArea(contour)
    
#     # Calculate perimeter
#     perimeter = cv2.arcLength(contour, True)
    
#     # Avoid division by zero
#     if perimeter == 0:
#         continue
    
#     # Calculate circularity
#     circularity = (4 * np.pi * area) / (perimeter * perimeter)
    
#     # Print circularity
#     print(f'Circularity: {circularity}')

#     # Optionally, draw the contour and circularity on the image
#     image3 = cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.putText(image, f'{circularity:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

#     image2 = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)
#     small = cv2.resize(image3, (0, 0), fx = 0.3, fy = 0.3)

# # Display the result
# cv2.imshow('Image with Circularity', image2)
# cv2.imshow('contours', small)
# cv2.waitKey(0)
# cv2.destroyAllWindows()