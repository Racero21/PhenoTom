import cv2 as cv2
import numpy as np

# image = cv2.imread("C:\\Users\\Ralph\\Downloads\\proj\\plant2.jpg")
# image = cv2.imread("C:\\Users\\Ralph\\Downloads\\download.jpg")
# image = cv2.imread("C:\\Users\\edwin\\Downloads\\pics\\2024 Tomato Pictures for Ralph\\p3.jpg")
image = cv2.imread("C:\\Users\\edwin\\Desktop\\PhenoTom\\new\\PhenoTom\\SP\\2024 Tomato Pictures for Ralph\\20240326_092617.jpg")
# # Get the dimensions of the image
# height, width, _ = image.shape
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
# Define the lower and upper bounds for the green color in HSV
lower_green = np.array([33, 33, 33])
upper_green = np.array([85, 255, 255])
yellows = np.array([255,255,0])

# Create a mask to extract the green regions
green_mask = cv2.inRange(hsv, lower_green, upper_green, yellows)

# Find contours in the binary green mask
contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

# Filter out outlier contours based on centroid positions
filtered_contours = []
if len(centroids) > 0:
    mean_centroid = np.mean(centroids, axis=0)
    std_dev_centroid = np.std(centroids, axis=0)
    threshold = 2  # Threshold for outlier detection

    for i, contour in enumerate(valid_contours):
        distance = np.linalg.norm(centroids[i] - mean_centroid)
        if distance < threshold * np.linalg.norm(std_dev_centroid):
            filtered_contours.append(contour)

# Calculate and sum up the area for each filtered contour
total_area = 0
total_perimeter = 0
for contour in filtered_contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    total_area += area
    total_perimeter += perimeter
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# total_area = 0
# total_perimeter = 0
# for contour in contours:
#     area = cv2.contourArea(contour)
#     perimeter = cv2.arcLength(contour, True)
#     total_area += area
#     total_perimeter += perimeter
    # if perimeter == 0:
    #     circularity = 0
    # else:
    #     circularity = 4 * np.pi * (area / (perimeter * perimeter))

circularity = 4 * np.pi * (total_area / (total_perimeter * total_perimeter))



print(f'Area: {total_area}, Perimeter: {total_perimeter}, Circularity: {circularity}')

# # Calculate the center coordinates of the image
# center_x = width // 2
# center_y = height // 2

# # Define the dimensions of the rectangle
# rectangle_width = width*1//3
# rectangle_height = height*1//3

# # Calculate the coordinates of the top-left corner of the rectangle
# top_left_x = center_x - rectangle_width // 2
# top_left_y = center_y - rectangle_height // 2

# # Calculate the coordinates of the bottom-right corner of the rectangle
# bottom_right_x = center_x + rectangle_width // 2
# bottom_right_y = center_y + rectangle_height // 2

# # Draw a rectangle in the middle of the image
# color = (0, 255, 0)  # Green color in BGR format
# thickness = -1  # Thickness of the rectangle border
# # cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)
# # cv2.circle(image, (width//2, height//2), 20, (255,0,0), thickness)
# # Display the image with the rectangle
# # cv2.imshow("Image with Rectangle", image)
# # Convert image to HSV color space
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define range of blue color in HSV
# lower_blue = np.array([90, 50, 50])
# upper_blue = np.array([130, 255, 255])

# # Threshold the HSV image to get only blue colors
# mask = cv2.inRange(hsv, lower_blue, upper_blue)

# # Apply the mask to the original image
# blue_masked_image = cv2.bitwise_and(image, image, mask=mask)

# # Display the original image and the blue mask
# # cv2.imshow('Original Image', image)
# half = cv2.resize(blue_masked_image, (0, 0), fx = 0.1, fy = 0.1)
# # cv2.imshow('Blue Mask', blue_masked_image)
# cv2.imshow('Blue Mask', half)
# # Convert the blue masked image to grayscale
# gray = cv2.cvtColor(blue_masked_image, cv2.COLOR_BGR2GRAY)

# divisor = 1
# while True:
# # Apply Hough Circle Transform
#     circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=width*1/divisor,
#                            param1=50, param2=30, minRadius=0, maxRadius=0)

#     if circles is not None and len(circles) == 1:
#         circles = np.round(circles[0, :]).astype("int")
#         count = 0
#         # Loop over the (x, y) coordinates and radius of the circles
#         for (x, y, r) in circles:
#             # Draw the circle outline
#             count += 1
#             cv2.circle(blue_masked_image, (x, y), r, (255, 255, 0), 4)

#         # Display the detected circles
#         half2 = cv2.resize(blue_masked_image, (0, 0), fx = 0.1, fy = 0.1)
#         cv2.imshow(f'Detected Circles - {count} - radius = {r}', blue_masked_image)
        
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         break
#     else:
#         divisor = divisor - 0.01
#         print("No circles detected.")

def getCoinScale(image_path):
    image = cv2.imread(image_path)

    # Get the dimensions of the image
    height, width, _ = image.shape

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Apply the mask to the original image
    blue_masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Display the original image and the blue mask
    # cv2.imshow('Original Image', image)
    half = cv2.resize(blue_masked_image, (0, 0), fx = 0.1, fy = 0.1)
    # cv2.imshow('Blue Mask', blue_masked_image)
    # cv2.imshow('Blue Mask', half)
    # Convert the blue masked image to grayscale
    gray = cv2.cvtColor(blue_masked_image, cv2.COLOR_BGR2GRAY)

    divisor = 1
    while True:
    # Apply Hough Circle Transform
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=width*1/divisor,
                            param1=50, param2=30, minRadius=0, maxRadius=0)

        if circles is not None and len(circles) == 1:
            circles = np.round(circles[0, :]).astype("int")
            count = 0
            # Loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # Draw the circle outline
                count += 1
                cv2.circle(blue_masked_image, (x, y), r, (255, 255, 0), 4)

            # Display the detected circles
            half2 = cv2.resize(blue_masked_image, (0, 0), fx = 0.1, fy = 0.1)
            return 2*r
            # cv2.imshow(f'Detected Circles - {count} - radius = {r}', half2)
            
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            break
        else:
            divisor = divisor - 0.01
            print("No circles detected.")
# # Ensure at least one circle was found
# if circles is not None:
#     # Convert the (x, y) coordinates and radius of the circles to integers
#     circles = np.round(circles[0, :]).astype("int")
#     count = 0
#     # Loop over the (x, y) coordinates and radius of the circles
#     for (x, y, r) in circles:
#         # Draw the circle outline
#         count += 1
#         cv2.circle(blue_masked_image, (x, y), r, (255, 255, 0), 4)

#     # Display the detected circles
#     half2 = cv2.resize(blue_masked_image, (0, 0), fx = 0.1, fy = 0.1)
#     cv2.imshow(f'Detected Circles - {count} - radius = {r}', half2)
    
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("No circles detected.")

cv2.waitKey(0)
cv2.destroyAllWindows()
