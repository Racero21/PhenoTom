import cv2
import numpy as np
import os
from test import getCoinScale


def extract_parameters(image_path, output_folder):
    image = cv2.imread(image_path)
    original = image.copy()

    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for the green color in HSV
    # lower_green = np.array([40, 40, 40])
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([80, 255, 255])
    yellows = np.array([255,255,0])

    # Create a mask to extract the green regions
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
    # cv2.imshow('Contours and Parameters', original)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    output_file_params = os.path.join(output_folder, f"contours_{os.path.basename(image_path)}")
    cv2.imwrite(output_file_params, original)

    coin = getCoinScale(image_path)

    print(f"Total Projected Area: {total_projected_area}, Extent X: {extent_x}, Extent Y: {extent_y}, Eccentricity: {eccentricity}, Hull Area: {hull_area}")
    return [total_projected_area, extent_x, extent_y, eccentricity, hull_area]
    # Read the image
    # image = cv2.imread('C:\\Users\\edwin\\Downloads\\t1.jpg')
    # image = cv2.imread(image_path)
    # original = image.copy()

    # # Convert the image to HSV
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # # Define the lower and upper bounds for the green color in HSV
    # lower_green = np.array([40, 40, 40])
    # upper_green = np.array([80, 255, 255])

    # # Create a mask to extract the green regions
    # green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # # Find contours in the binary green mask
    # contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # all_contours = np.concatenate(contours)

    # # Flatten the contour points for clustering
    # # contour_points = np.vstack([contour.squeeze() for contour in contours])

    # # # Perform hierarchical clustering to find connected contours
    # # linked_contours = linkage(contour_points, method='single', metric='euclidean')

    # # # Set a threshold distance to form clusters
    # # threshold_distance = 20
    # # clusters = fcluster(linked_contours, threshold_distance, criterion='distance')

    # # # Create a dictionary to store points for each cluster
    # # cluster_points = {cluster: [] for cluster in set(clusters)}
    # # for i, point in enumerate(contour_points):
    # #     cluster_points[clusters[i]].append(point)

    # # # Create a list of contours from the clustered points
    # # connected_contours = [np.array(cluster, dtype=np.int32).reshape((-1, 1, 2)) for cluster in cluster_points.values()]

    # # # Concatenate all connected contours
    # # all_contours = np.concatenate(connected_contours)

    # # Calculate area, object extent, eccentricity, and convex hull area for the combined contour
    # area = cv2.contourArea(all_contours)

    # # Calculate bounding rectangle
    # x, y, w, h = cv2.boundingRect(all_contours)
    # extent_x = float(area) / w
    # extent_y = float(area) / h

    # # Fit an ellipse if there are enough points
    # if len(all_contours) >= 5:
    #     ellipse = cv2.fitEllipse(all_contours)
    #     eccentricity = np.sqrt(1 - (ellipse[1][0] ** 2) / (ellipse[1][1] ** 2))
    # else:
    #     eccentricity = 0

    # # Calculate convex hull area
    # hull = cv2.convexHull(all_contours)
    # hull_area = cv2.contourArea(hull)

    # print(f"{area}, {extent_x}, {extent_y}, {eccentricity}, {hull_area}")
    # return area, extent_x, extent_y, eccentricity, hull_area
    # # Display original image with contours in a separate window
    # original_with_contours = original.copy()
    # cv2.drawContours(original_with_contours, [all_contours], -1, (0, 255, 0), 2)
    # cv2.imshow('Original Image with Contours', original_with_contours)
    # cv2.waitKey(0)

    # # Display original image with bounding rectangle in a separate window
    # original_with_extent = original.copy()
    # cv2.rectangle(original_with_extent, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow('Original Image with Bounding Rectangle', original_with_extent)
    # cv2.waitKey(0)

    # # Display original image with fitted ellipse in a separate window
    # original_with_ellipse = original.copy()
    # if len(all_contours) >= 5:
    #     cv2.ellipse(original_with_ellipse, ellipse, (0, 0, 255), 2)
    # cv2.imshow('Original Image with Fitted Ellipse', original_with_ellipse)
    # cv2.waitKey(0)

    # # Display original image with convex hull in a separate window
    # original_with_hull = original.copy()
    # cv2.drawContours(original_with_hull, [hull], 0, (255, 0, 0), 2)
    # cv2.imshow('Original Image with Convex Hull', original_with_hull)
    # cv2.waitKey(0)

    # # Wait for a key press and close all windows
    # cv2.destroyAllWindows()

# image = cv2.imread('C:\\Users\\edwin\\Downloads\\t1.jpg')
# # image = cv2.imread(image_path)
# original = image.copy()

# # Convert the image to HSV
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define the lower and upper bounds for the green color in HSV
# lower_green = np.array([40, 40, 40])
# upper_green = np.array([80, 255, 255])

# # Create a mask to extract the green regions
# green_mask = cv2.inRange(hsv, lower_green, upper_green)

# # Find contours in the binary green mask
# contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# total_projected_area = 0
# for contour in contours:
#     area = cv2.contourArea(contour)
#     total_projected_area += area

# # all_contours = np.concatenate(contours)

# # Flatten the contour points for clustering
# # contour_points = np.vstack([contour.squeeze() for contour in contours])

# # # Perform hierarchical clustering to find connected contours
# # linked_contours = linkage(contour_points, method='single', metric='euclidean')

# # # Set a threshold distance to form clusters
# # threshold_distance = 20
# # clusters = fcluster(linked_contours, threshold_distance, criterion='distance')

# # # Create a dictionary to store points for each cluster
# # cluster_points = {cluster: [] for cluster in set(clusters)}
# # for i, point in enumerate(contour_points):
# #     cluster_points[clusters[i]].append(point)

# # # Create a list of contours from the clustered points
# # connected_contours = [np.array(cluster, dtype=np.int32).reshape((-1, 1, 2)) for cluster in cluster_points.values()]

# # # Concatenate all connected contours
# # all_contours = np.concatenate(connected_contours)

# # Calculate area, object extent, eccentricity, and convex hull area for the combined contour
# area = cv2.contourArea(all_contours)

# # Calculate bounding rectangle
# x, y, w, h = cv2.boundingRect(all_contours)
# extent_x = float(area) / w
# extent_y = float(area) / h

# # Fit an ellipse if there are enough points
# if len(all_contours) >= 5:
#     ellipse = cv2.fitEllipse(all_contours)
#     eccentricity = np.sqrt(1 - (ellipse[1][0] ** 2) / (ellipse[1][1] ** 2))
# else:
#     eccentricity = 0

# # Calculate convex hull area
# hull = cv2.convexHull(all_contours)
# hull_area = cv2.contourArea(hull)

# print(f"{area}, {extent_x}, {extent_y}, {eccentricity}, {hull_area}")

# # Display original image with contours in a separate window
# original_with_contours = original.copy()
# cv2.drawContours(original_with_contours, [all_contours], -1, (0, 255, 0), 2)
# cv2.imshow('Original Image with Contours', original_with_contours)
# cv2.waitKey(0)

# # Display original image with bounding rectangle in a separate window
# original_with_extent = original.copy()
# cv2.rectangle(original_with_extent, (x, y), (x + w, y + h), (0, 255, 0), 2)
# cv2.imshow('Original Image with Bounding Rectangle', original_with_extent)
# cv2.waitKey(0)

# # Display original image with fitted ellipse in a separate window
# original_with_ellipse = original.copy()
# if len(all_contours) >= 5:
#     cv2.ellipse(original_with_ellipse, ellipse, (0, 0, 255), 2)
# cv2.imshow('Original Image with Fitted Ellipse', original_with_ellipse)
# cv2.waitKey(0)

# # Display original image with convex hull in a separate window
# original_with_hull = original.copy()
# cv2.drawContours(original_with_hull, [hull], 0, (255, 0, 0), 2)
# cv2.imshow('Original Image with Convex Hull', original_with_hull)
# cv2.waitKey(0)

# # Wait for a key press and close all windows
# cv2.destroyAllWindows()

############# START OF WORKING CODE ##############

# image = cv2.imread('C:\\Users\\edwin\\Downloads\\t1.jpg')
# original = image.copy()

# # Convert the image to HSV
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define the lower and upper bounds for the green color in HSV
# lower_green = np.array([40, 40, 40])
# upper_green = np.array([80, 255, 255])

# # Create a mask to extract the green regions
# green_mask = cv2.inRange(hsv, lower_green, upper_green)

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
# cv2.imshow('Contours and Bounding Rectangle', original)
# # Calculate object extent, eccentricity, and convex hull area for the combined contour
# extent_x = w
# extent_y = h

# if len(contours) >= 5:
#     ellipse = cv2.fitEllipse(np.concatenate(contours))
#     eccentricity = np.sqrt(1 - (ellipse[1][0] ** 2) / (ellipse[1][1] ** 2))
#     # Draw the fitted ellipse
#     cv2.ellipse(original, ellipse, (0, 0, 255), 2)
# else:
#     eccentricity = 0

# hull = cv2.convexHull(np.concatenate(contours))
# hull_area = cv2.contourArea(hull)
# color = (0,255,255)
# # Draw the convex hull on the original image
# cv2.drawContours(original, [hull], 0, (255, 0, 0), 2)
# cv2.line(original, (0,0), (0,364), color, 5)
# # Display the original image with contours, fitted ellipse, and convex hull
# cv2.imshow('Contours and Parameters', original)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(f"Total Projected Area: {total_projected_area}, Extent X: {extent_x}, Extent Y: {extent_y}, Eccentricity: {eccentricity}, Hull Area: {hull_area}")
# return total_projected_area, extent_x, extent_y, eccentricity, hull_area