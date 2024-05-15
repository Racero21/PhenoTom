# import cv2
# import numpy as np

# def detect_coins(image_path):
#     # Read the image
#     image = cv2.imread(image_path)
    
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (11, 11), 0)
#     cv2.imshow("blurred", blurred)

#     # Detect edges using Canny edge detector
#     edges = cv2.Canny(blurred, 30, 150)
#     cv2.imshow("edges", edges)

#     # Detect circles using Hough Circle Transform
#     # circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
#     #                            param1=50, param2=30, minRadius=30, maxRadius=200)
    
#     circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
#                                param1=50, param2=30, minRadius=30, maxRadius=100)
#     cv2.imshow("circles", circles)

#     if circles is not None:
#         # Convert the circle parameters to integer
#         circles = np.round(circles[0, :]).astype("int")
        
#         for (x, y, r) in circles:
#             # Draw the circle on the original image
#             cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            
#             # Measure the diameter of the circle (coin)
#             diameter = r * 2
#             print("Coin diameter:", diameter)
            
#         # Show the result
#         cv2.imshow("Coins Detected", image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     else:
#         print("No coins detected in the image.")

#     return 0

# # Path to the input image
# image_path = "C:\\Users\\edwin\\Downloads\\SP\\2024 Tomato Pictures for Ralph\\test1.jpg"

# # Detect coins and their diameters
# detect_coins(image_path)

import cv2
import numpy as np

def detect_coins(image_path, coin_diameter_mm):
    # Read the image
    image = cv2.imread(image_path)
    
    # Define ROI mask (exclude middle part)
    mask = np.zeros_like(image[:, :, 0])
    height, width = mask.shape[:2]
    roi = 0.3  # Percentage of image height to exclude from the middle
    mask[int(roi * height):int((1 - roi) * height), :] = 255
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Apply mask to the blurred image
    blurred[mask == 0] = 0
    
    # Detect edges using Canny edge detector
    edges = cv2.Canny(blurred, 30, 150)
    
    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=50, param2=30, minRadius=30, maxRadius=100)
    
    if circles is not None:
        # Convert the circle parameters to integer
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:
            # Draw the circle on the original image
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            
            # Measure the diameter of the circle (coin) in pixels
            diameter_pixels = r * 2
            print("Coin diameter (pixels):", diameter_pixels)
            
            # Calibrate to get the diameter in millimeters
            pixel_to_mm = coin_diameter_mm / diameter_pixels
            diameter_mm = diameter_pixels * pixel_to_mm
            print("Coin diameter (mm):", diameter_mm)
            
        # Show the result
        cv2.imshow("Coins Detected", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Ensure all windows are closed
        
    else:
        print("No coins detected in the image.")

# Path to the input image
image_path = "C:\\Users\\edwin\\Downloads\\SP\\2024 Tomato Pictures for Ralph\\test1.jpg"

# Known diameter of the coin in millimeters
coin_diameter_mm = 23.2  # Change this according to your coin's actual diameter

# Detect coins and their diameters
detect_coins(image_path, coin_diameter_mm)
