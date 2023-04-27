import cv2
import math
import numpy as np

# Load image
img = cv2.imread('Prueba3.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blur, 50, 150, apertureSize=3)

# Apply Hough line transform to detect lane lines
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

# Draw detected lines on the original image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Show the result
cv2.imshow('Lane lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def get_lane_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 50, minLineLength=50, maxLineGap=100)
    return lines

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_fresnel_zone_radius(lane_lines, camera_distance):
    distances = []
    for line in lane_lines:
        x1, y1, x2, y2 = line[0]
        distance = calculate_distance(x1, y1, x2, y2)
        distances.append(distance)
    lane_distance = sum(distances) / len(distances)
    midpoint = (lane_lines[0][0][0] + lane_lines[1][0][0]) / 2
    angle = math.atan(midpoint / camera_distance)
    fresnel_radius = (angle * camera_distance) / 2
    return fresnel_radius

# Load image and get lane lines

lane_lines = get_lane_lines(img)

# Calculate the radius of the Fresnel zone
camera_distance = 100  # Distance between the camera and the lane lines in meters
fresnel_radius = calculate_fresnel_zone_radius(lane_lines, camera_distance)

print("Fresnel zone radius:", fresnel_radius, "meters")