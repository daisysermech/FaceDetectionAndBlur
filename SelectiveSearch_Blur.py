import pickle

import cv2
import numpy as np
import selectivesearch

image_path="example.jpg"
image = cv2.imread(image_path)

def extract_gabor_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define Gabor filter parameters
    ksize = 5  # Size of the filter
    sigma = 1.0  # Standard deviation of the filter
    theta = np.pi / 4  # Orientation of the filter
    lambd = 2.0  # Wavelength of the sinusoidal factor
    gamma = 0.5  # Spatial aspect ratio

    gabor_filter = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)

    filtered_image = cv2.filter2D(gray, cv2.CV_8UC3, gabor_filter)

    features = filtered_image.flatten()

    return features


def interpolate_color(value):
    red = int(255 * (1 - value))
    green = int(255 * value)

    return red, green, 0

classifier = pickle.load(open('classifier.pkl', 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))
pca = pickle.load(open("pca.pkl", 'rb'))

img_lbl, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=10)

image_area = image.shape[0] * image.shape[1]
min_area_threshold = image_area * 0.1

filtered_proposals = []

for proposal in regions:
    x, y, w, h = proposal['rect']
    proposal_area = w * h

    if proposal_area >= min_area_threshold:
        filtered_proposals.append(proposal)

image_drawn = image

for proposal in filtered_proposals:
    x, y, w, h = proposal['rect']
    patch = image[y:y + w, x:x + h]
    patch = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_AREA)
    features = extract_gabor_features(patch)
    test_features = np.array(features).reshape(1, -1)
    test_features_standardized = scaler.transform(test_features)
    test_features_pca = pca.transform(test_features_standardized)

    faces = classifier.predict_proba(test_features_pca)
    if faces[0][1] > 0.5:
        face_region = image[y:y + h, x:x + w]
        blurred_face_region = cv2.GaussianBlur(face_region, (23, 23), 30)
        image[y:y + h, x:x + w] = blurred_face_region

cv2.imshow("RESULT", image)
cv2.waitKey(0)

cv2.imwrite("example result.jpg",image)
