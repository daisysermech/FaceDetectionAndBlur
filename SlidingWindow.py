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


import pickle
import cv2
import numpy as np

classifier = pickle.load(open('classifier.pkl', 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))
pca = pickle.load(open("pca.pkl", 'rb'))

image_path = "example.jpg"
image = cv2.imread(image_path)

import time
import cv2
import numpy as np

window_sizes = [128, 256, 512]
window_strides = [32, 64, 128]
threshold = 0.5

face_coords = []
confidences = []

import time
start_time = time.time()

for window_size in window_sizes:
    for window_stride in window_strides:
        for x in range(0, image.shape[1] - window_size, window_stride):
            for y in range(0, image.shape[0] - window_size, window_stride):
                patch = image[y:y + window_size, x:x + window_size]
                patch = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_AREA)
                features = extract_gabor_features(patch)
                test_features = np.array(features).reshape(1, -1)

                test_features_standardized = scaler.transform(test_features)
                test_features_pca = pca.transform(test_features_standardized)

                faces = classifier.predict_proba(test_features_pca)
                if faces[0][1] > threshold:
                    face_coords.append([x, y, window_size, window_size])
                    confidences.append(faces[0][1])

face_coords = np.array(face_coords)

fc = face_coords
fc = fc.astype(int).tolist()
indices = cv2.dnn.NMSBoxes(fc, confidences, .6, 0.2)

image = cv2.imread(image_path)
for i in indices:
    x, y, w, h = fc[i][0] ,fc[i][1], fc[i][2], fc[i][3]
    face_region = image[y:y + h, x:x + w]
    blurred_face_region = cv2.GaussianBlur(face_region, (23, 23), 30)
    image[y:y + h, x:x + w] = blurred_face_region

print("--- %s seconds ---" % (time.time() - start_time))
cv2.imshow("RESULT", image)
cv2.waitKey(0)

cv2.imwrite("example_result.jpg",image)
