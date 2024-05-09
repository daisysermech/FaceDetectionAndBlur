import cv2
import numpy as np
from skimage import io, color, exposure
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob, os

from pickle import dump

# Function to extract Gabor features from an image
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

np_samples = []
p_samples = []

person = 'face'
nonperson = 'nonface'

for file_path in glob.glob(os.path.join(nonperson, '*.jpg')):
    image = io.imread(file_path)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    features = extract_gabor_features(image)
    np_samples.append(features)

for file_path in glob.glob(os.path.join(person, '*.jpg')):
    image = io.imread(file_path)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    features = extract_gabor_features(image)
    p_samples.append(features)

labels = np.concatenate([np.ones(len(p_samples)), np.zeros(len(np_samples))])

all_samples = p_samples+np_samples

X = np.array(all_samples)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a Support Vector Machine (SVM) classifier
classifier = SVC(kernel='linear', C=1.0, decision_function_shape='ovr', probability=True,class_weight='balanced')
classifier.fit(X_train_pca, y_train)

y_pred = classifier.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

dump(classifier, open('classifier.pkl', 'wb'))
dump(scaler, open('scaler.pkl', 'wb'))
dump(pca, open('pca.pkl', 'wb'))
