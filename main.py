import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
data = []
labels = []
gestures = ["palm", "fist", "thumbs_up"]
for label, gesture in enumerate(gestures):
    folder = f"dataset/{gesture}"
    for file in os.listdir(folder):
        img = Image.open(os.path.join(folder, file))
        img = img.resize((64, 64))
        img = np.array(img).flatten()

        data.append(img)
        labels.append(label)
X = np.array(data)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = SVC(kernel="linear")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Gesture Recognition Accuracy:", accuracy * 100, "%")