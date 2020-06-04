import os
import face_recognition_api
import pickle
import numpy as np
import pandas as pd
from tkinter import *
from firebase import firebase

firebase = firebase.FirebaseApplication('https://face-recognition-b0b71.firebaseio.com/')

def get_prediction_images(prediction_dir):
    files = [x[2] for x in os.walk(prediction_dir)][0]
    l = []
    exts = [".jpg", ".jpeg", ".png"]
    for file in files:
        _, ext = os.path.splitext(file)
        if ext.lower() in exts:
            l.append(os.path.join(prediction_dir, file))

    return l


fname = 'classifier.pkl'
prediction_dir = './test-images'
root = Tk()
root.title("Face Recognition")

encoding_file_path = './encoded-images-data.csv'
df = pd.read_csv(encoding_file_path)
full_data = np.array(df.astype(float).values.tolist())

# Extract features and labels
# remove id column (0th column)
X = np.array(full_data[:, 1:-1])
y = np.array(full_data[:, -1:])

if os.path.isfile(fname):
    with open(fname, 'rb') as f:
        (le, clf) = pickle.load(f)
else:
    print("Classifier '{}' does not exist".format(fname))
    quit()

resultMessage = ''
 
for image_path in get_prediction_images(prediction_dir):
    # print colorful text with image name
    print("Predicting faces in '{}' ".format(image_path))

    resultMessage = resultMessage + "In " + image_path + "\n\n"
    img = face_recognition_api.load_image_file(image_path)
    X_faces_loc = face_recognition_api.face_locations(img)

    faces_encodings = face_recognition_api.face_encodings(img, known_face_locations=X_faces_loc)
    print("Found {} faces in the image".format(len(faces_encodings)))

    closest_distances = clf.kneighbors(faces_encodings, n_neighbors=1)

    is_recognized = [closest_distances[0][i][0] <= 0.5 for i in range(len(X_faces_loc))]
    
    S = Scrollbar(root, orient="vertical")
    T = Text(root, height = 20, width = 50) 
    T.configure(yscrollcommand=S.set)
    S.configure(command=T.yview)
    for pred, loc, rec in zip(clf.predict(faces_encodings), X_faces_loc, is_recognized):
        if rec:
            print(le.inverse_transform([int(pred)]), loc)
            result = firebase.get('Profiles',le.inverse_transform([int(pred)])[0])
            resultMessage = resultMessage + "    " + le.inverse_transform([int(pred)])[0] + "\n\n"
            for x,y in result.items():
                resultMessage = resultMessage + "        " + str(x) + ": " + str(y) + "\n\n"
        else:
            print(("Unknown", loc))
            resultMessage = resultMessage + "    " + "Unknown\n\n"
    print()

S.pack(side = "right", fill = "y")
T.pack()
T.insert(CURRENT, resultMessage)
root.mainloop()
            
