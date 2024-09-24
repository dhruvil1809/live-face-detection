import os
import face_recognition
import pickle

def load_known_faces(directory):
    known_face_encodings = []   
    known_face_names = []

    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(image_path)
            
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                known_face_encodings.append(face_encodings[0])
                name_parts = os.path.splitext(filename)[0].replace('_', ' ').split()[:2]
                name = ' '.join(name_parts) 
                known_face_names.append(name)  # Use filename as the name

    return known_face_encodings, known_face_names

known_faces_directory = 'known_faces'

known_face_encodings, known_face_names = load_known_faces(known_faces_directory)
print(known_face_encodings, known_face_names)

def save_known_faces_to_file(known_face_encodings, known_face_names, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

save_known_faces_to_file(known_face_encodings, known_face_names, 'known_faces.pkl')
