import io
from PIL import Image as im
import torch
from . models import *
from django.shortcuts import render, redirect
from .forms import MediaUploadForm

# def home(request):
#     form = ImageUploadForm()
#     if request.method == 'POST':
#         form = ImageUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             form.save()
#             # return redirect('home')
#             uploaded_img_qs = UploadedImage.objects.filter().last()
#             img_bytes = uploaded_img_qs.image.read()
#             img = im.open(io.BytesIO(img_bytes))

#             path_hubconfig = "yolov5-master"
#             # path_weightfile = "one_best.pt" # for one person
#             path_weightfile = "best.pt" # for group


#             model = torch.hub.load(path_hubconfig, 'custom',
#                                path=path_weightfile, source='local')
            
#             results = model(img, size=640)
#             results.render()
#             for img in results.ims:
#                 img_base64 = im.fromarray(img)
#                 img_base64.save("media/yolo_out/image0.jpg", format="JPEG")

#             inference_img = "/media/yolo_out/image0.jpg"

#             form = ImageUploadForm()
#             context = {
#                 "form": form,
#                 "inference_img": inference_img,
#                 "uploaded_img": uploaded_img_qs.image.url 
#             }
#             return render(request, 'index.html', context)
#     else:
#         form = ImageUploadForm()
#     return render(request, 'index.html', {'form': form})

import cv2
import face_recognition
import numpy as np
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import pickle 
import numpy
from django.http import HttpResponseServerError

def load_known_faces_from_file(file_path):
    with open(file_path, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces_from_file('known_faces.pkl')

colors = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 0, 255),(128, 128, 128),
          (255, 165, 0),(128, 0, 128),(0, 128, 128),(255, 192, 203),(255, 215, 0)]


# for video

def process_video(video_path):
    video_capture = cv2.VideoCapture(video_path)

    face_locations = []
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  

        frame_count += 1
        if frame_count % 2 == 0:  # Process every other frame
            continue

        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            highest_confidence = 0
            name = 'Unknown'
            for i, (match, distance) in enumerate(zip(matches, face_distances)):
                if match:
                    confidence = int((1 - distance) * 100)
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        name = known_face_names[i]

            if highest_confidence >= 50: # Set min confidence
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                text = f"{name} {highest_confidence}%"
                text_size = cv2.getTextSize(text, font, 1.0, 1)[0]
                text_width = text_size[0] + 10 
                text_height = text_size[1] + 10
                cv2.rectangle(frame, (left, top - text_height), (left + text_width, top), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, text, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Break the loop if 'q' is pressed

    video_capture.release()
    cv2.destroyAllWindows()

# for web camera

def use_webcam():
    video_capture = cv2.VideoCapture(0)  # Capture video from webcam

    face_locations = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  

        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            highest_confidence = 0
            name = 'Unknown'
            for i, (match, distance) in enumerate(zip(matches, face_distances)):
                if match:
                    confidence = int((1 - distance) * 100)
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        name = known_face_names[i]

            if highest_confidence >= 50:  # Set min confidence
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                text = f"{name} {highest_confidence}%"
                text_size = cv2.getTextSize(text, font, 1.0, 1)[0]
                text_width = text_size[0] + 10 
                text_height = text_size[1] + 10
                cv2.rectangle(frame, (left, top - text_height), (left + text_width, top), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, text, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Break the loop if 'q' is pressed

    video_capture.release()
    cv2.destroyAllWindows()



def home(request):
    form = MediaUploadForm()
    if request.method == 'POST':
        try:
            form = MediaUploadForm(request.POST, request.FILES)
            if form.is_valid():
                uploaded_media = form.save()
                media_path = uploaded_media.media.path

                if uploaded_media.media.name.endswith('.mp4'):
                    process_video(media_path) 
                    cv2.destroyAllWindows()

                else:
                    image = face_recognition.load_image_file(media_path)
                    face_locations = face_recognition.face_locations(image)
                    
                    if face_locations:
                        pil_image = Image.open(media_path)
                        draw = ImageDraw.Draw(pil_image)
                        
                        face_encodings = face_recognition.face_encodings(image, face_locations)
                        
                        color_index = 0 
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                            highest_confidence = 0
                            name = 'Unknown'
                            for i, (match, distance) in enumerate(zip(matches, face_distances)):
                                if match:
                                    confidence = int((1 - distance) * 100)
                                    if confidence > highest_confidence:
                                        highest_confidence = confidence
                                        name = known_face_names[i]
                            
                            if highest_confidence >= 50:  # Set min confidence
                                draw.rectangle(((left, top), (right, bottom)), outline=colors[color_index], width=2)
                                
                                font_size = 24
                                font = ImageFont.truetype("Signika-VariableFont_GRAD,wght.ttf", font_size)  
                                text = f"{name} {confidence}%"
                                text_width, text_height = draw.textsize(text, font=font)
                                draw.rectangle(((left, top - 25), (left + text_width, top - 25 + text_height)), fill=colors[color_index])
                                draw.text((left, top - 25), text, fill=(255,255,255), font=font)
                                
                            color_index = (color_index + 1) % len(colors)
                        
                        del draw
                        
                        recognized_img_path = 'media/recognized_img.jpg'
                        pil_image.save(recognized_img_path)
                        
                        context = {
                            "form": form,
                            "recognized_img": recognized_img_path
                        }
                        cv2.destroyAllWindows()
                        return render(request, 'index.html', context)
                    else:
                        context = {
                            "form": form,
                            "detected_name": "No face detected"
                        }
                        cv2.destroyAllWindows()
                        return render(request, 'index.html', context)
        except Exception as e:
            return HttpResponseServerError("An error occurred: {}".format(str(e)))
    else:
        form = MediaUploadForm()
    return render(request, 'index.html', {'form': form})


def activate_webcam(request):
    use_webcam()  
    return render(request, 'index.html')