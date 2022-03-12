# Create your views here.
import threading
# Create your views here.
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw  # add caption by using custom font
from decouple import config
from django.contrib import messages
from django.core.mail import send_mail
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators import gzip
from django.views.generic import ListView, DetailView
from tensorflow import keras

from surveillance.face_recognizer import match_face
from surveillance.models import Violence, Camera, Person

base_model = keras.applications.mobilenet.MobileNet(input_shape=(160, 160, 3),
                                                    include_top=False,
                                                    weights='imagenet', classes=2)

model = keras.models.load_model('static/MobileNet_Model.h5')


@gzip.gzip_page
def home(request):
    object_list = Camera.objects.all()
    context = {
        'object_list': object_list,
    }
    return render(request, 'base.html', context=context)


def video_feed(request, cid):
    return StreamingHttpResponse(gen(VideoCamera(cid=cid)),
                                 content_type='multipart/x-mixed-replace;boundary=frame')


def violence_video_feed(request, cid):
    return StreamingHttpResponse(gen(ViolenceVideoCamera(cid=cid)),
                                 content_type='multipart/x-mixed-replace;boundary=frame')


class VideoCamera(object):
    def __init__(self, cid):
        self.camera = Camera.objects.get(pk=cid)
        self.video = cv2.VideoCapture(self.camera.ip)
        (self.grabbed, self.frame) = self.video.read()
        if not self.grabbed:
            self.video = cv2.VideoCapture(int(self.camera.ip))
            (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while self.grabbed:
            (self.grabbed, self.frame) = self.video.read()


class ViolenceVideoCamera(object):
    def __init__(self, cid):
        self.camera = Camera.objects.get(pk=cid)
        self.video = cv2.VideoCapture(self.camera.ip)
        (self.grabbed, self.frame) = self.video.read()
        if not self.grabbed:
            self.video = cv2.VideoCapture(int(self.camera.ip))
            (self.grabbed, self.frame) = self.video.read()
        (self.W, self.H) = (None, None)
        self.i = 0  # Video seconds number. Iteration of the while loop
        self.Q = deque(maxlen=128)
        self.frame_counter = 0  # Number of frames per second. 1 to 30
        self.frame_list = []
        self.preds = None
        self.maxprob = None
        self.output_path = ""
        self.writer = None
        self.video_array = deque(maxlen=150)
        self.after_violence = 0
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while self.grabbed:
            (self.grabbed, self.frame) = self.video.read()

            self.frame_counter += 1

            if self.W is None or self.H is None:  # Frame image width (W), height (H) from the movie
                (self.H, self.W) = self.frame.shape[:2]

            output = self.frame.copy()  # Copy the video frame as is. As an .mp4 file to save/output
            self.frame = cv2.resize(self.frame, (160, 160))  # > Convert the input array to (160, 160, 3)
            self.frame_list.append(self.frame)  # Each frame array (160, 160, 3) is appended

            if self.frame_counter >= 30:  # The moment the frame counter reaches 30. The moment when len(
                # if self.i % 5 == 0:
                frame_ar = np.array(self.frame_list, dtype=float)

                if np.max(frame_ar) > 1:  # Scaling RGB values in a NumPy array
                    frame_ar = frame_ar / 255.0

                # MobileNetExtract features from an image array at frames per second with : (1 * 30, 5, 5, 1024)
                pred_imgarr = base_model.predict(frame_ar)  # > (30, 5, 5, 1024)
                # Transform the extracted feature arrays into one-dimensional : (1, 30, 5*5*1024)
                pred_imgarr_dim = pred_imgarr.reshape(1, pred_imgarr.shape[0], 5 * 5 * 1024)  # > (1, 30, 25600)
                # Store the predictive value of whether each frame is violent or not at 0
                self.preds = model.predict(pred_imgarr_dim)  # > (True, 0.99) : (Violence, Probability of Violence)
                print(f'preds:{self.preds}')
                self.Q.append(self.preds)  # > Deque Add predicted values to Q like a list

                # The average of the probability of violence over the past 5 seconds is taken as the result.
                if self.i < 5:
                    results = np.array(self.Q)[:self.i].mean(axis=0)
                else:
                    results = np.array(self.Q)[(self.i - 5):self.i].mean(axis=0)

                print(f'Results = {results}')  # > ex : (0.6, 0.650)

                # Maximum Violence Probability Value from Prediction Results
                self.maxprob = np.max(results)  # > Choose the highest value
                print(f'Maximum Probability : {self.maxprob}')
                print('')

                rest = 1 - self.maxprob
                diff = self.maxprob - rest
                th = 60

                if diff > 0.60:
                    th = diff

                if self.after_violence > 1:
                    threading.Thread(target=save_violence,
                                     args=(self.video_array.copy(), self.W, self.H, self.camera)).start()
                    self.video_array.clear()
                    self.after_violence = 0

                self.frame_list = []
                self.frame_counter = 0  # > Reset to frame_counter=0 since 1 second (30 frames) has elapsed
                self.i += 1  # > 1 second elapsed meaning

            font1 = ImageFont.truetype('static/fonts/Raleway-ExtraBold.ttf', int(0.05 * self.W))
            font2 = ImageFont.truetype('static/fonts/Raleway-ExtraBold.ttf', int(0.1 * self.W))

            if self.preds is not None and self.maxprob is not None:  # from the time the forecast is generated
                if (self.preds[0][1]) < th:  # > Normal if the probability of violence is less than th
                    text1_1 = 'Normal'
                    text1_2 = '{:.2f}%'.format(100 - (self.maxprob * 100))
                    img_pil = Image.fromarray(output)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((int(0.025 * self.W), int(0.025 * self.H)), text1_1, font=font1, fill=(0, 255, 0, 0))
                    draw.text((int(0.025 * self.W), int(0.095 * self.H)), text1_2, font=font2, fill=(0, 255, 0, 0))
                    output = np.array(img_pil)
                    if self.after_violence > 0:
                        self.after_violence += 1
                else:  # > If the probability of violence is greater than th, it is treated as violence.
                    text2_1 = 'Violence Alert!'
                    text2_2 = '{:.2f}%'.format(self.maxprob * 100)
                    img_pil = Image.fromarray(output)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((int(0.025 * self.W), int(0.025 * self.H)), text2_1, font=font1, fill=(0, 0, 255, 0))
                    draw.text((int(0.025 * self.W), int(0.095 * self.H)), text2_2, font=font2, fill=(0, 0, 255, 0))
                    output = np.array(img_pil)
                    self.after_violence = 1

            # Save subtitled video as writer
            # if writer is None:
            #     writer = cv2.VideoWriter(output_path, -1, 30, (self.W, self.H), True)
            # writer.write(output)  # Save the output object as output_path
            # Show the output in a new window
            # cv2.imshow('This is output', output)
            self.frame = output
            self.video_array.append(output)
            # print(self.frame_counter)
            # cv2.waitKey(round(1000 / fps))


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'


def save_violence(video_array, w, h, camera):
    path = 'media/violences/' + str(time.time()) + '.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(path, fourcc, 30, (w, h), True)
    violence = Violence()
    violence.camera = camera
    message = 'Dear concern,\n\nViolence detected in your camera ' + str(camera.pk) + '. Involved persons are:'
    violence.save()

    for frame in video_array:
        writer.write(frame)
        faces = match_face(frame)
        for face in faces:
            person = Person.objects.filter(pk=face).first()
            if person:
                violence.involved_persons.add(person)

    writer.release()
    violence.video.name = path
    violence.save()
    persons = violence.involved_persons.all()

    for person in persons:
        message += ' ' + person.name + ','

    print('save video finished')
    message += '. Video clip is saved in database naming ' + path + '. Please check admin site for more information.' \
                                                                    '\n\nRegards\nTeam Violence'
    send_mail(
        subject='New Violence Detected',
        message=message,
        from_email=config('EMAIL'),
        recipient_list=['jaber.siam@northsouth.edu', ],
    )


class CameraListView(ListView):
    def get_queryset(self):
        return Camera.objects.all()


class CameraDetailsView(DetailView):
    model = Camera
