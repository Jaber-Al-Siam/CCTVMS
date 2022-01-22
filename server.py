from collections import deque

import cv2  # openCV 4.5.1
import numpy as np
from PIL import Image, ImageFont, ImageDraw  # add caption by using custom font
from skimage.transform import resize
from tensorflow import keras

base_model = keras.applications.mobilenet.MobileNet(input_shape=(160, 160, 3),
                                                    include_top=False,
                                                    weights='imagenet', classes=2)

model = keras.models.load_model('static/MobileNet_Model.h5')
input_path = 0
output_path = 'output04.mp4'

vid = cv2.VideoCapture(input_path)
fps = vid.get(cv2.CAP_PROP_FPS)  # recognize frames per secone(fps) of input_path video file.
print(f'fps : {fps}')  # print fps.

writer = None
(W, H) = (None, None)
i = 0  # number of seconds in video = The number of times that how many operated while loop .
Q = deque(maxlen=128)

video_frm_ar = np.zeros((1, int(fps), 160, 160, 3), dtype=float)  # frames
frame_counter = 0  # frame number in 1 second. 1~30
frame_list = []
preds = None
maxprob = None

# . While loop : Until the end of input video, it read frame, extract features, predict violence True or False.
# ----- Reshape & Save frame img as (30, 160, 160, 3) Numpy array  -----
while True:
    frame_counter += 1
    grabbed, frm = vid.read()  # read each frame img. grabbed=True, frm=frm img. ex: (240, 320, 3)

    if not grabbed:
        print('There is no frame. Streaming ends.')
        break

    if fps != 30:
        print('Please set fps=30')
        break

    if W is None or H is None:  # W: width, H: height of frame img
        (H, W) = frm.shape[:2]

    output = frm.copy()  # It is necessary for streaming captioned output video, and to save that.

    frame = resize(frm, (160, 160, 3))  # > Resize frame img array to (160, 160, 3)
    frame_list.append(frame)  # Append each frame img Numpy array : element is (160, 160, 3) Numpy array.

    if frame_counter >= fps:  # fps=30 et al
        frame_ar = np.array(frame_list, dtype=np.float16)  # > (30, 160, 160, 3)
        frame_list = []

        if np.max(frame_ar) > 1:  # Scaling RGB value in Numpy array
            frame_ar = frame_ar / 255.0

        pred_imgarr = base_model.predict(frame_ar)
        pred_imgarr_dim = pred_imgarr.reshape(1, pred_imgarr.shape[0], 5 * 5 * 1024)

        preds = model.predict(pred_imgarr_dim)  # > (True, 0.99) : (Violence True or False, Probability of Violence)
        print(f'preds:{preds}')
        Q.append(preds)  # > Deque Q

        # Predict Result : Average of Violence probability in last 5 second
        if i < 5:
            results = np.array(Q)[:i].mean(axis=0)
        else:
            results = np.array(Q)[(i - 5):i].mean(axis=0)

        print(f'Results = {results}')  # > ex : (0.6, 0.650)

        maxprob = np.max(preds)  # > Select Maximum Probability
        print(f'Maximum Probability : {maxprob}')
        print('')

        rest = 1 - maxprob  # Probability of Non-Violence
        diff = maxprob - rest  # Difference between Probability of Violence and Non-Violence's
        th = 60

        if diff > 0.60:
            th = diff  # ?? What is supporting basis?

        frame_counter = 0  # > Initialize frame_counter to 0
        i += 1  # > 1 second elapsed

        # When frame_counter>=30, Initialize frame_counter to 0, and repeat above while loop.

    # ----- Setting caption option of output video -----
    # Renewed caption is added every 30 frames(if fps=30, it means 1 second.)
    font1 = ImageFont.truetype('static/fonts/Raleway-ExtraBold.ttf', 24)  # font option
    font2 = ImageFont.truetype('static/fonts/Raleway-ExtraBold.ttf', 48)  # font option

    if preds is not None and maxprob is not None:
        if (preds[0][1]) < th:  # > if violence probability < th, Violence=False (Normal, Green Caption)
            text1_1 = 'Normal'
            text1_2 = '{:.2f}%'.format(100 - (maxprob * 100))
            img_pil = Image.fromarray(output)
            draw = ImageDraw.Draw(img_pil)
            draw.text((int(0.025 * W), int(0.025 * H)), text1_1, font=font1, fill=(0, 255, 0, 0))
            draw.text((int(0.025 * W), int(0.095 * H)), text1_2, font=font2, fill=(0, 255, 0, 0))
            output = np.array(img_pil)

        else:  # > if violence probability > th, Violence=True (Violence Alert!, Red Caption)
            text2_1 = 'Violence Alert!'
            text2_2 = '{:.2f}%'.format(maxprob * 100)
            img_pil = Image.fromarray(output)
            draw = ImageDraw.Draw(img_pil)
            draw.text((int(0.025 * W), int(0.025 * H)), text2_1, font=font1, fill=(0, 0, 255, 0))
            draw.text((int(0.025 * W), int(0.095 * H)), text2_2, font=font2, fill=(0, 0, 255, 0))
            output = np.array(img_pil)

            # Save captioned video file by using 'writer'
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (W, H), True)

    cv2.imshow('This is output', output)  # View output in new Window.
    # writer.write(output)  # Save output in output_path

    key = cv2.waitKey(round(1000 / fps))  # time gap of frame and next frame
    if key == 27:  # If you press ESC key, While loop will be breaked and output file will be saved.
        print('ESC is pressed. Video recording ends.')
        break

print('Video recording ends. Release Memory.')  # Output file will be saved.
writer.release()
vid.release()
cv2.destroyAllWindows()
