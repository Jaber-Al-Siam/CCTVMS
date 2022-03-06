import cv2
import face_recognition
import numpy as np
import pickle

_KNOWN_FACE_ENCODINGS_FILE = 'known-face-encodings.pkl'
_KNOWN_FACE_IDS_FILE = 'known-face-ids.pkl'


# the two lists are parallel. meaning,
# face_encoding at index 0 of 'known_face_encodings' belongs to the face_id at index 0 of 'known_face_ids'
# known_face_encodings = []
# known_face_ids = []


def _read_list_of_objects_from_file(filepath):
    obj_list = []
    try:
        file = open(filepath, "rb")
        obj_list = pickle.load(file)
        file.close()
    except Exception as e:
        print(f'error inside _read_list_of_objects_from_file() function. filepath={filepath}, error={str(e)}')

    return obj_list


def _save_list_of_objects_to_file(filepath, obj_list=[]):
    try:
        file = open(filepath, "wb")
        pickle.dump(obj_list, file)
        file.close()
    except Exception as e:
        print(f'error inside _save_list_of_objects_to_file() function. filepath={filepath}, error={str(e)}')
        return False
    return True


def save_face_from_image_path(single_face_image_path, face_id):
    try:
        known_face_encodings = _read_list_of_objects_from_file(_KNOWN_FACE_ENCODINGS_FILE)
        known_face_ids = _read_list_of_objects_from_file(_KNOWN_FACE_IDS_FILE)

        face_image = face_recognition.load_image_file(single_face_image_path)
        face_encoding = face_recognition.face_encodings(face_image)[0]

        known_face_encodings.append(face_encoding)
        known_face_ids.append(face_id)

        _save_list_of_objects_to_file(filepath=_KNOWN_FACE_ENCODINGS_FILE, obj_list=known_face_encodings)
        _save_list_of_objects_to_file(filepath=_KNOWN_FACE_IDS_FILE, obj_list=known_face_ids)
    except Exception as e:
        print(f'error inside save_face() function. image_path={single_face_image_path}, error={str(e)}')


def save_face_from_video_path(video_path, face_id):
    try:
        known_face_encodings = _read_list_of_objects_from_file(_KNOWN_FACE_ENCODINGS_FILE)
        known_face_ids = _read_list_of_objects_from_file(_KNOWN_FACE_IDS_FILE)
        cap = cv2.VideoCapture(video_path)
        grabbed, frame = cap.read()
        frame_count = 0

        while grabbed:
            face_encoding = face_recognition.face_encodings(frame)
            if len(face_encoding) > 0:
                known_face_encodings.append(face_encoding[0])
                known_face_ids.append(face_id)
                print(frame_count)
            grabbed, frame = cap.read()
            frame_count += 1

        _save_list_of_objects_to_file(filepath=_KNOWN_FACE_ENCODINGS_FILE, obj_list=known_face_encodings)
        _save_list_of_objects_to_file(filepath=_KNOWN_FACE_IDS_FILE, obj_list=known_face_ids)
        print('Finish')
    except Exception as e:
        print(f'error inside save_face() function, error={str(e)}')


def save_face(face_image, face_id):
    try:
        known_face_encodings = _read_list_of_objects_from_file(_KNOWN_FACE_ENCODINGS_FILE)
        known_face_ids = _read_list_of_objects_from_file(_KNOWN_FACE_IDS_FILE)

        face_encoding = face_recognition.face_encodings(face_image)[0]

        known_face_encodings.append(face_encoding)
        known_face_ids.append(face_id)

        _save_list_of_objects_to_file(filepath=_KNOWN_FACE_ENCODINGS_FILE, obj_list=known_face_encodings)
        _save_list_of_objects_to_file(filepath=_KNOWN_FACE_IDS_FILE, obj_list=known_face_ids)
    except Exception as e:
        print(f'error inside save_face() function, error={str(e)}')


def match_face_from_image_path(image_path):
    # load known_face_encodings and known_face_ids from file
    known_face_encodings = _read_list_of_objects_from_file(_KNOWN_FACE_ENCODINGS_FILE)
    known_face_ids = _read_list_of_objects_from_file(_KNOWN_FACE_IDS_FILE)

    detected_face_ids = []

    face_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(face_image)
    unknown_face_encodings = face_recognition.face_encodings(face_image, face_locations)

    for unknown_face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)

        face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)
        best_match_index = np.argmin(face_distances)  # get index of minimum face_distance

        if matches[best_match_index]:
            detected_face_id = known_face_ids[best_match_index]
            detected_face_ids.append(detected_face_id)

        else:
            detected_face_ids.append(-1)

    return detected_face_ids


def match_face(image):
    # load known_face_encodings and known_face_ids from file
    known_face_encodings = _read_list_of_objects_from_file(_KNOWN_FACE_ENCODINGS_FILE)
    known_face_ids = _read_list_of_objects_from_file(_KNOWN_FACE_IDS_FILE)

    detected_face_ids = []

    face_locations = face_recognition.face_locations(image)
    unknown_face_encodings = face_recognition.face_encodings(image, face_locations)

    for unknown_face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)

        face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)
        best_match_index = np.argmin(face_distances)  # get index of minimum face_distance

        if matches[best_match_index]:
            detected_face_id = known_face_ids[best_match_index]
            detected_face_ids.append(detected_face_id)

        else:
            detected_face_ids.append(-1)

    return detected_face_ids
