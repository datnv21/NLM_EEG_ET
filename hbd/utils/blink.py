import os
import mne
import pandas as pd
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils
import cv2
# import dlib
# from dlib import cuda

# this folder path
dirname = os.path.dirname(__file__)

"""
Detect blinks from ET dataframe
Return a list of timestamps
"""
def detect_blink_ET(et_df: pd.DataFrame) -> list:
    # some temporary fixes
    et_df = et_df.rename(columns={'y': 'data', 'x': 'y', 'Data': 'x'})
    et_df['TimeStampNorm'] = et_df['TimeStamp'] - et_df['TimeStamp'][0]

    # get data corresponds to typing part
    typing_start = et_df[et_df['character typing'] == 'Typing'].tail(1).index.item()
    type_df = et_df[typing_start+1:]
    type_df = type_df[type_df['character typing'] != 'MainMenu']

    # get different of eye movement
    type_df['Xdiff'] = [*type_df['x'][1:], 0] - type_df['x']
    type_df['Ydiff'] = [*type_df['y'][1:], 0] - type_df['y']

    # blink corresponds to part where there are no eye movement
    blink_df = type_df[(type_df['Xdiff'] == 0) & (type_df['Ydiff'] ==0)]
    blink_df = blink_df.reset_index()

    start_blink = 0
    blinks = []
    for idx in range(1, len(blink_df)):
        if blink_df.loc[idx, 'index'] - blink_df.loc[idx-1, 'index'] > 1:
            # a missing of eye movement for at least 3 frames (50ms) and at most 31 frames (500ms) is considered a blink
            if idx - start_blink > 4 and idx - start_blink < 31:
                blinks.append((blink_df.loc[start_blink, 'TimeStamp'], blink_df.loc[idx-1, 'TimeStamp']))
            start_blink = idx

    return blinks

"""
Convert a list of blink timestamps to mne.Annotations for EEG data
    blinks: list of blink's timestamps
    eeg_ts: timestamps of EEG data
"""
def get_blink_annotations(blinks: list, eeg_ts) -> mne.Epochs:
    onset = []
    duration = []
    eeg_start_time = eeg_ts.loc[0].TimeStamp

    for blink in blinks:
        onset.append(eeg_ts[eeg_ts['TimeStamp'] >= blink[0]].head(1).TimeStamp.item() - eeg_start_time)
        duration.append(blink[1] - blink[0])

    annos = mne.Annotations(onset, duration, 'blink')
    return annos

"""
This part is for video
"""
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def getEAR(gray, rect, predictor):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return ear

def get_video_EAR(video_path):
    detector = dlib.get_frontal_face_detector()

    model_path = os.path.join(dirname, f'../../models/face_predictor_68_landmarks.dat')
    predictor = dlib.shape_predictor(model_path)

    cap = cv2.VideoCapture(video_path)
    ears = []
    while(True):
        try:
            ret, frame = cap.read()
            img = cv2.convertScaleAbs(frame, alpha=1.4, beta=20)
            img = imutils.resize(frame, width=500)
        except Exception as e:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            ear = getEAR(gray, rect, predictor)
            ears.append(ear)

    cap.release()
    return ears

def detect_blink_video(subject, sample):
    video_path = os.path.join(dirname, f'../../data/{subject}/{sample}/FaceGesture.avi')
    ears = get_video_EAR(video_path)

    list_blink = []
    count = 0
    EYE_AR_THRESHOLD = 0.2
    EYE_AR_CONSEC_FRAMES = 2

    for i in range(len(ears)):
        if ears[i] < EYE_AR_THRESHOLD:
            count += 1
        else:
            if count >= EYE_AR_CONSEC_FRAMES and count <= EYE_AR_CONSEC_FRAMES * 2:
                list_blink.append((i-count) / 30)
            count = 0
    return list_blink
