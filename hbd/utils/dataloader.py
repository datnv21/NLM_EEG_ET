from hbd.utils import blink
# import blink
import pandas as pd
import mne
import os

"""
Load EEG and ET signals
Currently only support 1 sample of 1 subject

Return EEG data as mne.Raw, ET data as pandas.DataFrame
"""
def get_typing_signals(subject, sample):
    if subject == None:
        raise(BaseException('subject cannot be empty'))

    if sample is None:
        raise(BaseException('sample cannot be empty'))

    dirname = os.path.dirname(__file__)
    sample_path = os.path.join(dirname, f'DataVIN/{subject}/{sample}/')
    if not os.path.exists(sample_path):
        raise(BaseException(f'Folder not found: {sample_path}'))

    raw_eeg = mne.io.read_raw_edf(sample_path + 'EEG.edf')
    raw_eeg.load_data().pick(['Fp1', 'Fp2']).filter(2., 20.)
    eeg_ts = pd.read_csv(sample_path + 'EEGTimeStamp.txt')

    # get typing annotation
    try:
        typing_annotation = next(filter(lambda a: a['description'] == 'Typing', raw_eeg.annotations))
    except StopIteration:
        print('Typing annotation not found for', sample_path)

    # to offset some misalignment in typing event in EEG and ET, we reduce the duration for 2 seconds, 1 at start, 1 at end
    eeg_start = typing_annotation['onset'] + 1
    eeg_end = eeg_start + typing_annotation['duration'] - 2
    timestamp_start = typing_annotation['onset'] + eeg_ts.loc[0].item() + 1
    timestamp_end = timestamp_start + typing_annotation['duration'] - 2
    print('Extracting EEG from', eeg_start, 'to',  eeg_end, 'duration', eeg_end - eeg_start)
    print('Extracting ET from', timestamp_start, 'to', timestamp_end, 'duration', timestamp_end - timestamp_start)

    # get EEG data
    typing_eeg = raw_eeg.crop(tmin=eeg_start, tmax=eeg_end)

    # get ET dataframe
    et_df = pd.read_csv(sample_path + 'ET.csv').rename(columns={'y': 'data', 'x': 'y', 'Data': 'x'})
    et_df['TimeStampNorm'] = et_df['TimeStamp'] - et_df['TimeStamp'][0]

    # get data corresponds to typing part
    start_index = et_df[et_df['TimeStamp'] >= timestamp_start].head(1).index.item()
    end_index = et_df[et_df['TimeStamp'] >= timestamp_end].head(1).index.item()
    type_df = et_df[start_index:end_index]

    return typing_eeg, type_df

"""
Load raw EEG data and ET data of blinks
    subjects: list of subject's ids
    samples: list of samples for each subject, when samples is None, get every samples
    tmin: time before blink onset
    tmax: time after blink onset

Return mne.Epochs of blinks
"""
def load_eeg_blinks(subjects: list, samples: list=None, tmin=-0.2, tmax=0.5) -> mne.io.BaseRaw:
    if (subjects == None or type(subjects) != list):
        raise(BaseException('Invalid parameter'))

    if samples is None:
        samples = [f'sample{i}' for i in range(1, 10)]
    filepaths = [f'DataVIN/{subject}/{sample}/' for sample in samples for subject in subjects]
    if any(map(lambda f: not os.path.exists(f), filepaths)):
        raise(BaseException(f'Folder not found. List folders: {filepaths}'))

    res = []
    epochs = []

    print(f'Loading data from: {filepaths}')

    for filepath in filepaths:
        et_df = pd.read_csv(filepath + 'ET.csv')
        eeg_ts = pd.read_csv(filepath + 'EEGTimeStamp.txt', names=['TimeStamp'])

        blinks = blink.detect_blink_ET(et_df)
        annos = blink.get_blink_annotations(blinks, eeg_ts)

        eeg_raw = mne.io.read_raw_edf(filepath + 'EEG.edf', verbose=0)
        eeg_raw.load_data().filter(2., 20.)
        eeg_raw.set_annotations(annos)

        events, event_id = mne.events_from_annotations(eeg_raw)
        epochs.append(mne.Epochs(raw=eeg_raw, baseline=None, tmin=-0, tmax=0.7, events=events, event_id=event_id, verbose=0))

    return mne.concatenate_epochs(epochs)
