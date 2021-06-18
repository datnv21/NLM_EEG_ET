import os
import sys
sys.path.append("utils")

import mne


DATA_DIR     = "DataVIN"
N_COMPONENTS = 32


def ica_instance():
    return mne.preprocessing.ICA(n_components=N_COMPONENTS, random_state=42)


def find_bad_indices(ica, raw_filter=None):
    bad_indices_eog = {}
    bad_indices_ecg = {}
    optimal_bad_indices = []
    indices_count = {}

    for i in range(N_COMPONENTS):
        indices_count[i] = 0

    for ch in ch_names:
        bad_indices_eog[ch] = []
        bad_indices_ecg[ch] = []

    for ch in ch_names:
        bad_idx, scores = ica.find_bads_eog(raw_filter, ch, threshold=2.0)
        bad_indices_eog[ch] = bad_idx
        
        bad_idx_, scores_ = ica.find_bads_ecg(raw_filter, ch, threshold="auto")
        bad_indices_ecg[ch] = bad_idx_

        for i in bad_idx: indices_count[i] += 1
        for j in bad_idx_: indices_count[i] += 1

    for k, v in indices_count.items(): 
        if v > (N_COMPONENTS // 4): optimal_bad_indices.append(k)
    
    return optimal_bad_indices


def main():
    global ch_names
    ch_names = None

    for sub_dir in os.listdir(DATA_DIR):
        if "HMI" not in sub_dir: continue
        sub_path = os.path.join(DATA_DIR, sub_dir)

        for sample in os.listdir(sub_path):
            sample_path = os.path.join(sub_path, sample)

            if os.path.isdir(sample_path):
                data_file = os.path.join(sample_path, "EEG.edf")
                raw_eeg = mne.io.read_raw_edf(data_file, preload=True)
                raw_eeg_data = raw_eeg.get_data()

                raw_eeg_bandpass = raw_eeg.filter(0.5, 40)
                if ch_names is None: ch_names = raw_eeg_bandpass.ch_names

                ica = ica_instance()
                ica.fit(raw_eeg_bandpass)
                optimal_bad_indices  = find_bad_indices(ica, raw_filter=raw_eeg_bandpass)
                raw_eeg_bandpass = ica.apply(raw_eeg_bandpass.copy(), exclude=optimal_bad_indices)
                try:
                    raw_eeg_bandpass.save(os.path.join(sample_path, "EEG_preprocessed.fif"), overwrite=True)
                except:
                    print("Error --- ", sample_path)

                print("{} EEG data preprocess successfully!".format(sample_path))
    return 1


if __name__=="__main__":
    main()
