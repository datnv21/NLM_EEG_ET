import os
import sys
sys.path.append("utils")
import re
import json

from glob import glob
from constant import VN_RE
from csv import DictReader


DATA_DIR = "DataVIN"
TEXT_TIMESTAMP_JSON = "text_timestamp.json"


def main():
    # et_text_data = {"data": [], "timestamp": []}
    et_text = {"text": [], "timestamp": []}

    for sub_dir in os.listdir(DATA_DIR):
        # if "HMI" not in sub_dir: continue
        if sub_dir == ".DS_Store": continue
        sub_path = os.path.join(DATA_DIR, sub_dir)

        for sample in os.listdir(sub_path):
            sample_path = os.path.join(sub_path, sample)
            if os.path.isdir(sample_path):
                data_file = os.path.join(sample_path, "ET.csv")

                with open(data_file, "r") as f:
                    csv_dict_reader = DictReader(f)
                    for idx, row in enumerate(csv_dict_reader):
                        if len(row.keys()) == 2:
                            timestamp = row["TimeStamp"]
                            text = row["Data"]
                            text = re.sub(VN_RE, '', text)
                            text = text.split(':')
                            sentence = text[-1].strip()
                        else:
                            timestamp = row["TimeStamp"]
                            text = row["character typing"]
                            sentence = text.strip()
                        
                        if len(sentence.split(" ")) > 1:
                            if sentence not in et_text["text"] and len(sentence.split(" ")[-1])>=2:
                                et_text["text"].append(sentence)
                                et_text["timestamp"].append(timestamp)
            
            print("Electing data from {} successfully!".format(sample_path))

    with open(TEXT_TIMESTAMP_JSON, "w") as f:
        json.dump(et_text, f)

    return 1

if __name__=="__main__":
    main()
