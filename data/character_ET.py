import os
import sys
sys.path.append("utils")
import re
import json

from glob import glob
from constant import VN_RE
from csv import DictReader


DATA_DIR = "DataVIN"
TEXT_TIMESTAMP_JSON = "character_et.json"


def main():
    # et_text_data = {"data": [], "timestamp": []}
    et_chr = {}

    for sub_dir in os.listdir(DATA_DIR):
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
                            text = row["Data"]
                            text = re.sub(VN_RE, '', text).lower()
                            text = text.split(':')
                            typing = text[1].strip()
                            cord = text[0].strip()
                            x_cord, y_cord, z_cord = cord.split(" ")
                            if z_cord == "0": 
                                x_cord, y_cord = float(x_cord), float(y_cord)
                        else:
                            x_cord = float(row["Data"])
                            y_cord = float(row["x"])
                            typing = row["y"].strip().lower()
                        
                        if (not len(typing)) or (typing == "EMPTY_ET_STREAM") or (typing == "START_ET_STREAM"): continue
                        if len(typing) != 1: typing = typing[-1]

                        if (typing == "␣" or typing == "▷" or typing == "⌫" or typing == "⌧" or typing == "⌂"): continue
                        if typing == "ắ": import pdb; pdb.set_trace()
                        if typing in et_chr.keys():
                            if x_cord != "none":
                                et_chr[typing].append([x_cord, y_cord])
                        else:
                            if x_cord != "none":
                                et_chr[typing] = [[x_cord, y_cord]]

            print("Electing data from {} successfully!".format(sample_path))

    with open(TEXT_TIMESTAMP_JSON, "w") as f:
        json.dump(et_chr, f)

    return 1

if __name__=="__main__":
    main()
