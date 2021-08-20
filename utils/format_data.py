from imutils import paths
import os
import pandas as pd

path_list_dir = []
path_ET = []
directory = '/Users/sonnguyen/Desktop/EEG/DataEEG'
list_dir = os.listdir(directory)

list_dir.remove('.DS_Store')
list_dir.remove('BV103_01')
list_dir.remove('HMI15')

def read_data(list_dir):
    for i in list_dir:
        path = os.path.join(directory, i)
        path_list_dir.append(path)

    for k, path in enumerate(path_list_dir):
        for i in list(paths.list_files(path_list_dir[k])):
            filename = os.path.basename(i)
            if filename == "ET.csv":
                path_ET.append(i)

    return path_ET

def csv_filter(path_ET):
    for k, j in enumerate(path_ET):
        x_ = []
        y_ = []
        _data = []
        _character_typing = []
        df = pd.read_csv(j)
        
        try:
            for i, value in enumerate(df['Data']):
                x, y, z = x,y,z = df['Data'][i].split(' : ')[0][3:-1].split(',')
                data = df['Data'][i].split(' : ')[1]
                character_typing = df['Data'][i].split(' : ')[2][:-2]
                x_.append(x)
                y_.append(y)
                _character_typing.append(character_typing)
                _data.append(data)
            final_data = pd.DataFrame({'TimeStamp':df['TimeStamp'],'x':x_, 'y':y_, 'Data':_data, 'character typing':_character_typing  })
            del x_
            del y_
            del _data
            del _character_typing
            new_path = os.path.splitext(j)[0][:-2]
            new_path = new_path + 'ET_new'+'.csv'
            print(new_path)
            final_data.to_csv(new_path, index = False, header=True)
        except ValueError:
            pass
        

if __name__=="__main__":
    path_ET = read_data(list_dir)
    csv_filter(path_ET)