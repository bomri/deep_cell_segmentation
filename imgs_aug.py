# to create image augmentation

import time
import os
from PIL import Image
root_data_dir = "/home/omri/omri_dl1/play_ground/data_aug"
train_csv_file = os.path.join(root_data_dir,"train.csv")
new_train_csv_file = os.path.join(root_data_dir,"train_aug.csv")


def rotate_and_save(path, deg):
    I = Image.open(os.path.join(root_data_dir, path))
    new_path = os.path.join(root_data_dir, path[:-4] + "_{0}.png".format(deg))
    Ir = I.rotate(deg).save(new_path)
    time.sleep(0.005)
    out_path = path[:-4] + "_{0}.png".format(deg)
    return out_path[1:]

def flip_rotate_and_save(path, deg):
    I = Image.open(os.path.join(root_data_dir, path))
    new_path = os.path.join(root_data_dir, path[:-4] + "_{0}_flip.png".format(deg))
    It = I.transpose(Image.FLIP_LEFT_RIGHT)
    Ir = It.rotate(deg).save(new_path)
    time.sleep(0.005)
    out_path = path[:-4] + "_{0}_flip.png".format(deg)
    return out_path[1:]


def main():
    import csv
    counter = 0
    with open(train_csv_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        new_csv_lines = []
        for row in reader:
        # for i,row in enumerate(reader):
        #     if i>10:
        #         break
            print ', '.join(row)

            new_csv_lines.append("{0},{1}".format(row[0], row[1]))
            counter += 1; print counter

            for d in [90, 180, 270]:
                new_raw = rotate_and_save(row[0], d)
                new_seg = rotate_and_save(row[1], d)
                new_csv_lines.append("{0},{1}".format(new_raw, new_seg))
                counter += 1; print counter

            # after flip
            for d in [0, 90, 180, 270]:
                new_raw = flip_rotate_and_save(row[0], d)
                new_seg = flip_rotate_and_save(row[1], d)
                new_csv_lines.append("{0},{1}".format(new_raw, new_seg))
                counter += 1; print counter

        with open(new_train_csv_file, 'wb') as newcsvfile:
            #     writer = csv.writer(newcsvfile)
            #     for line in new_csv_lines:
            #         writer.writerow(line)
            #     myfile = open(new_train_csv_file, 'wb')
            writer = csv.writer(newcsvfile, quoting=csv.QUOTE_NONE, delimiter='\n')
            writer.writerow(new_csv_lines)

if __name__ == '__main__':
    main()
