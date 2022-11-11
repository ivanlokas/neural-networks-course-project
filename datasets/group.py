import os
import glob
import shutil

MIN_AGE = 1
MAX_AGE = 116


def group_classes(src_path):
    dest_path = src_path + "_grouped"

    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)

    for age in range(MIN_AGE, MAX_AGE + 1):
        dest_dir = os.path.join(dest_path, str(age))
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)

        for photo in glob.iglob(os.path.join(src_path, str(age)) + "_*", recursive=True):
            name = os.path.basename(photo)
            shutil.copy(photo, dest_dir)

        if not os.listdir(dest_dir):
            os.rmdir(dest_dir)
