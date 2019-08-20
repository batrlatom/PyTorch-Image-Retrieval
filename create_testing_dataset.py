import os, glob



import sys
import shutil
import glob
path = sys.argv[1]
taget_folder_name = sys.argv[2]
import random
import os
from shutil import copyfile
from contextlib import ExitStack
import random

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def copy_file_rename(src, dst):
    shutil.copy(src, dst)


def main():

    import shutil

    #print(taget_folder_name)

    shutil.rmtree(taget_folder_name+"/reference", ignore_errors=True)
    shutil.rmtree(taget_folder_name+"/query", ignore_errors=True)
    os.makedirs(taget_folder_name+"/query")
    os.makedirs(taget_folder_name+"/reference")

    subs = get_immediate_subdirectories(path)

    subs = random.sample(subs, 10)

    for sub in subs:
        dir_name = sub
        sub = os.path.join(path, sub)
        #print(sub.split('_')[-1])

        files = glob.glob( os.path.join(sub, '*.jpg'))
        files = random.sample(files, 3)

        for index, file in enumerate(files):
            #print(file)
            if random.random() < 0.8:
                copy_file_rename(file, os.path.join(taget_folder_name, 'reference', sub.split('.')[-1] + '_' + str(index) + '.jpg'))
            else:
                copy_file_rename(file, os.path.join(taget_folder_name,'query', sub.split('.')[-1] + '_' + str(index) + '.jpg'))




if __name__== "__main__":
    main()
