import os
import shutil

"""miniImageNet"""

root = "./filelists/miniImagenet/images"

subdirectories = [root + x for x in os.listdir(root)]

for subdir in subdirectories:
    files = [subdir + "/" + x for x in os.listdir(subdir)]
    for file in files:
        shutil.move(file, root)
    os.rmdir(subdir)
