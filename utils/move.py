import os
import shutil

folders = os.listdir("filelists/inat/val")
for folder in folders:
    full_folder = os.path.join("filelists/inat/val", folder)
    files = os.listdir(full_folder)
    print("Processing %s" % (full_folder))

    for file in files:
        full_file = os.path.join(full_folder, file)
        shutil.move(full_file, "filelists/inat/images/" + file)
    
    shutil.rmtree(full_folder)