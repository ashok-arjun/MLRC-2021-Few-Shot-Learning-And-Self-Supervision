import os
import json
import shutil

json_files = list(os.listdir("filelists/domain"))

all_req_files = []

for file in json_files:
    f = open("filelists/domain/" + file)
    image_names = json.load(f)["image_names"]
    all_req_files.extend(image_names)

all_req_files = list(set(all_req_files))

os.makedirs("filelists/req_domain_files", exist_ok=True)
os.makedirs("filelists/req_domain_files/inat/images", exist_ok=True)
os.makedirs("filelists/req_domain_files/open-images/validation", exist_ok=True)

print(len(all_req_files))

for file in all_req_files:
    type_folder = file.split("/")[1] + "/" + file.split("/")[2] + "/" + file.split("/")[3]
    shutil.copy(file, os.path.join("filelists/req_domain_files", type_folder))
