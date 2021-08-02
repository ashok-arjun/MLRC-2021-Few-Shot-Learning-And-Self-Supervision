from PIL import Image
import os 

images = os.listdir("filelists/aircrafts/images")

for path in images:
    image = Image.open(os.path.join("filelists/aircrafts/images", path)).convert('RGB')
    image = image.crop((0,0,image.size[0],image.size[1]-20))
    image.save(os.path.join("filelists/aircrafts/images", path))