from re import match, DOTALL
from os import listdir, read, write
from json import load

config_path = "./keras-yolo3-master/config.json"
with open(config_path) as f:
    config = load(f)

annotations_path = config["train"]["train_annot_folder"]
images_path = config["train"]["train_image_folder"]

for file_path in listdir(annotations_path):
    with open(annotations_path+file_path) as file:
        content = file.read()
    old_path = match(r".*<path>(.*)</path", content, flags=DOTALL).group(1)
    file_name = old_path.split("\\")[-1]
    new_path = images_path + file_name
    content = content.replace(old_path, new_path)
    with open(annotations_path+file_path, "w") as file:
        file.write(content)
