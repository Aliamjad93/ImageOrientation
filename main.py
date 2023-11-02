
from datetime import datetime
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile
import numpy as np
import zipfile
import os
import torch
import numpy as np
import albumentations as albu

from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb
from check_orientation.pre_trained_models import create_model

from PIL import Image

app = FastAPI()

rotation_images="rotated_images"




def get_image_rotation_degree(image_path):
    try:
        # Open the image using PIL
        image = Image.open(image_path)

        # Check the Exif data for orientation information
        exif = image._getexif()
        if exif:
            orientation = exif.get(274)  # 274 corresponds to the orientation tag in Exif
            if orientation is not None:
                if orientation==1:
                    return 0
                elif orientation == 3:

                    return 180  # 180 degrees
                elif orientation == 6:

                    return 270  # 270 degrees (counterclockwise)
                elif orientation == 8:

                    return 90  # 90 degrees (counterclockwise)
            else:
                return 180

    except (AttributeError, KeyError, IndexError):
        # Handle exceptions in case there's no Exif data or orientation information
        pass

    # Default: No rotation
    return 0



def ImgOrientation(images,path_toSave,degree):
    if degree!=5 and degree!=180 and degree!=90 and degree!=270:

        return ('already rotated')

    else:




        img=Image.open(images)
        width, height = img.size


        model = create_model("swsl_resnext50_32x4d")
        model.eval()
        image = load_rgb(images)

        transform = albu.Compose([albu.Resize(height=224, width=224), albu.Normalize(p=1)], p=1)


        temp = []
        for k in [0, 1, 2, 3]:
          x = transform(image=np.rot90(image, k))["image"]
          temp += [tensor_from_rgb_image(x)]




        with torch.no_grad():
          prediction = model(torch.stack(temp)).numpy()


        '''for i in range(4):
          pred_1 = [str(round(tx, 2)) for tx in prediction[i]]
          print(pred_1)'''

        for i in range(4):
            val1 = [float(round(tx, 2)) for tx in prediction[i]]

            #fig, ax = plt.subplots(1, 1, figsize=(5, 5))  # Create a new figure and axis for each plot

            if val1[0] > 0.80:
                plt.imshow(np.rot90(image, i))
                plt.axis('off')  # Turn off the axes
                # Generate a timestamp in the format YYYYMMDDHHMMSS
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

                # Save the image with the timestamp as the filename

                filename =path_toSave+'/'+f"image_{timestamp}.jpg"


                plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
                img = Image.open(filename)
                resized_img=img.resize((width, height))
                resized_img.save(filename)

            else:
                continue









@app.post("/ImageRotation/")
async def highlight_backlit(image_directory):
    images_path=[]
    try:

        # Create directories if they don't exist
        # os.makedirs(uploaded_directory, exist_ok=True)
        # os.makedirs(image_directory, exist_ok=True)
        os.makedirs(rotation_images, exist_ok=True)

        # Save the uploaded zip file
        # zip_file_path = os.path.join(uploaded_directory, zip_file.filename)
        # with open(zip_file_path, "wb") as f:
        #     f.write(zip_file.file.read())

        # Extract the zip file
        # with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        #     zip_ref.extractall(image_directory)

        i = 0
        for root, _, files in os.walk(image_directory):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, filename)
                    images_path.append(image_path)
                    i += 1

                    pth = os.path.join(rotation_images, str(i) + '.jpg')
                    dgree=get_image_rotation_degree(image_path)


                    ImgOrientation(image_path,rotation_images,dgree)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return {'Rotated_Images':images_path}






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)




