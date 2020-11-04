import requests
import os
import io
from PIL import Image
import pprint

img_loc = os.path.join('..', 'images', 'golf.png')
headers = {'content-type': 'image/png',
           # Use your own token by calling `dploy token generate`!
           'Authorization': 'DployToken NDg4YmMyNzUtMTI2ZS11M2JmLTk0YjItYmVlMjUx'
                            'YWNlYjc4OlJNR1g0Mk5NSMVIUUpMSElJR0FBM0g0M1dB'
           }

# Annotated image
# Use your own model version URL!
annotated_img_url = 'https://object-detection-mobilenetv2-mx88d5k2u4p1-v1.users.dploy.ai/annotated_image'
img = open(img_loc, 'rb').read()
bin_image = requests.post(annotated_img_url, data=img, headers=headers).content
Image.open(io.BytesIO(bin_image)).show()

# Detected objects
# Use your own model version URL!
detect_objects_url = 'https://object-detection-mobilenetv2-mx88d5k2u4p1-v1.users.dploy.ai/detect_objects'
img = open(img_loc, 'rb').read()
detected_objects = requests.post(detect_objects_url, data=img, headers=headers).text
pp = pprint.PrettyPrinter(indent=4)
dict_detected_objects = eval(detected_objects)
pp.pprint(dict_detected_objects)
