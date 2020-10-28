## TensorFlow Object Detection using SSD based MobileNet V2
This model is an `SSD` based pre-trained [MobileNet V2](https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1) feature
extractor which is trained on [Open Images V4](https://storage.googleapis.com/openimages/web/download_v4.html)
dataset. It can detect up to 600 classes and outputs at most 100 detections. 
 
### Performance
As a performance metric mean Average Precision (mAP) is used and computed as `0.34` on 
Open Images V4 test set. Note that this is a light weight model and suitable for real-time feature extraction. If you 
need higher accuracy and execution time is not an issue then you can consider using `Faster-RCNN` based
[inception_resnet_v2](https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1).


### Endpoint Exposure
In object_detection.ipynb file, annotate_base64_image(body) takes a dictionary in `{image: "<Base64 Image String>", type: "<Image Type e.g. JPEG>"}` format and
returns
```
{
 image: "<Annotated Base64 Image String>",
 type: "<Image Type String>",
 objects: ["<Detected object 0>", .., "<Detected object n>"],
 scores: ["<Detected object score 0>", ..., "<Detected object score n>"],
 detection_class_entities: [<detection class names as Freebase MIDs>, .., ]
 detection_class_names: [<human-readable detection class name 0>, .., ]
 detection_class_labels: [<class indices 0>, .., ]
 detection_boxes: [<[ymin, xmin, ymax, xmax]>, .. <[ymin, xmin, ymax, xmax]>]
}
```

### Requirements
Users will have to provide `Application/Json` as `Content-Type`. The accepted image
types for this model are `PNG`, or `JPEG`.

### Steps For Local Testing
1. Install gcloud from [here](https://cloud.google.com/sdk/install)
2. Authenticate to gcloud with ```gcloud auth configure-docker```
3. Pull docker image of pybuilder ```docker pull eu.gcr.io/dploy-registry/build-wrappers/python-notebook-dpl:3.7```. This
image will use `annotate_base64_image(body)` method as endpoint because it is annotated with
`#' @dploy endpoint predict`
4. Change current directory to `py_tensorflow_notebook_object_detection`.
5. Run docker image:
    ``` dockerfile
    docker run \
    -p 8080:8080 -v "$(pwd)":/model \
    eu.gcr.io/dploy-registry/build-wrappers/python-notebook-dpl:3.7 \
    serve \
    --entrypoint object_detection.ipynb \
    --reqs requirements.txt \
    --location "./model"
   ```
6. Convert your base image to base64 string. Type ```base64 <image.jpeg> | tr -d '\n' | pbcopy``` for mac or type
 ```base64 <image.jpeg> | xclip``` or ```base64 <image.jpeg> | xsel``` for linux distributions if `xclip` or `xsel` is 
 installed. This code will encode image.jpeg in base64 and will copy it to your clipboard.
7. Now, you are ready to make an API call, run:
    ```curl
    curl --location --request POST 'localhost:8080/predict' \
    --header 'Content-Type: application/json' \
    --header 'Accept: application/json' \
    --data-raw '{
        "image": "<!!!!!PASTE YOUR CLIPBOARD TO HERE!!!!!>",
        "type": "png"
       }'
   ```
8. You will get the response in following format
    ```
    {
     image: "<Annotated Base64 Image String>",
     type: "<Image Type String>",
     objects: ["<Detected object 0>", .., "<Detected object n>"],
     scores: ["<Detected object score 0>", ..., "<Detected object score n>"],
     detection_class_entities: [<detection class names as Freebase MIDs>, .., ]
     detection_class_names: [<human-readable detection class name 0>, .., ]
     detection_class_labels: [<class indices 0>, .., ]
     detection_boxes: [<[ymin, xmin, ymax, xmax]>, .. <[ymin, xmin, ymax, xmax]>]
    }
    ```
