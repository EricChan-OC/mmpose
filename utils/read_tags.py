from sagemaker import get_execution_role
from sagemaker import ModelPackage
from urllib.parse import urlparse
from IPython.display import Image
from PIL import Image
import numpy as np
import boto3
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont
from io import BytesIO
import copy

def extract_read_tags(model, prediction, image, height, width):
      
    tag_names = []
    
    #may need to get move this logic to get_rekognition_annotation. 
    
    for pred in prediction['CustomLabels']:
        
        if pred['Name'] == 'Tag':
            
            bounding_box = pred['Geometry']['BoundingBox']
            left = width * bounding_box['Left']
            top = height * bounding_box['Top']
            width = width * bounding_box['Width']
            height = height * bounding_box['Height']
            
            img = Image.fromarray(image)
            canvas = copy.deepcopy(img) #in case making changes ruin the image
            
            #recognition cannot work with images less than 80 pixels
            width = max(width, 80)
            height = max(height, 80)
  
            cropped = canvas.crop((left, top, left+width, top+height)) #crop the images
            display(cropped)
        
            tags = read_tags(model,cropped)
            tag_names.append(tags)
            
        else:
            pass
    
    #we don't really need multiple labels. 
    tag = tag_names[0]
    
    return tag   

def read_tags(ocr_model, photo):
    
    """
    Triggers another rekognition model that is trained for OCR purposes
    Input will be images formulated using the trimmed coordinates of the tags
    """
    
    client=boto3.client('rekognition')
    
    print('reading tags....')
    
    im_file = BytesIO()
    photo.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes)
    
    base_64_binary = base64.decodebytes(im_b64)
    
    response = client.detect_custom_labels(Image={'Bytes': base_64_binary}, 
                                           MinConfidence=min_confidence, 
                                           ProjectVersionArn=ocr_model)
        
    tag = response['CustomLabels']
    
    if len(tag) == 0:
        tag = 'unknown_tag'
    else:
        tag = response['CustomLabels'][0]
    
    
    return tag

height = numpy_image.shape[0]
width = numpy_image.shape[1]
ocr_model='arn:aws:rekognition:us-east-1:617989056061:project/rekognition-custom-dataset/version/rekognition-custom-dataset.2021-05-05T16.53.07/1620258787633'

test = extract_read_tags(ocr_model,prediction, original_image, height, width)     