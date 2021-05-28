from sagemaker import ModelPackage
import sagemaker as sage
from sagemaker import get_execution_role
from sagemaker import ModelPackage
from urllib.parse import urlparse
from IPython.display import Image
from PIL import Image
import numpy as np
import boto3
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont
import os
import io
import multiprocessing as mp
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont
from io import BytesIO
import copy



def determine_color(name, BGR=False):
    
    if name == 'Tag':
        color = '#00FFFF'
    elif name == 'Head':
        color = '#376EAF'
    elif name == 'Knee':
        color = '#FFA500'
    elif name == 'Hoof':
        color = '#fffb00'
    elif name == 'Cow' or name == 'Sheep' or name == 'Horse':
        color = '#00FF00'
    elif name == 'Tail':
        color = '#49987A'
    elif name == 'Side Left':
        color = '#FF00FF'
    elif name == "Side Right":
        color = '#800080'
    elif name == "Head Left":
        color = '#DC143C'
    elif name == "Head Right":
        color = '#DB7093'
    else:
        color = '#000000'
    
    if BGR == True:
        color = ImageColor.getcolor(color, "RGB")
        #permute
        color = (color[2], color[1], color[0])    
    
    return color



def draw_animal_count(response, target_animal = 'Cow', frame=None):

    """
    A function that takes PIL image and draws the count as string from rekognition response.
    Returns a full image as PIL
    """
    
    count = 0

    for label in response['CustomLabels']:

        if label['Name'] == target_animal:
            count += 1
            
    image = copy.deepcopy(frame)
    number_of_animal = target_animal + ': ' + str(count)
    font = ImageFont.truetype('/usr/share/fonts/default/Type1/n019004l.pfb', 50)
    x, y = image.size
    text_width, text_height = font.getsize(number_of_animal)
    color = '#E4E1E1'
    
    button_size = (text_width +20, text_height +20)
    button_img = Image.new('RGBA', button_size, color)
    button_draw = ImageDraw.Draw(button_img)
    button_draw.text((10, 10), str(number_of_animal), fill ='#000000', font=font)
                                            
    image.paste(button_img, ((x//2)-text_width, y-text_height-20)) #bottom middle 
            
  
    return image


def visualize(bucket,photo,response, display_image=False):
    
    """
    Function that draws bounding boxes based on Rekognition response.
    """
    
    # Load image from S3 bucket
    s3_connection = boto3.resource('s3')

    s3_object = s3_connection.Object(bucket,photo)
    s3_response = s3_object.get()

    stream = io.BytesIO(s3_response['Body'].read())
    original_image=Image.open(stream) #just in case we need original image
    
    image = copy.deepcopy(original_image)
    imgWidth, imgHeight = image.size
    draw = ImageDraw.Draw(image)

    for customLabel in response['CustomLabels']:
        
        if 'Geometry' in customLabel:
            box = customLabel['Geometry']['BoundingBox']
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']

            fnt = ImageFont.truetype('/usr/share/fonts/default/Type1/n019004l.pfb', 20)
    
            text_width, text_height = fnt.getsize(customLabel['Name'])
            color = determine_color(customLabel['Name'])

            button_size = (text_width +20, text_height +20)
            button_img = Image.new('RGBA', button_size, color)
            button_draw = ImageDraw.Draw(button_img)
                        
            button_draw.text((10, 10), customLabel['Name'], fill ='#000000', font=fnt)
                                            
            image.paste(button_img, (int(left) - 2, int(top - text_height - 20)))    
    
            points = (
                (left,top),
                (left + width, top),
                (left + width, top + height),
                (left , top + height),
                (left, top))
            
            thickness = 5
            
            if customLabel['Name'] == 'cow':
                thickness = 7
            
            draw.line(points, fill=color, width=thickness)
    
    dimensions = [imgWidth, imgHeight]
    
    #draw annotations
    if display_image == True:
        image.show()

    img = np.asarray(image).copy()
    original_image = np.asarray(original_image).copy()
    
    return img, dimensions, original_image

def get_response(model,bucket,photo, min_confidence,display):
    
    """
    Uses images directly from S3 bucket to run Rekognition inference
    returns both images (with bounding boxes drawn) and the JSON responses.
    Rekognition fails when not in .jpg. If wrong format, convert it to binary. 
    """
    
    client=boto3.client('rekognition')
    s3 = boto3.resource('s3')
    
    response = client.detect_custom_labels(Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
                                               MinConfidence=min_confidence,
                                               ProjectVersionArn=model)  
           
    numpy_image, dimensions, original_image = visualize(bucket,photo,response, display_image=display)

    return response, dimensions, numpy_image, original_image

def upload_PIL_s3(image, bucket, file_name):
    
    """
    Takes numpy image array as input.Because numpy images cannot be uploaded directly. 
    Must be converted into memory file within the memory. 
    file_name here refers to the file name that ends with PNG or JPEG
    """
    
    img = Image.fromarray(image)
    
    s3_client = boto3.client('s3')
    
    buffer = BytesIO()
    img.save(buffer, get_safe_ext(file_name))
    buffer.seek(0)
    sent_data = s3_client.put_object(Bucket=bucket, Key=file_name, Body=buffer)
    
    if sent_data['ResponseMetadata']['HTTPStatusCode'] != 200:
        raise S3ImagesUploadFailed('Failed to upload image {} to bucket {}'.format(key, bucket))

def get_safe_ext(key):
    ext = os.path.splitext(key)[-1].strip('.').upper()
    if ext in ['JPG', 'JPEG']:
        return 'JPEG' 
    elif ext in ['PNG']:
        return 'PNG' 
    else:
        raise Exception('Invalid image file format for uploading to S3. You should check back if the image is JPG or PNG')
        
                 
   
