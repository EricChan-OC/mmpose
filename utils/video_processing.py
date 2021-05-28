import numpy as np
import boto3
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont
from matplotlib import pyplot as plt
from rekognition import determine_color, draw_animal_count
import cv2
import time
import math
import os

#this is for the video
def draw_response(frame,response, animal_target):
    
    original_image= Image.fromarray(frame).copy()

    #draw the target count here first
    image = draw_animal_count(response, animal_target, original_image)
    
    imgWidth, imgHeight = image.size
    
    draw = ImageDraw.Draw(image)

    for customLabel in response['CustomLabels']:

        if 'Geometry' in customLabel:
            box = customLabel['Geometry']['BoundingBox']
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']

            fnt = ImageFont.truetype('/usr/share/fonts/default/Type1/n019004l.pfb', 15)
                        
            if customLabel['Name'] != 'Tag':
                label = customLabel['Name']
            
            else:
                pass 
        
            # confidence = str(round(customLabel['Confidence'], 2))
#             text = '{} {}'.format(label, confidence)
            text = label
    
            text_width, text_height = fnt.getsize(label)
            color = determine_color(label, True)
        
            button_width = int(text_width + 20)
            button_height = int(text_height + 15)
            button_size = (button_width, button_height)
            button_img = Image.new('RGB', button_size, color)
            
            button_draw = ImageDraw.Draw(button_img)
            button_draw.text((10, 10), text, fill ='#000000', font=fnt)
                                
            image.paste(button_img, (int(left), int(top)))    

            points = (
                (left,top),
                (left + width, top),
                (left + width, top + height),
                (left , top + height),
                (left, top))
            
            thickness = 5
            
            if label == 'cow':
                thickness = 7
                
            draw.line(points, fill=color, width=thickness)
    
    img = np.asarray(image)[:,:,::-1].copy()
    
    return img


def analyzeVideo(src_file, output_file, projectVersionArn, fps=5, min_confidence=75):
    
    start = time.time()
    
    rekognition = boto3.client('rekognition')        
    cap = cv2.VideoCapture(src_file)
    frameRate = cap.get(fps) #frame rate
    
    #function to write a video
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    imgSize = (int(width), int(height))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_file, fourcc, frameRate, imgSize, True)
    
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        
        print("Processing frame id: {}".format(frameId))
        ret, frame = cap.read()
       
        if (ret != True):
            break
            
        # if (frameId % math.floor(frameRate) == 0): 
        else:
            #inference on the extracted frame
            hasFrame, imageBytes = cv2.imencode(".jpg", frame)

            if(hasFrame):
                response = rekognition.detect_custom_labels(
                    Image={
                        'Bytes': imageBytes.tobytes(),
                    },
                    MinConfidence=min_confidence,
                    ProjectVersionArn = projectVersionArn
                )
                

            inferred_frame = draw_response(frame, response, animal_target='Sheep')
            inferred_frame = cv2.cvtColor(inferred_frame, cv2.COLOR_BGR2RGB)    
            writer.write(inferred_frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    end = time.time()
    print('total time lapse', end - start)
    



if __name__ == "__main__" :

    sheep_arn = 'arn:aws:rekognition:us-east-1:617989056061:project/cow-detector/version/cow-detector.2021-05-25T13.10.30/1621973430621'
    # src_video = '../../annotations-sheep/video/401063242260627.mp4'
    # output_video = '../../annotations-sheep/output_vid/401063242260627.mp4'

    # analyzeVideo(src_video, output_video, sheep_arn)
    # print('finished analyzing the video')

    input_folder = '../../annotations-sheep/video/'
    output_folder = '../../annotations-sheep/output_vid'

    print('starting video generation')

    for files in os.listdir(input_folder):
        
        print('working on {}'.format(files))
        
        src_video = os.path.join(input_folder, files)
        output_video = os.path.join(output_folder, files)
        
        analyzeVideo(src_video, output_video, sheep_arn)
        
    print('finished processing all videos!')