import json 
import numpy as np
import boto3
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont
import os
from matplotlib import pyplot as plt
import iso8601 #we need this to correct dates


class CL_JSON_LINE:  
    def __init__(self,job, img):  

        #Get image info. Annotations are dealt with seperately
        sizes=[]
        image_size={}
        image_size["width"] = img["width"]
        image_size["depth"] = 3
        image_size["height"] = img["height"]
        sizes.append(image_size)

        bounding_box={}
        bounding_box["annotations"] = []
        bounding_box["image_size"] = sizes

        self.__dict__["source-ref"] = s3_path + img['path']
        self.__dict__[job] = bounding_box

        #get metadata
        metadata = {}
        metadata['job-name'] = job_name
        metadata['class-map'] = {}
        metadata['human-annotated']='yes'
        metadata['objects'] = [] 
        date_time_obj = iso8601.parse_date(img['date_captured'])
        metadata['creation-date']= date_time_obj.strftime('%Y-%m-%dT%H:%M:%S') 


        metadata['type']='groundtruth/object-detection'
        
        self.__dict__[job + '-metadata'] = metadata
    

def read_json_s3(bucket, file):

    obj = s3.Object(bucket, file)
    data = obj.get()['Body'].read().decode('utf-8')
    json_data = json.loads(data)

    return json_data

def upload_manifest(images_dict, cl_manifest_file):

    print('Writing Custom Labels to manifest...')

    for im in images_dict.values():
        with open(cl_manifest_file, 'a+') as outfile:
                json.dump(im.__dict__,outfile)
                outfile.write('\n')
                outfile.close()


if __name__ == "__main__":

    s3_bucket = 'annotations-cattle' #change this
    s3_key_path_manifest_file = 'manifest/'
    s3_key_path_images = ''
    s3_path='s3://' + s3_bucket  + '/' + s3_key_path_images
    s3 = boto3.resource('s3')

    #Source of COCO annotations
    src_bucket = 'annotations-cattle' #change this
    src_images_path = ''
    coco_manifest = 'COCO.json'
    coco_json_file = 's3://' + src_bucket  + '/' + coco_manifest
    job_name='Custom Labels'

    # A local name for the manifest file created after running the scripts
    label_attribute ='bounding-box'
    cwd = os.getcwd()
    cl_manifest_file = '10_classes_cows.manifest' #change this

    manifest_location = os.path.join(cwd, cl_manifest_file)

    open(manifest_location, 'w').close()

    print("Getting image, annotations, and categories from COCO file...")
    js = read_json_s3(src_bucket, coco_manifest)
    images = js['images']
    categories = js['categories']
    full_annotations = js['annotations']

    # change this based on COCO.JSON file
    annotations = []
    category_list = [1, 2, 3, 4, 5, 6, 7, 8, 9] #changing this is important -- head right, there is only one sample of this.

    for annotation in full_annotations:
        if annotation['category_id'] in category_list:
            annotations.append(annotation)
    
    print('Number of Images: ' + str(len(images)))
    print('Number of reduced annotations (only certain categories): ' + str(len(annotations)))
    print('Number of Full annotations: ' + str(len(full_annotations)))
    print('Number of categories: ' + str(len (categories)))
    print('==' * 20)

    print("Creating CL JSON lines...")
    
    images_dict = {image['id']: CL_JSON_LINE(label_attribute, image) for image in images}

    for annotation in annotations:

        image = images_dict[annotation['image_id']]

        cl_annotation = {}
        cl_class_map={}

        # get bounding box information
        cl_bounding_box={}
        cl_bounding_box['left'] = annotation['bbox'][0]
        cl_bounding_box['top'] = annotation['bbox'][1]
    
        cl_bounding_box['width'] = annotation['bbox'][2]
        cl_bounding_box['height'] = annotation['bbox'][3]
        cl_bounding_box['class_id'] = annotation['category_id']

        getattr(image, label_attribute)['annotations'].append(cl_bounding_box)

        for category in categories:
            if annotation['category_id'] == category['id']:
                getattr(image, label_attribute + '-metadata')['class-map'][category['id']]=category['name']
        
    
        cl_object={}
        cl_object['confidence'] = int(1)  #not currently used by Custom Labels
        getattr(image, label_attribute + '-metadata')['objects'].append(cl_object)

    print('Done parsing annotations')

    upload_manifest(images_dict, manifest_location)
    print('uploading completed manifest file')

    s3 = boto3.resource('s3')
    s3.Bucket(s3_bucket).upload_file(manifest_location, s3_key_path_manifest_file + cl_manifest_file)

    print('finished all process!')