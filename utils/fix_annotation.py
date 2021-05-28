import os
import json
import boto3

def fix_annotation(dataset_dir, json_dir, file_type='COCO'):

    """ 
    Fixes folder names and calls update_coco_file to update the change on 
    coco files as well as the DataTorch file
    """
    
    folders = os.listdir(dataset_dir)
    data_folder = (os.path.join(os.getcwd(), dataset_dir))
    update_count = 0
    file_update_count = 0

    # Fix messed up folder names
    for folder in folders:
        folder_name = os.path.join(data_folder, folder)
        new_name = folder_name.replace(" ", "_")

        if folder_name != new_name:
            try :
                os.rename(folder_name, new_name)
                update_count += 1
                print('successfully renamed {} to {} '.format(folder_name, new_name))

            except OSError as error:
                print('error found while attempting to rename folder: \n ', error)

        if folder_name.endswith('json'):
            pass

        else:
            #File names are messed up too
            for files in os.listdir(new_name):   

                file_name = os.path.join(folder_name, files)

                new_file_name = file_name.replace(" ", "_")

                if file_name !=new_file_name:

                    try :
                        os.rename(file_name, new_file_name)
                        file_update_count += 1
                        print('Renamed {} to {} '.format(file_name, new_file_name))

                    except OSError as error:
                        print('error found while attempting to rename file: \n ', error)


    print('Updated names of {} folders'.format(update_count))
    print('Updated names of {} files'.format(file_update_count))

    #update the annotations
    update_coco_file(json_dir, file_type)

#fix messed up json file names
def update_coco_file(src_path, file_type='COCO'):

    assert file_type == 'COCO' or file_type == 'DataTorch', 'invalid file type. choose b/w COCO or DataTorch'

    revision_count = 0

    with open(src_path) as f:
        data = json.load(f)
    
    # COCO and Datatorch have different file structures, so they need to be dealt seperately
    if file_type == 'COCO':
        for image in data['images']:
            new_path = image['path'].replace(" ", "_")
            if image['path'] != new_path:
                image['path'] = new_path
                revision_count += 1

    elif file_type == 'DataTorch':
        
        # update dataset key
        for record in data['datasets'].values():
            new_name = record['name'].replace(" ", "_")
            if record['name'] != new_name:
                record['name'] = new_name

        #update files key
        for record in data['files'].values():
            new_file = record['path'].replace(" ", "_")
            if record['path'] != new_file:
                record['path'] = new_file
                revision_count += 1

    else:
        print('invalid file type. choose b/w COCO or DataTorch')


    # write changes to the file
    with open(src_path, 'w') as jsonFile:
        json.dump(data, jsonFile, indent=4)

    print('updated {} records within the JSON file'.format(revision_count))


# function to upload cleaned images,folders,json file to destination S3 bucket
def upload_objects(bucket, folder_dir, region='us-east-1'):

    s3_resource = boto3.resource("s3", region_name=region)

    try:
        bucket_name = bucket #s3 bucket name
        root_path = folder_dir # local folder for upload

        my_bucket = s3_resource.Bucket(bucket_name)
        count = 0

        print('working on {}'.format(folder_dir))

        #take care of the json upload
        if folder_dir.endswith('json'):
            file_name = folder_dir.split('/')[1]
            my_bucket.upload_file(folder_dir, file_name)


        #take care of the images upload
        for path, subdirs, files in os.walk(root_path):

            path = path.replace("\\","/")

            directory_name = folder_dir.split('/')[1]

            for file in files:

                count += 1                
                if file.endswith('.json'):
                    print(file)

                my_bucket.upload_file(os.path.join(path, file), directory_name+'/'+file)

        print('completed copying {} files inside {} to {}'.format(count, folder_dir, bucket))

    except Exception as err:
        print(err)


if __name__ == "__main__":
    
    dataset_dir = 'annotations-sheep'
    src_coco = 'annotations-sheep/COCO.json'
    src_datatorch = 'annotations-sheep/DataTorch.json'


    fix_annotation(dataset_dir, src_coco)
    fix_annotation(dataset_dir, src_datatorch, file_type='DataTorch')

    #upload to S3
    for record in os.listdir(dataset_dir):

        files = os.path.join('annotations-sheep', record)

        upload_objects('annotations-sheep-us-east-1', files)


   
