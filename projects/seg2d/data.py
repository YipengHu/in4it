import os


data_path = '~/git/in4it_data/trus_processed_uint8/'
image_path = 'images'
label_path = ['seg_01','seg_02','seg_03']


image_files = os.listdir(os.path.join(data_path,image_path))


# find all the images to form a dataset that can output size and get subsset

