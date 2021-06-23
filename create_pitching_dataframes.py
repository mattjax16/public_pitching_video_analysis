'''
This Script is used after piching pre process and is used to take the data from the csv files and create them in a big
pandas data frame (might also make it a JSON style data frame later on)


'''
# initial importing of libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import bs4
import requests_html
import json
import cv2
import csv
import time
import concurrent.futures
import os


#this is a function to reshape images:
def unflatten_images(array_of_images,img_width, img_height):
    total_img_size = img_height*img_width

    #make sure image size is correct
    if array_of_images.shape[1] != total_img_size:
        print(f'Error The shape of {img_width} x {img_height} is not the size of the images in the array of '
              + f'{array_of_images.shape[1]}')
        raise ValueError

    reshaped_img_array = array_of_images.reshape(array_of_images.shape[0],img_width,img_height)
    return reshaped_img_array

def normalize_data_globaly(matrix,method = 'min-max'):
    '''

    :param matrix: (ndarray)
    :param method: (string) wither mean or min-max to chose the type of global normalization
    :return normalized matrix: (ndarray) the normalized matrix
    '''
    if method == 'mean':
        normalized_matrix = (matrix - matrix.mean())/matrix.std()
    elif method == 'min-max':
        normalized_matrix = (matrix - matrix.min())/(matrix.max()-matrix.min())
    else:
        print(f'Error method must be mean or min-max!!\nCurrently it is {method}')
        raise ValueError

    return normalized_matrix

def crop_frames(frames_matrix,top_left_corner,bottom_right_corner):

    #check to make sure frames matrix is 3 dimensions
    if frames_matrix.ndim != 3:
        print(f'Error frame matrix needs to be 3D!!! (frame,frame width, frame height)')
        raise ValueError

    #check to make sure size is ok:
    cropped_frames_matrix = frames_matrix[:,top_left_corner[1]:bottom_right_corner[1],
                            top_left_corner[0]:bottom_right_corner[0]]

    return cropped_frames_matrix


def display_clip(frames_matrix, img_width, img_height, number_of_frames=(4, 4), crop_the_frames=False,
                 top_left_corner=(0, 0), bottom_right_corner=(600, 600)):
    # unflatten the frames matrix
    frames_matrix = unflatten_images(frames_matrix, img_width, img_height)

    # crop frames if needed
    if crop_the_frames:
        frames_matrix = crop_frames(frames_matrix, top_left_corner, bottom_right_corner)

    total_num_frames_displayed = number_of_frames[0] * number_of_frames[1]

    frames_chosen_to_display = np.random.choice(np.arange(frames_matrix.shape[0]), total_num_frames_displayed,
                                                replace=False)

    # create the subplot for the frames
    fig, axes = plt.subplots(number_of_frames[0], number_of_frames[1], figsize=(20, 20))

    for frame, ax in zip(frames_chosen_to_display, axes.flatten()):
        ax.imshow(frames_matrix[frame, :, :], cmap='gray')

    plt.show()


def write_csv_clips_to_npy(data_dir_names):
    '''
    this functions writes this list of clips saved in csv format to npy format

    :param list_of_pitch_clips:
    :return:
    '''

    data_collected_dirs = [dirr for dirr in os.listdir('data')]

    for dir_name in data_dir_names:
        if dir_name not in data_collected_dirs:
            print(f'Error {dir_name} has not been collected')
            raise ValueError

        # loop through each session (dir)
    for dir_name in data_dir_names:
        print(dir_name)



        #make a directory for the numy files

        npy_dir_name = f'data/{dir_name}_npy'
        try:
            os.mkdir(npy_dir_name)
        except OSError as error:
            print(error)

        #get the file path of every file in the original session directory
        files_in_dir = [f'data/{dir_name}/{file}' for file in os.listdir(f'data/{dir_name}')]

        #loop through every file in the dir
        for file_path in files_in_dir:

            #load in the csv to a numpy array
            csv_array = np.genfromtxt(file_path, delimiter=',')

            npy_file_name = file_path.split('/')[-1]
            npy_file_name = npy_dir_name + '/' + npy_file_name.split('.')[0] + '.npy'

            #save the file to the new directory
            np.save(npy_file_name,csv_array)


    print(f'Done with write clips to npy method')



def concurrent_load_data(data_path,img_width = 480, img_height = 640,crop_the_frames = True,
                    top_left_corner=(100,30),bottom_right_corner=(450,420), data_type = 'float32'):


    #load in the clip for proper data type
    if data_path.split('.')[-1] == 'csv':
        frames_matrix = np.genfromtxt(data_path, delimiter=',')
    elif data_path.split('.')[-1] == 'npy':
        frames_matrix = np.load(data_path)
    else:
        print(f'Error data file must be a csv or npy file!!!\nCurrently it is {data_path.split(".")[-1]}')


    #set data type of frame matrix
    frames_matrix = frames_matrix.astype(data_type)


    #crop the clip if needed
    if crop_the_frames:

        #unflatten the frames matrix
        frames_matrix = unflatten_images(frames_matrix,img_width,img_height)

        frames_matrix = crop_frames(frames_matrix, top_left_corner=top_left_corner
                                        ,bottom_right_corner=bottom_right_corner)

        frames_matrix = frames_matrix.reshape(frames_matrix.shape[0],
                                                              frames_matrix.shape[1]*frames_matrix.shape[2])


    clip_df = pd.DataFrame(frames_matrix)

    #set the name of each column to pixels
    clip_df.columns = [f'pixel_{col}' for col in clip_df.columns]


    # Setting up labeling for each clip in the dataframe
    clip_vars = data_path.split('/')[-1]
    clip_vars = clip_vars.split('_')[::-1]
    clip_vars = clip_vars[:4]

    #getting rid of .csv
    clip_vars[0] = clip_vars[0].split('.')[0]

    pitcher = data_path.split('/')[-2]
    pitcher = pitcher.split('_')[:-1]
    pitcher = f'{pitcher[0]}_{pitcher[1]}'


    clip_df['camera_type'] = [clip_vars[0] for c in range(0,clip_df.shape[0])]
    clip_df['pitch_type'] = [clip_vars[2] for c in range(0,clip_df.shape[0])]
    clip_df['session_number'] = [clip_vars[3] for c in range(0,clip_df.shape[0])]
    clip_df['pitch_number'] = [clip_vars[1] for c in range(0,clip_df.shape[0])]
    clip_df['pitcher'] = [pitcher for c in range(0,clip_df.shape[0])]
    clip_df['frame_num'] = [c for c in range(0,clip_df.shape[0])]

    # print(f'Done with concurrent_load_data()')
    return clip_df

def load_clips_to_df(data_dir_names,img_width = 480, img_height = 640,crop_the_frames = True,
                    top_left_corner=(100,30),bottom_right_corner=(450,420)):

        data_collected_dirs = [dirr for dirr in os.listdir('data')]

        for dir_name in data_dir_names:
            if dir_name not in  data_collected_dirs:
                print(f'Error {dir_name} has not been collected')
                raise ValueError

        results_list = [] # a list to hold all the results of the helper function

        #loop through each session (dir) and run the function above on the clips)
        for dir_name in data_dir_names:

            #this is where multithreading will take place
            with concurrent.futures.ThreadPoolExecutor() as thread_executer:
                clips_list = os.listdir(f'data/{dir_name}/')

                clips_list = [f'data/{dir_name}/{clip_name}' for clip_name in clips_list]

                #use the helper function with the thread pool executer
                results = thread_executer.map(concurrent_load_data, clips_list)


                #loop through results and append them to the results list
                for results in results:
                    results_list.append(results)


        return results_list


'''
The main function used to test functions from this python script
'''
def main():

   # t_load = concurrent_load_data('data/will_t_0_npy/will_t_0_changeup_4_Wc1.npy')

   test_clips_df = load_clips_to_df(['will_t_0_npy'])

   #write_csv_clips_to_npy(['will_t_1','will_t_2','will_t_3'])

   t_c = pd.concat(test_clips_df)

   print(f'Done with creating_pitching_dataframes.py')


if __name__ == "__main__":
    main()
