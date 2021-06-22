"""
pitching_video_pre_process.py

This is a file to pre process all the pitching video data

"""
#TODO
# 1. Make it so dont have to use predefined number of pitches for a session with a while loop
#   instead of a for loop.

# initial importing of libraries needed
import csv
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import bs4
import requests_html
import json
import cv2
import os


def capture_video(capture_time = 3, cameras_list = ['webcam1','webcam2','oculus1','oculus2',], greyScale = True):
    '''
    This function is used to capture videos from the cameres input into the computer
    may need to go into script and change the video capture objects
    inputs:
    greyScale (bool): wether to grey scale the images or not
    capture_time (int) :number of seconds to capture the video for

    cameras_list [list of string] = list of the cameras used for recording
        options: [webcam1,webcam2,oculus1,oculus2]
    :return:
    numpy arrays of all the video captures for the desired ammount of time:
    '''
    #set bools for which cameras to use
    use_webcam1 = False
    use_webcam2 = False
    use_oculus1 = False
    use_oculus2 = False
    if 'webcam1' in cameras_list:
        use_webcam1 = True
    if 'webcam2' in cameras_list:
        use_webcam2 = True
    if 'oculus1' in cameras_list:
        use_oculus1 = True
    if 'oculus2' in cameras_list:
        use_oculus2 = True

    webcam1_cap = cv2.VideoCapture(4)
    webcam2_cap = cv2.VideoCapture(5)
    oculus_1_cap = cv2.VideoCapture(2)
    oculus_2_cap = cv2.VideoCapture(0)

    # will_pitch_vid_cap = cv2.VideoCapture('data/will_pitching/Changeups /IMG_0132_MOV.mp4')

    # buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    # getting base variables for creating video arrays

    frameCount = capture_time * 30 #assuming 30 fps

    if use_webcam1:
        webcam1_frameWidth = int(webcam1_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        webcam1_frameHeight = int(webcam1_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        webcam1_pixel_count = webcam1_frameWidth * webcam1_frameHeight
        webcam1_pic_stats = (webcam1_frameWidth,webcam1_frameHeight,webcam1_pixel_count)

    if use_webcam2:
        webcam2_frameWidth = int(webcam2_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        webcam2_frameHeight = int(webcam2_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        webcam2_pixel_count = webcam2_frameWidth * webcam2_frameHeight
        webcam2_pic_stats = (webcam2_frameWidth, webcam2_frameHeight, webcam2_pixel_count)

    if use_oculus1:
        oculus_1_frameWidth = int(oculus_1_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        oculus_1_frameHeight = int(oculus_1_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        oculus_1_pixel_count = oculus_1_frameWidth * oculus_1_frameHeight
        oculus_1_pic_stats = (oculus_1_frameWidth, oculus_1_frameHeight, oculus_1_pixel_count)

    if use_oculus2:
        oculus_2_frameWidth = int(oculus_2_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        oculus_2_frameHeight = int(oculus_2_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        oculus_2_pixel_count = oculus_2_frameWidth * oculus_2_frameHeight
        oculus_2_pic_stats = (oculus_2_frameWidth, oculus_2_frameHeight, oculus_2_pixel_count)

    #create lists for the webcam frames
    webcam1_frames_list = []
    oculus_1_frames_list = []
    webcam2_frames_list = []
    oculus_2_frames_list = []


    fc = 0
    while fc < frameCount:
        fc += 1

        ret_web1, frame_web1 = webcam1_cap.read()
        ret_web2, frame_web2 = webcam2_cap.read()
        ret_oc1, frame_oc1 = oculus_1_cap.read()
        ret_oc2, frame_oc2 = oculus_2_cap.read()
        # if use_webcam1:
        #     ret_web1, frame_web1 = webcam1_cap.read()
        # if use_webcam2:
        #     ret_web2, frame_web2 = webcam2_cap.read()
        # if use_oculus1:
        #     ret_oc1, frame_oc1 = oculus_1_cap.read()
        # if use_oculus2:
        #     ret_oc2, frame_oc2 = oculus_2_cap.read()


        #greScale the images or not
        if greyScale:
            if ret_web1 and use_webcam1:
                frame_web1 = cv2.cvtColor(frame_web1, cv2.COLOR_BGR2GRAY)
            if ret_web2 and use_webcam2:
                frame_web2 = cv2.cvtColor(frame_web2, cv2.COLOR_BGR2GRAY)
            if ret_oc1 and use_oculus1:
                frame_oc1 = cv2.cvtColor(frame_oc1, cv2.COLOR_BGR2GRAY)
            if ret_oc2 and use_oculus2:
                frame_oc2 = cv2.cvtColor(frame_oc2, cv2.COLOR_BGR2GRAY)

        #rotate oculus image
        if ret_oc1 and use_oculus1:
            frame_oc1 = np.rot90(frame_oc1,-1)
        if ret_oc2 and use_oculus2:
            frame_oc2 = np.rot90(frame_oc2,-1)

        if ret_web1 and use_webcam1:
            cv2.imshow('Test Webcam 1', frame_web1)
        if ret_web2 and use_webcam2:
            cv2.imshow('Test Webcam 2', frame_web2)
        if ret_oc1 and use_oculus1:
            cv2.imshow('Test Oculus 1', frame_oc1)
        if ret_oc2 and use_oculus2:
            cv2.imshow('Test Oculus 2', frame_oc2)


        # waits to quit out early of function
        if cv2.waitKey(1) == ord('q'):
            break

        #set up data to be put into the list by flattening it and add it to the list
        if greyScale:
            if ret_web1 and use_webcam1:
                flattened_frame_web1 = frame_web1.reshape(webcam1_pixel_count)
                webcam1_frames_list.append(flattened_frame_web1)
            if ret_web2 and use_webcam2:
                flattened_frame_web2 = frame_web2.reshape(webcam2_pixel_count)
                webcam2_frames_list.append(flattened_frame_web2)
            if ret_oc1 and use_oculus1:
                flattened_frame_oc1 = frame_oc1.reshape(oculus_1_pixel_count)
                oculus_1_frames_list.append(flattened_frame_oc1)
            if ret_oc2 and use_oculus2:
                flattened_frame_oc2 = frame_oc2.reshape(oculus_2_pixel_count)
                oculus_2_frames_list.append(flattened_frame_oc2)

        else:
            if ret_web1 and use_webcam1:
                flattened_frame_web1 = frame_web1.reshape(webcam1_pixel_count,3)
                webcam1_frames_list.append(flattened_frame_web1)
            if ret_web2 and use_webcam2:
                flattened_frame_web2 = frame_web2.reshape(webcam2_pixel_count, 3)
                webcam2_frames_list.append(flattened_frame_web2)
            if ret_oc1 and use_oculus1:
                flattened_frame_oc1 = frame_oc1.reshape(oculus_1_pixel_count,3)
                oculus_1_frames_list.append(flattened_frame_oc1)
            if ret_oc2 and use_oculus2:
                flattened_frame_oc2 = frame_oc2.reshape(oculus_2_pixel_count,3)
                oculus_2_frames_list.append(flattened_frame_oc2)

        # creating matrtix to hold the frames of each camera
        if ret_web1 and use_webcam1:
            webcam1_video_matrix = np.asarray(webcam1_frames_list)
        if ret_web2 and use_webcam2:
            webcam2_video_matrix = np.asarray(webcam2_frames_list)
        if ret_oc1 and use_oculus1:
            oculus_1_video_matrix = np.asarray(oculus_1_frames_list)
        if ret_oc2 and use_oculus2:
            oculus_2_video_matrix = np.asarray(oculus_2_frames_list)

        #increment frame count variable
        fc += 1




    # releases the video capture and cleans up from script
    if use_webcam1:
        webcam1_cap.release()
    if use_webcam2:
        webcam2_cap.release()
    if use_oculus1:
        oculus_1_cap.release()
    if use_oculus2:
        oculus_2_cap.release()





    cv2.destroyAllWindows()


    #set up the list of matricies to return
    return_list = []
    if use_webcam1:
        return_list.append(webcam1_video_matrix)
    if use_webcam2:
        return_list.append(webcam2_video_matrix)
    if use_oculus1:
        return_list.append(oculus_1_video_matrix)
    if use_oculus2:
        return_list.append(oculus_2_video_matrix)
    # print(f'Finished capture_video method')
    return return_list


def pitch_capture(player_name,session_number,capture_time = 3,number_of_pitches = 2,
                  pitchers_pitch_types = ['fastball','changeup', 'slider'],cameras_list = ['webcam1','webcam2','oculus1','oculus2',],
                  wait_for_input = True):
    '''
    This Function capture the number of pitches

    :param session_number: number of the recording session

    :param player_name: name of the player who is being recorded

    :param capture_time (int) , number of seconds to capture the video for:

    :param number_of_pitches: number of pitches a pitcher is throwing

    :param pitchers_pitch_types: list of strings of the types of pitches the pitcher throws


    :return:
    dictionary of all the clips
    '''

    # set bools for which cameras to use
    use_webcam1 = False
    use_webcam2 = False
    use_oculus1 = False
    use_oculus2 = False
    if 'webcam1' in cameras_list:
        use_webcam1 = True
    if 'webcam2' in cameras_list:
        use_webcam2 = True
    if 'oculus1' in cameras_list:
        use_oculus1 = True
    if 'oculus2' in cameras_list:
        use_oculus2 = True


    #set up a list of pitches
    pitch_type_list = ['fastball', 'twoSeamFastball', 'sinker', 'changeup', 'slider', 'curveball', 'cutter'
        ,'splitter', 'knuckleball']

    #check that all pitch types are in the list
    for pitch_type in pitchers_pitch_types:
        if pitch_type not in pitch_type_list:
            print(f'Error {pitch_type} must be in pitch types list!!!!\n{pitch_type_list}')
            raise ValueError

    #make an array to hold the count of each pitch
    pitch_count_array = np.zeros(len(pitchers_pitch_types))

    #make a list to hold the dictionary of clips form each pitch
    pitch_clips_list = []



    #loop through the number of pitches desired
    for pitch_count in np.arange(number_of_pitches):

        # make a dictionary to hold all the clips of each pitch from each camera
        pitch_clips_dictionary = {}
        if use_webcam1:
            pitch_clips_dictionary['webcam1'] = []
        if use_webcam2:
            pitch_clips_dictionary['webcam2'] = []
        if use_oculus1:
            pitch_clips_dictionary['oculus1'] = []
        if use_oculus2:
            pitch_clips_dictionary['oculus2'] = []

        pitch_to_throw_num = random.randint(0,len(pitchers_pitch_types)-1)
        pitch_to_throw = pitchers_pitch_types[pitch_to_throw_num]

        #add to the pitch count
        pitch_count_array[pitch_to_throw_num] = pitch_count_array[pitch_to_throw_num] + 1

        if wait_for_input:
            confirmation = input(f'\n\n{number_of_pitches-pitch_count} Pitch(s) Left\nThrow: {pitch_to_throw} \nPress any enter to start recording\n.')

        pitch_name = f'{player_name}_{session_number}_{pitch_to_throw}_{int(pitch_count_array[pitch_to_throw_num])}'

        pitch_clips = capture_video(capture_time,cameras_list=cameras_list)


        if use_webcam1:
            wc1_list = [f'{pitch_name}_Wc1']
            wc1_pixels_list = list(pitch_clips[0])
            for pixel in wc1_pixels_list:
                wc1_list.append(pixel)
            pitch_clips_dictionary['webcam1'] = wc1_list

        if use_webcam2:
            wc2_list = [f'{pitch_name}_Wc2']
            wc2_pixels_list = list(pitch_clips[0])
            for pixel in wc2_pixels_list:
                wc2_list.append(pixel)
            pitch_clips_dictionary['webcam2'] = wc2_list

        if use_oculus1:
            oc1_list = [f'{pitch_name}_Oc1']
            oc1_pixels_list = list(pitch_clips[1])
            for pixel in oc1_pixels_list:
                oc1_list.append(pixel)
            pitch_clips_dictionary['oculus1'] = oc1_list

        if use_oculus2:
            oc2_list = [f'{pitch_name}_Oc2']
            oc2_pixels_list = list(pitch_clips[1])
            for pixel in oc1_pixels_list:
                oc2_list.append(pixel)
            pitch_clips_dictionary['oculus2'] = oc2_list

        pitch_clips_list.append(pitch_clips_dictionary)


    #return the list of pitch clips and
    return pitch_clips_list


def write_clips_to_csv(list_of_pitch_clips, pitcher_name , session_number,
                       cameras_list = ['webcam1','webcam2','oculus1','oculus2']):
    '''['
    this functions writes this list of clips to their respective csv file

    :param list_of_pitch_clips:
    :return:
    '''

    # set bools for which cameras to use
    use_webcam1 = False
    use_webcam2 = False
    use_oculus1 = False
    use_oculus2 = False
    if 'webcam1' in cameras_list:
        use_webcam1 = True
    if 'webcam2' in cameras_list:
        use_webcam2 = True
    if 'oculus1' in cameras_list:
        use_oculus1 = True
    if 'oculus2' in cameras_list:
        use_oculus2 = True

    #set up list of clips
    if use_webcam1:
        list_of_wc1_clips = [clips['webcam1'] for clips in list_of_pitch_clips]
    if use_webcam2:
        list_of_wc2_clips = [clips['webcam2'] for clips in list_of_pitch_clips]
    if use_oculus1:
        list_of_oc1_clips = [clips['oculus1'] for clips in list_of_pitch_clips]
    if use_oculus2:
        list_of_oc2_clips = [clips['oculus2'] for clips in list_of_pitch_clips]

    # write csv files
    if use_webcam1:
        for clip in list_of_wc1_clips:
            #make new folder
            try:
                os.mkdir(f'data/{pitcher_name}_{session_number}')
            except OSError as error:
                print(error)

            with open(f'data/{pitcher_name}_{session_number}/{clip[0]}.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(clip[1:])
    if use_webcam2:
        for clip in list_of_wc2_clips:
            with open(f'data/{pitcher_name}_{session_number}/{clip[0]}.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(clip[1:])
    if use_oculus1:
        for clip in list_of_oc1_clips:
            with open(f'data/{pitcher_name}_{session_number}/{clip[0]}.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(clip[1:])
    if use_oculus2:
        for clip in list_of_oc2_clips:
            with open(f'data/{pitcher_name}_{session_number}/{clip[0]}.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(clip[1:])




    print(f'Done with write clips to csv method')

'''
The main function used to test functions from this python script
'''
def main():
    # test_video_cap = capture_video(10)
    #['webcam1','webcam2','oculus1','oculus2']

    # pitcher_name = 'will_t'
    pitcher_name = 'will_t'
    session_num = 4
    cameras_list = ['webcam1', 'oculus1']

    test_pitch_cap = pitch_capture(pitcher_name,session_num,capture_time = 10,wait_for_input=True,
                                   cameras_list=cameras_list, number_of_pitches=48)
    test_write_csv = write_clips_to_csv(test_pitch_cap,pitcher_name,session_num,cameras_list=cameras_list)
    print(f'Done Script')



if __name__ == "__main__":
    main()

