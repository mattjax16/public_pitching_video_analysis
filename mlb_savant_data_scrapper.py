"""
mlb_savant_data_scrapper.py

This is going to just be the initial scrapper for mlb savant data

will contain a variety of functions

NOTES ABOUT SITE:
1. contains years from 1950 to 2021 for all basic stats
    a. for stat cast advanced stats years 2015 and onwards

2. Here is an example of a url for searching for stats
    a. base url link :  https://baseballsavant.mlb.com/leaderboard/custom
        1. year modifier :year=2021,2020,2019&
        2. type of player modifier : type=pitcher&
        3. how to filter and sort the data modifiers :   filter=&sort=4&sortDir=asc&min=q&
        4.what stats to select from the stotal data set :  selections=player_age,p_game,p_formatted_ip,p_total_pa,p_ab,p_total_hits,p_single,
                p_double,p_triple,p_home_run,p_strikeout,p_walk,p_k_percent,p_bb_percent,batting_avg,slg_percent,
                on_base_percent,on_base_plus_slg,xba,xslg,woba,xwoba,xobp,xiso,exit_velocity_avg,launch_angle_avg,
                barrel_batted_rate,&chart=false&x=xba&y=xba&r=no&chartType=beeswarm

3. Here is my series of test links used to come to the best way to scrape, explore and analyize
   the data
    a. https://baseballsavant.mlb.com/leaderboard/custom?year=2021,2020,2019&type=pitcher&filter=&sort=4&sortDir=asc&min=q&selections=player_age,p_game,p_formatted_ip,p_total_pa,p_ab,p_total_hits,p_single,p_double,p_triple,p_home_run,p_strikeout,p_walk,p_k_percent,p_bb_percent,batting_avg,slg_percent,on_base_percent,on_base_plus_slg,xba,xslg,woba,xwoba,xobp,xiso,exit_velocity_avg,launch_angle_avg,barrel_batted_rate,&chart=false&x=xba&y=xba&r=no&chartType=beeswarm



"""

# initial importing of libraries needed
import pandas as pd
import numpy as np
import matplotlib as plt
import requests
import bs4
import requests_html
import json


# Set variables that will be used over all functions
current_year = 2021

def get_desired_stats_url(years = 'all'):
    '''
    This function gets the correct url for the desired stats from mlb savant

    :inputs:
    years:  str or tuple of ints range 1950 to current year to get the desired stats from (default all)

    :return:
        data_url: a url for the string of the desired data
    '''

    if years == 'all':
        years_array = np.arange((current_year - 1949))+1950
    elif type(years) == tuple:
        min_year = years[0]
        max_year = years[1]
        #check to make sure the years are correct
            #checking the min year
        if min_year <= 1950 or min_year >= current_year or min_year >= max_year:
            print(f'Error min_year must be >= 1950 and <= {max_year} and {current_year}'+
                  f'\nCurrently it is {min_year}')
            raise ValueError
        if max_year <= 1950 or max_year >= current_year or max_year <= min_year:
            print(f'Error max_year must be >= 1950 and <= {max_year} and {current_year}'+
                  f'\nCurrently it is {max_year}')
            raise ValueError

        years_array = np.arange((max_year - min_year-1)) + 1950


    #make the string for all the years
    years_string = 'year='
    for year_cnt ,year in enumerate(years_array):

        if year_cnt != years_array.size - 1:
            years_string = years_string + f'{year},'
        else:
            years_string = years_string + f'{year}'

    return years_string

def test_scrapper(web_page_url):
    session = requests_html.HTMLSession()

    t_get = get_desired_stats_url()

    initial_web_page_R = session.get(web_page_url)
    initial_web_page_html = initial_web_page_R.html

    return initial_web_page_R




'''
The main function used to test functions from this python script
'''
def main():

    mlb_savant_test_url = 'https://baseballsavant.mlb.com/leaderboard/custom?year=2021,2020,2019&type=pitcher&filter=&sort=4&sortDir=asc&min=q&selections=player_age,p_game,p_formatted_ip,p_total_pa,p_ab,p_total_hits,p_single,p_double,p_triple,p_home_run,p_strikeout,p_walk,p_k_percent,p_bb_percent,batting_avg,slg_percent,on_base_percent,on_base_plus_slg,xba,xslg,woba,xwoba,xobp,xiso,exit_velocity_avg,launch_angle_avg,barrel_batted_rate,&chart=false&x=xba&y=xba&r=no&chartType=beeswarm'

    test_scrapper(mlb_savant_test_url)
    pass

if __name__ == "__main__":
    main()