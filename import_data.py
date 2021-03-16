"""
Drake Watson & Drew Blik
CSE 163 A Winter 2021
Import and clean data for use in the analysis file.
"""

import pandas as pd


# Import, format, and merge data frames
def data():
    vgsales_2019_df = pd.read_csv('data/vgsales-12-4-2019.csv')
    vgsales_2019_df['Name'] = vgsales_2019_df['Name'] + ' ' + '(' + \
        vgsales_2019_df['Platform'] + ')'
    vgsales_2019_df = vgsales_2019_df.filter(['Name', 'Genre', 'ESRB_Rating',
                                              'Platform', 'Publisher',
                                              'Developer', 'Critic_Score',
                                              'User_Score', 'Total_Shipped',
                                              'Year'])
    vgsales_2016_df = pd.read_csv('data/vgsales.csv')
    vgsales_2016_df['Name'] = vgsales_2016_df['Name'] + ' ' + '(' + \
        vgsales_2016_df['Platform'] + ')'
    vgsales_2016_df = vgsales_2016_df.filter(['Name', 'NA_Sales', 'EU_Sales',
                                              'JP_Sales', 'Other_Sales',
                                              'Global_Sales'])
    merged_df = vgsales_2019_df.merge(vgsales_2016_df, left_on='Name',
                                      right_on='Name', how='left')
    return merged_df
