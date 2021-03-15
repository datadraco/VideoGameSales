import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Import the larger data set with more observations and ESRB/Score data, then
# filter for the columns we want to keep from that df
vgsales_2019_df = pd.read_csv('vgsales-12-4-2019.csv')
vgsales_2019_df = vgsales_2019_df.filter(['Name', 'Genre', 'ESRB_Rating',
                                          'Platform', 'Publisher', 'Developer',
                                          'Critic_Score', 'User_Score',
                                          'Total_Shipped', 'Year'])

# Import the smaller of the two data sets that has more complete sales data and
# then filter for the columns that we want to keep from that df
vgsales_2016_df = pd.read_csv('vgsales.csv')
vgsales_2016_df = vgsales_2016_df.filter(['Name', 'NA_Sales', 'EU_Sales',
                                          'JP_Sales', 'Other_Sales',
                                          'Global_Sales'])

# In order to merge later on, we must account for the fact that the 'Name'
# column has duplicates for games that have been released on several platforms,
# so we will rename the games to include what platform they are referring to
vgsales_2019_df['Name'] = vgsales_2019_df['Name'] + ' ' + '(' + \
                          vgsales_2019_df['Platform'] + ')'
vgsales_2016_df['Name'] = vgsales_2016_df['Name'] + ' ' + '(' + \
                          vgsales_2016_df['Platform'] + ')'

# Merge data frames in order to do joint analysis between the two data frames
# based upon the 'Names' column
merged_df = vgsales_2019_df.merge(vgsales_2016_df, left_on='Name',
                                  right_on='Name', how='left')

# Analysis Question 1

# Analyze the global trends by genre
global_genre_success_df = merged_df.filter(['Genre', 'Global_Sales']).dropna()
fig3 = px.pie(global_genre_success_df, values='Global_Sales', names='Genre')
fig3.update_traces(textposition='inside', textinfo='label')
fig3.update_layout(font_family="Rockwell",
                   title_text='Global Genre Popularity',
                   title_x=0.5, showlegend=False)

# Global ESRB trends
global_ESRB_success_df = merged_df.filter(['ESRB_Rating',
                                           'Global_Sales']).dropna()
fig5 = px.pie(global_ESRB_success_df, values='Global_Sales',
              names='ESRB_Rating')
fig5.update_traces(textposition='inside', textinfo='label')
fig5.update_layout(font_family="Rockwell", title_text='Global ESRB Popularity',
                   title_x=0.5, showlegend=False)

# Analyze global publisher success
global_pub_success_df = merged_df.filter(['Publisher', 'Genre',
                                          'Global_Sales']).dropna()
top_pubs = global_pub_success_df.groupby('Publisher')['Global_Sales'].sum()
top_pubs = top_pubs.reset_index()
top_10_pubs = top_pubs.sort_values(['Global_Sales'], ascending=False).head(10)
fig4 = px.bar(top_10_pubs, x='Global_Sales', y='Publisher', orientation='h',
              title='Global Publisher Success', template='plotly_white',
              labels={'Global_Sales': 'Global Sales (Millions)'})
fig4.update_layout(font_family="Rockwell", title_x=0.5)

# Analysis Question 2

# Analyze how these trends differ per the 3 largest regions (US, EU, JP)
na_genre_success_df = merged_df.filter(['Genre', 'NA_Sales']).dropna()
fig6 = px.pie(na_genre_success_df, values='NA_Sales', names='Genre')
fig6.update_traces(textposition='inside', textinfo='label')
fig6.update_layout(font_family="Rockwell",
                   title_text='Genre Popularity in North America',
                   title_x=0.5, showlegend=False)

eu_genre_success_df = merged_df.filter(['Genre', 'EU_Sales']).dropna()
fig7 = px.pie(eu_genre_success_df, values='EU_Sales', names='Genre')
fig7.update_traces(textposition='inside', textinfo='label')
fig7.update_layout(font_family="Rockwell",
                   title_text='Genre Popularity in Europe',
                   title_x=0.5, showlegend=False)

jp_genre_success_df = merged_df.filter(['Genre', 'JP_Sales']).dropna()
fig8 = px.pie(jp_genre_success_df, values='JP_Sales', names='Genre')
fig8.update_traces(textposition='inside', textinfo='label')
fig8.update_layout(font_family="Rockwell",
                   title_text='Genre Popularity in Japan',
                   title_x=0.5, showlegend=False)

# Analyze publisher success across the 3 markets
reg_pub_success_df = merged_df.filter(['Publisher', 'NA_Sales',
                                       'EU_Sales', 'JP_Sales',
                                       'Global_Sales']).dropna()
top_pubs_reg = reg_pub_success_df.groupby('Publisher').agg({'NA_Sales': 'sum',
                                                            'EU_Sales': 'sum',
                                                            'JP_Sales': 'sum',
                                                            'Global_Sales':
                                                            'sum'})
top_pubs_reg = top_pubs_reg.reset_index()
top_pubs_reg = top_pubs_reg.sort_values(['Global_Sales'],
                                        ascending=False).head(10)
top_pubs_reg = top_pubs_reg.filter(['Publisher', 'NA_Sales', 'EU_Sales',
                                    'JP_Sales'])
top_pubs_reg = top_pubs_reg.rename(columns={'NA_Sales': 'NA', 'EU_Sales': 'EU',
                                            'JP_Sales': 'JP'})
top_pubs_reg = pd.melt(top_pubs_reg, id_vars=['Publisher'], var_name='Region',
                       value_name='Sales')
fig9 = px.bar(top_pubs_reg, x='Publisher', y='Sales', color='Region',
              barmode='group', template='plotly_white')
fig9.update_layout(font_family="Rockwell",
                   title_text='Regional Publisher Popularity', title_x=0.5)

# Analysis Question 3

# First we'll look for any correlation between the critic score and the total
# units shipped
critic_success_df = merged_df.filter(['Critic_Score', 'Global_Sales', 'Name'])
critic_success_df = critic_success_df.dropna()
fig1 = px.scatter(critic_success_df, x='Critic_Score', y='Global_Sales',
                  trendline='ols', trendline_color_override="red",
                  labels={'Global_Sales': 'Global Sales (Millions)',
                          'Critic_Score': 'Metacritic Score'},
                  title='Do Critic Scores Influence Success?',
                  template='plotly_white')
fig1.add_annotation(text='Wii Sports!', x=7.7, y=82.86, arrowhead=1,
                    showarrow=True)
fig1.update_layout(font_family="Rockwell", title_x=0.5)

# Exclude outlier
no_wii_sports = critic_success_df[critic_success_df['Name'] !=
                                  'Wii Sports (Wii)']
fig2 = px.scatter(no_wii_sports, x='Critic_Score', y='Global_Sales',
                  trendline='ols', trendline_color_override="red",
                  labels={'Global_Sales': 'Global Sales (Millions)',
                          'Critic_Score': 'Metacritic Score'},
                  title='Do Critic Scores Influence Success? (No Wii Sports)',
                  template='plotly_white')
fig2.update_layout(font_family="Rockwell", title_x=0.5, showlegend=False)

# So now after evaluating the correlation between metacritic score and total
# units shipped (with and without the biggest outlier) we can see that the
# correlation is virtually nonexistent

# Analysis Question 4


def predict_country(data):
    # Filter data to columns needed
    temp = data[['Genre', 'Platform', 'Developer', 'Critic_Score',
                 'ESRB_Rating', 'NA_Sales', 'EU_Sales', 'JP_Sales']]
    # Drop Nan values
    temp = temp.dropna()
    # Add column for country with best sales
    best_country = temp[['NA_Sales', 'EU_Sales', 'JP_Sales']].idxmax(axis=1)
    temp['Best_Country'] = best_country
    # Process data for ML
    features = temp.loc[:, ['Genre', 'Platform', 'Developer', 'Critic_Score',
                            'ESRB_Rating']]
    features = pd.get_dummies(features)
    labels = temp['Best_Country']
    (features_train, features_test,
     labels_train, labels_test) = train_test_split(features,
                                                   labels, test_size=0.25)
    # Train model
    model = DecisionTreeClassifier()
    model.fit(features_train, labels_train)
    # Assess model
    train_pred = model.predict(features_train)
    train_acc = accuracy_score(labels_train, train_pred)
    test_predictions = model.predict(features_test)
    test_acc = accuracy_score(labels_test, test_predictions)
    print(train_acc, test_acc)


predict_country(merged_df)


def predict_sales(data):

    temp = data[['Genre', 'Platform', 'Developer', 'Critic_Score',
                 'ESRB_Rating', 'Global_Sales']]
    temp = temp.dropna()
    features = temp.loc[:, temp.columns != 'Global_Sales']
    features = pd.get_dummies(features)
    labels = temp['Global_Sales']
    (features_train, features_test,
     labels_train, labels_test) = train_test_split(features, labels,
                                                   test_size=0.25)
    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)
    train_pred = model.predict(features_train)
    test_predictions = model.predict(features_test)
    train_error = mean_squared_error(labels_train, train_pred)
    test_error = mean_squared_error(labels_test, test_predictions)
    print("Train error:", train_error, "Test_error:", test_error)


predict_sales(merged_df)
