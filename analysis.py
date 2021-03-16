"""
Drake Watson & Drew Blik
CSE 163 A Winter 2021
Analyzes video games sales data through the use of plotly visualizations
and machine learning models in order to get a better grasp on what determines
a video games success globally and regionally.
"""

from import_data import data
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import tree


def plot_global_genres(data):
    """
    Generate a pie chart displaying the global sales of each genre type
    that is present in the merged data frame.
    """
    global_genre_success_df = data.filter(['Genre', 'Global_Sales']).dropna()
    fig = px.pie(global_genre_success_df, values='Global_Sales', names='Genre')
    fig.update_traces(textposition='inside', textinfo='label')
    fig.update_layout(font_family="Rockwell",
                      title_text='Global Genre Popularity',
                      title_x=0.5, showlegend=False)
    fig.write_image('images/global_genres.png')


def plot_global_esrb(data):
    """
    Generate a pie chart displaying the global sales of each ESRB type
    that is present in the merged data frame.
    """
    global_ESRB_success_df = data.filter(['ESRB_Rating',
                                          'Global_Sales']).dropna()
    fig = px.pie(global_ESRB_success_df, values='Global_Sales',
                 names='ESRB_Rating')
    fig.update_traces(textposition='inside', textinfo='label')
    fig.update_layout(font_family="Rockwell",
                      title_text='Global ESRB Popularity',
                      title_x=0.5, showlegend=False)
    fig.write_image('images/global_esrbs.png')


def plot_global_publishers(data):
    """
    Generate a bar chart displaying the global sales of the top 10 publishers.
    """
    global_pub_success_df = data.filter(['Publisher', 'Genre',
                                         'Global_Sales']).dropna()
    top_pubs = global_pub_success_df.groupby('Publisher')['Global_Sales'].sum()
    top_pubs = top_pubs.reset_index()
    top_10_pubs = top_pubs.sort_values(['Global_Sales'],
                                       ascending=False).head(10)
    fig = px.bar(top_10_pubs, x='Global_Sales', y='Publisher',
                 orientation='h', title='Global Publisher Success',
                 template='plotly_white',
                 labels={'Global_Sales': 'Global Sales (Millions)'})
    fig.update_layout(font_family="Rockwell", title_x=0.5)
    fig.write_image('images/global_publishers.png')


def plot_na_genres(data):
    """
    Isolate the North American Sales and evaluate which genres are most popular
    by generating a pie chart.
    """
    na_genre_success_df = data.filter(['Genre', 'NA_Sales']).dropna()
    fig = px.pie(na_genre_success_df, values='NA_Sales', names='Genre')
    fig.update_traces(textposition='inside', textinfo='label')
    fig.update_layout(font_family="Rockwell",
                      title_text='Genre Popularity in North America',
                      title_x=0.5, showlegend=False)
    fig.write_image('images/north_american_genres.png')


def plot_eu_genres(data):
    """
    Isolate the European Sales and evaluate which genres are most popular
    by generating a pie chart.
    """
    eu_genre_success_df = data.filter(['Genre', 'EU_Sales']).dropna()
    fig = px.pie(eu_genre_success_df, values='EU_Sales', names='Genre')
    fig.update_traces(textposition='inside', textinfo='label')
    fig.update_layout(font_family="Rockwell",
                      title_text='Genre Popularity in Europe',
                      title_x=0.5, showlegend=False)
    fig.write_image('images/european_genres.png')


def plot_jp_genres(data):
    """
    Isolate the Japanese Sales and evaluate which genres are most popular
    by generating a pie chart.
    """
    jp_genre_success_df = data.filter(['Genre', 'JP_Sales']).dropna()
    fig = px.pie(jp_genre_success_df, values='JP_Sales', names='Genre')
    fig.update_traces(textposition='inside', textinfo='label')
    fig.update_layout(font_family="Rockwell",
                      title_text='Genre Popularity in Japan',
                      title_x=0.5, showlegend=False)
    fig.write_image('images/japanese_genres.png')


def plot_top_publishers_regional(data):
    """
    Using the merged data frame, find the top 10 publishers in the world
    and then create a grouped bar chart that displays how each of those
    publishers perform in each of the 3 regions of our sales data.
    """
    reg_pub_df = data.filter(['Publisher', 'NA_Sales',
                              'EU_Sales', 'JP_Sales',
                              'Global_Sales']).dropna()
    top_pubs_reg = reg_pub_df.groupby('Publisher').agg({'NA_Sales': 'sum',
                                                        'EU_Sales': 'sum',
                                                        'JP_Sales': 'sum',
                                                        'Global_Sales':
                                                        'sum'})
    top_pubs_reg = top_pubs_reg.reset_index()
    top_pubs_reg = top_pubs_reg.sort_values(['Global_Sales'],
                                            ascending=False).head(10)
    top_pubs_reg = top_pubs_reg.filter(['Publisher', 'NA_Sales', 'EU_Sales',
                                        'JP_Sales'])
    top_pubs_reg = top_pubs_reg.rename(columns={'NA_Sales': 'NA',
                                                'EU_Sales': 'EU',
                                                'JP_Sales': 'JP'})
    top_pubs_reg = pd.melt(top_pubs_reg, id_vars=['Publisher'],
                           var_name='Region',
                           value_name='Sales')
    fig = px.bar(top_pubs_reg, x='Publisher', y='Sales', color='Region',
                 barmode='group', template='plotly_white')
    fig.update_layout(font_family="Rockwell",
                      title_text='Regional Publisher Popularity', title_x=0.5)
    fig.write_image('images/top_publishers_regional.png')


def plot_critic_correlation(data):
    """
    Evaluate the relationship between critic review scores and global sales
    by generating a scatter plot with a regression line of all of the games
    that we have both sales data for and critic data for.
    """
    critic_success_df = data.filter(['Critic_Score', 'Global_Sales', 'Name'])
    critic_success_df = critic_success_df.dropna()
    fig = px.scatter(critic_success_df, x='Critic_Score', y='Global_Sales',
                     trendline='ols', trendline_color_override="red",
                     labels={'Global_Sales': 'Global Sales (Millions)',
                             'Critic_Score': 'Metacritic Score'},
                     title='Do Critic Scores Influence Success?',
                     template='plotly_white')
    fig.add_annotation(text='Wii Sports!', x=7.7, y=82.86, arrowhead=1,
                       showarrow=True)
    fig.update_layout(font_family="Rockwell", title_x=0.5)
    fig.write_image('images/critic_sales_correlation.png')


def predict_country(data):
    """
    Create a machine learning model that attempts to predict which region
    a game sold the most in. The features in the model will include genre,
    platform, developer, critic score, and esrb rating while the labels
    will be the sales data in each respective region. The labels will be
    converted into a single column that identifies which country the game
    performed best in.
    """
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
    print("Predicting Region With Most Sales:")
    print("Classifier Train Accuracy:", train_acc, "Classifier Test Accuracy:",
          test_acc)
    plt.figure(figsize=[15, 7], dpi=300)
    tree.plot_tree(model, max_depth=2, filled=True, proportion=True)
    plt.savefig('images/decision_tree.png')


def predict_sales(data):
    """
    Attempt to create a machine learning model that predicts the sales
    amount of a game using the genre, platform, developer, critic score,
    and esrb rating as features. This is likely to be unsuccessful due to
    the nuanced nature of sales and the lack of data in the data set.
    """
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
    print("Predicting Total Sales:")
    print("Regressor Train Error:", train_error, "Regressor Test Error:",
          test_error)


def main():
    merged_df = data()
    plot_global_genres(merged_df)
    plot_global_esrb(merged_df)
    plot_global_publishers(merged_df)
    plot_na_genres(merged_df)
    plot_eu_genres(merged_df)
    plot_jp_genres(merged_df)
    plot_top_publishers_regional(merged_df)
    plot_critic_correlation(merged_df)
    predict_country(merged_df)
    predict_sales(merged_df)


if __name__ == '__main__':
    main()
