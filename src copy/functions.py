#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:39:22 2020
final project
@author: caoxinyu
"""
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt


def function_to_import_data(data_set_name: str):
    """
    This function will import the dataset and clean it by dropping
    NA values. Then select the variables that I need and make them into a new
    dataset.
    """
    data_set = pd.read_csv(data_set_name)
    data_set.dropna()
    data_set.columns = ['rank', 'country', 'score', 'GDP', 'social_support',
                        'life_expectancy', 'freedom_to_make_choice',
                        'generosity', 'perception_of_corruption']
    new_df = data_set[['score', 'GDP', 'social_support', 'life_expectancy',
                       'freedom_to_make_choice', 'generosity',
                       'perception_of_corruption']]
    return new_df


def function_with_data_visualization():
    """ Take the new dataset, create different plots to look at the
    relationships between the dependent variable and the independent variables.
    Try to think if the relationships in the graphs make sense.
    """
    data_used = function_to_import_data('data_2019.csv')
    # pairplot to show correlations between all variables.
    pair_plot = sns.pairplot(data=data_used)
    plt.figure()
    # create scatter plot to see correlation between happiness score and GDP,
    # happiness score and life expectancy, and happiness score and social
    # support.
    scatter_gdp = sns.regplot(data=data_used, x="GDP", y="score")
    plt.figure()
    scatter_life = sns.regplot(data=data_used, x="life_expectancy",
                               y="score")
    plt.figure()
    scatter_support = sns.regplot(data=data_used, x="social_support",
                                  y="score")
    plt.figure()
    heat_map_info = data_used[['score', 'GDP', 'life_expectancy',
                               'social_support']]
    corr = heat_map_info.corr(method='pearson')
    heat_map = sns.heatmap(corr)
    return pair_plot, scatter_gdp, scatter_life, scatter_support, heat_map


def function_descriptive_stats():
    """ Use describe() function to generate descriptive statistics for the
    variables in the dataset
    """
    data_set_used = function_to_import_data('data_2019.csv')
    stats = data_set_used.describe()
    pd.set_option('display.max_columns', None)
    return stats


# Run the OLS estimation and modify the results into the dataframe
def results_summary_to_dataframe():
    '''take the result of an statsmodel results table and transforms it
        into a dataframe'''
    data_frame = function_to_import_data('data_2019.csv')
    x_value = data_frame[['GDP', 'social_support', 'life_expectancy',
                          'freedom_to_make_choice', 'generosity',
                          'perception_of_corruption']]
    y_value = data_frame['score']
    new_x = sm.add_constant(x_value)
    model = sm.OLS(y_value, new_x).fit()
    pvals = round(model.pvalues, 3)
    coeff = round(model.params, 4)
    conf_lower = model.conf_int()[0]
    conf_higher = model.conf_int()[1]
    results_df = pd.DataFrame({"pvals": pvals,
                               "coeff": coeff,
                               "conf_lower": round(conf_lower, 3),
                               "conf_higher": round(conf_higher, 3)})
    # Reordering...
    results_df = results_df[["coeff", "pvals", "conf_lower", "conf_higher"]]
    return results_df
