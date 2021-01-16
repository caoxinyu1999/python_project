#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:41:52 2020

@author: caoxinyu
"""
import src.functions as func


def test_function_should_return_data_set():
    """
    Test whether the imported dataset contains the same number of rows as
    the orginal.
    """
    assert func.function_to_import_data("data_2019.csv").shape[0] == 156


def test_function_should_return_plot():
    """
    Test that the the function should return 5 figures.
    """
    assert len(func.function_with_data_visualization()) == 5


def test_function_should_return_descriptive_stats():
    """
    Test whether the function returns the correct max for score.
    """
    max_score = func.function_to_import_data("data_2019.csv")['score'].max()
    assert round(func.function_descriptive_stats().iloc[7, 0], 3) == max_score


def test_function_should_return_estimation_of_model():
    """
    Test whether the coefficient for GDP is 0.7754.
    """
    assert round(func.results_summary_to_dataframe().iloc[1, 0], 4) == 0.7754
