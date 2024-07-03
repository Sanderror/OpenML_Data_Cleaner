import pandas as pd
from dash import dash_table, html, dcc
import numpy as np
import json
import re
import ast
import nltk
import wordninja
from typing import List, Tuple, Dict
from sklearn.metrics import classification_report
from nltk.stem.wordnet import WordNetLemmatizer
import random
from openai import OpenAI
import time
import copy
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pFAHES.common as common
import pFAHES.patterns as patterns
import pFAHES.DV_Detector as DV_Detector
import pFAHES.RandDMVD as RandDMVD
import pFAHES.OD as OD
from statistics import mean, median, mode
import copy
from sortinghatinf import get_sortinghat_types
from numpy import percentile
from deepchecks.tabular.checks import MixedDataTypes
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import StringMismatch
from PyNomaly import loop
import cleanlab
import numpy as np
from numpy.linalg import norm
from error_detection import *
import pandas as pd
from sklearn.metrics import accuracy_score
from dash import dash_table, html, dcc
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestClassifier
import openml
import numpy as np
import math
import json
import re
import nltk
import wordninja
from typing import List, Tuple, Dict
from sklearn.metrics import classification_report
from nltk.stem.wordnet import WordNetLemmatizer
import random
from openai import OpenAI
import time
import copy
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import pFAHES.common as common
import pFAHES.patterns as patterns
import pFAHES.DV_Detector as DV_Detector
import pFAHES.RandDMVD as RandDMVD
import pFAHES.OD as OD
from statistics import mean, median, mode
import copy
from sortinghatinf import get_sortinghat_types
from numpy import percentile
from deepchecks.tabular.checks import MixedDataTypes
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import StringMismatch, OutlierSampleDetection, SpecialCharacters, MixedNulls
from PyNomaly import loop
import cleanlab
import numpy as np
from numpy.linalg import norm
from error_correction import *
from dashapp import *

def create_mvs_section(all_mv_table, df_mvs_summary):
    ''' Function to create the missing values section for the error detection step'''
    # If there are no missing values, we return an alert
    if df_mvs_summary.empty:
        section = html.Div([
            dbc.Alert(
                "No missing values were detected in your dataset",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize":'16pt',
                    "textAlign": "center",
                    'color':'white'
                }
            ),
            html.Div(id='mv-table', style={'display':'none'}),
            html.Div(id='mv-table-dropdown-container', style={'display':'none'}),
            html.Div(id='selected-mvs', style={'display': 'none'})
            ])
    # If there are missing values, we return a table with the missing values marked in colored cells + a validation table
    else:
        section = html.Div([
        dbc.Badge("Disguised missing value", color="orange", className="me-1"),
        dbc.Badge("Regular missing value", color="tomato", className="me-1"),
        html.Div(all_mv_table),
        html.H4("Are the following values in their corresponding columns correctly identifed as (disguised) missing values?"),
        html.P("Select in the \"Missing Value?\" column in the table below \"Yes\" if the value is correctly identified as a (disguised) missing value and select \"No\" if the value is not a (disguised) missing value."),
        html.Div([dash_table.DataTable(
            id='mv-table',
            data=df_mvs_summary.to_dict('records'),
            columns=[
                {'name': 'Column', 'id': 'Column', 'editable': False},
                {'name': 'Value', 'id': 'Value', 'editable': False},
                {'name': 'Frequency', 'id': 'Frequency', 'editable': False},
                {'name': 'Missing Value?', 'id': 'Missing Value?', 'presentation': 'dropdown'}
            ],
            editable=True,
            dropdown={
                'Missing Value?': {
                    'options': [{'label': "Yes", 'value': 'yes'}, {'label': "No", 'value': 'no'}]
                }
            },
            page_size=20
        ),
            html.Div(id='mv-table-dropdown-container'),
            html.Div(id='selected-mvs', style={'display': 'none'})])
        ])
    return section

def create_mvs_correct_section(df, mv_list, ft_types):
    ''' Function to create the missing values section for the error correction step'''
    if len(mv_list) == 0:
        return html.Div([
            dbc.Alert(
                "No missing values were submitted during the error detection step",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            ),
            html.Div(id='dropdown-correct-mvs', style={'display':'none'}),
            html.Div(id='correct-methods-mvs', style={'display': 'none'}),
            html.Div(id='mv-correct-method', style={'display': 'none'}),
            html.Div(id='submit-mv-correct-method', style={'display': 'none'}),
            html.Div(id='display-mv-correction', style={'display': 'none'})
            ])
    else:
        imp_mv_mdl = imputation_model(df, mv_list, ft_types, False, False)
        return html.Div([
            html.H4('Recommendations'),
            html.P(
                f"Our recommended correction technique for the missing values is imputation using {imp_mv_mdl} on all columns"),
            html.P(
                "In the dropdown menu below you can select the correction technique. If you select \'Imputation\', a section pops up to select the imputation technique"),
            html.Div(dcc.Dropdown(id='dropdown-correct-mvs', options=[{'label': 'Imputation', 'value': 'imputation'},
                                                                      {'label': 'Remove rows', 'value': 'remove'},
                                                                      {'label': 'Keep all', 'value': 'keep'}]),
                     style={'display': 'inline-block', 'width': '30%'}),
            html.Div(id='correct-methods-mvs'),
            html.Div(id='mv-correct-method', style={'display': 'none'}),
            html.Div(id='submit-mv-correct-method'),
            html.Div(id='display-mv-correction')
        ])


def create_dup_row_section(all_dup_rows_table, list_idx_delete):
    ''' Function to create the duplicate instances section for the error detection step'''
    # If there are no duplicate instances, we return an alert
    if len(list_idx_delete) == 0:
        section = html.Div([
            html.P("These are the rows in your dataset that contain identical values across all columns (except for the index column)"),
            dbc.Alert(
                "No duplicate rows were detected in your dataset",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            )
        ])
    # If there are duplicate instances, we return the table containing the duplicate instances
    else:
        section = html.Div([
            html.P(
                "These are the rows/rows in your dataset that contain identical values across all columns (except for the index column). In the table below, we have displayed these errors by outlining pairs of duplicate rows with the same color."),
            html.Div(all_dup_rows_table)])
    return section

def create_dup_row_correct_section(dup_rows):
    ''' Function to create the duplicate instances section for the error correction step'''
    if len(dup_rows) == 0:
        return html.Div([
            dbc.Alert(
                "No duplicate rows were submitted during the error detection step",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            ),
            html.Div(id='dropdown-correct-dup-rows', style={'display':'none'}),
            html.Div(id='correct-methods-dup-rows', style={'display':'none'}),
            html.Div(id='submit-dup-row-correct-method', style={'display':'none'}),
            html.Div(id='dup-row-correct-method', style={'display': 'none'}),
            html.Div(id='dup-row-retain-checklist', style={'display':'none'}),
            html.Div(id='output-submit-dup-rows', style={'display':'none'})
        ])
    else:
        return html.Div([
            html.H4('Recommendations'),
            html.P(
                f"Our recommended correction technique for the duplicate rows is to keep the first row of each duplicate group and remove the other rows in that group."),
            html.P(
                "In the dropdown menu below you can choose the correction technique you want to apply. The option \'Select keep/remove\' allows you to choose for every group which row you want to keep and which you want to remove."),
            html.Div(dcc.Dropdown(id='dropdown-correct-dup-rows',
                                  options=[{'label': 'Keep first, remove rest', 'value': 'keep_first'},
                                           {'label': 'Remove all', 'value': 'remove'},
                                           {'label': 'Keep all', 'value': 'keep_all'},
                                           {'label': 'Select keep/remove', 'value': 'select'}]),
                     style={'display': 'inline-block', 'width': '30%'}),
            html.Div(id='correct-methods-dup-rows'),
            html.Div(id='submit-dup-row-correct-method'),
            html.Div(id='dup-row-correct-method', style={'display': 'none'}),
            html.Div(id='dup-row-retain-checklist'),
            html.Div(id='output-submit-dup-rows')
        ])


def create_dup_col_section(all_dup_cols_table, list_col_delete):
    ''' Function to create the duplicate attributes section of the error detection step'''
    # If there are no duplicate attributes, we return an alert
    if len(list_col_delete) == 0:
        section = html.Div([
            html.P("Similar to the duplicate rows, duplicate attributes are attributes/columns in your dataset that contain identical values across all rows"),
            dbc.Alert(
                "No duplicate attributes were detected in your dataset",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            )
        ])
    # If there are duplicate attributes, we return the table containing the duplicate attributes
    else:
        section = html.Div([
            html.P(
                "Similar to the duplicate rows, duplicate attributes are attributes/columns in your dataset that contain identical values across all rows. In the table below, we have displayed these errors by outlining pairs of duplicate attributes with the same color."),
            html.Div(all_dup_cols_table)])
    return section

def create_dup_col_correct_section(dup_cols):
    ''' Function to create the duplicate attributes section of the error correction step'''
    if len(dup_cols) == 0:
        return html.Div([
            dbc.Alert(
                "No duplicate columns were submitted during the error detection step",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            ),
            html.Div(id='dropdown-correct-dup-cols', style={'display': 'none'}),
            html.Div(id='correct-methods-dup-cols', style={'display': 'none'}),
            html.Div(id='submit-dup-col-correct-method', style={'display': 'none'}),
            html.Div(id='dup-col-correct-method', style={'display': 'none'}),
            html.Div(id='dup-col-retain-checklist', style={'display': 'none'}),
            html.Div(id='output-submit-dup-cols', style={'display': 'none'})
        ])
    else:
        return html.Div([
            html.H4("Recommendations"),
            html.P(
                "Our recommended correction technique for the duplicate columns is to keep the first column of each duplicate group and remove the other column in that group."),
            html.P(
                "In the dropdown menu below you can choose the correction technique you want to apply. The option \'Select keep/remove\' allows you to choose for every column which column you want to keep and which you want to remove."),
            html.Div(dcc.Dropdown(id='dropdown-correct-dup-cols',
                                  options=[{'label': 'Keep first, remove rest', 'value': 'keep_first'},
                                           {'label': 'Remove all', 'value': 'remove'},
                                           {'label': 'Keep all', 'value': 'keep_all'},
                                           {'label': 'Select keep/remove', 'value': 'select'}]),
                     style={'display': 'inline-block', 'width': '30%'}),
            html.Div(id='correct-methods-dup-cols'),
            html.Div(id='submit-dup-col-correct-method'),
            html.Div(id='dup-col-correct-method', style={'display': 'none'}),
            html.Div(id='dup-col-retain-checklist'),
            html.Div(id='output-submit-dup-cols')
        ])

def create_out_val_section(outlier_val_table, df_outs_summary):
    ''' Function to create the outlier vales section for the error detection step'''
    # If there are no outlier values, return an alert
    if df_outs_summary.empty:
        section = html.Div([
            dbc.Alert(
                "No outlier values were detected in your dataset",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            ),
            html.Div(id='outs-table', style={'display':'none'}),
            html.Div(id='outs-table-dropdown-container', style={'display':'none'}),
            html.Div(id='selected-outs', style={'display': 'none'})
        ])
    # if there are outlier values, return a table with the outlier marked in colored cells
    else:
        section = html.Div([
            dbc.Badge("Close outlier", color="orange", className="me-1"),
            dbc.Badge("Far outlier", color="tomato", className="me-1"),
            html.Div(outlier_val_table),  # Display outliers result
            html.P(
                "An outlier value does not necessarily have to be an error. For example, a value of 108 in a column \"age\" is rare, but not necessarily incorrect. Therefore, it is important that you manually check the detected outlier values yourself and correct them if needed."),
            html.H4(
                "Are the following outlier values in their corresponding columns actual errors?"),
            html.P(
                "Select in the \"Erroneous Outlier?\" column in the table below \"Yes\" if the outlier value is an actual data error and select \"No\" if the outlier value is a valid value."),
            html.Div([dash_table.DataTable(
                id='outs-table',
                data=df_outs_summary.to_dict('records'),
                columns=[
                    {'name': 'Column', 'id': 'Column', 'editable': False},
                    {'name': 'Type', 'id': 'Type', 'editable': False},
                    {'name': 'Value', 'id': 'Value', 'editable': False},
                    {'name': 'Erroneous Outlier?', 'id': 'Erroneous Outlier?', 'presentation': 'dropdown'}
                ],
                editable=True,
                dropdown={
                    'Erroneous Outlier?': {
                        'options': [{'label': "Yes", 'value': 'yes'}, {'label': "No", 'value': 'no'}]
                    }
                },
                page_size=20
            ),
                html.Div(id='outs-table-dropdown-container'),
                html.Div(id='selected-outs', style={'display': 'none'})]
            )])
    return section

def create_out_val_correct_section(df, out_vals, ft_types):
    ''' Function to create the outlier vales section for the error correction step'''
    if len(out_vals) == 0:
        return html.Div([
            dbc.Alert(
                "No outlier values were submitted during the error detection step",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            ),
            html.Div(id='dropdown-correct-outs', style={'display': 'none'}),
            html.Div(id='correct-methods-outs', style={'display': 'none'}),
            html.Div(id='out-correct-method', style={'display': 'none'}),
            html.Div(id='submit-out-correct-method', style={'display': 'none'}),
            html.Div(id='display-out-correction', style={'display': 'none'}),
        ])
    else:
        imp_out_mdl = imputation_model(df, out_vals, ft_types, True, False)
        return html.Div([
            html.H4("Recommendations"),
            html.P(
                f"Our recommended correction technique for the outlier values is imputation using {imp_out_mdl} on all columns"),
            html.P(
                "In the dropdown menu below you can select the correction technique. If you select \'Imputation\', a section pops up to select the imputation technique. If you select \'Type specific\', a section pops up where you can choose the correction method per outlier type (i.e. close or far)."),
            html.Div(dcc.Dropdown(id='dropdown-correct-outs', options=[{'label': 'Imputation', 'value': 'imputation'},
                                                                       {'label': 'Remove rows', 'value': 'remove'},
                                                                       {'label': 'Keep all', 'value': 'keep'},
                                                                       {'label': 'Type specific', 'value': 'type'}]),
                     style={'display': 'inline-block', 'width': '30%'}),
            html.Div(id='correct-methods-outs'),
            html.Div(id='out-correct-method', style={'display': 'none'}),
            html.Div(id='submit-out-correct-method'),
            html.Div(id='display-out-correction'),
        ])

def create_out_row_correct_section(out_rows):
    ''' Function to create the outlier instances section for the error correction step'''
    if len(out_rows) == 0:
        return html.Div([
            dbc.Alert(
                "No outlier rows were submitted during the error detection step",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            ),
            html.Div(id='dropdown-correct-out-rows', style={'display': 'none'}),
            html.Div(id='correct-methods-out-rows', style={'display': 'none'}),
            html.Div(id='out-row-correct-method', style={'display': 'none'}),
            html.Div(id='submit-out-row-correct-method', style={'display': 'none'}),
            html.Div(id='out-row-retain-checklist', style={'display': 'none'}),
            html.Div(id='output-submit-out-rows', style={'display': 'none'})
        ])
    else:
        return html.Div([
            html.H4("Recommendations"),
            html.P(
                "Our recommended correction technique for the outlier rows is to remove these rows from your dataset."),
            html.P(
                "In the dropdown menu below you can select the correction technique. The option \'Select keep/remove\' allows you to choose for every row which row you want to keep and which you want to remove."),
            html.Div(dcc.Dropdown(id='dropdown-correct-out-rows', options=[{'label': 'Remove all', 'value': 'remove'},
                                                                           {'label': 'Keep all', 'value': 'keep'},
                                                                           {'label': 'Select keep/remove',
                                                                            'value': 'select'}]),
                     style={'display': 'inline-block', 'width': '30%'}),
            html.Div(id='correct-methods-out-rows'),
            html.Div(id='out-row-correct-method', style={'display': 'none'}),
            html.Div(id='submit-out-row-correct-method'),
            html.Div(id='out-row-retain-checklist'),
            html.Div(id='output-submit-out-rows')
        ])

def create_cryp_correct_section(df, cryp_cols):
    ''' Function to create the cryptic attribute names section for the error correction step'''
    if len(cryp_cols) == 0:
        return html.Div([
            dbc.Alert(
                "No cryptic column names were submitted during the error detection step",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            ),
            html.Div(id='dropdown-correct-cryp', style={'display': 'none'}),
            html.Div(id='cryp-correct-table', style={'display': 'none'}),
            html.Div(id='correct-methods-cryp', style={'display': 'none'}),
            html.Div(id='cryp-correct-method', style={'display': 'none'}),
            html.Div(id='submit-cryp-correct-method', style={'display': 'none'}),
            html.Div(id='cryp-retain-checklist', style={'display': 'none'}),
            html.Div(id='output-submit-cryp', style={'display': 'none'})
        ])
    else:
        cryp_dtb = create_cryp_correct(df, cryp_cols)
        return html.Div([
            html.H4("Recommendations"),
            html.P(
                "Our recommended correction technique for the cryptic column names is to change them to the suggested names of GPT-3.5."),
            html.P(
                "Below you can see the suggested new column names instead of the cryptic names as suggested by GPT-3.5. You can also change the names in the \'Corrected name\' column yourself."),
            cryp_dtb,
            html.P(
                "In the dropdown menu below you can select the correction technique. The option \'Select keep/change\' allows you to choose for every column whether you want to keep the previous name or change it to the suggested name."),
            html.Div(dcc.Dropdown(id='dropdown-correct-cryp',
                                  options=[{'label': 'Change to suggested names', 'value': 'suggested'},
                                           {'label': 'Keep original names', 'value': 'keep'},
                                           {'label': 'Select keep/change', 'value': 'select'}]),
                     style={'display': 'inline-block', 'width': '30%'}),
            html.Div(id='correct-methods-cryp'),
            html.Div(id='cryp-correct-method', style={'display': 'none'}),
            html.Div(id='submit-cryp-correct-method'),
            html.Div(id='cryp-retain-checklist'),
            html.Div(id='output-submit-cryp')
        ])

def create_sv_correct_section(sv_cols):
    ''' Function to create the single value attributes section for the error correction step'''
    if len(sv_cols) == 0:
        return html.Div([
            dbc.Alert(
                "No single value columns were submitted during the error detection step",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            ),
            html.Div(id='dropdown-correct-svs', style={'display': 'none'}),
            html.Div(id='correct-methods-svs', style={'display': 'none'}),
            html.Div(id='svs-correct-method', style={'display': 'none'}),
            html.Div(id='submit-svs-correct-method', style={'display': 'none'}),
            html.Div(id='sv-retain-checklist', style={'display': 'none'}),
            html.Div(id='output-submit-svs', style={'display': 'none'}),
            html.Div(id='renamed-cols', style={'display': 'none'})
        ])
    else:
        return html.Div([
            html.H4("Recommendations"),
            html.P(
                "Our recommended correction technique for the single value columns is to remove them from the dataset."),
            html.P(
                "In the dropdown menu below you can select the correction technique. The option \'Select keep/remove\' allows you to choose which single value columns you want to keep and which you want to remove."),
            html.Div(dcc.Dropdown(id='dropdown-correct-svs', options=[{'label': 'Remove all', 'value': 'remove'},
                                                                      {'label': 'Keep all', 'value': 'keep'},
                                                                      {'label': 'Select keep/remove',
                                                                       'value': 'select'}]),
                     style={'display': 'inline-block', 'width': '30%'}),
            html.Div(id='correct-methods-svs'),
            html.Div(id='svs-correct-method', style={'display': 'none'}),
            html.Div(id='submit-svs-correct-method'),
            html.Div(id='sv-retain-checklist'),
            html.Div(id='output-submit-svs'),
            html.Div(id='renamed-cols', style={'display': 'none'})
        ])

def create_md_correct_section(df, mix_dct, ft_types):
    ''' Function to create the mixed data type attributes section for the error correction step'''
    if len(mix_dct) == 0:
        return html.Div([
            dbc.Alert(
                "No mixed data type columns were submitted during the error detection step",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            ),
            html.Div(id='dropdown-mds-major-minor', style={'display': 'none'}),
            html.Div(id='md-col-specific', style={'display': 'none'}),
            html.Div(id='md-dropdown-section', style={'display': 'none'}),
            html.Div(id='dropdown-correct-mds', style={'display': 'none'}),
            html.Div(id='correct-methods-mds', style={'display': 'none'}),
            html.Div(id='mds-correct-method', style={'display': 'none'}),
            html.Div(id='submit-mds-correct-method', style={'display': 'none'}),
            html.Div(id='complete-md-list', style={'display': 'none'}),
            html.Div(id='display-mds-correction', style={'display': 'none'})
        ])
    else:
        imp_md_mdl = imputation_model(df, mix_dct, ft_types, False, True)
        return html.Div([
            html.H4("Recommendations"),
            html.P(
                f"Our recommended correction technique for the mixed data type columns is to convert the minority data type to the majority data type using {imp_md_mdl} imputation."),
            html.P(
                "In the dropdown menu below, you must first select whether you want to correct all minority types, all majority types, or correct them column-specific."),
            html.Div(dcc.Dropdown(id='dropdown-mds-major-minor', options=[{'label': 'Minority type', 'value': 'minor'},
                                                                          {'label': 'Majority type', 'value': 'major'},
                                                                          {'label': 'Column-specific types',
                                                                           'value': 'column'}]),
                     style={'display': 'inline-block', 'width': '30%'}),
            html.Div(id='md-col-specific', style={'display': 'none'}),
            html.Div(id='md-dropdown-section'),
            html.Div(id='dropdown-correct-mds'),
            html.Div(id='correct-methods-mds'),
            html.Div(id='mds-correct-method', style={'display': 'none'}),
            html.Div(id='submit-mds-correct-method'),
            html.Div(id='complete-md-list', style={'display': 'none'}),
            html.Div(id='display-mds-correction')
        ])

def create_ml_correct_section(ml_list):
    ''' Function to create the incorrect labels section for the error correction step'''
    if len(ml_list) == 0:
        return html.Div([
            dbc.Alert(
                "No incorrect labels were submitted during the error detection step",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            ),
            html.Div(id='dropdown-correct-mls', style={'display': 'none'}),
            html.Div(id='correct-methods-mls', style={'display': 'none'}),
            html.Div(id='ml-correct-method', style={'display': 'none'}),
            html.Div(id='submit-ml-correct-method', style={'display': 'none'}),
            html.Div(id='display-ml-correction', style={'display': 'none'})
        ])
    else:
        return html.Div([
            html.H4("Recommendations"),
            html.P(
                "Our recommended correction technique for the incorrect labels is to convert them to the suggested correct label."),
            html.P("In the dropdown menu below, you can select the correction technique."),
            html.Div(dcc.Dropdown(id='dropdown-correct-mls',
                                  options=[{'label': 'Convert to correct label', 'value': 'convert'},
                                           {'label': 'Remove rows', 'value': 'remove'},
                                           {'label': 'Keep all', 'value': 'keep'}]),
                     style={'display': 'inline-block', 'width': '30%'}),
            html.Div(id='correct-methods-mls'),
            html.Div(id='ml-correct-method', style={'display': 'none'}),
            html.Div(id='submit-ml-correct-method'),
            html.Div(id='display-ml-correction')
        ])

def create_mm_correct_section(mm_list):
    ''' Function to create the string mismatches section for the error correction step'''
    if len(mm_list) == 0:
        return html.Div([
            dbc.Alert(
                "No string mismatches were submitted during the error detection step",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            ),
            html.Div(id='dropdown-correct-mms', style={'display': 'none'}),
            html.Div(id='correct-methods-mms', style={'display': 'none'}),
            html.Div(id='mm-correct-method', style={'display': 'none'}),
            html.Div(id='submit-mm-correct-method', style={'display': 'none'}),
            html.Div(id='display-mm-correction', style={'display': 'none'})
        ])
    else:
        return html.Div([
            html.H4("Recommendations"),
            html.P(
                "Our recommended correction technique for the string mismatches is to convert the mismatch variations to the suggested base form."),
            html.P(
                "In the dropdown menu below, you can select the correction technique. The \'Convert to mode\' option converts the mismatch variations to the most frequently occurring variation of that mismatch."),
            html.Div(dcc.Dropdown(id='dropdown-correct-mms',
                                  options=[{'label': 'Convert to base form', 'value': 'convert_base'},
                                           {'label': 'Convert to mode', 'value': 'convert_mode'},
                                           {'label': 'Remove rows', 'value': 'remove'},
                                           {'label': 'Keep all', 'value': 'keep'}]),
                     style={'display': 'inline-block', 'width': '30%'}),
            html.Div(id='correct-methods-mms'),
            html.Div(id='mm-correct-method', style={'display': 'none'}),
            html.Div(id='submit-mm-correct-method'),
            html.Div(id='display-mm-correction')
        ])

def badge_creator_mvs(imp, remove, keep):
    ''' Function to create the badges above the correction sections for
    the missing values, outliers and mixed data types'''
    badge = html.Div("")
    # If values are imputed, we display the imputation badge
    # If instances are removed, we display the remove badge
    # If values are retained, we display the retain badge
    if imp:
        if remove:
            if keep:
                badge = html.Div([dbc.Badge("Imputed value", color="#50C878", className="me-1"),
                                    dbc.Badge("Removed row", color="tomato", className="me-1"),
                                  dbc.Badge("Retained value", color='#0080FF', className='me-1')])
            else:
                badge = html.Div([dbc.Badge("Imputed value", color="#50C878", className="me-1"),
                                    dbc.Badge("Removed row", color="tomato", className="me-1")])
        else:
            if keep:
                badge = html.Div([dbc.Badge("Imputed value", color="#50C878", className="me-1"),
                                  dbc.Badge("Retained value", color='#0080FF', className='me-1')])
            else:
                badge = html.Div([dbc.Badge("Imputed value", color="#50C878", className="me-1")])
    else:
        if remove:
            if keep:
                badge = html.Div([dbc.Badge("Removed row", color="tomato", className="me-1"),
                                  dbc.Badge("Retained value", color='#0080FF', className='me-1')])
            else:
                badge = html.Div([dbc.Badge("Removed row", color="tomato", className="me-1")])
        else:
            if keep:
                badge = html.Div([dbc.Badge("Retained value", color='#0080FF', className='me-1')])
            else:
                badge = html.Div("")
    return badge

def create_dup_group_section(dup_groups, df, type):
    ''' Function to create the select keep/remove section for the duplicate
    intances and attributes for the error correction step'''
    count = 0
    all_sections = []
    # For every duplicate group we generate a section where the user can select
    # which instances/attributes to keep for that specific group
    for dups in dup_groups:
        str_dups = str(dups).strip("[").strip("]")
        if type == 'row':
            keep_type = 'Indices'
            section_df = df.loc[dups].reset_index()
        else:
            keep_type = 'Columns'
            section_df = df.T.loc[dups].reset_index()
        section = html.Div([
            html.H4(f"Duplicate Group {count + 1} ({keep_type}: {str_dups})"),
            dash_table.DataTable(
                data=section_df.to_dict('records'),
                columns=[{"name": str(i), "id": str(i)} for i in section_df.columns],
                page_size=20
            ),
            html.H5(f"{keep_type} to keep:"),
            dbc.Checklist(id={'type': f'dup-{type}-retain-checklist', 'index': count},
            options=[{'label': str(i), 'value': i} for i in dups],
            value=[dups[0]],  # By default, keep the first occurrence
            labelStyle={'display': 'inline-block', 'margin-right': '10px'},
            switch=True),
        ])
        count += 1
        all_sections.append(section)
    return html.Div(all_sections)

def create_out_row_section(out_rows, df):
    ''' Function to create the select keep/remove for the outlier
    instances for the error correction step'''
    count = 0
    all_sections = []
    # For every outlier instance we generate a section where the user can select
    # whether they want to keep that outlier instance in the dataset or not
    for out_idx, out_prob in out_rows:
        section_df = df.loc[[out_idx]].reset_index()
        section = html.Div([
            html.H4(f"Outlier row {count + 1} (index: {out_idx}, probability: {out_prob})"),
            dash_table.DataTable(
                data=section_df.to_dict('records'),
                columns=[{"name": str(i), "id": str(i)} for i in section_df.columns],
                page_size=20
            ),
            html.H5("Keep this row?"),
            dbc.Checklist(id={'type': 'out-row-retain-checklist', 'index': count},
                          options=[{'label': str(i), 'value': i} for i in ['Yes']],
                          value=['Yes'],
                          labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                          switch=True),
        ])
        count += 1
        all_sections.append(section)
    return html.Div(all_sections)

def create_cryp_correct(df, cryptic_cols,  title=False, description=False, n_rows=0):
    ''' Function to generate non-cryptic names for the cryptic attribute names
    using OpenAI's GPT-3.5 and display the suggestions'''
    # Here, you have to fill in your API key yourself
    client = OpenAI(api_key="")

    # Following our query experiment, by default the query does not use example instances
    # nor a title, nor a description
    if n_rows == 0:
        rows = pd.DataFrame()
    else:
        # Obtain n random rows of content for the cryptic columns
        sliced_df = df[cryptic_cols]
        indices = random.sample(range(0, len(sliced_df)), n_rows)
        rows = sliced_df.iloc[indices]

    # Generate the OpenAI query
    query = query_cryp_cols(cryptic_cols, title=title, description=description, content_df=rows)

    # Call OpenAI's GPT 3.5 Turbo LLM model to generate better column names
    completion = client.chat.completions.create(model="gpt-3.5-turbo", temperature=0.0,
                                                messages=[{"role": "user", "content": query}])
    message = completion.choices[0].message
    new_col_names = message.content.split(" | ")

    # # temp fix because no openai credits left
    # new_col_names = ['Final Weight', 'Redundant Single Value Column']

    if len(new_col_names) != len(cryptic_cols):
        error_new_cols = ['Something went wrong' for i in range(len(cryptic_cols))]
        new_col_names = error_new_cols

    # Display the cryptic names and suggestions next to each other in a table
    # The user can change the suggestion of GPT-3.5 in this table
    df_dtb = pd.DataFrame()
    df_dtb['Cryptic name'] = cryptic_cols
    df_dtb['Corrected name'] = new_col_names
    dtb = dash_table.DataTable(
        id='cryp-correct-table',
        data=df_dtb.to_dict('records'),
        columns=[{'name': 'Cryptic name', 'id': 'Cryptic name', 'editable':False}, {'name':'Corrected name', 'id':'Corrected name'}],
        editable=True,
        page_size=20
    )
    return html.Div([dtb])

def create_cryp_select_section(cryp_table):
    ''' Function to create the select keep/change for the cryptic
     attribute names for the error correction step'''
    count = 0
    all_sections = []
    # For every cryptic attribute name we generate a section where the user can select
    # whether they want to use the cryptic or corrected attribute name in the dataset
    for cryp_dct in cryp_table:
        cols = list(cryp_dct.values())
        cryp_name = cols[0]
        corr_name = cols[1]
        section = html.Div([
            html.H4(f"Cryptic Attribute Name {count + 1} (cryptic: {cryp_name}, corrected: {corr_name})"),
            html.H5("Select the name you want in your dataset"),
            dbc.RadioItems(id={'type': 'cryp-retain-checklist', 'index': count},
                          options=[{'label': i, 'value': i} for i in [cryp_name, corr_name]],
                          value=corr_name,  # By default, choose for the corrected names
                          labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                          switch=True),
        ])
        count += 1
        all_sections.append(section)
    return html.Div(all_sections)

def create_sv_select_section(sv_cols, df):
    ''' Function to create the select keep/remove for the single
         value attributes for the error correction step'''
    count = 0
    all_sections = []
    # For every sv attribute we generate a section where the user can select
    # whether they want to keep that sv attribute in the dataset or not
    for sv_col in sv_cols:
        section_df = df.T.loc[[sv_col]].reset_index()
        section = html.Div([
            html.H4(f"Single Value Attribute {count + 1} (column: {sv_col})"),
            dash_table.DataTable(
                data=section_df.to_dict('records'),
                columns=[{"name": str(i), "id": str(i)} for i in section_df.columns],
                page_size=20
            ),
            html.H5("Keep this attribute?"),
            dbc.Checklist(id={'type': 'sv-retain-checklist', 'index': count},
                          options=[{'label': str(i), 'value': i} for i in ['Yes']],
                          value=['Yes'],
                          labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                          switch=True),
        ])
        count += 1
        all_sections.append(section)
    return html.Div(all_sections)

def create_md_select_type(md_dct, method='column'):
    ''' Function to display a data table where the user can select
    per mixed data type column the data type to correct. Furthermore,
    it also stores the to be corrected types for the other two methods: minor and major'''
    dtb_dct = dict()
    df_dtb = pd.DataFrame()
    # If the column-specific method is chosen, we need to display the percentage of
    # each data type in that column, therefore the
    for col, val_dct in md_dct.items():
        # If the column-specific method is chosen, we need to display the percentage of
        # each data type in that column, therefore the labels are different then for minor and major
        if method == 'column':
            dtb_dct[col] = [{'label': f"strings ({round(val_dct['strings'] * 100, 2)}%)", 'value': 'strings'},
                            {'label': f"numbers ({round(val_dct['numbers'] * 100, 2)}%)", 'value': 'numbers'}]
        else:
            dtb_dct[col] = [{'label': 'strings', 'value': 'strings'}]
        # If there are more numeric values in the column, the majority method will
        # replace the numeric values, while the minority will replace the strings
        # For the data table, the column-specific method gets initially assigned the minority method
        if val_dct['strings'] < val_dct['numbers']:
            if method == 'major':
                df_dtb[col] = ['numbers']
            else:
                df_dtb[col] = ['strings']
        else:
            if method == 'major':
                df_dtb[col] = ['strings']
            else:
                df_dtb[col] = ['numbers']

    return dash_table.DataTable(
        id='md-col-specific',
        data=df_dtb.to_dict('records'),
        columns=[{'name':i, 'id':i, 'presentation':'dropdown'} for i in df_dtb.columns],
        dropdown={
            col :{'options':dtb_dct[col]} for col in df_dtb.columns
        },
        editable=True
    )

def generate_md_section(df, md_list, method, model, global_method, button_style):
    '''  Generates the correction technique section for the mixed data types
    after the user has chosen what data type to address (per column).
    md_list has for every column the values to be imputed and the frequency of those
    values stored in a list, like this: [['education', 0, 10], ['mix_d_types', 'MD1', 5] etc.].
    '''

    # Depending on the data type to address, different outputs will be displayed
    if global_method == 'minor' or global_method == 'major':
        output_string = f" of the {global_method}ity data type"
        after_string = ""
    else:
        output_string = ""
        after_string = "according to the column-specific types chosen above "


    if method == 'imputation':
        types = dict()
        for md in md_list:
            val = md[1]
            # if the to be corrected values are strings, the column type will be numeric and vice versa
            if isinstance(val, str) and not val.isdigit():
                types[md[0]] = 'number'
            else:
                types[md[0]] = 'string'
        unique_cols = list(set([md[0] for md in md_list]))
        dct_imp = {col:[model] for col in unique_cols}
        df_imp = pd.DataFrame(dct_imp)
        numeric_models = ['KNN', 'RF', 'MLP', 'CART', 'Mean', 'Median', 'Mode', 'Remove', 'Keep']
        numeric_options = [{'label':model, 'value':model} for model in numeric_models]
        categorical_models = ['KNN', 'RF', 'MLP', 'CART', 'Mode', 'Remove', 'Keep']
        categorical_options = [{'label': model, 'value': model} for model in categorical_models]
        # for each mix data type column we display the alternative imputation techinques in a table
        # where the user can select the one he/she likes per column
        dtb = dash_table.DataTable(
            id='mds-correct-method',
            data=df_imp.to_dict('records'),
            columns=[{'name':i, 'id':i, 'presentation':'dropdown'} for i in df_imp.columns],
            dropdown={col: {'options': numeric_options} if types[col] == 'number' else {'options': categorical_options} for col in df_imp.columns},
            editable=True
        )
        return html.Div([
            html.P(f"You have selected \'Imputation\' as correction technique. This means that all values in the mixed data type columns{output_string} will be imputed {after_string}using the {model} model."),
            html.P("Below you can select the imputation method per column."),
            dtb,
            html.P("Please press the button below to submit your imputation methods."),
            html.Button("Submit", id='submit-mds-correct-method', n_clicks=0, style=button_style)
        ])

    # For the remove and keep options, we dont need to construct a table. We simply display some explanation text
    elif method == 'remove':
        return html.Div([
            dcc.Store(id='mds-correct-method', data=method, storage_type='memory'),
            html.P(f"You have chosen the option \'Remove rows\'. This means that all rows in the mixed data type columns{output_string} will be removed from the dataset{after_string}"),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-mds-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'keep':
        return html.Div([
            dcc.Store(id='mds-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Keep all\'. This means that all detected mixed data type columns will be retained in the dataset"),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-mds-correct-method', n_clicks=0, style=button_style)
        ])
    else:
        method = 'nothing'
        return html.Div([
            dcc.Store(id='mds-correct-method', data=method, storage_type='memory'),
            html.Button("Submit", id='submit-mds-correct-method', n_clicks=0, style={'display':'none'})
        ])

def create_badge_numbers(df, num_df, ft_types, target, mv_list, dup_rows, dup_cols, out_vals, out_rows, cryp_cols, sv_cols, mix_data_dct, ml_list, mm_list):
    ''' Function to create the updated numbers and colors of the badges in the headers of the error sections'''
    error_summary = dict()

    # missing values
    mv_count = sum([mv[2] for mv in mv_list])
    mv_pct = round((mv_count / df.size) * 100, 2)
    mv_color = '#50C878' if mv_pct == 0 else 'orange' if mv_pct < 10 else 'tomato'
    error_summary['mvs'] = [mv_count, mv_pct, mv_color]

    # dup rows
    dup_row_count = len([dup for dups in dup_rows for dup in dups])
    dup_row_pct = round((dup_row_count / len(df)) * 100, 2)
    dup_row_color = '#50C878' if dup_row_pct == 0 else 'orange' if dup_row_pct < 10 else 'tomato'
    error_summary['dup_rows'] = [dup_row_count, dup_row_pct, dup_row_color]

    # dup cols
    dup_col_count = len([dup for dups in dup_cols for dup in dups])
    dup_col_pct = round((dup_col_count / len(df.columns)) * 100, 2)
    dup_col_color = '#50C878' if dup_col_pct == 0 else 'orange' if dup_col_pct < 10 else 'tomato'
    error_summary['dup_cols'] = [dup_col_count, dup_col_pct, dup_col_color]

    # out vals
    out_val_lens = []
    for col, typ, val in out_vals:
        out_val_lens.append(len(df[df[col] == val]))
    out_val_count = sum(out_val_lens)
    out_val_pct = round((out_val_count / df.size) * 100, 2)
    out_val_color = '#50C878' if out_val_pct == 0 else 'orange' if out_val_pct < 10 else 'tomato'
    error_summary['out_vals'] = [out_val_count, out_val_pct, out_val_color]

    # out rows
    out_row_count = len(out_rows)
    out_row_pct = round((out_row_count / len(df)) * 100, 2)
    out_row_color = '#50C878' if out_row_pct == 0 else 'orange' if out_row_pct < 10 else 'tomato'
    error_summary['out_rows'] = [out_row_count, out_row_pct, out_row_color]

    # cryp cols
    cryp_count = len(cryp_cols)
    cryp_pct = round((cryp_count / len(df.columns)) * 100, 2)
    cryp_color = '#50C878' if cryp_pct == 0 else 'orange' if cryp_pct < 30 else 'tomato'
    error_summary['cryps'] = [cryp_count, cryp_pct, cryp_color]

    # sv cols
    sv_count = len(sv_cols)
    sv_pct = round((sv_count / len(df.columns)) * 100, 2)
    sv_color = '#50C878' if sv_pct == 0 else 'orange' if sv_pct < 10 else 'tomato'
    error_summary['svs'] = [sv_count, sv_pct, sv_color]

    # md cols
    md_count = len(mix_data_dct)
    md_pct = round((md_count / len(df.columns)) * 100, 2)
    md_color = '#50C878' if md_pct == 0 else 'orange' if md_pct < 10 else 'tomato'
    error_summary['mds'] = [md_count, md_pct, md_color]

    # mls
    ml_count = len(ml_list)
    ml_pct = round((ml_count / len(df)) * 100, 2)
    ml_color = '#50C878' if ml_pct == 0 else 'orange' if ml_pct < 10 else 'tomato'
    error_summary['mls'] = [ml_count, ml_pct, ml_color]

    # mms
    mm_count = len(mm_list)
    mm_pct = round((mm_count / len(df)) * 100, 2)
    mm_color = '#50C878' if mm_pct == 0 else 'orange' if mm_pct < 10 else 'tomato'
    error_summary['mms'] = [mm_count, mm_pct, mm_color]

    return error_summary

