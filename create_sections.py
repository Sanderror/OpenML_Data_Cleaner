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

def create_dup_row_section(all_dup_rows_table, list_idx_delete):
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
    else:
        section = html.Div([
            html.P(
                "These are the rows/rows in your dataset that contain identical values across all columns (except for the index column). In the table below, we have displayed these errors by outlining pairs of duplicate rows with the same color."),
            html.Div(all_dup_rows_table)])
    return section

def create_dup_col_section(all_dup_cols_table, list_col_delete):
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
    else:
        section = html.Div([
            html.P(
                "Similar to the duplicate rows, duplicate attributes are attributes/columns in your dataset that contain identical values across all rows. In the table below, we have displayed these errors by outlining pairs of duplicate attributes with the same color."),
            html.Div(all_dup_cols_table)])
    return section

def create_out_val_section(outlier_val_table, df_outs_summary):
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


def badge_creator_mvs(imp, remove, keep):
    badge = html.Div("")
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

def create_dup_group_section(dup_rows, df):
    count = 0
    all_sections = []
    for dups in dup_rows:
        str_dups = str(dups).strip("[").strip("]")
        section_df = df.loc[dups].reset_index()
        section = html.Div([
            html.H4(f"Duplicate Group {count + 1} (indices: {str_dups})"),
            dash_table.DataTable(
                data=section_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in section_df.columns],
                page_size=20
            ),
            html.H5("Indices to keep:"),
            dbc.Checklist(id={'type': 'dup-row-retain-checklist', 'index': count},
            options=[{'label': str(i), 'value': i} for i in dups],
            value=[dups[0]],  # By default, keep the first occurrence
            labelStyle={'display': 'inline-block', 'margin-right': '10px'},
            switch=True),
        ])
        count += 1
        all_sections.append(section)
    return html.Div(all_sections)


def create_dup_col_select_section(dup_cols, df):
    count = 0
    all_sections = []
    for dups in dup_cols:
        str_dups = str(dups).strip("[").strip("]")
        section_df = df.T.loc[dups].reset_index()
        section = html.Div([
            html.H4(f"Duplicate Group {count + 1} (columns: {str_dups})"),
            dash_table.DataTable(
                data=section_df.to_dict('records'),
                columns=[{"name": str(i), "id": str(i)} for i in section_df.columns],
                page_size=20
            ),
            html.H5("Columns to keep:"),
            dbc.Checklist(id={'type': 'dup-col-retain-checklist', 'index': count},
                          options=[{'label': str(i), 'value': i} for i in dups],
                          value=[dups[0]],  # By default, keep the first occurrence
                          labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                          switch=True),
        ])
        count += 1
        all_sections.append(section)
    return html.Div(all_sections)


def create_out_row_section(out_rows, df):
    count = 0
    all_sections = []
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
                          value=['Yes'],  # By default, keep the first occurrence
                          labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                          switch=True),
        ])
        count += 1
        all_sections.append(section)
    return html.Div(all_sections)

def create_correct_cryp_section(df, cryptic_cols,  title=False, description=False, n_rows=0):
    client = OpenAI(api_key="") # put your api key here

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
    count = 0
    all_sections = []
    for cryp_dct in cryp_table:
        cols = list(cryp_dct.values())
        cryp_name = cols[0]
        corr_name = cols[1]
        section = html.Div([
            html.H4(f"Cryptic Attribute Name {count + 1} (cryptic: {cryp_name}, corrected: {corr_name})"),
            html.H5("Select the name you want in your dataset"),
            dbc.RadioItems(id={'type': 'cryp-retain-checklist', 'index': count},
                          options=[{'label': i, 'value': i} for i in [cryp_name, corr_name]],
                          value=corr_name,  # By default, keep the first occurrence
                          labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                          switch=True),
        ])
        count += 1
        all_sections.append(section)
    return html.Div(all_sections)

def create_sv_select_section(sv_cols, df):
    count = 0
    all_sections = []
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
                          value=['Yes'],  # By default, keep the first occurrence
                          labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                          switch=True),
        ])
        count += 1
        all_sections.append(section)
    return html.Div(all_sections)

def create_md_select_type(md_dct, method='column'):
    dtb_dct = dict()
    df_dtb = pd.DataFrame()
    if method == 'column':
        for col, val_dct in md_dct.items():
            dtb_dct[col] = [{'label': f"strings ({round(val_dct['strings'] * 100, 2)}%)", 'value': 'strings'}, {'label': f"numbers ({round(val_dct['numbers'] * 100, 2)}%)", 'value': 'numbers'}]
            if val_dct['strings'] < val_dct['numbers']:
                df_dtb[col] = ['strings']
            else:
                df_dtb[col] = ['numbers']
    elif method == 'minor':
        for col, val_dct in md_dct.items():
            if val_dct['strings'] < val_dct['numbers']:
                dtb_dct[col] = [{'label':'strings', 'value':'strings'}]
                df_dtb[col] = ['strings']
            else:
                dtb_dct[col] = [{'label':'numbers', 'value':'numbers'}]
                df_dtb[col] = ['numbers']
    elif method == 'major':
        for col, val_dct in md_dct.items():
            if val_dct['numbers'] < val_dct['strings']:
                dtb_dct[col] = [{'label':'strings', 'value':'strings'}]
                df_dtb[col] = ['strings']
            else:
                dtb_dct[col] = [{'label':'numbers', 'value':'numbers'}]
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
    elif method == 'remove':
        return html.Div([
            dcc.Store(id='mds-correct-method', data=method, storage_type='memory'),
            html.P(f"You have chosen the option \'Remove rows\'. This means that all rows in the mixed data type columns{output_string} will be removed from the dataset{after_string}"),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-mds-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'keep':
        return html.Div([
            dcc.Store(id='mds-correct-method', data=method, storage_type='memory', style=button_style),
            html.P(
                "You have chosen the option \'Keep all\'. This means that all detected mixed data type columns will be retained in the dataset"),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-mds-correct-method', n_clicks=0)
        ])
    else:
        method = 'nothing'
        return html.Div([
            dcc.Store(id='mds-correct-method', data=method, storage_type='memory'),
            html.Button("Submit", id='submit-mds-correct-method', n_clicks=0, style={'display':'none'})
        ])