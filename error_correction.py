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
from sklearn.neighbors import KNeighborsClassifier
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



def convert_mix_to_nan(df, mix_dct, method='minor', col_spec=False):
    nan_values = []
    seen_dct = dict()
    for col, dct in mix_dct.items():
        if method == 'minor':
            if dct['numbers'] < dct['strings']:
                #numbers to np.nan
                for idx, val in df[col].items():
                    if isinstance(val, str) and val.isdigit():
                        nan_values.append([col, val, 1])
                    elif isinstance(val, (int, float)):
                        nan_values.append([col, val, 1])
            else:
                #strings to np.nan
                for idx, val in df[col].items():
                    if isinstance(val, str) and not val.isdigit():
                        nan_values.append([col, val, 1])
        elif method == 'major':
            if dct['numbers'] < dct['strings']:
                # strings to np.nan
                for idx, val in df[col].items():
                    if isinstance(val, str) and not val.isdigit():
                        nan_values.append([col, val, 1])
            else:
                # numbers to np.nan
                for idx, val in df[col].items():
                    if isinstance(val, str) and val.isdigit():
                        nan_values.append([col, val, 1])
                    elif isinstance(val, (int, float)):
                        nan_values.append([col, val, 1])
    if method == 'column':
        if col_spec:
            for col, correct in col_spec.items():
                if correct == 'strings':
                    for idx, val in df[col].items():
                        if isinstance(val, str) and not val.isdigit():
                            nan_values.append([col, val, 1])
                if correct == 'numbers':
                    for idx, val in df[col].items():
                        if isinstance(val, str) and val.isdigit():
                            nan_values.append([col, val, 1])
                        elif isinstance(val, (int, float)):
                            nan_values.append([col, val, 1])
    for md in nan_values:
        md_loc = tuple(md[:2])
        if md_loc in seen_dct:
            seen_dct[md_loc] += 1
        else:
            seen_dct[md_loc] = 1
    md_list = []
    for loc, freq in seen_dct.items():
        md_list.append([loc[0], loc[1], freq])
    return md_list


def query_cryp_cols(cryp_list, title, description, content_df):
    col_query = " | ".join(str(item) for item in cryp_list)

    if not content_df.empty:
        num_rows = len(content_df)
        content_list = [content_df.loc[idx].to_list() for idx in content_df.index]
        contents = "\n".join([" | ".join(str(item) for item in lst) for lst in content_list])
        content_bool = True
    else:
        content_bool = False

    if title == False:
        title_query = ""
    else:
        title_query = f"title: {title}"
    if description == False:
        desc_query = ""
    else:
        desc_query = f"description: {description}"
    if content_bool == False:
        content_query = ""
    else:
        content_query = f"contents of {num_rows} random rows:\n{contents}"

    if title == False and description == False and content_bool == False:
        subquery = ""
    else:
        subquery = f"""As abbreviations of column names from a table with characteristics:
{title_query}
{desc_query}
{content_query}
        """

    query = f"""As abbreviations of column names from a table, the column names c_name | pCd | dt stand for Customer Name | Product Code | Date.
{subquery}
The column names {col_query} stand for"""
    print(query)
    return query

def create_mv_corr_styles(list_of_points, color="#50C878"):
    style = []
    for idx, col in list_of_points:
        style_dict ={
            'if':{
                'column_id':col,
                'filter_query':f'{"{index}"} eq {str(idx)}'
            },
            "backgroundColor":color,
            "color":"white"
        }
        style.append(style_dict)
    return style

def create_dup_corr_styles(indices, df):
    style = []
    for idx in indices:
        if idx != len(df):
            style_dict = {
                'if' : {
                    'filter_query':f'{"{index}"} eq {str(idx + 1)}'
                },
                'borderTop':"3px solid red",
                'color':'black'
            }
        else:
            style_dict = {
                'if': {
                    'filter_query': f'{"{index}"} eq {str(idx - 1)}'
                },
                'borderBottom': "3px solid red",
                'color': 'black'
            }
        style.append(style_dict)
    return style

def create_dup_col_corr_styles(df, dup_cols):
    df.index = df.index.set_names(["index"])
    df = df.reset_index()
    style = []
    loc_dup = None
    for dup_col in dup_cols:
        idx = 0
        for col in df.columns:
            if col == dup_col:
                loc_dup = idx - 1
            idx += 1
        if loc_dup != None:
            if loc_dup != -1:
                border_col = df.columns[loc_dup]
                style_dict = {
                    'if':{
                        'column_id':border_col
                    },
                    'borderRight':'3px solid red'
                }
            else:
                border_col = df.columns[loc_dup + 1]
                style_dict = {
                    'if': {
                        'column_id': border_col
                    },
                    'borderLeft': '3px solid red'
                }
            style.append(style_dict)
    return style

def create_cryp_header(cryp_map):
    style = []
    for col in cryp_map.values():
        style_dict = {
            'if':{
                'column_id': col
            },
            'backgroundColor':'#0080FF',
            'color':'white'
        }
        style.append(style_dict)
    return style

def create_combi_mv_styles(impute_list, remove_list, keep_list, df):
    style = []
    style.extend(create_dup_corr_styles(remove_list, df))
    style.extend(create_mv_corr_styles(impute_list))
    style.extend(create_mv_corr_styles(keep_list, "#0080FF"))
    return style


def complete_corr_styles(new_df, old_df, remove_indices, remove_cols, imputed_locations):
    row_styles = create_dup_corr_styles(remove_indices, new_df)
    col_styles = create_dup_col_corr_styles(old_df, remove_cols)
    val_styles = create_mv_corr_styles(imputed_locations)
    all_styles = row_styles + col_styles + val_styles
    return all_styles

def mv_method(df, col, method):
    """ Applies the selected imputation method on the selected column
    input:
        df: Pandas DataFrame containing a column on which imputation will be applied
        col: String of the column name on which the imputation will be applied
        method: String of the imputation method that is selected to impute the MVs
        in the selected column. Could be either: Mode, MLP, CART, RF, KNN, Delete record
        and Do not impute (and Mean and Median too for numerical columns)
    output:
        curr_col: Pandas Series of the imputed column
    """
    curr_col = df[col]
    if method == "Mode":
        print("this is the mode: ", mode(curr_col))
        imputed_col = curr_col.fillna(mode(curr_col))
    elif method == 'Mean':
        imputed_col = curr_col.fillna(mean(curr_col))
    elif method == 'Median':
        imputed_col = curr_col.fillna(median(curr_col))
    elif method == 'RF' or method == "MLP" or method == 'CART' or method == 'KNN':
        imputed_col = ml_imputation(df, col, method)
    elif method == 'Keep':
        imputed_col = curr_col.astype(str)
    else:
        print(f"Method {method} is not known")

    return imputed_col


def ml_imputation(df, col, method):
    """ Imputation of a column with a Machine Learning (ML) technique
    input:
        df: Pandas DataFrame containing a column on which ML-imputation will be applied
        col: String of the column name on which the ML-imputation will be applied
        method: String of the ML-imputation method that is selected to impute the MVs
        in the selected column. Could be either: MLP, CART or RF.
    output:
        curr_col: Pandas Series of the ML-imputed column
    """
    curr_col = df[col]
    # Transform categorical string columns in dataframe to categorical labels
    temp_df, encoder = categorical_to_label(df)

    # Mask over the NaN values in the to be imputed column
    nan_mask = curr_col.isnull()

    # We use all other columns (X) to predict the missing values of the to be imputed column (y)
    X = temp_df.loc[:, temp_df.columns != col]
    y = curr_col

    # We split the data based on the indices for which y is a NaN
    X_test = X[nan_mask]
    X_train = X[~nan_mask]
    y_test = y[nan_mask]
    y_train = y[~nan_mask]

    if method == 'MLP':
        model = MLPClassifier(solver='lbfgs', alpha=1e-5,
                              hidden_layer_sizes=(100,), random_state=1)
    elif method == 'CART':
        model = DecisionTreeClassifier()
    elif method == 'RF':
        model = RandomForestClassifier(max_depth=2, random_state=0)
    elif method == 'KNN':
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        print(f"Method {method} is not included as a ML-imputation technique")
    print(f"the model {model} is being used for column {col}, because method is {method}")
    model.fit(X_train, y_train)
    pred_values = model.predict(X_test)

    # Map the indices to the predicted values in order to impute the values into the column
    indices = [idx for idx in X_test.index]
    mapping = {indices[i]: pred_values[i] for i in range(len(indices))}
    imputed_col = copy.deepcopy(curr_col)
    for i, j in mapping.items():
        imputed_col.loc[i] = j
    return imputed_col


def impute_mv(df, feature_types, mv_list):
    df = df.astype(str)
    if len(mv_list) == 0:
        return df, 0, html.Div([dbc.Alert(
            "No missing values were detected in your dataset",
            color="#50C878",
            is_open=True,
            style={
                "fontWeight": "bold",
                "fontSize": '16pt',
                "textAlign": "center",
                'color': 'white'
            }
        ), html.H4("The error correction step for the missing values is unavailable, since no missing values were detected in the error detection step.")])

    else:
        new_mv_list = [[str(mv[0]), str(mv[1]), mv[2]] for mv in mv_list]
        df_copy = copy.deepcopy(df)
        cols_containing_mvs = []
        list_of_points = []
        mv_indices = set()
        # Change all dmvs to actual nans
        for mv in new_mv_list:
            indices = df_copy[df_copy[mv[0]] == mv[1]].index
            for idx in indices:
                mv_indices.add(idx)
            df_copy.loc[indices, mv[0]] = np.nan
            if mv[0] not in cols_containing_mvs:
                cols_containing_mvs.append(mv[0])
        rows_with_na = df_copy.isna().any(axis=1)
        new_df = df[rows_with_na].reset_index()
        for col, val, freq in new_mv_list:
            indices = new_df[new_df[col] == val].index
            for idx in indices:
                list_of_points.append([idx, col])

        # Determine imputation method
        feature_type_set = set(feature_types.values())
        dataset_size = df_copy.size
        dtypes_dct = {col :df.dtypes[col] for col in df}

        print(f"Number of feature types: {len(feature_type_set)}")
        if len(feature_type_set) == 1:
            # not mixed data
            data_type = list(feature_type_set)[0]
            print(f"Data type: {data_type}")
            print(f"Dataset size: {dataset_size}")
            if data_type == 'numeric':
                if dataset_size < 10000:
                    method = "MLP"
                else:
                    method = "CART"
            else:
                if dataset_size < 10000:
                    method = "RF"
                else:
                    method = "CART"
        else:
            # mixed data
            nan_values = df_copy.isnull().sum().sum()
            missingness_ratio = nan_values / dataset_size
            print(f"MR: {missingness_ratio}")
            if missingness_ratio <= 0.2:
                method = "KNN"
            else:
                print(f"Dataset size: {dataset_size}")
                if dataset_size < 10000:
                    method = "RF"
                else:
                    method = "CART"
        print(f"Method: {method}")

        if method == "KNN":
            labeled_df, dct_encoder = categorical_to_label(df_copy, encode_nans=False)
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            output = imputer.fit_transform(labeled_df)
            df_imputed = pd.DataFrame(output, columns=labeled_df.columns)
            df_final = copy.deepcopy(df_copy)
            for col in df_final:
                if df_final[col].isnull().any(): # only take the imputed column for columns that got missing values
                    if col in dct_encoder:
                        # object columns
                        col_encoding = dct_encoder[col]
                        imputed_col = col_encoding.inverse_transform(list(round(df_imputed[col]).values.astype(int)))
                    else:
                        if dtypes_dct[col] == pd.Int64Dtype():
                            # integer columns should not contain floats
                            print(dtypes_dct[col], print(type(dtypes_dct[col])))
                            imputed_col = pd.Series(list(round(df_imputed[col]).values.astype(int)))
                            print(imputed_col)
                        else:
                            # float columns
                            imputed_col = df_imputed[col]
                    df_final[col] = imputed_col.astype(dtypes_dct[col])
            mv_indices_list = [x for x in mv_indices]
            mv_indices_list = sorted(mv_indices_list)
            df_dtb = df_final.loc[mv_indices_list].reset_index()
            dtb = dash_table.DataTable(
                data=df_dtb.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_dtb.columns],
                style_data_conditional=create_mv_corr_styles(list_of_points),
                page_size=20
            )
            df_choose_imputation = pd.DataFrame(columns=cols_containing_mvs)
            for col in cols_containing_mvs:
                df_choose_imputation[col] = ['KNN']
            additional_dtb = dash_table.DataTable(
                data=df_choose_imputation.to_dict("records"),
                columns=[{"name": i, "id": i, 'presentation':'dropdown', "editable":True} for i in df_choose_imputation.columns],
                editable=True,
                dropdown={
                    col: {
                        'options': [{'label': "KNN", 'value': 'KNN'}, {'label': "MLP", 'value': 'MLP'},{'label': "CART", 'value': 'CART'}, {'label': "RF", 'value': 'RF'},{'label': "mode", 'value': 'mode'}, {'label': "Do not impute", 'value': 'Do not impute'},{'label': "Delete records", 'value': 'Delete records'}]
                    } for col in df_choose_imputation.columns
                }
            )
            comparison = df != df_final
            imputed_vals = comparison.sum().sum()
            return df_final, imputed_vals, html.Div([dbc.Badge("Imputed value", color="#50C878", className="me-1"),
                dtb,
            html.H4("Do you want to change the imputation method for any of the columns?"),
            html.P("Select the imputation method using the dropdown menu for every column. If you do not want to change the imputation method, skip this step."),
            additional_dtb])
        else:
            for col in df_copy.columns:
                if col in cols_containing_mvs:
                    # Apply the method on every to be imputed column
                    df_copy[col] = mv_method(df_copy, col, method)
            mv_indices_list = [x for x in mv_indices]
            mv_indices_list = sorted(mv_indices_list)
            df_dtb = df_copy.loc[mv_indices_list].reset_index()
            dtb = dash_table.DataTable(
                data=df_dtb.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_dtb.columns],
                style_data_conditional=create_mv_corr_styles(list_of_points),
                page_size=20
            )
            df_choose_imputation = pd.DataFrame(columns=cols_containing_mvs)
            for col in cols_containing_mvs:
                df_choose_imputation[col] = [method]
            additional_dtb = dash_table.DataTable(
                data=df_choose_imputation.to_dict("records"),
                columns=[{"name": i, "id": i, 'presentation': 'dropdown', 'editable':True} for i in df_choose_imputation.columns],
                editable=True,
                dropdown={
                    col: {
                        'options': [{'label': "KNN", 'value': 'KNN'}, {'label': "MLP", 'value': 'MLP'},
                                    {'label': "CART", 'value': 'CART'}, {'label': "RF", 'value': 'RF'},
                                    {'label': "mode", 'value': 'mode'},
                                    {'label': "Do not impute", 'value': 'Do not impute'},
                                    {'label': "Delete records", 'value': 'Delete records'}]
                    } for col in df_choose_imputation.columns
                }
            )
        comparison = df != df_copy
        imputed_vals = comparison.sum().sum()
        return df_copy, imputed_vals, html.Div([dbc.Badge("Imputed value", color="#50C878", className="me-1"), dtb,
            html.H4("Do you want to change the imputation method for any of the columns?"),
            html.P("Select the imputation method using the dropdown menu for every column. If you do not want to change the imputation method, skip this step."),
            additional_dtb])

def correct_dup_row(df, duplicate_rows):
    print("df: ", df)
    print("duplicate rows: ", duplicate_rows)
    if len(duplicate_rows) == 0:
        return html.Div([dbc.Alert(
            "No duplicate rows were detected in your dataset",
            color="#50C878",
            is_open=True,
            style={
                "fontWeight": "bold",
                "fontSize": '16pt',
                "textAlign": "center",
                'color': 'white'
            }
        ), html.H4("The error correction step for the duplicate rows is unavailable, since no duplicate rows were detected in the error detection step.")]), df, duplicate_rows
    else:
        new_df = df.drop(index=duplicate_rows)
        df_dtb = new_df.copy()
        df_dtb.index = df_dtb.index.set_names(["index"])
        df_dtb = df_dtb.reset_index()
        dtb = dash_table.DataTable(
            data=df_dtb.to_dict("records"),
            columns=[{"name": i, "id": i} for i in df_dtb.columns],
            style_data_conditional=create_dup_corr_styles(duplicate_rows, df),
            page_size=20
        )
        return html.Div([dbc.Badge("Removed row", color="tomato", className="me-1"), dtb]), new_df, duplicate_rows

def correct_dup_col(df, ft_types_dct, dup_cols):
    if len(dup_cols) == 0:
        return html.Div([dbc.Alert(
            "No duplicate columns were detected in your dataset",
            color="#50C878",
            is_open=True,
            style={
                "fontWeight": "bold",
                "fontSize": '16pt',
                "textAlign": "center",
                'color': 'white'
            }
        ), html.H4("The error correction step for the duplicate columns is unavailable, since no duplicate columns were detected in the error detection step.")]), df, ft_types_dct, 0
    else:
        new_df = df.drop(columns=dup_cols)
        dtb = dash_table.DataTable(
            data=new_df.to_dict('records'),
            columns=[{'name':i, 'id':i} for i in new_df.columns],
            style_data_conditional=create_dup_col_corr_styles(df, dup_cols),
            page_size=20
        )
        print(ft_types_dct)
        for col in dup_cols:
            print(col)
            del ft_types_dct[col]
        print(ft_types_dct)
        removed_cols = len(dup_cols)
        return html.Div([dbc.Badge("Removed column", color="tomato", className="me-1"), dtb]), new_df, ft_types_dct, removed_cols

def correct_out_val(df, ft_types, out_vals):
    if len(out_vals) == 0:
        return df, 0, html.Div([dbc.Alert(
            "No outlier values were detected in your dataset",
            color="#50C878",
            is_open=True,
            style={
                "fontWeight": "bold",
                "fontSize": '16pt',
                "textAlign": "center",
                'color': 'white'
            }
        ), html.H4("The error correction step for the outlier values is unavailable, since no outlier values were detected in the error detection step.")])
    else:
        new_out_vals = []
        for out in out_vals:
            if out[0] in ft_types.keys():
                new_out_vals.append([out[0], out[2], out[1]])
        print("New out vals", new_out_vals)
        df_imputed, imputed_vals, dtb = impute_mv(df, ft_types, new_out_vals)
        return df_imputed, imputed_vals, dtb

def correct_out_row(df, out_rows, removed_rows):
    if len(out_rows) == 0:
        return df, html.Div([dbc.Alert(
            "No outlier rows were detected in your dataset",
            color="#50C878",
            is_open=True,
            style={
                "fontWeight": "bold",
                "fontSize": '16pt',
                "textAlign": "center",
                'color': 'white'
            }
        ), html.H4("The error correction step for the outlier rows is unavailable, since no outlier rows were detected in the error detection step.")]), out_rows
    else:
        out_indices = []
        for out in out_rows:
            not_same = True
            prev_indices_count = 0
            for removed in removed_rows:
                if out[0] == removed:
                    not_same = False
                elif not_same:
                    if out[0] < removed:
                        prev_indices_count += 1
            if not_same:
                out_indices.append(out[0])

        new_df = df.drop(index=out_indices)
        reset_index_df = new_df.copy()
        reset_index_df.index = reset_index_df.index.set_names(["index"])
        reset_index_df = reset_index_df.reset_index()
        dtb = dash_table.DataTable(
            data=reset_index_df.to_dict("records"),
            columns=[{'name':i, 'id':i} for i in reset_index_df.columns],
            style_data_conditional=create_dup_corr_styles(out_indices, df),
            page_size=20
        )
        return new_df, html.Div([dbc.Badge("Removed row", color="tomato", className="me-1"), dtb]), out_indices


def correct_cryp_cols(df, cryptic_cols, ft_types, sv_cols,  title=False, description=False, n_rows=0):
    # Set API key, uncomment to test
    client = OpenAI(api_key="") # add your API key here

    # Identify the cryptic column names
    num_cols = len(cryptic_cols)

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

    # temporary solve
    new_col_names = ['Final Weight', 'Redundant Single Value Column']

    rename_dct = dict()
    for old_col, new_col in zip(cryptic_cols, new_col_names):
        if old_col in sv_cols:
            sv_cols.append(new_col)
            sv_cols.remove(old_col)
        rename_dct[old_col] = new_col
        ft_types[new_col] = ft_types.pop(old_col)
    new_df = df.copy().rename(columns=rename_dct)
    df_dtb = pd.DataFrame()
    df_dtb['Cryptic name'] = cryptic_cols
    df_dtb['Corrected name'] = new_col_names
    dtb = dash_table.DataTable(
        data=df_dtb.to_dict('records'),
        columns=[{'name':i,'id':i} for i in df_dtb.columns],
        page_size=20
    )
    nr_of_corr_cryp = len(new_col_names)
    return new_df, dtb, ft_types, sv_cols, nr_of_corr_cryp, rename_dct

def correct_sv_col(df, sv_cols, ft_types):
    if len(sv_cols) == 0:
        return df, html.Div([dbc.Alert(
            "No single value columns were detected in your dataset",
            color="#50C878",
            is_open=True,
            style={
                "fontWeight": "bold",
                "fontSize": '16pt',
                "textAlign": "center",
                'color': 'white'
            }
        ), html.H4("The error correction step for the single value columns is unavailable, since no single value columns were detected in the error detection step.")]), ft_types, 0
    else:
        new_df = df.drop(columns=sv_cols)
        df_dtb = new_df.copy().reset_index()
        dtb = dash_table.DataTable(
            data=df_dtb.to_dict("records"),
            columns=[{'name':i,'id':i} for i in df_dtb.columns],
            style_data_conditional=create_dup_col_corr_styles(df, sv_cols),
            page_size=20
        )
        for sv_col in sv_cols:
            del ft_types[sv_col]
        removed_svs = len(sv_cols)
        return new_df, html.Div([dbc.Badge('Removed column', color='tomato', className="me-1"), dtb]), ft_types, removed_svs

def correct_mixed_data(df, mix_dct, feature_types):
    if len(mix_dct) == 0:
        return df, 0, html.Div([dbc.Alert(
            "No mixed data type columns were detected in your dataset",
            color="#50C878",
            is_open=True,
            style={
                "fontWeight": "bold",
                "fontSize": '16pt',
                "textAlign": "center",
                'color': 'white'
            }
        ), html.H4("The error correction step for the mixed data type columns is unavailable, since no mixed data type columns were detected in the error detection step.")])
    else:
        df_copy = copy.deepcopy(df)
        nan_values = convert_mix_to_nan(df_copy, mix_dct)
        df_imputed, imputed_vals, dtb = impute_mv(df, feature_types, nan_values)
        return df_imputed, imputed_vals, dtb

def correct_ml(df, ml_list, target, removed_rows):
    if len(ml_list) == 0:
        return df, html.Div([dbc.Alert(
            "No incorrect labels were detected in your dataset",
            color="#50C878",
            is_open=True,
            style={
                "fontWeight": "bold",
                "fontSize": '16pt',
                "textAlign": "center",
                'color': 'white'
            }
        ), html.H4("The error correction step for the incorrect labels is unavailable, since no incorrect labels were detected in the error detection step.")]), 0
    else:
        new_df = df.copy()
        indices = []
        new_ml_list = []

        for ml in ml_list:
            not_same = True
            prev_indices_count = 0
            for removed in removed_rows:
                if ml[0] == removed:
                    not_same = False
                elif not_same:
                    if ml[0] < removed:
                        prev_indices_count += 1
            if not_same:
                new_idx = ml[0]
                indices.append(new_idx)
                new_ml_list.append([new_idx, ml[1], ml[2]])
        print("This is the ml lsit: ", ml_list)
        print("This is the new ml list; ", new_ml_list)
        print("This is df: ", df, len(df), df.size)
        for ml in new_ml_list:
            new_df.loc[ml[0], target] = ml[2]
        print("This is new_df: ", new_df, len(new_df), new_df.size)
        df_dtb = new_df.loc[indices].reset_index()
        dtb = dash_table.DataTable(
            data=df_dtb.to_dict("records"),
            columns=[{'name':i,'id':i} for i in df_dtb.columns],
            style_data_conditional=[{
                'if': {
                    'column_id':target
                },
                "backgroundColor":"#50C878",
                "color":"white"
            }],
            page_size=20
        )
        comparison = new_df != df
        nr_of_corr_mls = comparison.sum().sum()
        return new_df, html.Div([dbc.Badge("Corrected label", color="#50C878", className="me-1"), dtb]), nr_of_corr_mls

def correct_mm(df, mm_list):
    if len(mm_list) == 0:
        return df, html.Div([dbc.Alert(
            "No string mismatches were detected in your dataset",
            color="#50C878",
            is_open=True,
            style={
                "fontWeight": "bold",
                "fontSize": '16pt',
                "textAlign": "center",
                'color': 'white'
            }
        ), html.H4("The error correction step for the string mismatches is unavailable, since no string mismatches were detected in the error detection step.")]), 0
    else:
        new_df = df.copy()
        list_of_points = []
        slice_indices = []
        for mm in mm_list:
            col = mm[0]
            base = mm[1]
            variations = ast.literal_eval(mm[2])
            if base in variations:
                variations.remove(base)
            variation_idx = new_df[new_df[col].isin(variations)].index
            new_df.loc[variation_idx, col] = base
            style_df = new_df.reset_index(drop=True)
            style_indices = style_df[style_df[col].isin(variations)].index
            for idx in variation_idx:
                slice_indices.append(idx)
                list_of_points.append([idx, col])
        new_points = []
        indice = 0
        for idx, col in sorted(list_of_points):
            new_points.append([indice, col])
            indice += 1
        df_dtb = new_df.loc[slice_indices].reset_index()
        dtb = dash_table.DataTable(
            data=df_dtb.to_dict('records'),
            columns=[{'name':i,'id':i} for i in df_dtb.columns],
            style_data_conditional=create_mv_corr_styles(new_points),
            page_size=20
        )
        comparison = new_df != df
        nr_of_corr_mms = comparison.sum().sum()
        print("Nr of corrections: ", nr_of_corr_mms)
        print("Nr of indices: ", slice_indices)
        print("Df and new_df: ", comparison)
        return new_df, html.Div([dbc.Badge("Corrected mismatch", color="#50C878", className="me-1") ,dtb]), nr_of_corr_mms


def show_final_df(clean_df, dirty_df_cryp, cryp_mapping):
    dirty_df = dirty_df_cryp.copy().rename(columns=cryp_mapping).astype(str)
    removed_row_indices = [idx for idx in dirty_df.index if idx not in clean_df.index]
    removed_cols = [col for col in dirty_df.columns if col not in clean_df.columns]
    imputed_value_locations = []
    print(dirty_df.dtypes)
    for idx in clean_df.index:
        for col in clean_df.columns:
            if clean_df.at[idx, col] != dirty_df.at[idx, col]:
                print(idx, col)
                offset = len([row for row in removed_row_indices if row < idx])
                imputed_value_locations.append([idx - offset, col])
    print(imputed_value_locations)
    df_dtb = clean_df.copy()
    df_dtb.index = df_dtb.index.set_names(["index"])
    df_dtb = df_dtb.reset_index()
    final_dtb = dash_table.DataTable(
        id='final_table',
        data=df_dtb.to_dict("records"),
        columns=[{'name':i,'id':i} for i in df_dtb.columns],
        style_data_conditional=complete_corr_styles(df_dtb, dirty_df, removed_row_indices, removed_cols, imputed_value_locations),
        page_size=20
    )
    return html.Div([dbc.Badge("Imputed values", color="#50C878", className="me-1"), dbc.Badge("Removed rows/columns", color="tomato", className="me-1"), dbc.Badge("Corrected column name", color="#0080FF", className="me-1"), final_dtb]), removed_row_indices, removed_cols, imputed_value_locations
