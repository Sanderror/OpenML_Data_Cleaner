## put here the error detection functions, that can then be imported to main file and used
#All imports
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

class CrypticIdentifier:
    """Module to identify any cryptic forms in a column header.
    Example usage:
        identifier = CrypticIdentifier(vocab_file)
        identifier.iscryptic("newyorkcitytotalpopulation") --> False
        identifier.iscryptic("tot_revq4") --> True
    """

    def __init__(self, vocab_file=None, word_rank_file=None, k_whole=4, k_split=2):
        """
        Args:
            vocab_file (str, optional): json file containing the vocabulary. Defaults to None.
            k_whole (int, optional): length threshold for a whole string to be considered non-cryptic if it fails the first round of check (i.e.
            _iscryptic returns True). Defaults to 4.
            k_split (int, optional): length threshold for each word split (wordninja.split()) from the string to be considered non-cryptic, if the pre-split string fails the first round of check (i.e.
            _iscryptic returns True). Defaults to 2.
        """
        if vocab_file is not None:
            with open(vocab_file, "r") as fi:
                self.vocab = json.load(fi)
        #                 print("#vocab={}".format(len(self.vocab)))
        else:
            self.vocab = None

        self.k_whole = k_whole
        self.k_split = k_split
        if word_rank_file is None:
            self.splitter = wordninja
        else:
            self.splitter = wordninja.LanguageModel(word_rank_file)
        self.lem = WordNetLemmatizer()

    def split_rm_punc(self, text: str) -> list:
        return re.sub(r'[^\w\s]', ' ', text).split()

    def separate_camel_case(self, text: str) -> list:
        return re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', text))

    def convert2base(self, text: str) -> str:
        return self.lem.lemmatize(text)

    def _split(self, text: str) -> list:
        text = text.replace('_', ' ')
        words = self.split_rm_punc(self.separate_camel_case(text))
        return words

    def _iscryptic(self, text: str) -> bool:
        words = self._split(text)
        if all([word.isnumeric() for word in words]):
            return True
        if self.vocab is None:
            self.vocab = nltk.corpus.wordnet.words('english')
        return any([self.convert2base(w.lower()) not in self.vocab for w in words])

    def doublecheck_cryptic(self, text: str) -> Tuple[bool, List[str]]:
        """Double-check whether a column header contains cryptic terms. For example in some cases where neither
        delimiters between tokens nor camelcases is available

        Args:
            text (str): column header

        Returns:
            Tuple[
                    bool: whether header is cryptic
                    List[str]: splitted tokens from the header
                ]
        """

        # stopwords = nltk.corpus.stopwords.words('english')

        def split_check(words: List[str]) -> Tuple[bool, List[str]]:
            l_cryptic = []
            for ele in words:
                if ele.isdigit():
                    l_cryptic.append(False)
                ## Cornercases includes stopwords like "I", "for", etc.
                elif len(ele) < self.k_split:  # and ele.lower() not in stopwords:
                    l_cryptic.append(True)
                ## Second round check
                else:
                    l_cryptic.append(self._iscryptic(ele))
            return any(l_cryptic), words

        if len(text) >= self.k_whole:
            if self._iscryptic(text):
                split = self.splitter.split(text)
                return split_check(split)
            else:
                # return (False, self.splitter.split(text))
                return (False, self._split(text))
        else:
            with open('lookups/words.txt', 'r') as file:
                # Read the content of the file
                content = file.read()

                # Split the content into words based on whitespace
                vocabulair = content.split()
            # Print the list of words
            if text in vocabulair:
                return (False, [text])
            return (True, [text])

    def iscryptic(self, text: str) -> bool:
        return self.doublecheck_cryptic(text)[0]

    def split_results(self, text: str) -> List[str]:
        return self.doublecheck_cryptic(text)[1]

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = norm(a)
    norm_b = norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity
def create_mv_styles(output_list):
    ''' Styles for dash table of mvs'''
    style = []
    for output in output_list:
        if output[1] == 'nan':
            column = "{" + str(output[0]) + "}"
            style_dict = {
                'if':{
                    'column_id':output[0],
                    'filter_query':f'{column} eq {output[1]}',
                },
                'backgroundColor':'tomato',
                'color':'white'
            }
            style.append(style_dict)
        else:
            column = "{" + str(output[0]) + "}"
            style_dict = {
                'if':{
                    'column_id':output[0],
                    'filter_query':f'{column} eq {output[1]}',
                },
                'backgroundColor':'orange',
                'color':'white'
            }
            style.append(style_dict)
    return style

def create_out_styles(outlier_list):
    ''' Styles for dash table of outs'''
    style = []
    for outlier_data in outlier_list:
        if outlier_data[1] == 'far':
            column = "{" + str(outlier_data[0]) + "}"
            style_dict = {
                'if': {
                    'column_id': outlier_data[0],
                    'filter_query': f'{column} eq {outlier_data[2]}',
                },
                'backgroundColor': 'tomato',
                'color': 'white'
            }
            style.append(style_dict)
        else:
            column = "{" + str(outlier_data[0]) + "}"
            style_dict = {
                'if': {
                    'column_id': outlier_data[0],
                    'filter_query': f'{column} eq {outlier_data[2]}',
                },
                'backgroundColor': 'orange',
                'color': 'white'
            }
            style.append(style_dict)
    print(style)
    return style
def create_dup_row_styles(df):
    ''' Styles for dup rows'''
    style = []
    dup_dict = dict()
    color_dict = dict()
    df = df.reset_index(drop=True)
    first_col, last_col = df.columns[0], df.columns[-1]
    rgb_encodings = []
    for idx, row in df.iterrows():
        colors_too_similar = True
        values = tuple(row.values)[1:]
        while colors_too_similar:
            not_similar_count = 0
            rgb_encoding = [random.randint(50, 255) for _ in range(3)]
            for other_rgb in rgb_encodings:
                if cosine_similarity(rgb_encoding, other_rgb) < 0.975:
                    not_similar_count += 1
            if len(rgb_encodings) == not_similar_count:
                colors_too_similar = False
                rgb_encodings.append(rgb_encoding)
        if values not in dup_dict:
            dup_dict[values] = [idx]
            color_dict[idx] = "3px solid rgb" + str(tuple(rgb_encoding))
        else:
            dup_dict[values].append(idx)
            color = color_dict[dup_dict[values][0]]
            color_dict[idx] = color
    print(color_dict)
    for row, indices in dup_dict.items():
        for idx in indices:
            style_dict = {
                'if': {
                    'row_index': idx,
                },
                'borderTop': color_dict[idx],
                'borderBottom':color_dict[idx],
                'color':'black'
            }
            style.append(style_dict)
            style_dict_left = {
                'if':{
                    'row_index':idx,
                    'column_id':first_col,
                },
                'borderLeft':color_dict[idx]
            }
            style.append(style_dict_left)
            style_dict_right = {
                'if':{
                    'row_index':idx,
                    'column_id':last_col,
                },
                'borderRight':color_dict[idx]
            }
            style.append(style_dict_right)
    return style

def create_dup_col_styles(df, pagesize=20):
    style = []
    dup_dict = dict()
    color_dict = dict()
    rgb_encodings = []
    df = df.iloc[:,1:]
    for name, col in df.items():
        values = tuple(col.values)
        colors_too_similar = True
        while colors_too_similar:
            not_similar_count = 0
            rgb_encoding = [random.randint(100, 255) for _ in range(3)]
            for other_rgb in rgb_encodings:
                if cosine_similarity(rgb_encoding, other_rgb) < 0.975:
                    not_similar_count += 1
            if len(rgb_encodings) == not_similar_count:
                colors_too_similar = False
                rgb_encodings.append(rgb_encoding)
        if values not in dup_dict:
            dup_dict[values] = [name]
            color_dict[name] = "3px solid rgb" + str(tuple(rgb_encoding))
        else:
            dup_dict[values].append(name)
            color = color_dict[dup_dict[values][0]]
            color_dict[name] = color
    for col, names in dup_dict.items():
        for name in names:
            style_dict = {
                'if': {
                    'column_id': name,
                },
                'borderLeft': color_dict[name],
                'borderRight':color_dict[name],
                'color':'black'
            }
            style.append(style_dict)
            style_dict_special = {
                'if': {
                    'column_id':name,
                    'row_index':pagesize - 1
                },
                'borderBottom': color_dict[name]
            }
            style.append(style_dict_special)
            style_dict_special = {
                'if': {
                    'column_id': name,
                    'row_index': 0
                },
                'borderTop': color_dict[name]
            }
            style.append(style_dict_special)
    return style

def categorical_to_label(df, encode_nans=True):
    """ Converts categorical columns in a dataframe to labels (numerical)
    input:
        df: Pandas DataFrame in which the categorical features will be converted to labels
        encode_nans: True, if the NaNs should be encoded as a label as well (needed for most
        imputation techniques because do not work if there are NaNs in the non-target columns). False,
        if the NaNs should not be encoded as a label (e.g. for KNN imputation)
    output:
        df_copy: Copy of the original df, but now containing labels for the categorical columns
        dct: A dictionary containing the column names (of categorical columns) as keys and the
        LabelEncoder used for that column as value. This is needed to convert the labels back to
        the original categorical values
    """
    dct = dict()
    df_copy = copy.deepcopy(df)
    if encode_nans:
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].astype(str)
                encoder = LabelEncoder()
                df_copy[col] = encoder.fit_transform(df_copy[col])
                dct[col] = encoder
    else:
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                nan_mask = df_copy[col].isnull()
                nan_rows = df_copy[col][nan_mask]
                cat_rows = df_copy[col][~nan_mask].astype(str)
                encoder = LabelEncoder()
                cat_as_label_rows = encoder.fit_transform(cat_rows)
                df_copy[col][~nan_mask] = cat_as_label_rows
                dct[col] = encoder
    return df_copy, dct

def detect_mv(df):
    """ Detects whether there are missing values in a column
    input:
        series: Pandas Series of a column in a dataframe
    output:
        boolean: True if missing values in the column, False if not
    """
    # nan_df = df[df.isna().any(axis=1)].astype(str)
    nan_columns = df.columns[df.isna().any()].tolist()
    output_list = [[col,'nan', len(df[df[col].isna()])] for col in nan_columns]
    # dtb = dash_table.DataTable(
    #     data=nan_df.to_dict('records'),
    #     columns=[{"name": i, "id": i} for i in nan_df.columns],
    #     editable=True,
    #     style_data_conditional=create_mv_styles(output_list, True)
    # )
    # print(dtb)
    return output_list

def detect_dmv(df, tool_id="5"):
    """ Detects disguised missing values (DMVs) in a dataframe
    input:
        df: Pandas DataFrame that will be checked on DMVs
        tool_id: String of 1, 2, 3, 4 or 5, which determines what type of DMVs
        will be checked for. Default=5; pattern discovery + numerical outlier detection
    output:
        output_list: List containing the DMVs detected stored as lists within that list
        in the format: Column name, DMV, Frequency (of DMV), Detection tool (that found the DMV)
    """

    df_str = df.astype(str)
    sus_dis_values = []

    if tool_id == '1':
        sus_dis_values, ptrns = patterns.find_all_patterns(df_str, sus_dis_values)
        sus_dis_values = DV_Detector.check_non_conforming_patterns(df_str, sus_dis_values)
    elif tool_id == '2':
        sus_dis_values = RandDMVD.find_disguised_values(df_str, sus_dis_values)
    elif tool_id == '3':
        sus_dis_values = OD.detect_outliers(df_str, sus_dis_values)
    elif tool_id == '4':
        sus_dis_values, ptrns = patterns.find_all_patterns(df_str, sus_dis_values)
        sus_dis_values = DV_Detector.check_non_conforming_patterns(df_str, sus_dis_values)
        sus_dis_values = RandDMVD.find_disguised_values(df_str, sus_dis_values)
        sus_dis_values = OD.detect_outliers(df_str, sus_dis_values)
    elif tool_id == '5':
        sus_dis_values, ptrns = patterns.find_all_patterns(df_str, sus_dis_values)
        sus_dis_values = DV_Detector.check_non_conforming_patterns(df_str, sus_dis_values)
        sus_dis_values = OD.detect_outliers(df_str, sus_dis_values)
    else:
        print("Unkown option ..", tool_id)

    output_str = "Column Name, Value, Frequency, Detection Tool \n"
    output_list = []
    for sus_dis in sus_dis_values:
        output_str = output_str + f"{sus_dis.attr_name}, {sus_dis.value}, {sus_dis.frequency}, {sus_dis.tool_name}\n"
        output_list.append([sus_dis.attr_name, sus_dis.value, sus_dis.frequency])

    mv_specs = detect_specs(df)
    mv_specs = [dmv for dmv in mv_specs if dmv not in output_list]
    mv_mix_null = detect_mix_nulls(df)
    mv_mix_null = [dmv for dmv in mv_mix_null if dmv not in output_list and dmv not in mv_specs]

    all_dmvs = output_list + mv_specs + mv_mix_null
    # indices = set()
    # for dmv in output_list:
    #     index = df[df[dmv[0]] == dmv[1]].index
    #     for idx in index:
    #         indices.add(idx)
    # indice_list = sorted([x for x in indices])
    # new_df = df.loc[indice_list]
    # dtb = dash_table.DataTable(
    #     data=new_df.to_dict('records'),
    #     columns=[{"name": i, "id": i} for i in new_df.columns],
    #     editable=True,
    #     style_data_conditional=create_mv_styles(output_list, False)
    # )
    return all_dmvs

def detect_specs(df):
    spec_mvs = []
    check = SpecialCharacters(n_most_common=10000, n_top_columns=10000).run(df)
    if check.display:
        result = check.display[1]
        for idx, specs in zip(result.index, result['Most Common Special-Only Samples']):
            for spec in specs:
                freq = len(df[df[idx] == spec])
                spec_mvs.append([idx, spec, freq])
    return spec_mvs

def detect_mix_nulls(df):
    mixed_nulls = []
    check = MixedNulls(n_top_columns=10000).run(df)
    if check.display:
        result = check.display[1].reset_index()
        for i in range(len(result)):
            if not result['Value'][i].endswith('.nan') and result['Value'][i] != 'None':
                col = result.loc[i, 'Column Name']
                val = result.loc[i, 'Value']
                freq = result.loc[i, 'Count']
                mixed_nulls.append([col, val, freq])
    return mixed_nulls

def detect_all_mvs(df):
    # Actual nans
    output_mv = detect_mv(df)
    print(output_mv)
    # DMVs
    output_dmv = detect_dmv(df, tool_id='5')
    output_dmv = [dmv for dmv in output_dmv if dmv not in output_mv]
    print(output_dmv)
    output_list = output_mv + output_dmv
    frequency_mvs = [mv[2] for mv in output_list]
    nr_of_mvs = sum(frequency_mvs)
    pct_of_mvs = round((nr_of_mvs / df.size) * 100, 2)
    df = df.astype(str)

    indices = set()
    for mv in output_list:
        index = df[df[mv[0]] == mv[1]].index
        for idx in index:
            indices.add(idx)
    indice_list = sorted([x for x in indices])
    new_df = df.loc[indice_list].reset_index()

    dtb = dash_table.DataTable(
        data=new_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in new_df.columns],
        style_data_conditional=create_mv_styles(output_list),
        page_size=20)
    print(dtb)
    output_df = pd.DataFrame(output_list, columns=['Column', 'Value', 'Frequency'])
    output_df['Missing Value?'] = 'yes'
    print(output_df)
    if pct_of_mvs == 0:
        mvs_color = '#50C878'
    elif pct_of_mvs < 20:
        mvs_color = 'orange'
    elif pct_of_mvs >= 20:
        mvs_color = 'tomato'
    return dtb, output_df, nr_of_mvs, pct_of_mvs, mvs_color

def detect_dup_row(df):
    df = df.astype(str)
    dup_row_mask = df.duplicated(keep=False)
    duplicates = df[dup_row_mask]
    if not duplicates.empty:
        duplicate_groups = duplicates.groupby(list(df.columns)).apply(lambda x: list(x.index)).tolist()
    else:
        duplicate_groups = []
    dup_df = duplicates.reset_index()
    nr_of_dup_rows = len(dup_df)
    pct_of_dup_rows = round((nr_of_dup_rows / len(df)) * 100, 2)
    if len(df[df.duplicated()]) > 1 and len(df[df.duplicated()]) <= 20:
        dtb = dash_table.DataTable(
            data=dup_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in dup_df.columns],
            style_data_conditional=create_dup_row_styles(dup_df),
            page_size=20
        )
    else:
        dtb = dash_table.DataTable(
            data=dup_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in dup_df.columns],
            page_size=20
        )
    idx_to_delete = df[df.duplicated()].index.to_list()
    if pct_of_dup_rows == 0:
        dup_rows_color = '#50C878'
    elif pct_of_dup_rows < 10:
        dup_rows_color = 'orange'
    elif pct_of_dup_rows >= 10:
        dup_rows_color = 'tomato'
    print("dup dict: ", duplicate_groups)
    return dtb, duplicate_groups, nr_of_dup_rows, pct_of_dup_rows, dup_rows_color

def detect_dup_col(df):
    pagesize = 20
    dup_col_mask = df.T.duplicated(keep=False)
    duplicates = df.loc[:, dup_col_mask].T
    if duplicates.empty:
        duplicate_groups = []
    else:
        duplicate_groups = duplicates.groupby(list(df.index)).apply(lambda x: list(x.index)).tolist()
    # dup_col_names = df.columns[dup_col_mask].to_list()
    dup_df = duplicates.T.reset_index()
    nr_of_dup_cols = len(duplicates)
    pct_of_dup_cols = round((nr_of_dup_cols / len(df.columns)) * 100, 2)
    if len(df.T[df.T.duplicated()]) > 0 and len(df.T[df.T.duplicated()]) <= 10:
        dtb = dash_table.DataTable(
            data=dup_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in dup_df.columns],
            style_data_conditional=create_dup_col_styles(dup_df, pagesize),
            page_size=pagesize
        )
    else:
        dtb = dash_table.DataTable(
            data=dup_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in dup_df.columns],
            page_size=pagesize
        )
    # cols_to_delete = df.columns[df.T.duplicated()].to_list()
    if pct_of_dup_cols == 0:
        dup_cols_color = '#50C878'
    elif pct_of_dup_cols < 10:
        dup_cols_color = 'orange'
    elif pct_of_dup_cols >= 10:
        dup_cols_color = 'tomato'
    return dtb, duplicate_groups, nr_of_dup_cols, pct_of_dup_cols, dup_cols_color

def detect_outlier_val(df, ft_types, k=2):
    outliers_list = []
    for name, series in df.items():
        if ft_types[name] == 'numeric':
            q25, q75 = percentile(series, 25), percentile(series,75)
            IQR_3 = 3 * (q75 - q25)
            IQR_2 = 2 * (q75 - q25)
            lb_3, ub_3 = q25 - IQR_3, q75 + IQR_3
            lb_2, ub_2 = q25 - IQR_2, q75 + IQR_2
            print("UB_2", ub_2)
            print("UB_3", ub_3)
            far_outliers = set([val for val in series if val < lb_3 or val > ub_3])
            close_outliers = set([val for val in series if (val < lb_2 or val > ub_2) and val not in far_outliers])
            if len(far_outliers) != 0:
                for outlier in far_outliers:
                    outliers_list.append([name, 'far', outlier])
            if len(close_outliers) != 0:
                for outlier in close_outliers:
                    outliers_list.append([name, 'close', outlier])
    print(outliers_list)
    outlier_freqs = [len(df[df[outlier[0]] == outlier[2]]) for outlier in outliers_list]
    nr_of_out_vals = sum(outlier_freqs)
    pct_of_out_vals = round((nr_of_out_vals / df.size) * 100, 2)
    indices = set()
    for outliers in outliers_list:
        index = df[df[outliers[0]] == outliers[2]].index
        for idx in index:
            indices.add(idx)
    indice_list = sorted([x for x in indices])
    new_df = df.loc[indice_list].reset_index()
    print(indice_list)
    if new_df.empty:
        dtb = html.P("No outliers were found")
    else:
        dtb = dash_table.DataTable(
            data=new_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in new_df.columns],
            style_data_conditional=create_out_styles(outliers_list),
            page_size=20)
    output_df = pd.DataFrame(outliers_list, columns=['Column', 'Type', 'Value'])
    output_df['Erroneous Outlier?'] = 'yes'
    if pct_of_out_vals == 0:
        out_vals_color = '#50C878'
    elif pct_of_out_vals < 20:
        out_vals_color = 'orange'
    elif pct_of_out_vals >= 20:
        out_vals_color = 'tomato'
    return dtb, output_df, outliers_list, nr_of_out_vals, pct_of_out_vals, out_vals_color

def detect_outlier_row(df, threshold=0.8):
    # df_temp, dct_encoding = categorical_to_label(df)
    # model = loop.LocalOutlierProbability(df_temp).fit()
    # scores = model.local_outlier_probabilities
    # mask = scores > threshold
    # new_df = df[mask].reset_index()
    # outlier_indices = new_df.iloc[:,0].to_list()
    # nr_of_out_rows = len(new_df)
    # pct_of_out_rows = round((nr_of_out_rows / len(df)) * 100, 2)

    if math.sqrt(len(df)) < 5:  # minimum number of nearest neighbors
        nearest_neighbors_percent = 5 / len(df)
        if nearest_neighbors_percent >= 1:
            nearest_neighbors_percent = 0.99
        if nearest_neighbors_percent == 0:
            nearest_neighbors_percent = 0.01
    else:
        nearest_neighbors_percent = round(((math.sqrt(len(df))) / len(df)), 2)
        if nearest_neighbors_percent == 0:
            nearest_neighbors_percent = 0.01
        if nearest_neighbors_percent >= 1:
            nearest_neighbors_percent = 0.99
    check = OutlierSampleDetection(n_to_show=10000, nearest_neighbors_percent=nearest_neighbors_percent,
                                   extent_parameter=3, timeout=120)

    result = check.run(df)
    if result.display:
        output = result.display[1]
        output['Outlier Probability Score'] = [round(val, 2) for val in output['Outlier Probability Score']]
        outlier_probabilities = output[output['Outlier Probability Score'] > 0.5]['Outlier Probability Score'].to_list()
        print("outlier probs", outlier_probabilities)
        outlier_indices = output[output['Outlier Probability Score'] > 0.5].index
        print("Detected outlier rows: ", outlier_indices)

    new_df = df.loc[outlier_indices].reset_index()
    nr_of_out_rows = len(new_df)
    pct_of_out_rows = round((nr_of_out_rows / len(df)) * 100, 2)

    df_out_summary = pd.DataFrame()
    df_out_summary['Index'] = outlier_indices
    df_out_summary['Probability'] = outlier_probabilities
    df_out_summary['Outlier row?'] = 'yes'
    print("df out summary, ", df_out_summary)

    if new_df.empty:
        dtb = dbc.Alert(
                "No outlier rows were detected in your dataset",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            )
        check_dtb = html.Div([
            html.Div(id='out-rows-table', style={'display':'none'})
        ])
    else:
        dtb = dash_table.DataTable(
            data=new_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in new_df.columns],
            page_size=20),
        check_dtb = html.Div([
                        html.H4(
                            "Are the following outlier rows actual errors?"),
                        html.P(
                            "Select in the \"Outlier row?\" column in the table below \"Yes\" if the row is indeed an outlier row and select \"No\" if the row is a valid row."),
                        html.Div([dash_table.DataTable(
                            id='out-rows-table',
                            data=df_out_summary.to_dict('records'),
                            columns=[
                                {'name': 'Index', 'id': 'Index', 'editable': False},
                                {'name': 'Probability', 'id': 'Probability', 'editable': False},
                                {'name': 'Outlier row?', 'id': 'Outlier row?', 'presentation': 'dropdown'}
                            ],
                            editable=True,
                            dropdown={
                                'Outlier row?': {
                                    'options': [{'label': "Yes", 'value': 'yes'}, {'label': "No", 'value': 'no'}]
                                }
                            },
                            page_size=20
                        )])])
    if pct_of_out_rows == 0:
        out_rows_color = '#50C878'
    elif pct_of_out_rows < 10:
        out_rows_color = 'orange'
    elif pct_of_out_rows >= 10:
        out_rows_color = 'tomato'
    return dtb, check_dtb, df_out_summary, output, outlier_indices, nr_of_out_rows, pct_of_out_rows, out_rows_color

def detect_cryp_cols(df):
    identifier = CrypticIdentifier("./lookups/wordnet.json", "./lookups/wordninja_words_alpha.txt.gz")
    cryptic_cols = [col for col in df.columns if identifier.doublecheck_cryptic(col)[0]==True]
    nr_of_cryps = len(cryptic_cols)
    pct_of_cryps = round((nr_of_cryps / len(df.columns)) * 100, 2)
    cryp_col_checklist = dbc.Card(
        [
            dbc.CardHeader("Cryptic Column Names", style={'backgroundColor':'#0080FF','color':'white', 'fontWeight':'bold', 'fontSize':'16pt'}),
            dbc.CardBody([
                dbc.Checklist(
                        options=[{'label': f"{col}", 'value': col} for col in df.columns],
                        value=cryptic_cols,
                        id='cryptic-cols',
                        switch=True
                    )
            ])
        ]
    )
    if pct_of_cryps == 0:
        cryp_color = '#50C878'
    elif pct_of_cryps < 30:
        cryp_color = 'orange'
    elif pct_of_cryps >= 30:
        cryp_color = 'tomato'
    return cryp_col_checklist, cryptic_cols, nr_of_cryps, pct_of_cryps, cryp_color

def detect_sv_col(df):
    single_value_cols_list = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            single_value_cols_list.append(col)
    nr_of_svs = len(single_value_cols_list)
    pct_of_svs = round((nr_of_svs / len(df.columns)) * 100, 2)
    if len(single_value_cols_list) != 0:
        df_sv = pd.DataFrame(columns=['Column', 'Single Value', 'Frequency', 'NaNs'])
        df_sv['Column'] = single_value_cols_list
        df_sv['Single Value'] = [df[col].unique()[0] for col in single_value_cols_list]
        df_sv['Frequency'] = [len(df[df[col] == df[col].unique()[0]]) for col in single_value_cols_list]
        df_sv['NaNs'] = [len(df[df[col].isna()]) for col in single_value_cols_list]
        dtb = dash_table.DataTable(
            id='sv_table',
            data=df_sv.to_dict('records'),
            columns=[{'name':i, 'id':i} for i in df_sv.columns],
            page_size=20
        )
    else:
        dtb = dbc.Alert(
                "No single value columns were detected in your dataset",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            )
    if pct_of_svs == 0:
        svs_color = '#50C878'
    elif pct_of_svs < 10:
        svs_color = 'orange'
    elif pct_of_svs >= 10:
        svs_color = 'tomato'
    return dtb, single_value_cols_list, nr_of_svs, pct_of_svs, svs_color

def detect_mixed_data(df):
    result = MixedDataTypes().run(df)
    mixed_dct = result.value
    print("Result value mixed: ", mixed_dct)
    columns = []
    string_pct = []
    number_pct = []
    string_ex = []
    number_ex = []
    delete_cols = []

    for col, mixed in mixed_dct.items():
        if len(mixed) != 0:
            columns.append(col)
            string_pct.append(mixed['strings'])
            number_pct.append(mixed['numbers'])
            string_ex.append(str(mixed['strings_examples']))
            number_ex.append(str(mixed['numbers_examples']))
        else:
            delete_cols.append(col)

    for col in delete_cols:
        del mixed_dct[col]

    nr_of_mds = len(mixed_dct)
    pct_of_mds = round((nr_of_mds / len(df.columns)) * 100, 2)

    new_df = pd.DataFrame()
    new_df['Column'] = columns
    new_df['Strings (%)'] = string_pct
    new_df['Numbers (%)'] = number_pct
    new_df['String Examples'] = string_ex
    new_df['Number Examples'] = number_ex

    if new_df.empty:
        dtb = dbc.Alert(
                "No mixed data type columns were detected in your dataset",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            )
    else:
        dtb = dash_table.DataTable(
            data=new_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in new_df.columns],
            page_size=20)
    if pct_of_mds == 0:
        mds_color = '#50C878'
    elif pct_of_mds < 10:
        mds_color = 'orange'
    elif pct_of_mds >= 10:
        mds_color = 'tomato'
    return dtb, mixed_dct, nr_of_mds, pct_of_mds, mds_color


def detect_mislabels(df, target):
    print(target)
    if target != "No target column":
        print("Zijn we hier?")
        model_RF = RandomForestClassifier()
        df_numeric, encoding = categorical_to_label(df)
        X = df_numeric.drop(columns=[target])
        y = df_numeric[target]
        stratified_splits = StratifiedKFold(n_splits=5)
        print("Dit is de X", X)
        print("Starting cross_val_predict")
        pred_probs = cross_val_predict(model_RF, X, y, cv=stratified_splits, method='predict_proba')
        print("Cross_val_predict completed")
        preds = np.argmax(pred_probs, axis=1)
        accuracy_score_xgbc = round((accuracy_score(preds, y) * 100), 1)
        print("This is the accuracy score: ", accuracy_score_xgbc)
        cl = cleanlab.classification.CleanLearning()
        print("Finding label issues")
        df_label_issues = cl.find_label_issues(X=None, labels=y, pred_probs=pred_probs)
        print("Label issues: ", df_label_issues)
        print("encoding: ", encoding[target].inverse_transform(df_label_issues['predicted_label'].to_list()))
        df_label_issues['predicted_label'] = encoding[target].inverse_transform(
            df_label_issues['predicted_label'].to_list())
        print(df_label_issues['predicted_label'])
        issue_indices = list(df_label_issues[df_label_issues['is_label_issue'] == True].index)
        print(issue_indices)
        new_df = df.iloc[issue_indices].reset_index()
        new_df[f'{target} (previous)'] = new_df[target]
        new_df = new_df.drop(columns=[target])
        new_df[f'{target} (predicted)'] = list(df_label_issues[df_label_issues['is_label_issue'] == True]['predicted_label'])
        if not new_df.empty:
            dtb = html.Div([
                dbc.Badge("Previous label", color="tomato", className="me-1"),
                dbc.Badge("Predicted label", color="#50C878", className="me-1"),
                dash_table.DataTable(
                data=new_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in new_df.columns],
                style_data_conditional=[{
                'if': {
                    'column_id': f"{target} (previous)",
                },
                'backgroundColor': 'tomato',
                'color': 'white'
                },
                    {
                        'if':{
                            'column_id': f"{target} (predicted)",
                        },
                        'backgroundColor':"#50C878",
                        "color":"white"
                    }],
                editable=False,
                page_size=20
            )])
        else:
            dtb = dbc.Alert(
                "No incorrect labels were detected in your dataset",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            )
        check_df = pd.DataFrame(columns=['Original label', 'Predicted label', 'Incorrect label?'])
        check_df['Index'] = new_df['index']
        check_df['Original label'] = new_df[f'{target} (previous)']
        check_df['Predicted label'] = new_df[f'{target} (predicted)']
        check_df['Incorrect label?'] = 'yes'
    else:
        dtb = dbc.Alert(
                "No target column was selected in the begin menu. Therefore, this check is not available.",
                color="orange",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            )
        check_df = pd.DataFrame()
        issue_indices = []
    nr_of_mislabels = len(issue_indices)
    pct_of_mislabels = round((nr_of_mislabels / len(df)) * 100, 2)
    print(dtb)
    if pct_of_mislabels == 0:
        mislabels_color = '#50C878'
    elif pct_of_mislabels < 10:
        mislabels_color = 'orange'
    elif pct_of_mislabels >= 10:
        mislabels_color = 'tomato'
    return dtb, check_df, issue_indices, nr_of_mislabels, pct_of_mislabels, mislabels_color

def detect_mismatch(df, target):
    result = StringMismatch().run(df)
    print("Result mismatch: ", result)
    print("Result value mismatch: ", result.value)
    dct_mismatch = result.value
    delete_cols = []
    df_dct = dict()
    for col, mismatches in dct_mismatch.items():
        if len(mismatches) == 0:
            delete_cols.append(col)
    for col in delete_cols:
        del dct_mismatch[col]
    if target in dct_mismatch:
        del dct_mismatch[target]
    nr_of_mms = len(dct_mismatch)
    pct_of_mms = round((nr_of_mms / len(df.columns)) * 100, 2)
    if len(dct_mismatch) != 0:
        for col, data in dct_mismatch.items():
            for base, var in data.items():
                new_df = pd.DataFrame(var)
                df_dct[(col, base)] = new_df

        mismatch_summary = []
        for key, value in dct_mismatch.items():
            for subkey, subvalue in value.items():
                mismatch_summary.append({'Column': key, 'Base': subkey, 'Variations': str([item['variant'] for item in subvalue]),
                             'String Mismatch?': 'yes'})
        df_mismatch_summary = pd.DataFrame(mismatch_summary)

        dtbs = html.Div(children=[
            html.Div([
                html.H4(f"String mismatch in column \"{key[0]}\" with base variant \"{key[1]}\""),
                dash_table.DataTable(
                    data=new_df.to_dict("records"),
                    columns=[{'name': i, 'id': i} for i in new_df.columns],
                    page_size=20
                )
            ]) for key, new_df in df_dct.items()
        ])
        additional_dtb = html.Div([
            html.H4("Check the string mismatches"),
            html.P("If the detected string mismatch is incorrect (i.e. it is not a string mismatch), select \"No\" in the dropdown menu of column \"String Mismatch?\". If the string mismatch is correctly identified, but the base form is incorrect, select \"Yes\" in the dropdown menu, and just change the value in the \"Base\" column to the correct base form and hit enter."),
            dash_table.DataTable(
                id='string-mismatch-table',
                data=df_mismatch_summary.to_dict('records'),
                columns=[{'name': 'Column', 'id': 'Column', 'editable': False},
                         {'name': 'Base', 'id': 'Base'},
                         {'name': 'Variations', 'id': 'Variations', 'editable': False},
                         {'name': 'String Mismatch?', 'id': 'String Mismatch?', 'presentation': 'dropdown'}],
                editable=True,
                dropdown={
                    'String Mismatch?': {
                        'options': [{'label': "Yes", 'value': 'yes'}, {'label': "No", 'value': 'no'}]
                    }
                },
                page_size=20
            )
        ])
        print(dtbs)
        print(additional_dtb)
        dtb = html.Div([dtbs,additional_dtb])
    else:
        dtb = dbc.Alert(
                "No string mismatches were detected in your dataset",
                color="#50C878",
                is_open=True,
                style={
                    "fontWeight": "bold",
                    "fontSize": '16pt',
                    "textAlign": "center",
                    'color': 'white'
                }
            )
    if pct_of_mms == 0:
        mms_color = '#50C878'
    elif pct_of_mms < 10:
        mms_color = 'orange'
    elif pct_of_mms >= 10:
        mms_color = 'tomato'
    return dtb, dct_mismatch, nr_of_mms, pct_of_mms, mms_color