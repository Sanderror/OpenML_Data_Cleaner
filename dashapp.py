import ast
import copy
import pickle

import dash
import numpy as np
from dash import dcc, html, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate
import pandas as pd
import dash_bootstrap_components as dbc
import io
import base64
from error_detection import *
import os
import uuid
import sortinghatinf
import time
from error_correction import *
from create_sections import *

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css" #dbc stylesheet
external_stylesheets = [dbc.themes.BOOTSTRAP, dbc_css]
button_style = {
                        'color': 'white', 'height': '50px', 'width': '90px',
                        'backgroundColor': '#0080FF', 'fontWeight': 'bold',
                        'textAlign': 'center', "borderRadius": "5px", "marginTop": "20px"
                    }

def imputation_model(df, mv_list, ft_types, outs=False, mix_data=False, method='minor', col_spec=False):
    if outs:
        new_mv_list = []
        for out in mv_list:
            freq = len(df[df[out[0]] == out[2]])
            new_mv_list.append([out[0], out[2], freq])
        mv_list = new_mv_list
    if mix_data:
        df_copy = copy.deepcopy(df)
        mv_list = convert_mix_to_nan(df_copy, mv_list, method, col_spec)
    print("MV list impu: ", mv_list)
    feature_type_set = set(ft_types.values())
    df_size = df.size
    mr_rate = sum([mv[2] for mv in mv_list]) / df_size
    print("MR rate: ", mr_rate)
    if len(feature_type_set) == 1:
        if list(feature_type_set)[0] == 'numeric':
            if df_size < 10000:
                model = 'MLP'
            else:
                model = 'CART'
        else:
            if df_size < 10000:
                model = 'RF'
            else:
                model = 'CART'
    else:
        if mr_rate < 0.1:
            model = 'KNN'
        else:
            if df_size < 10000:
                model = 'RF'
            else:
                model = 'CART'
    return model


def generate_filepath(uploaded_filename):
    """"helper function for obtaining the filepath for storing the dataset offline (to prevent issues with large dataset storage in-browser)"""
    session_id = str(uuid.uuid4())
    uploaded_filename = os.path.splitext(uploaded_filename)[0] #remove the file extension
    filename = f"{session_id}_{uploaded_filename}.pkl"
    filepath = os.path.join(cache_dir, filename)
    return filepath

def fetch_data(filepath):
    """"helper function to obtain the data"""
    if os.path.exists(filepath):
        # If the file already exists, load the DataFrame from the cache
        df = pd.read_pickle(filepath)
        return df

def df_to_dash_table(df):
    return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], page_size=20)


app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

current_directory = os.path.dirname(os.path.realpath('dashapp.py'))
cache_dir = os.path.join(current_directory, 'cached_files')

app.layout = html.Div([
    html.H1("OpenML Data Cleaner"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    dcc.Loading(
        id='loading-detections',
        type='default',
        children=html.Div(id='output-error-detection')
    )
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    filepath = generate_filepath(filename)
    try:
        if 'csv' in filename:
            # Assuming the format of the CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return html.Div([
                'Unsupported file format'
            ])
        df.to_pickle(filepath)

        feature_types = sortinghatinf.get_sortinghat_types(df)
        feature_type_dct = {df.columns[i]: feature_types[i] for i in range(len(df.columns))}
        ft_type_filepath = "cached_files/ft_type_" + filename
        with open(ft_type_filepath, 'wb') as handle:
            pickle.dump(feature_type_dct, handle)

        num_cols = [col for col, ft_type in feature_type_dct.items() if ft_type == 'numeric']
        numeric_df = df[num_cols]
        num_filename = "numeric_" + filename
        num_filepath = generate_filepath(num_filename)
        numeric_df.to_pickle(num_filepath)

        return html.Div([
            html.P(f"File {filename} succesfully uploaded."),
            dcc.Store(id='stored-df', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-mvs', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-dup-rows', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-dup-cols', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-out-vals', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-out-rows', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-cryps', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-svs', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-mds', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-mls', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-mms', data=filepath, storage_type='memory'),
            dcc.Store(id='stored-feature-types', data=ft_type_filepath, storage_type='memory'),
            dcc.Store(id='stored-num-df', data=num_filepath, storage_type='memory'),
            html.H4("Select the target column"),
            html.P("Select the target column of the dataset in the dropdown menu below. If there is no target column, or you do not know which one it is, select \"No target column\"."),
            dcc.Dropdown(id='target-col', options=[{'label':'No target column', 'value':'No target column'}] + [{'label':x, 'value':x} for x in df.columns], value = 'No target column'),
            html.Button('Submit', id='submit-target', n_clicks=0, style=button_style)
        ])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])



def error_detection(df, num_df, ft_types, target):
    ''' The main function that generates the whole error detection section of the tool'''
    # Perform error checks
    all_mv_table, df_mvs_summary, nr_of_mvs, pct_of_mvs, mvs_color = detect_all_mvs(df)
    detected_mvs_section = create_mvs_section(all_mv_table, df_mvs_summary)
    all_dup_rows_table, dup_row_list, nr_of_dup_rows, pct_of_dup_rows, dup_rows_color = detect_dup_row(df)
    detected_dup_rows_section = create_dup_row_section(all_dup_rows_table, dup_row_list)
    all_dup_cols_table, dup_col_list, nr_of_dup_cols, pct_of_dup_cols, dup_cols_color = detect_dup_col(df)
    detected_dup_cols_section = create_dup_col_section(all_dup_cols_table, dup_col_list)
    outlier_val_table, df_outs_summary, outlier_val_list, nr_of_out_vals, pct_of_out_vals, out_vals_color = detect_outlier_val(df, ft_types)
    detected_out_vals_section = create_out_val_section(outlier_val_table, df_outs_summary)
    outlier_row_table, out_row_check_table, df_out_row_summary, df_out_probs, outlier_row_indices, nr_of_out_rows, pct_of_out_rows, out_rows_color = detect_outlier_row(df)
    out_row_probs_fp = "cached_files/out_row_probs.csv"
    with open(out_row_probs_fp, 'wb') as handle:
        pickle.dump(df_out_probs, handle)

    cryp_col_checklist, cryp_col_list_original, nr_of_cryps, pct_of_cryps, cryps_color = detect_cryp_cols(df)

    sv_col_table, sv_col_list, nr_of_svs, pct_of_svs, svs_color = detect_sv_col(df)

    md_table, md_dct, nr_of_mds, pct_of_mds, mds_color = detect_mixed_data(df)

    md_fp = "cached_files/mixed_data_dct.csv"
    with open(md_fp, 'wb') as handle:
        pickle.dump(md_dct, handle)

    mislabel_table, df_mls_summary, mislabel_idx_list, nr_of_mislabels, pct_of_mislabels, mislabels_color = detect_mislabels(df, target)


    if not df_mls_summary.empty:
        mislabel_dtb = html.Div([
            html.H4("Are the following values correctly identified as incorrect labels?"),
            html.P(
                "In the table below, select \"Yes\" in the \"Incorrect label?\" column if the original label was indeed incorrect and select \"No\" if it was correct."),
            html.P(
                "It is possible to select a different label than the predicted label in the \"Predicted label\" column using the dropdown menu, if the original label was indeed incorrect, but the predicted label was also not correct."),
            html.Div([dash_table.DataTable(
                id='mislabels-table',
                data=df_mls_summary.to_dict('records'),
                columns=[{'name': 'Index', 'id': 'Index', 'editable': False},
                         {'name': 'Original label', 'id': 'Original label', 'editable': False},
                         {'name': 'Predicted label', 'id': 'Predicted label', 'presentation': 'dropdown'},
                         {'name': 'Incorrect label?', 'id': 'Incorrect label?', 'presentation': 'dropdown'}],
                editable=True,
                dropdown={
                    'Incorrect label?': {
                        'options': [{'label': "Yes", 'value': 'yes'}, {'label': "No", 'value': 'no'}]
                    },
                    'Predicted label': {
                        'options': [{'label': i, 'value': i} for i in df[target].unique()]
                    }
                },
                page_size=20
            )]),
            html.Div(id='mislabels-table-container'),
            html.Div(id='selected-mislabels', style={'display': 'none'}),
            html.H4(
                "Are there any additional incorrect labels in the dataset that we did not detect?"),
            html.Div([
                html.P(
                    "Choose using the dropdown menu below the index which contains the incorrect label. Then, select in the next dropdown menu what the correct label is for that row."),
                html.H4('Index'),
                html.Div(dcc.Dropdown(list(df.index), df.index[0], id='ml-idx-dropdown'), style={'display': 'inline-block', 'width': '30%'}),
                html.Div(id='ml-idx-dropdown-output'),
                html.H4('Correct label'),
                html.Div(id='ml-lbl-dropdown', style={'display': 'none'}),
                html.Div([
                    html.Div(id='ml-lbl-dropdown-container', style={'display': 'inline-block', 'width': '30%'}),
                    html.Div([html.Button('Submit', id='ml-lbl-submit', n_clicks=0, style=button_style)], style={'display': 'inline-block', 'width': '20%', 'paddingLeft': '10px'})],
                style={'display': 'flex', 'alignItems': 'center'}),
                html.Div(id='output-ml-lbl-submit')
            ]),
            html.Div(id='added-ml-list', style={'display': 'none'}),
            html.Div(id='complete-ml-output'),
            html.Div(id='complete-ml-table'),
            html.Div(id='ml-table-output'),
            html.Div(id='complete-ml-list', style={'display': 'none'})])
    else:
        mislabel_dtb = html.Div([
            html.Div(id='mislabels-table'),
            html.Div(id='mislabels-table-container'),
            html.Div(id='selected-mislabels', style={'display': 'none'}),
                dcc.Dropdown(list(df.index), df.index[0], id='ml-idx-dropdown', style={'display':'none'}),
                html.Div(id='ml-idx-dropdown-output'),
                html.Div(id='ml-lbl-dropdown', style={'display': 'none'}),
                html.Div([
                    html.Div(id='ml-lbl-dropdown-container', style={'display': 'inline-block', 'width': '30%'}),
                    html.Div([
                        html.Button('Submit', id='ml-lbl-submit', n_clicks=0, style={'display':'none'})],
                         style={'display': 'inline-block', 'width': '20%', 'paddingLeft': '10px'})],
                    style={'display': 'flex', 'alignItems': 'center'}),
                html.Div(id='output-ml-lbl-submit'),
            html.Div(id='added-ml-list', style={'display': 'none'}),
            html.Div(id='complete-ml-output'),
            html.Div(id='complete-ml-table'),
            html.Div(id='ml-table-output'),
            html.Div(id='complete-ml-list', style={'display': 'none'})])

    mismatch_table, mismatch_dct, nr_of_mms, pct_of_mms, mms_color = detect_mismatch(df, target)


    if not num_df.empty:
        add_outs_vals = html.Div([html.H4(
            "Are there any additional erroneous outlier values in the dataset that we did not detect?"),
        html.Div([
            html.P(
                'Choose using the dropdown menu below the column containing the erroneous outlier value, then choose the outlier value and press the submit button. Only numerical columns are available'),
            # html.H4('Column'),
            # dcc.Dropdown(list(num_df.columns), num_df.columns[0], id='out-col-dropdown'),
            # html.H4('Erroneous Outlier'),
            html.Div([
                html.Div([
                    html.H5('Column'),
                    dcc.Dropdown(list(num_df.columns), num_df.columns[0], id='out-col-dropdown'),
                ], style={'display': 'inline-block', 'width': '30%'}),

                html.Div([
                    html.H5('Erroneous Outlier'),
                    html.Div(id='out-val-dropdown-container'),
                    html.Div(id='out-val-dropdown', style={'display': 'none'})
                ], style={'display': 'inline-block', 'width': '30%', 'paddingLeft': '10px'}),

                html.Div([
                    html.Button('Submit', id='out-val-submit', n_clicks=0, style=button_style)
                ], style={'display': 'inline-block', 'width': '20%', 'paddingLeft': '10px'})
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ]),
            html.Div(id='output-out-submit'),
            html.Div(id='added-out-list', style={'display': 'none'}),
            html.Div(id='complete-out-output'),
            html.Div(id='complete-out-table'),
            html.Div(id='out-table-output'),
            html.Div(id='complete-out-list', style={'display': 'none'}),
        ])
    else:
        add_outs_vals = html.Div([
            html.H4("The dataset does not contain any numerical columns, so no outlier values can be added."),
            html.Div(id='out-col-dropdown', style={'display':'none'}),
            html.Div(id='out-val-dropdown-container', style={'display':'none'}),
            html.Div(id='out-val-dropdown', style={'display': 'none'}),
            html.Div(id='out-val-submit', style={'display':'none'}),
            html.Div(id='output-out-submit', style={'display': 'none'}),
            html.Div(id='added-out-list', style={'display': 'none'}),
            html.Div(id='complete-out-output', style={'display': 'none'}),
            html.Div(id='complete-out-table', style={'display': 'none'}),
            html.Div(id='out-table-output', style={'display': 'none'}),
            html.Div(id='complete-out-list', style={'display': 'none'}),
        ])

    return html.Div([
        dcc.Store(id='duplicate-rows', data=dup_row_list, storage_type='memory'),
        dcc.Store(id='duplicate-cols', data=dup_col_list, storage_type='memory'),
        dcc.Store(id='out-row-probs', data=out_row_probs_fp, storage_type='memory'),
        dcc.Store(id='sv-col-list', data=sv_col_list, storage_type='memory'),
        dcc.Store(id='mix-data-dct-fp', data=md_fp, storage_type='memory'),
        html.H2("Error Detection"),
        html.P("The first step of the OpenML Data Cleaner is to detect all data errors present in your dataset."),
        html.P("The tool detects 8 different error types. Each error type has its own section, where the tool's detections are shown."),
        html.P("At some sections, you can correct the tool's detections or add data errors of the concerning error type yourself."),
        # Accordion for each error check
        dbc.Accordion([
            dbc.AccordionItem(
                [
                    html.H2("(Disguised) Missing Values"),
                    html.P("Missing values are values in your dataset that do not contain a value. This is often represented as NaN (not a number), None, NA (not applicable) or just by an empty cell."),
                    html.P("Disguised missing values are also missing values but these values are not easily distinguishable from regular correct values. For example, the value 12345678 in a column named \"phone number\", is probably not a legitimate value. As one can understand it can be difficult to detect such disguised missing values, therefore we ask you kindly to correct the incorrect detections of our algorithm."),
                    detected_mvs_section,
                    html.H4(
                        "Are there any additional (disguised) missing values in the dataset that we did not detect?"),
                    html.Div([
                        html.P('Choose using the dropdown menu below the column containing the missing value, then choose the missing value and press the submit button.'),

                        html.Div([
                            html.Div([
                                html.H5('Column'),
                                dcc.Dropdown(list(df.columns), df.columns[0], id='mv-col-dropdown'),
                            ], style={'display': 'inline-block', 'width': '30%'}),

                            html.Div([
                                html.H5('Missing Value'),
                                html.Div(id='mv-val-dropdown-container'),
                                html.Div(id='mv-val-dropdown', style={'display': 'none'})
                            ], style={'display': 'inline-block', 'width': '30%', 'paddingLeft': '10px'}),

                            html.Div([
                                html.Button('Submit', id='mv-val-submit', n_clicks=0, style=button_style)
                            ], style={'display': 'inline-block', 'width': '20%', 'paddingLeft': '10px'})
                        ], style={'display': 'flex', 'alignItems': 'center'}),

                        # Removed code since code above is prettier
                        # html.H5('Column'),
                        # dcc.Dropdown(list(df.columns), df.columns[0], id='mv-col-dropdown'),
                        # html.H5('Missing Value'),
                        # html.Div(id='mv-val-dropdown-container'),
                        # html.Div(id='mv-val-dropdown', style={'display': 'none'}),
                        # html.Button('Submit', id='mv-val-submit', n_clicks=0, style={'color':'white', 'height': '50px', 'width':'80px', 'backgroundColor':'#0080FF','fontWeight':'bold', 'textAlign':'center', "borderRadius": "5px", "marginTop": "20px"}),
                        html.Div(id='output-mv-submit')
                    ]),
                        html.Div(id='added-mv-list', style={'display': 'none'}),
                        html.Div(id='complete-mv-output'),
                        html.Div(id='complete-mv-table'),
                        html.Div(id='mv-table-output'),
                        html.Div(id='complete-mv-list', style={'display':'None'})
                    # html.H5("Disguised Missing Values"),
                    # html.Div(dmv_result)  # Display missing values result
                ],
                title=dbc.Row(
                [
                    dbc.Col("Missing Values", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(dbc.Row(
                            [dbc.Badge(f"{nr_of_mvs} missing values; {pct_of_mvs}%", color=mvs_color, style={'fontSize': '1.1rem', 'marginRight':"20px"})
                             ], className='no-gutters flex-nowrap justify-content-end'),
                        className="col-2")
                ],
                className="w-100"
            )
            ),
            dbc.AccordionItem(
                [
                    html.H2("Duplicate Rows"),
                    detected_dup_rows_section, # Display duplicates result
                    html.Hr(),
                    html.H2("Duplicate Columns"),
                    detected_dup_cols_section  # Display duplicates result
                ],
                title=dbc.Row(
                [
                    dbc.Col("Duplicates", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Badge(f"{nr_of_dup_rows} duplicate rows; {pct_of_dup_rows}%", color=dup_rows_color,
                                          style={'fontSize': '1.1rem', 'marginRight': '10px'}),
                                dbc.Badge(f"{nr_of_dup_cols} duplicate columns; {pct_of_dup_cols}%",
                                          color=dup_cols_color, style={'fontSize': '1.1rem', 'marginRight':'20px'}),
                            ],
                            className="no-gutters flex-nowrap justify-content-end"
                        ),
                        className="col-2"
                    ),
                ],
                className="w-100"
            )
            ),
            dbc.AccordionItem(
                [
                    html.H2("Outlier Values"),
                    html.P("Outlier values in your dataset are values that significantly differ from the majority of the data in the corresponding column. They can be unusually high or low values that do not fit the general pattern of the data in the column. Outlier values can only be detected in numerical columns."),
                    html.P("We divide the outlier values into close and far values. Close outliers are closer to the distribution of the values in the column than far outliers. Therefore, we are less sure that these outliers are acutal errors compared to far outliers."),
                    detected_out_vals_section,
                    add_outs_vals,
                    html.Hr(),
                    html.H2("Outlier Rows"),
                    html.P("Similar to outlier values, outlier rows are occurences that differ significantly from the data. But instead of column-wise, outlier rows are rows that differ significantly from the other rows in the dataset."),
                    html.P("Again, we ask you kindly to check the detected outlier rows and evaluate whether they are actual errors or just rare (but legitimate) combinations of values."),
                    html.Div(outlier_row_table),
                    out_row_check_table,
                    html.Div(id='out-rows-table-dropdown-container'),
                    html.Div(id='selected-out-rows', style={'display': 'none'}),
                    html.H4(
                        "Are there any additional outlier rows in the dataset that we did not detect?"),
                    html.Div([
                        html.P(
                            'Choose using the dropdown menu below the index of the outlier row. The selected outlier row will be shown below the menu.'),
                        html.H4('Index'),
                        html.Div(dcc.Dropdown(list(df.index), df.index[0], id='out-row-idx-dropdown'), style={'display': 'inline-block', 'width': '30%'}),
                        html.Div(id='out-row-idx-dropdown-container'),
                        html.Button('Submit', id='out-row-idx-submit', n_clicks=0, style=button_style),
                        html.Div(id='output-out-row-submit')
                    ]),
                    html.Div(id='added-out-row-list', style={'display': 'none'}),
                    html.Div(id='complete-out-row-output'),
                    html.Div(id='complete-out-row-table'),
                    html.Div(id='out-row-table-output'),
                    html.Div(id='complete-out-row-list', style={'display': 'none'}),

        ],
                title=dbc.Row(
                [
                    dbc.Col("Outliers", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Badge(f"{nr_of_out_vals} outlier values; {pct_of_out_vals}%", color=out_vals_color,
                                          style={'fontSize': '1.1rem', 'marginRight': '10px'}),
                                dbc.Badge(f"{nr_of_out_rows} outlier rows; {pct_of_out_rows}%",
                                          color=out_rows_color, style={'fontSize': '1.1rem', 'marginRight':'20px'}),
                            ],
                            className="no-gutters flex-nowrap justify-content-end"
                        ),
                        className="col-2"
                    ),
                ],
                className="w-100"
            ),
            ),
            dbc.AccordionItem(
                [
                    html.H2("Cryptic Column Names"),
                    html.P(
                        "A cryptic column name is a column name in a dataset that is unclear (for someone that does not know the dataset). Examples of types of cryptic column names are abbreviations, misspellings and non-English words."),
                    html.P(
                        "Names with whitespaces separated by underscores are not cryptic. For example, test_column_gender is not crytic, but test_col_gndr is."),
                    html.H4("Are the following column names correctly identified as being \"cryptic\"?"),
                    html.P("The switches of the column names that are switched on below are detected as being cryptic by our algorithm."),
                    html.P("Please correct any mistakes our algorithm made. That is, if a column name's box is ticked, but the name is not cryptic, please untick it and if a column name's box is unticked, but the name is cryptic, please tick it."),
                    html.Div(cryp_col_checklist),
                    html.Div(id='cryp-col-list', style={'display': 'none'}),
                    html.Div(id='cryp-col-output')
                ],
                title=dbc.Row(
                [
                    dbc.Col("Cryptic Column Names", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(dbc.Row(
                            [dbc.Badge(f"{nr_of_cryps} cryptic column names; {pct_of_cryps}%", color=cryps_color, style={'fontSize': '1.1rem', 'marginRight':"20px"})
                             ], className='no-gutters flex-nowrap justify-content-end'),
                        className="col-2")
                ],
                className="w-100"
            )
            ),
            dbc.AccordionItem(
                [
                    html.H2("Single Value Columns"),
                    html.P("These are the columns with only one unique value in all the rows. If a column contains one \"real\" value and missing values then it is also classified as a single value column."),
                    html.Div(sv_col_table)
                ],
                title=dbc.Row(
                [
                    dbc.Col("Single Value Columns", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(dbc.Row(
                            [dbc.Badge(f"{nr_of_svs} single value columns; {pct_of_svs}%", color=svs_color, style={'fontSize': '1.1rem', 'marginRight':"20px"})
                             ], className='no-gutters flex-nowrap justify-content-end'),
                        className="col-2")
                ],
                className="w-100"
            )
            ),
            dbc.AccordionItem(
                [
                    html.H2("Mixed Data Type Columns"),
                    html.P("These are columns that contain both numeric values as well as strings (or text) values. For example, a column named \"age\" containing numbers (e.g. 21, 43, 76) and strings (e.g. twenty-one, forty-three, seventy-six)."),
                    html.Div(md_table)
                ],
                title=dbc.Row(
                [
                    dbc.Col("Mixed Data Type Columns", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(dbc.Row(
                            [dbc.Badge(f"{nr_of_mds} mixed data type columns; {pct_of_mds}%", color=mds_color, style={'fontSize': '1.1rem', 'marginRight':"20px"})
                             ], className='no-gutters flex-nowrap justify-content-end'),
                        className="col-2")
                ],
                className="w-100"
            )
            ),
            dbc.AccordionItem(
                [
                    html.H2("Incorrect Labels"),
                    html.P("Incorrect labels are rows that got assigned the wrong label/value of the target column. For example, a dataset with target column \"country\" and in the dataset there is a column named \"capital\". If the value in the target column for an row with capital \"Amsterdam\" is \"Germany\", then we have an incorrect label (since it should be \"The Netherlands\")."),
                    html.Div(mislabel_table),
                    mislabel_dtb
                ],
                title=dbc.Row(
                [
                    dbc.Col("Incorrect Labels", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(dbc.Row(
                            [dbc.Badge(f"{nr_of_mislabels} incorrect labels; {pct_of_mislabels}%", color=mislabels_color, style={'fontSize': '1.1rem', 'marginRight':"20px"})
                             ], className='no-gutters flex-nowrap justify-content-end'),
                        className="col-2")
                ],
                className="w-100"
            )
            ),
            dbc.AccordionItem(
                [
                    html.H2("String Mismatches"),
                    html.P("String mismatches are cases where two different text/string values in a column refer to the same entity. For example, in a column named country \"Belgium\" and \"belgium\" refer to the same country, but the values are not identical (one starts with a capital and the other not)."),
                    html.P("In the tables below, we display the string mismatches we found in your dataset. The base of these string mismatches can be found in the header above the table."),
                    html.Div(mismatch_table),
                    html.Div(id='mismatch-table-dropdown-container'),
                    html.Div(id='selected-mismatches', style={'display':'None'}),
                    html.H4("Add any additional string mismatches"),
                    html.P(
                        "First select using the dropdown menus the values in a column which are variations of each other. Then press \"Submit variations\" when done and fill in the base form of the variations in the text box and press the \"Submit mismatch\" button."),
                    html.H4("Column"),
                    html.Div(dcc.Dropdown(id='mismatch-col-dropdown', options=[{'label': x, 'value': x} for x in df.columns],
                                 value=df.columns[0]), style={'display': 'inline-block', 'width': '30%'}),
                    html.Div(id='mismatch-col-dropdown-container'),
                    html.Div(id='mismatch-val-checklist', style={'display': 'None'}),
                    html.Div("Add variation", id='add-variation', n_clicks=0, style={'display': 'None'}),
                    html.Div('Reset variations', id='reset-btn', n_clicks=0, style={'display':'None'}),
                    html.Div(id='remove-value-mismatch', style={'display':'None'}),
                    html.Div(id='value-to-remove-mismatch', style={'display':'None'}),
                    html.Div(id='output-mismatch-submit'),
                    html.Div(id='remove-value-mismatch-output'),
                    html.Div(id='variations', style={'display':'None'}),
                    html.Div("Submit variations", id='submit-variations', n_clicks=0, style={'display': 'None'}),
                    html.Div(id='variations-output'),
                    html.Div(id='added-mismatch-output'),
                    dcc.Input(id='base-variations', style={'display':'None'}),
                    html.Button('Submit base', id='submit-base-var', n_clicks=0, style={'display':'None'}),
                    html.Div(id='added-mm-list', style={'display':'none'}),
                    html.Div(id='mm-table-output'),
                    html.Div(id='complete-mm-list', style={'display':'none'})
                ],
                title=dbc.Row(
                [
                    dbc.Col("String Mismatches", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(dbc.Row(
                            [dbc.Badge(f"{nr_of_mms} string mismatch columns; {pct_of_mms}%", color=mms_color, style={'fontSize': '1.1rem', 'marginRight':"20px"})
                             ], className='no-gutters flex-nowrap justify-content-end'),
                        className="col-2")
                ],
                className="w-100"
            )
            )
        ],
            start_collapsed=True,
        ),
        html.Div([
            html.H4("Error Detection step completed"),
            html.P(
                "Press the button below if you have checked all errors detected by the tool and corrected them if needed.")
        ]),
        dcc.ConfirmDialogProvider(
            children=html.Button("Submit", style=button_style),
            id='confirm-errors',
            message='Are you sure that you have checked all errors detected by the tool?'
        ),
        dcc.Loading(
            type="default",
            id='loading-corrections',
            children=html.Div(id='output-submitted-errors')
        )
    ])

def recommendations(df, num_df, ft_types, target, mv_list, dup_rows, dup_cols, out_vals, out_rows, cryp_cols, sv_cols, mix_data_dct, ml_list, mm_list):
    ''' This is the main section where the recommendations for the corrections of the errors is generated'''
    error_summary = create_badge_numbers(df, num_df, ft_types, target, mv_list, dup_rows, dup_cols, out_vals, out_rows, cryp_cols, sv_cols, mix_data_dct, ml_list, mm_list)
    mvs_section = create_mvs_correct_section(df, mv_list, ft_types)
    dup_rows_section = create_dup_row_correct_section(dup_rows)
    dup_cols_section = create_dup_col_correct_section(dup_cols)
    out_vals_section = create_out_val_correct_section(df, out_vals, ft_types)
    out_rows_section = create_out_row_correct_section(out_rows)
    cryp_section = create_cryp_correct_section(df, cryp_cols)
    sv_section = create_sv_correct_section(sv_cols)
    md_section = create_md_correct_section(df, mix_data_dct, ft_types)
    ml_section = create_ml_correct_section(ml_list)
    mm_section = create_mm_correct_section(mm_list)

    return html.Div([
        html.H2("Error Correction"),
        html.P(
            "Now that all errors in your dataset have been detected, we will provide recommendations on how to correct these errors. You can change the correction technique yourself and directly apply it to your dataset."),
        dbc.Accordion([
            dbc.AccordionItem([
                html.H2('(Disguised) Missing Values'),
                mvs_section
                ],
                title=dbc.Row(
                [
                    dbc.Col("Missing Values", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(dbc.Row(
                            [dbc.Badge(f"{error_summary['mvs'][0]} missing values; {error_summary['mvs'][1]}%", color=error_summary['mvs'][2], style={'fontSize': '1.1rem', 'marginRight':"20px"})
                             ], className='no-gutters flex-nowrap justify-content-end'),
                        className="col-2")
                ],
                className="w-100"
            )),
            dbc.AccordionItem([
                html.H2("Duplicate Rows"),
                dup_rows_section,
                html.Hr(),
                html.H2('Duplicate Columns'),
                dup_cols_section
            ],
                title=dbc.Row(
                [
                    dbc.Col("Duplicates", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Badge(f"{error_summary['dup_rows'][0]} duplicate rows; {error_summary['dup_rows'][1]}%", color=error_summary['dup_rows'][2],
                                          style={'fontSize': '1.1rem', 'marginRight': '10px'}),
                                dbc.Badge(f"{error_summary['dup_cols'][0]} duplicate columns; {error_summary['dup_cols'][1]}%",
                                          color=error_summary['dup_cols'][2], style={'fontSize': '1.1rem', 'marginRight':'20px'}),
                            ],
                            className="no-gutters flex-nowrap justify-content-end"
                        ),
                        className="col-2"
                    ),
                ],
                className="w-100"
            )),
            dbc.AccordionItem([
                html.H2("Outlier Values"),
                out_vals_section,
                html.Hr(),
                html.H2("Outlier Rows"),
                out_rows_section
            ],
            title=dbc.Row(
                [
                    dbc.Col("Outliers", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Badge(f"{error_summary['out_vals'][0]} outlier values; {error_summary['out_vals'][1]}%", color=error_summary['out_vals'][2],
                                          style={'fontSize': '1.1rem', 'marginRight': '10px'}),
                                dbc.Badge(f"{error_summary['out_rows'][0]} outlier rows; {error_summary['out_rows'][1]}%",
                                          color=error_summary['out_rows'][2], style={'fontSize': '1.1rem', 'marginRight':'20px'}),
                            ],
                            className="no-gutters flex-nowrap justify-content-end"
                        ),
                        className="col-2"
                    ),
                ],
                className="w-100"
            )),
            dbc.AccordionItem([
                html.H2("Cryptic Column Names"),
                cryp_section
            ],
            title=dbc.Row(
                [
                    dbc.Col("Cryptic Column Names", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(dbc.Row(
                            [dbc.Badge(f"{error_summary['cryps'][0]} cryptic column names; {error_summary['cryps'][1]}%", color=error_summary['cryps'][2], style={'fontSize': '1.1rem', 'marginRight':"20px"})
                             ], className='no-gutters flex-nowrap justify-content-end'),
                        className="col-2")
                ],
                className="w-100"
            )),
            dbc.AccordionItem([
                html.H2("Single Value Columns"),
                sv_section
            ],
            title=dbc.Row(
                [
                    dbc.Col("Single Value Columns", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(dbc.Row(
                            [dbc.Badge(f"{error_summary['svs'][0]} single value columns; {error_summary['svs'][1]}%", color=error_summary['svs'][2], style={'fontSize': '1.1rem', 'marginRight':"20px"})
                             ], className='no-gutters flex-nowrap justify-content-end'),
                        className="col-2")
                ],
                className="w-100"
            )),
            dbc.AccordionItem([
                html.H2("Mixed Data Type Columns"),
                md_section
            ],
            title=dbc.Row(
                [
                    dbc.Col("Mixed Data Type Columns", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(dbc.Row(
                            [dbc.Badge(f"{error_summary['mds'][0]} mixed data type columns; {error_summary['mds'][1]}%", color=error_summary['mds'][2], style={'fontSize': '1.1rem', 'marginRight':"20px"})
                             ], className='no-gutters flex-nowrap justify-content-end'),
                        className="col-2")
                ],
                className="w-100"
            )),
            dbc.AccordionItem([
                html.H2("Incorrect Labels"),
                ml_section
            ],
            title=dbc.Row(
                [
                    dbc.Col("Incorrect Labels", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(dbc.Row(
                            [dbc.Badge(f"{error_summary['mls'][0]} incorrect labels; {error_summary['mls'][1]}%", color=error_summary['mls'][2], style={'fontSize': '1.1rem', 'marginRight':"20px"})
                             ], className='no-gutters flex-nowrap justify-content-end'),
                        className="col-2")
                ],
                className="w-100"
            )),
            dbc.AccordionItem([
                html.H2("String Mismatches"),
                mm_section
            ],
            title=dbc.Row(
                [
                    dbc.Col("String Mismatches", className="col-10", style={'fontSize': '1.25rem', 'fontWeight':'bold'}),
                    dbc.Col(dbc.Row(
                            [dbc.Badge(f"{error_summary['mms'][0]} string mismatch columns; {error_summary['mms'][1]}%", color=error_summary['mms'][2], style={'fontSize': '1.1rem', 'marginRight':"20px"})
                             ], className='no-gutters flex-nowrap justify-content-end'),
                        className="col-2")
                ],
                className="w-100"
            ))
            ]),
        html.Hr(),
        html.H4("Error correction step completed"),
        html.P("If you have corrected all detected errors, please press the button below to submit all corrections. Then, your corrected dataset will be shown."),
        html.Button("Submit", id='submit-all-corrections', n_clicks=0, style=button_style),
        html.Div(id='fully-cleaned-dataset')
    ])


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               State('upload-data', 'filename')])
def process_dataset(contents, filename):
    if contents is None:
        return html.Div(
            html.H4("Welcome to the Data Cleaner of OpenML. Upload your dataset to automatically clean it.")
        )
    else:
        return parse_contents(contents, filename)


@app.callback(Output('output-error-detection', 'children'),
              [State('stored-df', 'data'),
               State('stored-num-df', 'data'),
               State('stored-feature-types', 'data'),
              Input('target-col', 'value'),
               Input('submit-target', 'n_clicks')])
def display_error_detection_section(df_path, num_df_path, ft_types_path, target, n_clicks):
    for mv in missing_values:
        missing_values.remove(mv)
    for out in outlier_values:
        outlier_values.remove(out)
    for mm in mismatches:
        mismatches.remove(mm)

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-target' in changed_id:
        df = fetch_data(df_path)
        num_df = fetch_data(num_df_path)
        with open(ft_types_path, 'rb') as handle:
            ft_types = pickle.load(handle)
        return error_detection(df, num_df, ft_types, target)


@app.callback([Output('mv-table-dropdown-container', 'children'),
               Output('selected-mvs', 'data')],
              [Input('mv-table', 'data')]
              )
def display_mv_values(data):
    selected_mvs = []
    if data:
        for row in data:
            if row['Missing Value?'] == 'yes':
                mv_chars = [row['Column'], row['Value'], row['Frequency']]
                selected_mvs.append(mv_chars)
        selected_mvs_text = "The selected missing values are: " + ', '.join([f"\"{mvs[1]}\" in column \"{mvs[0]}\" occuring {mvs[2]} times" for mvs in selected_mvs])
        selected_mvs_fp = "cached_files/selected_mvs.csv"
        with open(selected_mvs_fp, 'wb') as handle:
            pickle.dump(selected_mvs, handle)
        return html.Div(selected_mvs_text), dcc.Store(id='selected-mvs', data=selected_mvs_fp, storage_type='memory')
    else:
        selected_mvs_fp = "cached_files/selected_mvs.csv"
        with open(selected_mvs_fp, 'wb') as handle:
            pickle.dump(selected_mvs, handle)
        return html.P('No missing values were found'), dcc.Store(id='selected-mvs', data=selected_mvs_fp, storage_type='memory')

@app.callback(
    Output('mv-val-dropdown-container', 'children'),
    [Input('mv-col-dropdown', 'value'),
     State('stored-df', 'data')]
)
def generate_value_dropdown(value, filepath):
    df = fetch_data(filepath).astype(str)
    col_vals = list(df[value].unique())
    return dcc.Dropdown(col_vals, col_vals[0], id='mv-val-dropdown')

@app.callback(
    [Output('output-mv-submit', 'children'),
     Output('added-mv-list', 'data')],
    [State('mv-val-dropdown', 'value'),
     State('stored-df', 'data'),
     State('mv-col-dropdown', 'value'),
     Input('mv-val-submit', 'n_clicks'),
     Input('selected-mvs', 'data')]
)
def update_missing_values(value, filepath, column, n_clicks, selected_mvs_data):
    added_mvs_fp = "cached_files/added_mvs.csv"
    selected_mvs_fp = selected_mvs_data['props']['data']
    with open(selected_mvs_fp, 'rb') as handle:
        selected_mvs = pickle.load(handle)
    df = fetch_data(filepath).astype(str)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if n_clicks > 0 and 'mv-val-submit' in changed_id:
        freq = len(df[df[column] == value])
        mv_char = [column, value, freq]
        if mv_char not in missing_values and mv_char not in selected_mvs:
            missing_values.append(mv_char)
            with open(added_mvs_fp, 'wb') as handle:
                pickle.dump(missing_values, handle)
            print("Missing values", missing_values)
            return (html.Div([
                html.Label('All Missing Values identified: ')
            ]), dcc.Store(id='added-mv-list', data=added_mvs_fp, storage_type='memory'))
        else:
            with open(added_mvs_fp, 'wb') as handle:
                pickle.dump(missing_values, handle)
            print("Missing values", missing_values)
            return (html.Div([
                html.P('This value is already in the missing value list')
            ]), dcc.Store(id='added-mv-list', data=added_mvs_fp, storage_type='memory'))
    else:
        with open(added_mvs_fp, 'wb') as handle:
            pickle.dump(missing_values, handle)
        print("Missing values", missing_values)
        return (html.Div([
            html.Label('All Missing Values identified: ')
        ]), dcc.Store(id='added-mv-list', data=added_mvs_fp, storage_type='memory'))
#hallo

@app.callback(
    Output('complete-mv-output', 'children'),
    [Input('added-mv-list', 'data'),
     Input('selected-mvs','data')]
)
def display_complete_mvs(added_mvs_data, selected_mvs_data):
    selected_mvs_fp = selected_mvs_data['props']['data']
    added_mvs_fp = added_mvs_data['props']['data']
    with open(selected_mvs_fp, 'rb') as handle:
        selected_mvs = pickle.load(handle)
    with open(added_mvs_fp, 'rb') as handle_2:
        added_mvs = pickle.load(handle_2)
    complete_mv_list = selected_mvs + added_mvs
    print("COmplete mv list: ", complete_mv_list)
    unique_set = set(tuple(sublist) for sublist in complete_mv_list)
    print("Unique set: ", unique_set)
    unique_list = [list(subtuple) for subtuple in unique_set]
    print("Unique list: ", unique_list)
    mv_df = pd.DataFrame(unique_list, columns=['Column', "Value", 'Frequency'])
    mv_list_filepath = "cached_files/mv_list.csv"
    with open(mv_list_filepath, 'wb') as h:
        pickle.dump(unique_list, h)
    return html.Div([dcc.Store(id='complete-mv-list', data=mv_list_filepath, storage_type='memory'),
        dash_table.DataTable(
            id='complete-mv-table',
            data=mv_df.to_dict('records'),
            columns=[
                {'name': 'Column', 'id': 'Column'},
                {'name': 'Value', 'id': 'Value'},
                {'name': 'Frequency', 'id': 'Frequency'}
            ],
            editable=False,
            row_deletable=True
        )
    ])

@app.callback(Output('mv-table-output', 'children'),
              [Input("complete-mv-table", 'data_previous'),
                  State('complete-mv-table', 'data'),
               Input('selected-mvs', 'data'),
               Input('added-mv-list', 'data')])
def store_all_mvs(mv_table_previous, mv_table, selected_mvs_data, added_mvs_data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'complete-mv-table' in changed_id:
        selected_mvs_fp = selected_mvs_data['props']['data']
        added_mvs_fp = added_mvs_data['props']['data']
        with open(selected_mvs_fp, 'rb') as handle:
            selected_mvs = pickle.load(handle)
        with open(added_mvs_fp, 'rb') as handle:
            added_mvs = pickle.load(handle)
        complete_mv_list = []
        row_to_delete = []
        if mv_table_previous:
            for row in mv_table_previous:
                if row not in mv_table:
                    row_to_delete = [row['Column'], row['Value'], row['Frequency']]
            if row_to_delete in selected_mvs:
                selected_mvs.remove(row_to_delete)
            elif row_to_delete in missing_values:
                missing_values.remove(row_to_delete)
            else:
                print("This row is neither in selected mvs nor in missing_values: ", row_to_delete)
        for row in mv_table:
            complete_mv_list.append([row['Column'], row['Value'], row['Frequency']])
        for mv in selected_mvs:
            if mv not in complete_mv_list:
                selected_mvs.remove(mv)
        for mv in missing_values:
            if mv not in complete_mv_list:
                missing_values.remove(mv)
                if mv in added_mvs:
                    added_mvs.remove(mv)
        with open(selected_mvs_fp, 'wb') as handle:
            pickle.dump(selected_mvs, handle)
        with open(added_mvs_fp, 'wb') as handle:
            pickle.dump(added_mvs, handle)
        mv_list_filepath = "cached_files/mv_list.csv"
        with open(mv_list_filepath, 'wb') as handle:
            pickle.dump(complete_mv_list, handle)
        return html.Div(dcc.Store(id='complete-mv-list', data=mv_list_filepath, storage_type='memory'))

#
# Example of using selected-mvs stored as: [column, value, frequency]
# @app.callback(
#     Output('display-selected-mvs', 'children'),
#     [Input('selected-mvs', 'data')]
# )
# def use_selected_mvs(data):
#     if data:
#         return html.Div(f"selected mvs: {data}")

@app.callback([Output('out-rows-table-dropdown-container', 'children'),
               Output('selected-out-rows','data')],
              [Input('out-rows-table', 'data')]
              )
def display_outs_rows(data):
    selected_out_rows = []
    if data:
        for row in data:
            if row['Outlier row?'] == 'yes':
                out_row_chars = [row['Index'], row['Probability']]
                selected_out_rows.append(out_row_chars)
        selected_out_rows_text = f"The selected outlier rows are: " + ', '.join([f"row with index \"{outs[0]}\" with probability {outs[1]}" for outs in selected_out_rows])
        selected_out_rows_fp = "cached_files/selected_out_rows.csv"
        with open(selected_out_rows_fp, 'wb') as handle:
            pickle.dump(selected_out_rows, handle)
        return html.Div(selected_out_rows_text), dcc.Store(id='selected-out-rows', data=selected_out_rows_fp, storage_type='memory')
    else:
        selected_out_rows_fp = "cached_files/selected_out_rows.csv"
        with open(selected_out_rows_fp, 'wb') as handle:
            pickle.dump(selected_out_rows, handle)
        return html.P(), dcc.Store(id='selected-out-rows', data=selected_out_rows_fp, storage_type='memory')

@app.callback(
    Output('out-row-idx-dropdown-container', 'children'),
    [Input('out-row-idx-dropdown', 'value'),
     State('stored-df', 'data')]
)
def generate_out_rows_dropdown(idx, filepath):
    df = fetch_data(filepath)
    df_idx = df.loc[[idx]].reset_index()
    dtb = dash_table.DataTable(
        data=df_idx.to_dict('records'),
        columns=[{'name':i,'id':i} for i in df_idx.columns],
        page_size=20
    )
    return html.Div(dtb)

@app.callback(
    [Output('output-out-row-submit', 'children'),
     Output('added-out-row-list', 'data')],
    [State('out-row-idx-dropdown', 'value'),
     Input('out-row-idx-submit', 'n_clicks'),
     Input('selected-out-rows', 'data'),
     Input('out-row-probs', 'data')]
)
def update_outlier_values(idx, n_clicks, selected_out_rows_data, out_probs_fp):
    added_out_rows_fp = 'cached_files/added_out_rows.csv'
    if selected_out_rows_data != None:
        selected_out_rows_fp = selected_out_rows_data['props']['data']
        with open(selected_out_rows_fp, 'rb') as handle:
            selected_out_rows = pickle.load(handle)
    if out_probs_fp != None:
        with open(out_probs_fp, 'rb') as handle:
            df_out_probs = pickle.load(handle)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if n_clicks > 0 and 'out-row-idx-submit' in changed_id:
        prob = df_out_probs.loc[idx, 'Outlier Probability Score']
        out_char = [idx, prob]
        selected_idx = [out[0] for out in selected_out_rows]
        outlier_rows_idx = [out[0] for out in outlier_rows]
        if idx not in outlier_rows_idx and idx not in selected_idx:
            outlier_rows.append(out_char)
            with open(added_out_rows_fp, 'wb') as handle:
                pickle.dump(outlier_rows, handle)
            return (html.Div([
                html.Label('All outlier rows identified: ')
            ]), dcc.Store(id='added-out-row-list', data=added_out_rows_fp, storage_type='memory'))
        else:
            with open(added_out_rows_fp, 'wb') as handle:
                pickle.dump(outlier_rows, handle)
            return (html.Div([
                html.P(f'This row with index {idx} is already in the list of outlier rows.')
            ]), dcc.Store(id='added-out-row-list', data=added_out_rows_fp, storage_type='memory'))
    else:
        with open(added_out_rows_fp, 'wb') as handle:
            pickle.dump(outlier_rows, handle)
        return (html.Div([
            html.Label('All outlier rows identified: ')
        ]), dcc.Store(id='added-out-row-list', data=added_out_rows_fp, storage_type='memory'))
#hallo

@app.callback(
    Output('complete-out-row-output', 'children'),
    [Input('added-out-row-list', 'data'),
     Input('selected-out-rows','data')]
)
def display_complete_outs(added_out_rows_data, selected_out_rows_data):
    selected_out_rows = []
    added_out_rows = []
    if selected_out_rows_data != None:
        selected_out_rows_fp = selected_out_rows_data['props']['data']
        with open(selected_out_rows_fp, 'rb') as handle:
            selected_out_rows = pickle.load(handle)
    if added_out_rows_data != None:
        added_out_rows_fp = added_out_rows_data['props']['data']
        with open(added_out_rows_fp, 'rb') as handle_2:
            added_out_rows = pickle.load(handle_2)
    complete_out_row_list = selected_out_rows + added_out_rows
    unique_set = set(tuple(sublist) for sublist in complete_out_row_list)
    unique_list = [list(subtuple) for subtuple in unique_set]
    out_df = pd.DataFrame(unique_list, columns=['Index', "Probability"])
    out_row_list_filepath = "cached_files/out_row_list.csv"
    with open(out_row_list_filepath, 'wb') as handle:
        pickle.dump(unique_list, handle)
    return html.Div([dcc.Store(id='complete-out-row-list', data=out_row_list_filepath, storage_type='memory'),
        dash_table.DataTable(
            id='complete-out-row-table',
            data=out_df.to_dict('records'),
            columns=[
                {'name': 'Index', 'id': 'Index'},
                {'name': 'Probability', 'id': 'Probability'}
            ],
            editable=False,
            row_deletable=True
        )
    ])

@app.callback(Output('out-row-table-output', 'children'),
              [Input("complete-out-row-table", 'data_previous'),
                  State('complete-out-row-table', 'data'),
               Input('selected-out-rows', 'data'),
               Input('added-out-row-list', 'data')])
def store_all_out_rows(out_table_previous, out_table, selected_out_rows_data, added_out_rows_data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'complete-out-row-table' in changed_id:
        selected_out_rows_fp = selected_out_rows_data['props']['data']
        added_out_rows_fp = added_out_rows_data['props']['data']
        with open(selected_out_rows_fp, 'rb') as handle:
            selected_out_rows = pickle.load(handle)
        with open(added_out_rows_fp, 'rb') as handle:
            added_out_rows = pickle.load(handle)
        complete_out_row_list = []
        row_to_delete = []
        if out_table_previous:
            for row in out_table_previous:
                if row not in out_table:
                    row_to_delete = [row['Index'], row['Probability']]
            if row_to_delete in selected_out_rows:
                selected_out_rows.remove(row_to_delete)
            elif row_to_delete in outlier_rows:
                outlier_rows.remove(row_to_delete)
            else:
                print("This row is neither in selected outs nor in outlier_values: ", row_to_delete)
        for row in out_table:
            complete_out_row_list.append([row['Index'], row['Probability']])
        for out in selected_out_rows:
            if out not in complete_out_row_list:
                selected_out_rows.remove(out)
        for out in outlier_rows:
            if out not in complete_out_row_list:
                outlier_rows.remove(out)
                if out in added_out_rows:
                    added_out_rows.remove(out)
        with open(selected_out_rows_fp, 'wb') as handle:
            pickle.dump(selected_out_rows, handle)
        with open(added_out_rows_fp, 'wb') as handle:
            pickle.dump(added_out_rows, handle)
        out_row_list_filepath = "cached_files/out_row_list.csv"
        with open(out_row_list_filepath, 'wb') as handle:
            pickle.dump(complete_out_row_list, handle)
        return html.Div(dcc.Store(id='complete-out-row-list', data=out_row_list_filepath, storage_type='memory'))



@app.callback([Output('outs-table-dropdown-container', 'children'),
               Output('selected-outs','data')],
              [Input('outs-table', 'data')]
              )
def display_outs_values(data):
    selected_outs = []
    if data:
        for row in data:
            if row['Erroneous Outlier?'] == 'yes':
                out_chars = [row['Column'], row['Type'], row['Value']]
                selected_outs.append(out_chars)
        selected_outs_text = f"The selected erroneous outlier values are: " + ', '.join([f"\"{outs[2]}\" in column \"{outs[0]}\" being a {outs[1]} outlier" for outs in selected_outs])
        selected_outs_fp = "cached_files/selected_outs.csv"
        with open(selected_outs_fp, 'wb') as handle:
            pickle.dump(selected_outs, handle)
        return html.Div(selected_outs_text), dcc.Store(id='selected-outs', data=selected_outs_fp, storage_type='memory')
    else:
        selected_outs_fp = "cached_files/selected_outs.csv"
        with open(selected_outs_fp, 'wb') as handle:
            pickle.dump(selected_outs, handle)
        return html.P('No outliers were found'), dcc.Store(id='selected-outs', data=selected_outs_fp, storage_type='memory')

@app.callback(
    Output('out-val-dropdown-container', 'children'),
    [Input('out-col-dropdown', 'value'),
     State('stored-num-df', 'data')]
)
def generate_outs_dropdown(value, filepath):
    if value != None:
        df = fetch_data(filepath)
        col_vals = list(df[value].unique())
        return dcc.Dropdown(col_vals, col_vals[0], id='out-val-dropdown')
    else:
        return dcc.Dropdown([], id='out-val-dropdown')

@app.callback(
    [Output('output-out-submit', 'children'),
     Output('added-out-list', 'data')],
    [State('out-val-dropdown', 'value'),
     State('out-col-dropdown', 'value'),
     Input('out-val-submit', 'n_clicks'),
     Input('selected-outs', 'data')]
)
def update_outlier_values(value, column, n_clicks, selected_outs_data):
    added_outs_fp = 'cached_files/added_outs.csv'
    selected_outs_fp = selected_outs_data['props']['data']
    with open(selected_outs_fp, 'rb') as handle:
        selected_outs = pickle.load(handle)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if n_clicks != None:
        if n_clicks > 0 and 'out-val-submit' in changed_id:
            out_char = [column, 'User selected', value]
            trimmed_char = [out_char[0], out_char[2]]
            trimmed_outs = [[out[0], out[2]] for out in selected_outs]
            if out_char not in outlier_values and trimmed_char not in trimmed_outs:
                outlier_values.append(out_char)
                with open(added_outs_fp, 'wb') as handle:
                    pickle.dump(outlier_values, handle)
                return (html.Div([
                    html.Label('All Erroneous Outliers identified: ')
                ]), dcc.Store(id='added-out-list', data=added_outs_fp, storage_type='memory'))
            else:
                with open(added_outs_fp, 'wb') as handle:
                    pickle.dump(outlier_values, handle)
                return (html.Div([
                    html.P('This value is already in the erroneous outliers list')
                ]), dcc.Store(id='added-out-list', data=added_outs_fp, storage_type='memory'))
        else:
            with open(added_outs_fp, 'wb') as handle:
                pickle.dump(outlier_values, handle)
            return (html.Div([
                html.Label('All Erroneous Outliers identified: ')
            ]), dcc.Store(id='added-out-list', data=added_outs_fp, storage_type='memory'))
    else:
        with open(added_outs_fp, 'wb') as handle:
            pickle.dump(outlier_values, handle)
        return (html.Div([
            html.P()
        ]), dcc.Store(id='added-out-list', data=added_outs_fp, storage_type='memory'))
#hallo

@app.callback(
    Output('complete-out-output', 'children'),
    [Input('added-out-list', 'data'),
     Input('selected-outs','data')]
)
def display_complete_outs(added_outs_data, selected_outs_data):
    selected_outs_fp = selected_outs_data['props']['data']
    added_outs_fp = added_outs_data['props']['data']
    with open(selected_outs_fp, 'rb') as handle:
        selected_outs = pickle.load(handle)
    with open(added_outs_fp, 'rb') as handle_2:
        added_outs = pickle.load(handle_2)
    complete_out_list = selected_outs + added_outs
    unique_set = set(tuple(sublist) for sublist in complete_out_list)
    unique_list = [list(subtuple) for subtuple in unique_set]
    out_df = pd.DataFrame(unique_list, columns=['Column', "Type", 'Value'])
    out_list_filepath = "cached_files/out_list.csv"
    with open(out_list_filepath, 'wb') as handle:
        pickle.dump(unique_list, handle)
    return html.Div([dcc.Store(id='complete-out-list', data=out_list_filepath, storage_type='memory'),
        dash_table.DataTable(
            id='complete-out-table',
            data=out_df.to_dict('records'),
            columns=[
                {'name': 'Column', 'id': 'Column'},
                {'name': 'Type', 'id': 'Type'},
                {'name': 'Value', 'id': 'Value'}
            ],
            editable=False,
            row_deletable=True
        )
    ])

@app.callback(Output('out-table-output', 'children'),
              [Input("complete-out-table", 'data_previous'),
                  State('complete-out-table', 'data'),
               Input('selected-outs', 'data'),
               Input('added-out-list', 'data')])
def store_all_outs(out_table_previous, out_table, selected_outs_data, added_outs_data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'complete-out-table' in changed_id:
        selected_outs_fp = selected_outs_data['props']['data']
        added_outs_fp = added_outs_data['props']['data']
        with open(selected_outs_fp, 'rb') as handle:
            selected_outs = pickle.load(handle)
        with open(added_outs_fp, 'rb') as handle:
            added_outs = pickle.load(handle)
        complete_out_list = []
        row_to_delete = []
        if out_table:
            if out_table_previous:
                for row in out_table_previous:
                    if row not in out_table:
                        row_to_delete = [row['Column'], row['Type'], row['Value']]
                if row_to_delete in selected_outs:
                    selected_outs.remove(row_to_delete)
                elif row_to_delete in outlier_values:
                    outlier_values.remove(row_to_delete)
                else:
                    print("This row is neither in selected outs nor in outlier_values: ", row_to_delete)
            for row in out_table:
                complete_out_list.append([row['Column'], row['Type'], row['Value']])
            for out in selected_outs:
                if out not in complete_out_list:
                    selected_outs.remove(out)
            for out in outlier_values:
                if out not in complete_out_list:
                    outlier_values.remove(out)
                    if out in added_outs:
                        added_outs.remove(out)
            with open(selected_outs_fp, 'wb') as handle:
                pickle.dump(selected_outs, handle)
            with open(added_outs_fp, 'wb') as handle:
                pickle.dump(added_outs, handle)
            out_list_filepath = "cached_files/out_list.csv"
            with open(out_list_filepath, 'wb') as handle:
                pickle.dump(complete_out_list, handle)
            return html.Div(dcc.Store(id='complete-out-list', data=out_list_filepath, storage_type='memory'))

@app.callback(
    [
        Output("cryp-col-list", 'data'),
        Output('cryp-col-output', 'children')
     ],
    Input('cryptic-cols', 'value')
)
def display_and_store_cryps(data):
    cryp_list = []
    for val in data:
        cryp_list.append(val)
    if len(cryp_list) != 0:
        cryp_text = "The selected cryptic column names are: " + ', '.join(
            [f"{col}" for col in cryp_list])
    else:
        cryp_text = "No column names have been selected as being cryptic"
    return dcc.Store(id= 'cryp-col-list', data=cryp_list, storage_type='memory'), html.Div(cryp_text)


@app.callback([Output('mislabels-table-container', 'children'),
               Output('selected-mislabels', 'data')],
              [Input('mislabels-table', 'data')]
              )
def display_ml_values(data):
    selected_mls = []
    if data:
        for row in data:
            if row['Incorrect label?'] == 'yes':
                ml_chars = [row['Index'], row['Original label'], row['Predicted label']]
                selected_mls.append(ml_chars)
        selected_mls_text = "The selected incorrect labels are: " + ', '.join([f"\"{mls[1]}\" which should be \"{mls[2]}\" on index \"{mls[0]}\"" for mls in selected_mls])
        selected_mls_fp = "cached_files/selected_mislabels.csv"
        with open(selected_mls_fp, 'wb') as handle:
            pickle.dump(selected_mls, handle)
        return html.Div(selected_mls_text), dcc.Store(id='selected-mislabels', data=selected_mls_fp, storage_type='memory')
    else:
        selected_mls_fp = "cached_files/selected_mislabels.csv"
        with open(selected_mls_fp, 'wb') as handle:
            pickle.dump(selected_mls, handle)
        return html.P(''), dcc.Store(id='selected-mislabels', data=selected_mls_fp, storage_type='memory')

@app.callback(
    [Output('ml-idx-dropdown-output', 'children'),
     Output('ml-lbl-dropdown-container', 'children')],
    [Input('ml-idx-dropdown', 'value'),
     State('stored-df', 'data'),
     State('target-col', 'value')]
)
def generate_mls_dropdown(value, filepath, target):
    print(value, filepath, target)
    if target != 'No target column':
        df = fetch_data(filepath)
        df_idx = df.loc[[value]].reset_index()
        print("DF idx", df_idx)
        all_labels = df[target].unique()
        print("all labels", all_labels)
        dtb = dash_table.DataTable(
            data=df_idx.to_dict('records'),
            columns=[{'name':i, 'id':i} for i in df_idx.columns],
            page_size=20
        )
        return html.Div(dtb), html.Div([dcc.Dropdown(all_labels, all_labels[0], id='ml-lbl-dropdown')])
    else:
        return "", ""

@app.callback(
    [Output('output-ml-lbl-submit', 'children'),
     Output('added-ml-list', 'data')],
    [State('ml-lbl-dropdown', 'value'),
     State('ml-idx-dropdown', 'value'),
     Input('ml-lbl-submit', 'n_clicks'),
     Input('selected-mislabels', 'data'),
     State('stored-df', 'data'),
     State('target-col', 'value')]
)
def update_mislabel_values(value, idx, n_clicks, selected_mls_data, df_fp, target):
    added_mls_fp = 'cached_files/added_mislabels.csv'
    selected_mls_fp = selected_mls_data['props']['data']
    if target != 'No target column':
        df = fetch_data(df_fp)
        prev_value = df.loc[idx, target]
        with open(selected_mls_fp, 'rb') as handle:
            selected_mls = pickle.load(handle)
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if n_clicks > 0 and 'ml-lbl-submit' in changed_id:
            if prev_value != value:
                ml_char = [idx, prev_value, value]
                mislabels_idx = [ml[0] for ml in mislabels]
                selected_mls_idx = [ml[0] for ml in selected_mls]
                if idx not in mislabels_idx and idx not in selected_mls_idx:
                    mislabels.append(ml_char)
                    with open(added_mls_fp, 'wb') as handle:
                        pickle.dump(mislabels, handle)
                    return (html.Div([
                        html.Label('All incorrect labels identified: ')
                    ]), dcc.Store(id='added-ml-list', data=added_mls_fp, storage_type='memory'))
                else:
                    with open(added_mls_fp, 'wb') as handle:
                        pickle.dump(mislabels, handle)
                    return (html.Div([
                        html.P(f'This row with index {idx} is already identified as an incorrect label. If you want to change the correct label of this row, first remove the row in the table below.')
                    ]), dcc.Store(id='added-ml-list', data=added_mls_fp, storage_type='memory'))
            else:
                with open(added_mls_fp, 'wb') as handle:
                    pickle.dump(mislabels, handle)
                return (html.Div([
                    html.P(
                        f'The selected corrected label is the same as the previous incorrect label: {prev_value} = {value}. Please, select a different label if you want to add this row with index {idx} to the list of incorrect labels.')
                ]), dcc.Store(id='added-ml-list', data=added_mls_fp, storage_type='memory'))
        else:
            with open(added_mls_fp, 'wb') as handle:
                pickle.dump(mislabels, handle)
            return (html.Div([
                html.Label('All incorrect labels identified: ')
            ]), dcc.Store(id='added-ml-list', data=added_mls_fp, storage_type='memory'))
    else:
        with open(added_mls_fp, 'wb') as handle:
            pickle.dump(mislabels, handle)
        return (html.Div(""), dcc.Store(id='added-ml-list', data=added_mls_fp, storage_type='memory'))
#hallo

@app.callback(
    Output('complete-ml-output', 'children'),
    [Input('added-ml-list', 'data'),
     Input('selected-mislabels','data')]
)
def display_complete_mls(added_mls_data, selected_mls_data):
    selected_mls_fp = selected_mls_data['props']['data']
    if added_mls_data != None:
        added_mls_fp = added_mls_data['props']['data']

        with open(selected_mls_fp, 'rb') as handle:
            selected_mls = pickle.load(handle)
        with open(added_mls_fp, 'rb') as handle_2:
            added_mls = pickle.load(handle_2)
        complete_ml_list = selected_mls + added_mls
        unique_set = set(tuple(sublist) for sublist in complete_ml_list)
        unique_list = [list(subtuple) for subtuple in unique_set]
        mls_df = pd.DataFrame(unique_list, columns=['Index', "Previous label", 'Correct label'])
        return html.Div([
            dash_table.DataTable(
                id='complete-ml-table',
                data=mls_df.to_dict('records'),
                columns=[{'name':i, 'id':i} for i in mls_df.columns],
                editable=False,
                row_deletable=True
            )
        ])
    else:
        return html.P("", id='complete-ml-table')

@app.callback(Output('ml-table-output', 'children'),
              [Input("complete-ml-table", 'data_previous'),
                  State('complete-ml-table', 'data'),
               Input('selected-mislabels', 'data'),
               Input('added-ml-list', 'data')])
def store_all_mls(ml_table_previous, ml_table, selected_mls_data, added_mls_data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if ml_table:
        if 'complete-ml-table' in changed_id:
            selected_mls_fp = selected_mls_data['props']['data']
            added_mls_fp = added_mls_data['props']['data']
            with open(selected_mls_fp, 'rb') as handle:
                selected_mls = pickle.load(handle)
            with open(added_mls_fp, 'rb') as handle:
                added_mls = pickle.load(handle)
            complete_ml_list = []
            row_to_delete = []
            if ml_table:
                if ml_table_previous:
                    for row in ml_table_previous:
                        if row not in ml_table:
                            row_to_delete = [row['Index'], row['Previous label'], row['Correct label']]
                    if row_to_delete in selected_mls:
                        selected_mls.remove(row_to_delete)
                    elif row_to_delete in mislabels:
                        mislabels.remove(row_to_delete)
                    else:
                        print("This row is neither in selected mls nor in mislabels: ", row_to_delete)
                for row in ml_table:
                    complete_ml_list.append([row['Index'], row['Previous label'], row['Correct label']])
                for ml in selected_mls:
                    if ml not in complete_ml_list:
                        selected_mls.remove(ml)
                for ml in mislabels:
                    if ml not in complete_ml_list:
                        mislabels.remove(ml)
                        if ml in added_mls:
                            added_mls.remove(ml)
                with open(selected_mls_fp, 'wb') as handle:
                    pickle.dump(selected_mls, handle)
                with open(added_mls_fp, 'wb') as handle:
                    pickle.dump(added_mls, handle)
                mls_list_filepath = "cached_files/mislabels_list.csv"
                with open(mls_list_filepath, 'wb') as handle:
                    pickle.dump(complete_ml_list, handle)
                return html.Div(dcc.Store(id='complete-ml-list', data=mls_list_filepath, storage_type='memory'))
    else:
        mls_list_filepath = "cached_files/mislabels_list.csv"
        complete_ml_list = []
        with open(mls_list_filepath, 'wb') as handle:
            pickle.dump(complete_ml_list, handle)
        return html.Div(dcc.Store(id='complete-ml-list', data=mls_list_filepath, storage_type='memory'))

@app.callback([Output('mismatch-table-dropdown-container', 'children'),
               Output('selected-mismatches','data')],
              [Input('string-mismatch-table', 'data')]
              )
def display_mismatches(data):
    if data:
        selected_mismatches = []
        for row in data:
            if row['String Mismatch?'] == 'yes':
                mismatch_chars = [row['Column'], row['Base'], row['Variations']]
                selected_mismatches.append(mismatch_chars)
        selected_mismatches_text = "The selected string mismatches are: " + ', '.join([f"\"{mismatch[2]}\" in column \"{mismatch[0]}\" with base form {mismatch[1]}" for mismatch in selected_mismatches])
        sel_mm_filepath = "cached_files/selected_mismatches.csv"
        with open(sel_mm_filepath, 'wb') as handle:
            pickle.dump(selected_mismatches, handle)
        return html.Div(selected_mismatches_text), dcc.Store(id='selected_mismatches', data=sel_mm_filepath, storage_type='memory')
    else:
        selected_mismatches = []
        sel_mm_filepath = "cached_files/selected_mismatches.csv"
        with open(sel_mm_filepath, 'wb') as handle:
            pickle.dump(selected_mismatches, handle)
        return html.P('No string mismatches were found'), dcc.Store(id='selected_mismatches', data=sel_mm_filepath, storage_type='memory')

@app.callback(
    Output('mismatch-col-dropdown-container', 'children'),
     [Input('mismatch-col-dropdown', 'value'),
     State('stored-df', 'data')]
)
def generate_value_dropdown(value, filepath):
    df = fetch_data(filepath).astype(str)
    col_vals = list(df[value].unique())
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'mismatch-col-dropdown' in changed_id:
        return html.Div([
            html.Div([
            html.Div([html.H4("Value"), dcc.Dropdown(col_vals, id='mismatch-val-checklist')], style={'display': 'inline-block', 'width': '30%'})]),
            html.Button("Add", id='add-variation', n_clicks=0, style=button_style),
            html.Button('Reset', id='reset-btn', n_clicks=0, style=button_style)
        ])

@app.callback(
    [Output('output-mismatch-submit', 'children'),
    Output('variations', 'data'),
     Output('value-to-remove-mismatch', 'data')],
    [State('stored-df', 'data'),
    State('mismatch-val-checklist', 'value'),
     State('mismatch-col-dropdown', 'value'),
     State('variations', 'data'),
     Input('add-variation', 'n_clicks'),
     Input('reset-btn', 'n_clicks'),
     Input('selected-mismatches', 'data')]
) # to do morgen: new-mismatch verwijderen en proberen selected-mismatches naar pickle file te doen en dan inladen miss dan geen probleem
def update_mismatches(filepath, value, column, variations, add_clicks, reset_clicks, selected_mms_data):
    if 'props' not in selected_mms_data or 'data' not in selected_mms_data['props']:
        print("Error: selected_mms_data is improperly formatted first occurence")
    selected_mismatches_fp = selected_mms_data['props']['data']
    with open(selected_mismatches_fp, 'rb') as handle:
        selected_mismatches = pickle.load(handle)
    df = fetch_data(filepath).astype(str)
    all_mismatches = selected_mismatches + mismatches
    wrong_val = None
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered]
    if 'reset-btn.n_clicks' in changed_id:
        variations = []
    if 'add-variation.n_clicks' in changed_id and value != None:
        for mismatch in all_mismatches:
            if column == mismatch[0] and value in ast.literal_eval(mismatch[2]):
                wrong_val = (column, value, mismatch)
                return [html.Div([html.P(f"This value {value} is already present as a variation of column {column} of one of the string mismatches. It is not possible for a value to be part of two different string mismatches, since it should have only one correct base value. If you want to add this value to the string mismatch you are currently creating, then please first remove the value from any other string mismatches using the table below."),
                                  html.P(f"The string mismatch that already contains the value is the following: {mismatch}"),
                                  html.P(f"Do you want to remove the value from this string mismatch? Then it can be added to the string mismatch you are currently creating"),
                                  html.P(f"If you wish to remove the whole string mismatch, instead of just this value {value}, scroll down to the table containing all string mismatches and remove the corresponding mismatch."),
                                  html.P(f"If you do not want to remove the value from the mismatch nor want to remove the whole mismatch, please continue adding new values to your current mismatch and submit it afterwards"),
                                  html.Button("Remove", id='remove-value-mismatch', n_clicks=0, style=button_style),
                                  html.Button("Submit", id='submit-variations', n_clicks=0, style=button_style)]), variations, wrong_val]
        if isinstance(variations, list):
            for val in variations:
                if str(val) not in df[column].to_list():#dksajfl
                    variations = []
            if value not in variations:
                variations.append(value)
        else:
            variations = [value]
        var_text = f"The selected variations in column {column} are: " + ', '.join(
            [f"{val}" for val in variations])
        return [html.Div([
            html.P(var_text),
            html.Button("Submit", id='submit-variations', n_clicks=0, style=button_style),
            html.Button("Remove", id='remove-value-mismatch', n_clicks=0, style={'display':'None'})
        ]), variations, wrong_val]
    else:
        return [html.Div([html.P(""), html.Button("Submit", id='submit-variations', n_clicks=0, style={'display':'None'}), html.Button("Remove", id='remove-value-mismatch', n_clicks=0, style={'display':'None'})
                ]), variations, wrong_val]

@app.callback(Output('remove-value-mismatch-output', 'children'),
              [Input('selected-mismatches', 'data'),
                  Input('remove-value-mismatch', 'n_clicks'),
              Input('value-to-remove-mismatch', 'data')])
def remove_value_mismatch(selected_mms_data, n_clicks, remove_val):
    if 'props' not in selected_mms_data or 'data' not in selected_mms_data['props']:
        print("Error: selected_mms_data is improperly formatted second occurence")
    selected_mms_fp = selected_mms_data['props']['data']
    with open(selected_mms_fp, 'rb') as handle:
        selected_mms = pickle.load(handle)
    if remove_val != None:
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered]
        idx = 0
        new_mismatch = ""
        if 'remove-value-mismatch.n_clicks' in changed_id and n_clicks != None and n_clicks > 0:
            column = remove_val[0]
            value = remove_val[1]
            mismatch = remove_val[2]
            for mismatch_2 in mismatches:
                if mismatch == mismatch_2:
                    variations = ast.literal_eval(mismatch[2])
                    variations.remove(value)
                    new_mismatch = [mismatch[0], mismatch[1], str(variations)]
                    mismatches[idx] = new_mismatch
                idx += 1
            idx = 0
            for selected_mm in selected_mms:
                if mismatch == selected_mm:
                    variations = ast.literal_eval(mismatch[2])
                    variations.remove(value)
                    new_mismatch = [mismatch[0], mismatch[1], str(variations)]
                    selected_mms[idx] = new_mismatch
            with open(selected_mms_fp, 'wb') as handle:
                pickle.dump(selected_mms, handle)
            return html.Div([
                html.P(f"The value {value} was removed from string mismatch {mismatch}. The changed mismatch is now: {new_mismatch}. You can now add the value {value} to your new string mismatch.")
            ])
    else:
        return ""



@app.callback(Output('variations-output', 'children'),
              [Input('submit-variations', 'n_clicks'),
               Input('variations', 'data'),
               State('mismatch-col-dropdown', 'value')])
def submit_variations(n_clicks, variations, column):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered]
    if 'submit-variations.n_clicks' in changed_id and n_clicks > 0:
        if len(variations) != 0:
            return html.Div([
                html.P(f"You have submitted the following variations {variations} for the column {column}"),
                html.P("Please fill in the base form of these variations in the textbox below"),
                dcc.Input(id='base-variations'),
                html.Button("Submit", id='submit-base-var', n_clicks=0, style=button_style)
            ])
        else:
            return html.P("Please first add variations before you try to submit any.")
    else:
        return ""

@app.callback([Output('added-mismatch-output', 'children'),
               Output('added-mm-list', 'data')],
              [State('base-variations', 'value'),
               Input('submit-base-var', 'n_clicks'),
               Input('variations', 'data'),
               State('mismatch-col-dropdown', 'value'),
               Input('selected-mismatches', 'data')])
def output_added_mismatch(base, n_clicks, variations, column, selected_mms_data):
    if 'props' not in selected_mms_data or 'data' not in selected_mms_data['props']:
        print("Error: selected_mms_data is improperly formatted third occurence")
    selected_mms_fp = selected_mms_data['props']['data']
    added_mms_fp = 'cached_files/added_mismatches.csv'
    with open(selected_mms_fp, 'rb') as handle:
        selected_mms = pickle.load(handle)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    idx = 0
    if 'submit-base-var' in changed_id and base != None and n_clicks > 0:
        mismatch = [column, base, str(variations)]
        for mm in mismatches:
            if column == mm[0] and str(variations) == mm[2]:
                mismatches[idx] = mismatch
            idx += 1
        if mismatch not in mismatches:
            mismatches.append(mismatch)
        with open(added_mms_fp, 'wb') as handle:
            pickle.dump(mismatches, handle)
        complete_mm_list = selected_mms + mismatches
        mm_df = pd.DataFrame(complete_mm_list, columns=['Column', "Base", 'Variations'])
        mms_list_filepath = "cached_files/mismatches_list.csv"
        with open(mms_list_filepath, 'wb') as handle:
            pickle.dump(complete_mm_list, handle)
        return html.Div([dcc.Store(id='complete-mm-list', data=mms_list_filepath, storage_type='memory'),
            html.P(f"You have added this mismatch {mismatch}"),
                dash_table.DataTable(
                    id='complete-mm-table',
                    data=mm_df.to_dict('records'),
                    columns=[
                        {'name': 'Column', 'id': 'Column'},
                        {'name': 'Base', 'id': 'Base'},
                        {'name': 'Variations', 'id': 'Variations'}
                    ],
                    editable=False,
                    row_deletable=True
                )
            ]), dcc.Store(id='added-mm-list', data=added_mms_fp, storage_type='memory')
    else:
        with open(added_mms_fp, 'wb') as handle:
            pickle.dump(mismatches, handle)
        complete_mm_list = selected_mms + mismatches
        mm_df = pd.DataFrame(complete_mm_list, columns=['Column', "Base", 'Variations'])
        mms_list_filepath = "cached_files/mismatches_list.csv"
        with open(mms_list_filepath, 'wb') as handle:
            pickle.dump(complete_mm_list, handle)
        return html.Div([dcc.Store(id='complete-mm-list', data=mms_list_filepath, storage_type='memory'),
            html.H4("These are the string mismatches currently detected."),
            dash_table.DataTable(
                id='complete-mm-table',
                data=mm_df.to_dict('records'),
                columns=[
                    {'name': 'Column', 'id': 'Column'},
                    {'name': 'Base', 'id': 'Base'},
                    {'name': 'Variations', 'id': 'Variations'}
                ],
                editable=False,
                row_deletable=True
            )
        ]), dcc.Store(id='added-mm-list', data=added_mms_fp, storage_type='memory')

@app.callback(Output('mm-table-output', 'children'),
              [Input("complete-mm-table", 'data_previous'),
                  State('complete-mm-table', 'data'),
               Input('selected-mismatches', 'data'),
               Input('added-mm-list', 'data')])
def store_all_mls(mm_table_previous, mm_table, selected_mms_data, added_mms_data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'complete-mm-table' in changed_id:
        selected_mms_fp = selected_mms_data['props']['data']
        added_mms_fp = added_mms_data['props']['data']
        with open(selected_mms_fp, 'rb') as handle:
            selected_mms = pickle.load(handle)
        with open(added_mms_fp, 'rb') as handle:
            added_mms = pickle.load(handle)
        complete_mm_list = []
        row_to_delete = []

        if mm_table_previous:
            for row in mm_table_previous:
                if row not in mm_table:
                    row_to_delete = [row['Column'], row['Base'], row['Variations']]
            if row_to_delete in selected_mms:
                selected_mms.remove(row_to_delete)
            elif row_to_delete in mismatches:
                mismatches.remove(row_to_delete)
            else:
                print("This row is neither in selected mms nor in mismatches: ", row_to_delete)
        for row in mm_table:
            complete_mm_list.append([row['Column'], row['Base'], row['Variations']])
        for mm in selected_mms:
            if mm not in complete_mm_list:
                selected_mms.remove(mm)
        for mm in mismatches:
            if mm not in complete_mm_list:
                mismatches.remove(mm)
                if mm in added_mms:
                    added_mms.remove(mm)
        with open(selected_mms_fp, 'wb') as handle:
            pickle.dump(selected_mms, handle)
        with open(added_mms_fp, 'wb') as handle:
            pickle.dump(added_mms, handle)
        mms_list_filepath = "cached_files/mismatches_list.csv"
        with open(mms_list_filepath, 'wb') as handle:
            pickle.dump(complete_mm_list, handle)
        return html.Div(dcc.Store(id='complete-mm-list', data=mms_list_filepath, storage_type='memory'))


### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### ERROR CORRECTION STEP
### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Callback when submit detected errors is clicked which will load the error correction step where all detected errors will be corrected
@app.callback(Output('output-submitted-errors', 'children'),
              [Input('confirm-errors', 'submit_n_clicks'),
               State('stored-df', 'data'),
               State('stored-num-df', 'data'),
               State('stored-feature-types', 'data'),
               State('target-col', 'value'),
               Input("complete-mv-list", 'data'),
               Input('duplicate-rows', 'data'),
               Input('duplicate-cols','data'),
               Input('complete-out-list', 'data'),
               Input('complete-out-row-list', 'data'),
               Input('cryp-col-list', 'data'),
               Input('sv-col-list', 'data'),
               Input('mix-data-dct-fp', 'data'),
               Input('complete-ml-list', 'data'),
               Input('complete-mm-list', 'data')])
def process_error_detections(submit_n_clicks, df_fp, num_df_fp, ft_types_fp, target, mv_list_fp, dup_rows, dup_cols, out_vals_fp, out_rows_fp, cryp_cols_data, sv_cols, mix_data_fp,ml_list_fp, mm_list_fp):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'confirm-errors' in changed_id:
        df = fetch_data(df_fp)
        num_df = fetch_data(num_df_fp)
        with open(ft_types_fp, 'rb') as handle:
            ft_types = pickle.load(handle)
        with open(mv_list_fp, 'rb') as h:
            mv_list = pickle.load(h)
        with open(out_vals_fp, 'rb') as h:
            out_vals = pickle.load(h)
        with open(out_rows_fp, 'rb') as h:
            out_rows = pickle.load(h)
        cryp_cols = cryp_cols_data['props']['data']
        with open(mix_data_fp, 'rb') as h:
            mix_data_dct = pickle.load(h)
        with open(ml_list_fp, 'rb') as h:
            ml_list = pickle.load(h)
        with open(mm_list_fp, 'rb') as h:
            mm_list = pickle.load(h)
        print("All detected data errors: ")
        print("MVs: ", mv_list)
        print("Dup rows: ", dup_rows)
        print("Dup cols: ", dup_cols)
        print('Out vals: ', out_vals)
        print("Out rows: ", out_rows)
        print("Crypc cols: ", cryp_cols)
        print("SV cols ", sv_cols)
        print("Mix data: ", mix_data_dct)
        print("ML list: ", ml_list)
        print("MMs: ", mm_list)
        return recommendations(df, num_df, ft_types, target, mv_list, dup_rows, dup_cols, out_vals, out_rows, cryp_cols, sv_cols, mix_data_dct, ml_list, mm_list)
        # return error_correction(df, num_df, ft_types, target, mv_list, dup_rows, dup_cols, out_vals, out_rows, cryp_cols, sv_cols, mix_data_dct, ml_list, mm_list)
    else:
        return ""

@app.callback(Output('correct-methods-mvs', 'children'),
              [State('stored-df', 'data'),
                State('stored-feature-types', 'data'),
               Input('complete-mv-list', 'data'),
                  Input('dropdown-correct-mvs', 'value')])
def recommend_mvs(df_fp, ft_types_fp, mv_list_fp, method):
    df = fetch_data(df_fp)
    with open(mv_list_fp, 'rb') as f:
        mv_list = pickle.load(f)
    with open(ft_types_fp, 'rb') as f:
        ft_types = pickle.load(f)

    if method == 'imputation':
        model = imputation_model(df, mv_list, ft_types)
        unique_cols = list(set([mv[0] for mv in mv_list]))
        dct_imp = {col:[model] for col in unique_cols}
        df_imp = pd.DataFrame(dct_imp)
        numeric_models = ['KNN', 'RF', 'MLP', 'CART', 'Mean', 'Median', 'Mode', 'Remove', 'Keep']
        numeric_options = [{'label':model, 'value':model} for model in numeric_models]
        categorical_models = ['KNN', 'RF', 'MLP', 'CART', 'Mode', 'Remove', 'Keep']
        categorical_options = [{'label': model, 'value': model} for model in categorical_models]
        dtb = dash_table.DataTable(
            id='mv-correct-method',
            data=df_imp.to_dict('records'),
            columns=[{'name':i, 'id':i, 'presentation':'dropdown'} for i in df_imp.columns],
            dropdown={col: {'options': numeric_options} if ft_types[col] == 'numeric' else {'options': categorical_options} for col in df_imp.columns},
            editable=True
        )
        return html.Div([
            html.P("You have selected imputation as correction technique. Below you can select the imputation method per column."),
            dtb,
            html.P("Please press the button below to submit your imputation methods."),
            html.Button("Submit", id='submit-mv-correct-method', n_clicks=0, style=button_style)
        ])

    elif method == 'remove':
        return html.Div([
            dcc.Store(id='mv-correct-method', data=method, storage_type='memory'),
            html.P("You have chosen the option \'Remove rows\'. This means that all rows containing a detected missing value will be removed from the dataset."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-mv-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'keep':
        return html.Div([
            dcc.Store(id='mv-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Keep all\'. This means that all detected missing values will be retained in the dataset"),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-mv-correct-method', n_clicks=0, style=button_style)
        ])

@app.callback(Output('display-mv-correction', 'children'),
              [Input('submit-mv-correct-method', 'n_clicks'),
               State('stored-df', 'data'),
               State('stored-feature-types', 'data'),
               Input('complete-mv-list', 'data'),
               Input('mv-correct-method', 'data')])
def display_mv_correction(n_clicks, df_fp, ft_types_fp, mv_fp, mv_methods):
    print("Mv methods: ", mv_methods)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-mv-correct-method' in changed_id and n_clicks > 0:
        df = fetch_data(df_fp).astype(str)
        with open(mv_fp, 'rb') as f:
            mv_list = pickle.load(f)
        if mv_methods == 'keep':
            new_mv_list = [[str(mv[0]), str(mv[1]), mv[2]] for mv in mv_list]
            df_copy = copy.deepcopy(df)
            list_of_points = []
            for mv in new_mv_list:
                indices = df_copy[df_copy[mv[0]] == mv[1]].index
                df_copy.loc[indices, mv[0]] = np.nan

            for col, val, freq in new_mv_list:
                indices = df[df[col] == val].index
                for idx in indices:
                    list_of_points.append([idx, col])

            df_dtb = df.copy()
            df_dtb.index = df_dtb.index.set_names(["index"])
            df_dtb = df_dtb.reset_index()
            mv_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict('records'),
                columns=[{'name':i, 'id':i} for i in df_dtb.columns],
                style_data_conditional=create_mv_corr_styles(list_of_points, '#0080FF'),
                page_size=20
            )
            mv_corr_badge = dbc.Badge('Retained value', color='#0080FF', className='me-1')
            updated_df = df.copy()
        elif mv_methods == 'remove':
            indices_to_remove = set()
            for col, mv, _ in mv_list:
                indices = df.index[df[col] == mv].tolist()
                indices_to_remove.update(indices)
            remove_indices = list(indices_to_remove)
            df_dtb = df.drop(remove_indices).reset_index()
            mv_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_dtb.columns],
                style_data_conditional=create_dup_corr_styles(remove_indices, df),
                page_size=20
            )
            mv_corr_badge = dbc.Badge('Removed row', color='tomato', className='me-1')
            updated_df = df.drop(remove_indices)
        else:
            new_mv_list = [[str(mv[0]), str(mv[1]), mv[2]] for mv in mv_list]
            df_copy = copy.deepcopy(df)
            list_of_points = []
            mv_indices = set()
            # Change all dmvs to actual nans
            for mv in new_mv_list:
                indices = df_copy[df_copy[mv[0]] == mv[1]].index
                for idx in indices:
                    mv_indices.add(idx)
                df_copy.loc[indices, mv[0]] = np.nan

            rows_with_na = df_copy.isna().any(axis=1)
            new_df = df[rows_with_na].reset_index()
            for col, val, freq in new_mv_list:
                indices = new_df[new_df[col] == val].index
                for idx in indices:
                    list_of_points.append([idx, col])

            remove_cols = []
            keep_list = []
            impute_list = []
            remove_method = False
            keep_method = False
            imp_method = False
            for col, method in mv_methods[0].items():
                if method != 'Remove':
                    if method == 'Keep':
                        indices = list(df_copy[df_copy[col].isna()].index)
                        for x in indices:
                            keep_list.append([x, col])
                        df_copy[col] = df[col]
                        keep_method = True
                    else:
                        indices = list(df_copy[df_copy[col].isna()].index)
                        for x in indices:
                            impute_list.append([x, col])
                        df_copy[col] = mv_method(df_copy, col, method)
                        imp_method = True
                else:
                    remove_method = True
                    remove_cols.append(col)

            removed_indices = []
            for col in remove_cols:
                indices = list(df_copy[df_copy[col].isna()].index)
                df_copy = df_copy.drop(index=indices)
                for x in indices:
                    removed_indices.append(x)

            df_dtb = df_copy.reset_index()
            mv_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_dtb.columns],
                style_data_conditional=create_combi_mv_styles(impute_list, removed_indices, keep_list, df_dtb),
                page_size=20
            )
            mv_corr_badge = badge_creator_mvs(imp_method, remove_method, keep_method)
            updated_df = df_copy.copy()
        filepath = 'cached_files/df-no-mvs.pkl'
        with open(filepath, 'wb') as h:
            pickle.dump(updated_df, h)
        return html.Div([dcc.Store(id='df-no-mvs', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-dup-rows', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-dup-cols', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-out-vals', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-out-rows', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-cryps', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-svs', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-mds', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-mls', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-mms', data=filepath, storage_type='memory'),
                         html.Hr(),
                        html.H4("Corrections"),
                         html.P("Your correction technique has been applied on the dataset and the updated dataset is shown below."),
                         mv_corr_badge,
                            mv_corr_dtb])


@app.callback(Output('correct-methods-dup-rows', 'children'),
              [State('df-no-mvs', 'data'),
                State('stored-feature-types', 'data'),
               Input('duplicate-rows', 'data'),
                  Input('dropdown-correct-dup-rows', 'value')])
def recommend_dups(df_fp, ft_types_fp, dup_rows, method):
    df = fetch_data(df_fp)
    with open(ft_types_fp, 'rb') as f:
        ft_types = pickle.load(f)

    if method == 'select':
        new_dup = []
        for y in dup_rows:
            in_df = []
            for x in y:
                if x in df.index:
                    in_df.append(x)
            if len(in_df) != 0:
                new_dup.append(in_df)
        dup_rows = new_dup
        dtb = create_dup_group_section(dup_rows, df, 'row')
        return html.Div([
            dcc.Store(id='dup-row-correct-method', data=method, storage_type='memory'),
            html.P("You have selected \'Select keep/remove\' as correction technique. Below you can select which rows you want to keep for every group."),
            dtb,
            html.P("Please press the button below to submit your corrections."),
            html.Button("Submit", id='submit-dup-row-correct-method', n_clicks=0, style=button_style)
        ])

    elif method == 'remove':
        return html.Div([
            dcc.Store(id='dup-row-correct-method', data=method, storage_type='memory'),
            html.P("You have chosen the option \'Remove all\'. This means that all duplicate rows will be removed from the dataset."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-dup-row-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'keep_first':
        return html.Div([
            dcc.Store(id='dup-row-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Keep first, remove rest\'. This means that all duplicate rows will be removed from the dataset except for the first one."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-dup-row-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'keep_all':
        return html.Div([
            dcc.Store(id='dup-row-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Keep all\'. This means that all duplicate rows will be retained in the dataset."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-dup-row-correct-method', n_clicks=0, style=button_style)
        ])

@app.callback(
    Output('output-submit-dup-rows', 'children'),
    [State('df-no-mvs', 'data'),
     Input('submit-dup-row-correct-method', 'n_clicks'),
     State({'type': 'dup-row-retain-checklist', 'index': dash.dependencies.ALL}, 'value'),
     Input('duplicate-rows', 'data'),
     State('dup-row-correct-method', 'data')])
def apply_dup_rows(df_fp, n_clicks, retain_checklist_values, dup_rows, method):
    df = fetch_data(df_fp)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    # To update the dup_rows with already deleted rows
    new_dup = []
    for y in dup_rows:
        in_df = []
        for x in y:
            if x in df.index:
                in_df.append(x)
        new_dup.append(in_df)
    dup_rows = new_dup
    if 'submit-dup-row-correct-method' in changed_id and n_clicks > 0:
        if method == 'remove':
            remove_indices = set([x for y in dup_rows for x in y])
            new_df = df.drop(index=remove_indices)
            df_dtb = new_df.reset_index()
        elif method == 'keep_all':
            remove_indices = []
            new_df = df.drop(index=remove_indices)
            df_dtb = new_df.reset_index()
        elif method == 'keep_first':
            remove_indices = [x for y in dup_rows for x in y if x != min(y)]
            new_df = df.drop(index=remove_indices)
            df_dtb = new_df.reset_index()
        elif method == 'select':
            retained_indices = [index for sublist in retain_checklist_values for index in sublist]
            remove_indices = set([x for y in dup_rows for x in y if x not in retained_indices])
            new_df = df.drop(index=remove_indices)
            df_dtb = new_df.reset_index()
    else:
        return dash.no_update
    filepath = 'cached_files/df-no-dup-rows.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(new_df, f)
    return html.Div([
        dcc.Store(id='df-no-dup-rows', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-dup-cols', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-out-vals', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-out-rows', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-cryps', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-svs', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-mds', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-mls', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-mms', data=filepath, storage_type='memory'),
        html.Hr(),
        html.H4("Corrections"),
        html.P("Your selected correction technique has been applied on your dataset, resulting in the dataset below."),
        dbc.Badge("Removed row", color='tomato', className='me-1'),
        dash_table.DataTable(
            data=df_dtb.to_dict("records"),
            columns=[{'name': i, 'id': i} for i in df_dtb.columns],
            style_data_conditional=create_dup_corr_styles(remove_indices, df),
            page_size=20
        )
    ])

@app.callback(Output('correct-methods-dup-cols', 'children'),
              [State('df-no-dup-rows', 'data'),
               Input('duplicate-cols', 'data'),
                  Input('dropdown-correct-dup-cols', 'value')])
def recommend_dups_cols(df_fp, dup_cols, method):
    df = fetch_data(df_fp)
    if method == 'select':
        dtb = create_dup_group_section(dup_cols, df, 'col')
        return html.Div([
            dcc.Store(id='dup-col-correct-method', data=method, storage_type='memory'),
            html.P("You have selected \'Select keep/remove\' as correction technique. Below you can select which rows you want to keep for every group."),
            dtb,
            html.P("Please press the button below to submit your corrections."),
            html.Button("Submit", id='submit-dup-col-correct-method', n_clicks=0, style=button_style)
        ])

    elif method == 'remove':
        return html.Div([
            dcc.Store(id='dup-col-correct-method', data=method, storage_type='memory'),
            html.P("You have chosen the option \'Remove all\'. This means that all duplicate columns will be removed from the dataset."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-dup-col-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'keep_all':
        return html.Div([
            dcc.Store(id='dup-col-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Keep all\'. This means that all duplicate columns will be retained in your dataset."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-dup-col-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'keep_first':
        return html.Div([
            dcc.Store(id='dup-col-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Keep first, remove rest\'. This means that all duplicate columns will be removed except for the first occuring column."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-dup-col-correct-method', n_clicks=0, style=button_style)
        ])

@app.callback(
    Output('output-submit-dup-cols', 'children'),
    [State('df-no-dup-rows', 'data'),
     Input('submit-dup-col-correct-method', 'n_clicks'),
     State({'type': 'dup-col-retain-checklist', 'index': dash.dependencies.ALL}, 'value'),
     Input('duplicate-cols', 'data'),
     State('dup-col-correct-method', 'data')])
def apply_dup_cols(df_fp, n_clicks, retain_checklist_values, dup_cols, method):
    df = fetch_data(df_fp)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'submit-dup-col-correct-method' in changed_id and n_clicks > 0:
        if method == 'remove':
            remove_cols = set([x for y in dup_cols for x in y])
            new_df = df.drop(columns=remove_cols)
            df_dtb = new_df.reset_index()
        elif method == 'keep_all':
            remove_cols = []
            new_df = df.drop(columns=remove_cols)
            df_dtb = new_df.reset_index()
        elif method == 'keep_first':
            remove_cols = [x for y in dup_cols for x in y if x != y[0]]
            new_df = df.drop(columns=remove_cols)
            df_dtb = new_df.reset_index()
        elif method == 'select':
            retained_cols = [col for sublist in retain_checklist_values for col in sublist]
            remove_cols = set([x for y in dup_cols for x in y if x not in retained_cols])
            new_df = df.drop(columns=remove_cols)
            df_dtb = new_df.reset_index()
    else:
        return dash.no_update
    filepath = 'cached_files/df-no-dup-cols.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(new_df, f)
    return html.Div([
        dcc.Store(id='df-no-dup-cols', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-out-vals', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-out-rows', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-cryps', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-svs', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-mds', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-mls', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-mms', data=filepath, storage_type='memory'),
        html.Hr(),
        html.H4("Corrections"),
        html.P("Your selected correction technique has been applied on your dataset, resulting in the dataset below."),
        dbc.Badge("Removed column", color='tomato', className='me-1'),
        dash_table.DataTable(
            data=df_dtb.to_dict("records"),
            columns=[{'name': i, 'id': i} for i in df_dtb.columns],
            style_data_conditional=create_dup_col_corr_styles(df, remove_cols),
            page_size=20
        )
    ])

@app.callback(Output('correct-methods-outs', 'children'),
              [State('df-no-dup-cols', 'data'),
                State('stored-feature-types', 'data'),
               Input('complete-out-list', 'data'),
                  Input('dropdown-correct-outs', 'value')])
def recommend_outs(df_fp, ft_types_fp, out_list_fp, method):
    df = fetch_data(df_fp)
    with open(out_list_fp, 'rb') as f:
        out_list = pickle.load(f)
    with open(ft_types_fp, 'rb') as f:
        ft_types = pickle.load(f)
    out_list = [x for x in out_list if x[0] in df.columns]
    if method == 'imputation':
        model = imputation_model(df, out_list, ft_types, True)
        unique_cols = list(set([out[0] for out in out_list]))
        dct_imp = {col:[model] for col in unique_cols}
        df_imp = pd.DataFrame(dct_imp)
        numeric_models = ['KNN', 'RF', 'MLP', 'CART', 'Mean', 'Median', 'Mode', 'Remove', 'Keep']
        numeric_options = [{'label':model, 'value':model} for model in numeric_models]
        categorical_models = ['KNN', 'RF', 'MLP', 'CART', 'Mode', 'Remove', 'Keep']
        categorical_options = [{'label': model, 'value': model} for model in categorical_models]
        dtb = dash_table.DataTable(
            id='out-correct-method',
            data=df_imp.to_dict('records'),
            columns=[{'name':i, 'id':i, 'presentation':'dropdown'} for i in df_imp.columns],
            dropdown={col: {'options': numeric_options} if ft_types[col] == 'numeric' else {'options': categorical_options} for col in df_imp.columns},
            editable=True
        )
        return html.Div([
            html.P("You have selected \'Imputation\' as correction technique. Below you can select the imputation method per column."),
            dtb,
            html.P("Please press the button below to submit your imputation methods."),
            html.Button("Submit", id='submit-out-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'type':
        model = imputation_model(df, out_list, ft_types, True)
        types = ['Close outliers', 'Far outliers']
        dct_imp = {typ: [model] for typ in types}
        df_imp = pd.DataFrame(dct_imp)
        numeric_models = ['KNN', 'RF', 'MLP', 'CART', 'Mean', 'Median', 'Mode', 'Remove', 'Keep']
        numeric_options = [{'label': model, 'value': model} for model in numeric_models]
        dtb = dash_table.DataTable(
            id='out-correct-method',
            data=df_imp.to_dict('records'),
            columns=[{'name': i, 'id': i, 'presentation': 'dropdown'} for i in df_imp.columns],
            dropdown={
                col: {'options': numeric_options} for col in df_imp.columns},
            editable=True
        )
        return html.Div([
            html.P(
                "You have selected \'Type specific\' as correction technique. Below you can select the correction technique per outlier type."),
            dtb,
            html.P("Please press the button below to submit your correction techniques."),
            html.Button("Submit", id='submit-out-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'remove':
        return html.Div([
            dcc.Store(id='out-correct-method', data=method, storage_type='memory'),
            html.P("You have chosen the option \'Remove rows\'. This means that all rows containing a detected missing value will be removed from the dataset."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-out-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'keep':
        return html.Div([
            dcc.Store(id='out-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Keep all\'. This means that all detected missing values will be retained in the dataset"),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-out-correct-method', n_clicks=0, style=button_style)
        ])

@app.callback(Output('display-out-correction', 'children'),
              [Input('submit-out-correct-method', 'n_clicks'),
               State('df-no-dup-cols', 'data'),
               State('stored-feature-types', 'data'),
               Input('complete-out-list', 'data'),
               Input('out-correct-method', 'data')])
def display_out_correction(n_clicks, df_fp, ft_types_fp, out_fp, out_methods):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-out-correct-method' in changed_id and n_clicks > 0:
        df = fetch_data(df_fp).astype(str)
        with open(out_fp, 'rb') as f:
            out_list = pickle.load(f)
        with open(ft_types_fp, 'rb') as f:
            ft_types = pickle.load(f)
        out_list = [x for x in out_list if x[0] in df.columns]
        if out_methods == 'keep':
            new_out_list = [[str(out[0]), str(out[2]), out[1]] for out in out_list]
            df_copy = copy.deepcopy(df)
            list_of_points = []
            for out in new_out_list:
                indices = df_copy[df_copy[out[0]] == out[1]].index
                df_copy.loc[indices, out[0]] = np.nan

            for col, val, typ in new_out_list:
                indices = df[df[col] == val].index
                for idx in indices:
                    list_of_points.append([idx, col])

            df_dtb = df.copy()
            df_dtb.index = df_dtb.index.set_names(["index"])
            df_dtb = df_dtb.reset_index()
            out_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict('records'),
                columns=[{'name':i, 'id':i} for i in df_dtb.columns],
                style_data_conditional=create_mv_corr_styles(list_of_points, '#0080FF'),
                page_size=20
            )
            out_corr_badge = dbc.Badge('Retained value', color='#0080FF', className='me-1')
            updated_df = df.copy()
        elif out_methods == 'remove':
            indices_to_remove = set()
            for col, typ, out in out_list:
                indices = df.index[df[col] == str(out)].tolist()
                indices_to_remove.update(indices)
            remove_indices = list(indices_to_remove)
            df_dtb = df.drop(remove_indices).reset_index()
            out_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_dtb.columns],
                style_data_conditional=create_dup_corr_styles(remove_indices, df),
                page_size=20
            )
            out_corr_badge = dbc.Badge('Removed row', color='tomato', className='me-1')
            updated_df = df.drop(remove_indices)
        elif type(out_methods) == list:
            if list(out_methods[0].keys()) == ['Close outliers', 'Far outliers']:
                close_method = out_methods[0]['Close outliers']
                far_method = out_methods[0]['Far outliers']
                close_list = [[str(out[0]), str(out[2]), out[1]] for out in out_list if out[1] == 'close']
                far_list = [[str(out[0]), str(out[2]), out[1]] for out in out_list if out[1] == 'far']
                df_copy = copy.deepcopy(df)
                keep_list = []
                remove_list = []
                impute_list = []
                imp_method = False
                remove_method = False
                keep_method = False

                if close_method == 'Keep':
                    keep_method = True
                    for col, val, typ in close_list:
                        indices = df_copy[df_copy[col] == val].index
                        for x in indices:
                            keep_list.append([x,col])
                elif close_method == 'Remove':
                    remove_method = True
                    for col, val, typ in close_list:
                        indices = df_copy[df_copy[col] == val].index
                        df_copy = df_copy.drop(index=indices)
                        for x in indices:
                            remove_list.append(x)
                else:
                    imp_method = True
                    close_cols = set()
                    for col, val, typ in close_list:
                        indices = df_copy[df_copy[col] == val].index
                        df_copy.loc[indices, col] = np.nan
                        close_cols.add(col)
                        for x in indices:
                            impute_list.append([x,col])
                    for col in close_cols:

                        df_copy[col] = mv_method(df_copy, col, close_method)
                if far_method == 'Keep':
                    keep_method = True
                    for col, val, typ in far_list:
                        indices = df_copy[df_copy[col] == val].index
                        for x in indices:
                            keep_list.append([x, col])
                elif far_method == 'Remove':
                    remove_method = True
                    for col, val, typ in far_list:
                        indices = df_copy[df_copy[col] == val].index
                        df_copy = df_copy.drop(index=indices)
                        for x in indices:
                            remove_list.append(x)
                else:
                    imp_method = True
                    far_cols = set()
                    for col, val, typ in far_list:
                        indices = df_copy[df_copy[col] == val].index
                        df_copy.loc[indices, col] = np.nan
                        far_cols.add(col)
                        for x in indices:
                            impute_list.append([x,col])
                    for col in far_cols:
                        df_copy[col] = mv_method(df_copy, col, far_method)

                df_dtb = df_copy.reset_index()
                out_corr_dtb = dash_table.DataTable(
                    data=df_dtb.to_dict("records"),
                    columns=[{"name": i, "id": i} for i in df_dtb.columns],
                    style_data_conditional=create_combi_mv_styles(impute_list, remove_list, keep_list, df_dtb),
                    page_size=20
                )
                out_corr_badge = badge_creator_mvs(imp_method, remove_method, keep_method)
                updated_df = df_copy.copy()
            else:
                new_out_list = [[str(out[0]), str(out[2]), out[1]] for out in out_list]
                df_copy = copy.deepcopy(df)
                out_indices = set()
                # Change all dmvs to actual nans
                for out in new_out_list:
                    indices = df_copy[df_copy[out[0]] == out[1]].index
                    for idx in indices:
                        out_indices.add(idx)
                    df_copy.loc[indices, out[0]] = np.nan

                remove_cols = []
                keep_list = []
                impute_list = []
                remove_method = False
                keep_method = False
                imp_method = False
                for col, method in out_methods[0].items():
                    if method != 'Remove':
                        if method == 'Keep':
                            indices = list(df_copy[df_copy[col].isna()].index)
                            for x in indices:
                                keep_list.append([x, col])
                            df_copy[col] = df[col]
                            keep_method = True
                        else:
                            indices = list(df_copy[df_copy[col].isna()].index)
                            for x in indices:
                                impute_list.append([x, col])
                            if ft_types[col] == 'numeric':
                                df_copy[col] = df_copy[col].astype(float)
                            df_copy[col] = mv_method(df_copy, col, method)
                            imp_method = True
                    else:
                        remove_method = True
                        remove_cols.append(col)

                removed_indices = []
                for col in remove_cols:
                    indices = list(df_copy[df_copy[col].isna()].index)
                    df_copy = df_copy.drop(index=indices)
                    for x in indices:
                        removed_indices.append(x)

                df_dtb = df_copy.reset_index()
                out_corr_dtb = dash_table.DataTable(
                    data=df_dtb.to_dict("records"),
                    columns=[{"name": i, "id": i} for i in df_dtb.columns],
                    style_data_conditional=create_combi_mv_styles(impute_list, removed_indices, keep_list, df_dtb),
                    page_size=20
                )
                out_corr_badge = badge_creator_mvs(imp_method, remove_method, keep_method)
                updated_df = df_copy.copy()
        filepath = 'cached_files/df-no-out-vals.pkl'
        with open(filepath, 'wb') as h:
            pickle.dump(updated_df, h)
        return html.Div([dcc.Store(id='df-no-out-vals', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-out-rows', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-cryps', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-svs', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-mds', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-mls', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-mms', data=filepath, storage_type='memory'),
                         html.Hr(),
                        html.H4("Corrections"),
                         html.P("Your correction technique has been applied on the dataset and the updated dataset is shown below."),
                         out_corr_badge,
                            out_corr_dtb])

@app.callback(Output('correct-methods-out-rows', 'children'),
              [State('df-no-out-vals', 'data'),
                State('stored-feature-types', 'data'),
               Input('complete-out-row-list', 'data'),
                  Input('dropdown-correct-out-rows', 'value')])
def recommend_out_rows(df_fp, ft_types_fp, out_rows_fp, method):
    df = fetch_data(df_fp)
    with open(out_rows_fp, 'rb') as h:
        out_rows = pickle.load(h)
    out_rows = out_rows[::-1]
    new_out = []
    for x in out_rows:
        in_df = []
        if x[0] in df.index:
            new_out.append(x)
    out_rows = new_out
    if method == 'select':
        dtb = create_out_row_section(out_rows, df)
        return html.Div([
            dcc.Store(id='out-row-correct-method', data=method, storage_type='memory'),
            html.P("You have selected \'Select keep/remove\' as correction technique. Below you can select which rows you want to keep."),
            dtb,
            html.P("Please press the button below to submit your corrections."),
            html.Button("Submit", id='submit-out-row-correct-method', n_clicks=0, style=button_style)
        ])

    elif method == 'remove':
        return html.Div([
            dcc.Store(id='out-row-correct-method', data=method, storage_type='memory'),
            html.P("You have chosen the option \'Remove all\'. This means that all outlier rows will be removed from the dataset."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-out-row-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'keep':
        return html.Div([
            dcc.Store(id='out-row-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Keep all\'. This means that all outlier rows will be retained in your dataset."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-out-row-correct-method', n_clicks=0, style=button_style)
        ])

@app.callback(
    Output('output-submit-out-rows', 'children'),
    [State('df-no-out-vals', 'data'),
     Input('submit-out-row-correct-method', 'n_clicks'),
     State({'type': 'out-row-retain-checklist', 'index': dash.dependencies.ALL}, 'value'),
     Input('complete-out-row-list', 'data'),
     Input('out-row-correct-method', 'data')])
def apply_out_rows(df_fp, n_clicks, retain_checklist_values, out_rows_fp, method):
    df = fetch_data(df_fp)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    with open(out_rows_fp, 'rb') as f:
        out_rows = pickle.load(f)
    new_out = []
    for x in out_rows:
        in_df = []
        if x[0] in df.index:
            new_out.append(x)
    out_rows = new_out
    if 'submit-out-row-correct-method' in changed_id and n_clicks > 0:
        if method == 'remove':
            remove_indices = [x[0] for x in out_rows]
            new_df = df.drop(index=remove_indices)
            df_dtb = new_df.reset_index()
        elif method == 'keep':
            remove_indices = []
            new_df = df.drop(index=remove_indices)
            df_dtb = new_df.reset_index()
        elif method == 'select':
            count = 0
            retained_indices = []
            for val in retain_checklist_values:
                if len(val) != 0:
                    retained_indices.append(out_rows[count][0])
                count += 1
            remove_indices = [x[0] for x in out_rows if x[0] not in retained_indices]
            new_df = df.drop(index=remove_indices)
            df_dtb = new_df.reset_index()
    else:
        return dash.no_update
    filepath = 'cached_files/df-no-out-rows.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(new_df, f)
    return html.Div([
        dcc.Store(id='df-no-out-rows', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-cryps', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-svs', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-mds', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-mls', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-mms', data=filepath, storage_type='memory'),
        html.Hr(),
        html.H4("Corrections"),
        html.P("Your selected correction technique has been applied on your dataset, resulting in the dataset below."),
        dbc.Badge("Removed row", color='tomato', className='me-1'),
        dash_table.DataTable(
            data=df_dtb.to_dict("records"),
            columns=[{'name': i, 'id': i} for i in df_dtb.columns],
            style_data_conditional=create_dup_corr_styles(remove_indices, df),
            page_size=20
        )
    ])

@app.callback(Output('correct-methods-cryp', 'children'),
              [State('df-no-out-rows', 'data'),
                State('stored-feature-types', 'data'),
               Input('cryp-correct-table', 'data'),
                  Input('dropdown-correct-cryp', 'value')])
def recommend_cryp(df_fp, ft_types_fp, cryp_table, method):
    df = fetch_data(df_fp)
    if method == 'keep':
        return html.Div([
            dcc.Store(id='cryp-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Keep original names\'. This means that all original column names will be retained in your dataset."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-cryp-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'suggested':
        return html.Div([
            dcc.Store(id='cryp-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Change to suggested names\'. This means that all cryptic column names will be changed to the suggested names."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-cryp-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'select':
        dtb = create_cryp_select_section(cryp_table)
        return html.Div([
            dcc.Store(id='cryp-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have selected \'Select keep/change\' as correction technique. Below you can select which version of each column name you want to use."),
            dtb,
            html.P("Please press the button below to submit your corrections."),
            html.Button("Submit", id='submit-cryp-correct-method', n_clicks=0, style=button_style)
        ])


@app.callback(
    Output('output-submit-cryp', 'children'),
    [State('df-no-out-rows', 'data'),
     Input('submit-cryp-correct-method', 'n_clicks'),
     State({'type': 'cryp-retain-checklist', 'index': dash.dependencies.ALL}, 'value'),
     Input('cryp-correct-table', 'data'),
     Input('cryp-correct-method', 'data'),
     Input('sv-col-list', 'data'),
     Input('mix-data-dct-fp', 'data'),
     Input('complete-ml-list', 'data'),
     Input('complete-mm-list', 'data'),
     State('stored-feature-types', 'data'),
     State('target-col', 'value'),
     ])
def apply_cryp(df_fp, n_clicks, retain_checklist_values, cryp_table, method, sv_cols, md_fp, ml_fp, mm_fp, ft_types_fp, target):
    df = fetch_data(df_fp)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-cryp-correct-method' in changed_id and n_clicks > 0:
        rename_dct = dict()
        if method == 'keep':
            new_df = df.copy()
            df_dtb = new_df.reset_index()
        elif method == 'suggested':
            rename_dct = {list(cryp_dct.values())[0] : list(cryp_dct.values())[1] for cryp_dct in cryp_table}
            new_df = df.rename(columns=rename_dct)
            df_dtb = new_df.reset_index()
        elif method == 'select':
            cryp_cols = []
            corr_cols = []
            rename_dct = dict()
            for cryp_dct in cryp_table:
                for col_id, name in cryp_dct.items():
                    if col_id == 'Cryptic name':
                        cryp_cols.append(name)
                    else:
                        corr_cols.append(name)
            for idx in range(len(retain_checklist_values)):
                if retain_checklist_values[idx] == corr_cols[idx]:
                    rename_dct[cryp_cols[idx]] = corr_cols[idx]
            new_df = df.rename(columns=rename_dct)
            df_dtb = new_df.reset_index()

        filepath = 'cached_files/df-no-cryps.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(new_df, f)
        return html.Div([
            dcc.Store(id='renamed-cols', data=rename_dct, storage_type='memory'),
            dcc.Store(id='df-no-cryps', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-svs', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-mds', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-mls', data=filepath, storage_type='memory'),
            dcc.Store(id='df-no-mms', data=filepath, storage_type='memory'),
            html.Hr(),
            html.H4("Corrections"),
            html.P("Your selected correction technique has been applied on your dataset, resulting in the dataset below."),
            dbc.Badge("Corrected column name", color='#0080FF', className='me-1'),
            dash_table.DataTable(
                data=df_dtb.to_dict("records"),
                columns=[{'name': i, 'id': i} for i in df_dtb.columns],
                style_header_conditional=create_cryp_header(rename_dct),
                page_size=20
            )
        ])

@app.callback(Output('correct-methods-svs', 'children'),
              [State('df-no-cryps', 'data'),
               Input('sv-col-list', 'data'),
                  Input('dropdown-correct-svs', 'value'),
               Input('renamed-cols', 'data')])
def recommend_svs(df_fp, sv_cols, method, rename_dct):
    df = fetch_data(df_fp)
    if rename_dct != None:
        sv_cols = [rename_dct[col] if col in rename_dct.keys() else col for col in sv_cols]
        sv_cols = [col for col in sv_cols if col in df.columns]
    if method == 'select':
        dtb = create_sv_select_section(sv_cols, df)
        return html.Div([
            dcc.Store(id='svs-correct-method', data=method, storage_type='memory'),
            html.P("You have selected \'Select keep/remove\' as correction technique. Below you can select which single value columns you want to keep."),
            dtb,
            html.P("Please press the button below to submit your corrections."),
            html.Button("Submit", id='submit-svs-correct-method', n_clicks=0, style=button_style)
        ])

    elif method == 'remove':
        return html.Div([
            dcc.Store(id='svs-correct-method', data=method, storage_type='memory'),
            html.P("You have chosen the option \'Remove all\'. This means that all single value columns will be removed from the dataset."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-svs-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'keep':
        return html.Div([
            dcc.Store(id='svs-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Keep all\'. This means that all single value columns will be retained in your dataset."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-svs-correct-method', n_clicks=0, style=button_style)
        ])

@app.callback(
    Output('output-submit-svs', 'children'),
    [State('df-no-cryps', 'data'),
     Input('submit-svs-correct-method', 'n_clicks'),
     State({'type': 'sv-retain-checklist', 'index': dash.dependencies.ALL}, 'value'),
     Input('sv-col-list', 'data'),
     State('svs-correct-method', 'data'),
     Input("renamed-cols", 'data')])
def apply_svs(df_fp, n_clicks, retain_checklist_values, sv_cols, method, rename_dct):
    df = fetch_data(df_fp)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-svs-correct-method' in changed_id and n_clicks > 0:
        if rename_dct != None:
            sv_cols = [rename_dct[col] if col in rename_dct.keys() else col for col in sv_cols]
            sv_cols = [col for col in sv_cols if col in df.columns]
        if method == 'remove':
            remove_cols = sv_cols
            new_df = df.drop(columns=remove_cols)
            df_dtb = new_df.reset_index()
        elif method == 'keep':
            remove_cols = []
            new_df = df.copy()
            df_dtb = new_df.reset_index()
        elif method == 'select':
            count = 0
            retained_cols = []
            for val in retain_checklist_values:
                if len(val) != 0:
                    retained_cols.append(sv_cols[count])
                count += 1
            remove_cols = [x for x in sv_cols if x not in retained_cols]
            new_df = df.drop(columns=remove_cols)
            df_dtb = new_df.reset_index()
    else:
        return dash.no_update
    filepath = 'cached_files/df-no-svs.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(new_df, f)
    return html.Div([
        dcc.Store(id='df-no-svs', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-mds', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-mls', data=filepath, storage_type='memory'),
        dcc.Store(id='df-no-mms', data=filepath, storage_type='memory'),
        html.Hr(),
        html.H4("Corrections"),
        html.P("Your selected correction technique has been applied on your dataset, resulting in the dataset below."),
        dbc.Badge("Removed column", color='tomato', className='me-1'),
        dash_table.DataTable(
            data=df_dtb.to_dict("records"),
            columns=[{'name': i, 'id': i} for i in df_dtb.columns],
            style_data_conditional=create_dup_col_corr_styles(df, remove_cols),
            page_size=20
        )
    ])

@app.callback(Output('md-dropdown-section', 'children'),
              [Input('df-no-svs', 'data'),
                Input('mix-data-dct-fp', 'data'),
               Input('dropdown-mds-major-minor', 'value'),
               Input('renamed-cols', 'data')],
              prevent_initial_call=True)
def md_correct_what(df_fp, md_fp, method, rename_dct):
    df = fetch_data(df_fp)
    with open(md_fp, 'rb') as f:
        md_dct = pickle.load(f)
    if rename_dct != None:
        new_md_dct = dict()
        for old_key, value in md_dct.items():
            if old_key in rename_dct:
                new_key = rename_dct[old_key]
            else:
                new_key = old_key
            new_md_dct[new_key] = value
        md_dct = new_md_dct

    dtb = create_md_select_type(md_dct, method)
    if method == 'minor' or method == 'major':
        return html.Div([
            html.Div(dtb, style={'display':'none'}),
            html.P(f"You have chosen to correct the {method}ity types in all columns."),
            html.P(
                "In the dropdown menu below you can select the correction technique. If you select the option \'Imputation\', a section pops up to select the imputation technique."),
            html.Div(dcc.Dropdown(id='dropdown-correct-mds',
                         options=[{'label': 'Imputation', 'value': 'imputation'},
                                  {'label': 'Remove rows', 'value': 'remove'},
                                  {'label': 'Keep all',
                                   'value': 'keep'}]), style={'display': 'inline-block', 'width': '30%'}),
        ])
    elif method == 'column':
        return html.Div([
            html.P("You have chosen for the option to choose per specific column the type that will be corrected."),
            html.P("Select in the table below per column the data type that you want to correct. The percentage is shown of how often that type occurs in the column."),
            dtb,
            html.P(
                "In the dropdown menu below you can select the correction technique. If you select the option \'Imputation\', a section pops up to select the imputation technique."),
            html.Div(dcc.Dropdown(id='dropdown-correct-mds',
                         options=[{'label': 'Imputation', 'value': 'imputation'},
                                  {'label': 'Remove rows', 'value': 'remove'},
                                  {'label': 'Keep all',
                                   'value': 'keep'}]), style={'display': 'inline-block', 'width': '30%'}),
        ])


@app.callback(Output('correct-methods-mds', 'children'),
              [State('df-no-svs', 'data'),
                State('stored-feature-types', 'data'),
               Input('mix-data-dct-fp', 'data'),
                  Input('dropdown-correct-mds', 'value'),
               Input('dropdown-mds-major-minor', 'value'),
               Input('md-col-specific', 'data'),
               Input('renamed-cols', 'data')])
def recommend_mds(df_fp, ft_types_fp, md_fp, local_method, global_method, col_specific_data, rename_dct):
    df = fetch_data(df_fp)
    with open(md_fp, 'rb') as f:
        md_dct = pickle.load(f)
    if rename_dct != None:
        new_md_dct = dict()
        for old_key, value in md_dct.items():
            if old_key in rename_dct:
                new_key = rename_dct[old_key]
            else:
                new_key = old_key
            new_md_dct[new_key] = value
        md_dct = new_md_dct

    with open(ft_types_fp, 'rb') as f:
        ft_types = pickle.load(f)
    md_list = []
    if col_specific_data == None:
        model = imputation_model(df, md_dct, ft_types, False, True, global_method, False)
    else:
        model = imputation_model(df, md_dct, ft_types, False, True, global_method, col_specific_data[0])

    if global_method == 'minor':
        md_list = convert_mix_to_nan(df, md_dct, global_method, False)
        md_list = [x for x in md_list if x[0] in df.columns]
        return html.Div([dcc.Store(id='complete-md-list', data=md_list, storage_type='memory'),generate_md_section(df, md_list, local_method, model, global_method, button_style)])
    elif global_method == 'major':
        md_list = convert_mix_to_nan(df, md_dct, global_method, False)
        md_list = [x for x in md_list if x[0] in df.columns]
        return html.Div([dcc.Store(id='complete-md-list', data=md_list, storage_type='memory'),generate_md_section(df, md_list, local_method, model, global_method, button_style)])
    elif global_method == 'column':
        if col_specific_data != None:
            md_list = convert_mix_to_nan(df, md_dct, global_method, col_specific_data[0])
            md_list = [x for x in md_list if x[0] in df.columns]
            return html.Div([dcc.Store(id='complete-md-list', data=md_list, storage_type='memory'),
                generate_md_section(df, md_list, local_method, model, global_method, button_style)])
    else:
        return html.Div([dcc.Store(id='complete-md-list', data=md_list, storage_type='memory'),
                generate_md_section(df, md_list, local_method, model, global_method, button_style)])

@app.callback(Output('display-mds-correction', 'children'),
              [Input('submit-mds-correct-method', 'n_clicks'),
               State('df-no-svs', 'data'),
               State('stored-feature-types', 'data'),
               Input('complete-md-list', 'data'),
               Input('mds-correct-method', 'data'),
               Input('md-col-specific', 'data')])
def display_md_correction(n_clicks, df_fp, ft_types_fp, md_list, method, col_specific_data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-mds-correct-method' in changed_id and n_clicks > 0:
        df = fetch_data(df_fp).astype(str)
        if method == 'keep':
            new_md_list = [[str(md[0]), str(md[1]), md[2]] for md in md_list]
            df_copy = copy.deepcopy(df)
            list_of_points = []
            for md in new_md_list:
                indices = df_copy[df_copy[md[0]] == md[1]].index
                df_copy.loc[indices, md[0]] = np.nan

            for col, val, freq in new_md_list:
                indices = df[df[col] == val].index
                for idx in indices:
                    list_of_points.append([idx, col])

            df_dtb = df.copy()
            df_dtb.index = df_dtb.index.set_names(["index"])
            df_dtb = df_dtb.reset_index()
            md_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict('records'),
                columns=[{'name':i, 'id':i} for i in df_dtb.columns],
                style_data_conditional=create_mv_corr_styles(list_of_points, '#0080FF'),
                page_size=20
            )
            md_corr_badge = dbc.Badge('Retained value', color='#0080FF', className='me-1')
            updated_df = df.copy()
        elif method == 'remove':
            indices_to_remove = set()
            for col, val, freq in md_list:
                indices = df.index[df[col] == str(val)].tolist()
                indices_to_remove.update(indices)
            remove_indices = list(indices_to_remove)
            df_dtb = df.drop(remove_indices).reset_index()
            md_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_dtb.columns],
                style_data_conditional=create_dup_corr_styles(remove_indices, df),
                page_size=20
            )
            md_corr_badge = dbc.Badge('Removed row', color='tomato', className='me-1')
            updated_df = df.drop(remove_indices)
        elif type(method) == list:
            df_copy = copy.deepcopy(df)
            new_md_list = [[str(md[0]), str(md[1]), md[2]] for md in md_list]
            md_indices = set()
            for md in new_md_list:
                indices = df_copy[df_copy[md[0]] == md[1]].index
                for idx in indices:
                    md_indices.add(idx)
                df_copy.loc[indices, md[0]] = np.nan

            remove_cols = []
            keep_list = []
            impute_list = []
            remove_method = False
            keep_method = False
            imp_method = False
            for col, method in method[0].items():
                if method != 'Remove':
                    if method == 'Keep':
                        indices = list(df_copy[df_copy[col].isna()].index)
                        for x in indices:
                            keep_list.append([x, col])
                        df_copy[col] = df[col]
                        keep_method = True
                    else:
                        indices = list(df_copy[df_copy[col].isna()].index)
                        for x in indices:
                            impute_list.append([x, col])
                        if col_specific_data[0][col] == 'strings': # if strings is data type to correct, then convert col to float
                            df_copy[col] = df_copy[col].astype(float)
                        df_copy[col] = mv_method(df_copy, col, method)
                        imp_method = True
                else:
                    remove_method = True
                    remove_cols.append(col)

            removed_indices = []
            for col in remove_cols:
                indices = list(df_copy[df_copy[col].isna()].index)
                df_copy = df_copy.drop(index=indices)
                for x in indices:
                    removed_indices.append(x)

            df_dtb = df_copy.reset_index()
            md_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_dtb.columns],
                style_data_conditional=create_combi_mv_styles(impute_list, removed_indices, keep_list, df_dtb),
                page_size=20
            )
            md_corr_badge = badge_creator_mvs(imp_method, remove_method, keep_method)
            updated_df = df_copy.copy()
        filepath = 'cached_files/df-no-mds.pkl'
        with open(filepath, 'wb') as h:
            pickle.dump(updated_df, h)
        return html.Div([dcc.Store(id='df-no-mds', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-mls', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-mms', data=filepath, storage_type='memory'),
                         html.Hr(),
                        html.H4("Corrections"),
                         html.P("Your correction technique has been applied on the dataset and the updated dataset is shown below."),
                         md_corr_badge,
                            md_corr_dtb])

@app.callback(Output('correct-methods-mls', 'children'),
              [Input('dropdown-correct-mls', 'value')])
def recommend_mls(method):
    if method == 'convert':
        return html.Div([
            dcc.Store(id='ml-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Convert to correct label\'. This means that all detected incorrect labels will be converted to the correct label."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-ml-correct-method', n_clicks=0, style=button_style)
        ])

    elif method == 'remove':
        return html.Div([
            dcc.Store(id='ml-correct-method', data=method, storage_type='memory'),
            html.P("You have chosen the option \'Remove rows\'. This means that all rows containing an incorrect label will be removed from the dataset."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-ml-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'keep':
        return html.Div([
            dcc.Store(id='ml-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Keep all\'. This means that all detected incorrect labels will be retained in the dataset"),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-ml-correct-method', n_clicks=0, style=button_style)
        ])

@app.callback(Output('display-ml-correction', 'children'),
              [Input('submit-ml-correct-method', 'n_clicks'),
               State('df-no-mds', 'data'),
               State('stored-feature-types', 'data'),
               Input('complete-ml-list', 'data'),
               Input('ml-correct-method', 'data'),
               Input('target-col', 'value'),
               Input('renamed-cols', 'data')])
def apply_mls(n_clicks, df_fp, ft_types_fp, ml_list_fp, method, target, rename_dct):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-ml-correct-method' in changed_id and n_clicks > 0:
        with open(ml_list_fp, 'rb') as f:
            ml_list = pickle.load(f)
        if rename_dct != None:
            if target in rename_dct:
                target = rename_dct[target]
        df = fetch_data(df_fp).astype(str)
        if method == 'keep':
            list_of_points = [[ml[0], target] for ml in ml_list]

            df_dtb = df.copy()
            df_dtb.index = df_dtb.index.set_names(["index"])
            df_dtb = df_dtb.reset_index()
            ml_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_dtb.columns],
                style_data_conditional=create_mv_corr_styles(list_of_points, '#0080FF'),
                page_size=20
            )
            ml_corr_badge = dbc.Badge('Retained value', color='#0080FF', className='me-1')
            updated_df = df.copy()
        elif method == 'remove':
            remove_indices = [ml[0] for ml in ml_list if ml[0] in df.index]
            new_df = df.drop(index=remove_indices)
            df_dtb = new_df.copy()
            df_dtb.index = df_dtb.index.set_names(["index"])
            df_dtb = df_dtb.reset_index()
            ml_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_dtb.columns],
                style_data_conditional=create_dup_corr_styles(remove_indices, df),
                page_size=20
            )
            ml_corr_badge = dbc.Badge('Removed row', color='tomato', className='me-1')
            updated_df = new_df.copy()
        elif method == 'convert':
            df_copy = copy.deepcopy(df)
            ml_list = [ml for ml in ml_list if ml[0] in df.index]
            list_of_points = []
            for ml in ml_list:
                df_copy.loc[ml[0], target] = ml[2]
                list_of_points.append([ml[0], target])
            df_dtb = df_copy.copy()
            df_dtb.index = df_dtb.index.set_names(["index"])
            df_dtb = df_dtb.reset_index()
            ml_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_dtb.columns],
                style_data_conditional=create_mv_corr_styles(list_of_points),
                page_size=20
            )
            ml_corr_badge = dbc.Badge('Corrected value', color='#50C878', className='me-1')
            updated_df = df_copy.copy()
        filepath = 'cached_files/df-no-mls.pkl'
        with open(filepath, 'wb') as h:
            pickle.dump(updated_df, h)
        return html.Div([dcc.Store(id='df-no-mls', data=filepath, storage_type='memory'),
                         dcc.Store(id='df-no-mms', data=filepath, storage_type='memory'),
                         html.Hr(),
                         html.H4("Corrections"),
                         html.P(
                             "Your correction technique has been applied on the dataset and the updated dataset is shown below."),
                         ml_corr_badge,
                         ml_corr_dtb])

@app.callback(Output('correct-methods-mms', 'children'),
              [Input('df-no-mls', 'data'),
                  Input('dropdown-correct-mms', 'value'),
               Input('complete-mm-list', 'data'),
               Input('renamed-cols', 'data')])
def recommend_mls(df_fp, method, mm_list_fp, rename_dct):
    df = fetch_data(df_fp)
    with open(mm_list_fp, 'rb') as f:
        mm_list = pickle.load(f)
    if rename_dct != None:
        new_mm_list = []
        for mm in mm_list:
            if mm[0] in rename_dct:
                new_col = rename_dct[mm[0]]
            else:
                new_col = mm[0]
            new_mm_list.append([new_col, mm[1], mm[2]])
        mm_list = new_mm_list
    if method == 'convert_base':
        output_str = "The base forms for all string mismatches are: " + ', '.join([f"{mm[1]} for variations {mm[2]} in column {mm[0]}" for mm in mm_list])
        return html.Div([
            dcc.Store(id='mm-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Convert to base form\'. This means that all detected string mismatches will be converted to their respective base form."),
            html.Div(output_str),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-mm-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'convert_mode':
        mode_dct = dict()
        for col, base, str_vars in mm_list:
            vars = ast.literal_eval(str_vars)
            mode = 0
            mode_var = ""
            for var in vars:
                amount = len(df[df[col] == var])
                if amount > mode:
                    mode = amount
                    mode_var = var
            mode_dct[(col, base, str_vars)] = mode_var
        output_str = "The modes for all string mismatches are: " + ', '.join([f"{val} for variations {key[2]} in column {key[0]}" for key, val in mode_dct.items()])
        return html.Div([
            dcc.Store(id='mm-correct-method', data=method, storage_type='memory'),
            html.P("You have chosen the option \'Convert to mode\'. This means that all detected string mismatches will be converted to the most frequent occurring variation in that mismatch."),
            html.Div(output_str, style={'whiteSpace': 'pre-line'}),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-mm-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'remove':
        return html.Div([
            dcc.Store(id='mm-correct-method', data=method, storage_type='memory'),
            html.P("You have chosen the option \'Remove rows\'. This means that all rows containing string mismatches will be removed from the dataset."),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-mm-correct-method', n_clicks=0, style=button_style)
        ])
    elif method == 'keep':
        return html.Div([
            dcc.Store(id='mm-correct-method', data=method, storage_type='memory'),
            html.P(
                "You have chosen the option \'Keep all\'. This means that all detected string mismatches will be retained in the dataset"),
            html.P("Please press the button below to submit your correction technique."),
            html.Button("Submit", id='submit-mm-correct-method', n_clicks=0, style=button_style)
        ])

@app.callback(Output('display-mm-correction', 'children'),
              [Input('submit-mm-correct-method', 'n_clicks'),
               State('df-no-mls', 'data'),
               State('stored-feature-types', 'data'),
               Input('complete-mm-list', 'data'),
               Input('mm-correct-method', 'data'),
               Input('renamed-cols', 'data')])
def apply_mms(n_clicks, df_fp, ft_types_fp, mm_list_fp, method, rename_dct):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-mm-correct-method' in changed_id and n_clicks > 0:
        df = fetch_data(df_fp)
        with open(mm_list_fp, 'rb') as f:
            mm_list = pickle.load(f)
        if rename_dct != None:
            new_mm_list = []
            for mm in mm_list:
                if mm[0] in rename_dct:
                    new_col = rename_dct[mm[0]]
                else:
                    new_col = mm[0]
                new_mm_list.append([new_col, mm[1], mm[2]])
            mm_list = new_mm_list
        if method == 'remove':
            remove_indices_set = set()
            for col, base, str_vars in mm_list:
                vars = ast.literal_eval(str_vars)
                for var in vars:
                    indices = df[df[col] == var].index
                    for idx in indices:
                        remove_indices_set.add(idx)
            remove_indices = sorted(list(remove_indices_set))
            new_df = df.drop(index=remove_indices)
            df_dtb = new_df.copy()
            df_dtb.index = df_dtb.index.set_names(["index"])
            df_dtb = df_dtb.reset_index()
            mm_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_dtb.columns],
                style_data_conditional=create_dup_corr_styles(remove_indices, df),
                page_size=20
            )
            mm_corr_badge = dbc.Badge('Removed row', color='tomato', className='me-1')
            updated_df = new_df.copy()
        elif method == 'keep':
            list_of_points = []
            for col, base, str_vars in mm_list:
                vars = ast.literal_eval(str_vars)
                for var in vars:
                    indices = df[df[col] == var].index
                    for idx in indices:
                        list_of_points.append([idx, col])

            df_dtb = df.copy()
            df_dtb.index = df_dtb.index.set_names(["index"])
            df_dtb = df_dtb.reset_index()
            mm_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_dtb.columns],
                style_data_conditional=create_mv_corr_styles(list_of_points, '#0080FF'),
                page_size=20
            )
            mm_corr_badge = dbc.Badge('Retained value', color='#0080FF', className='me-1')
            updated_df = df.copy()
        elif method == 'convert_base':
            df_copy = copy.deepcopy(df)
            list_of_points = []
            for col, base, str_vars in mm_list:
                vars = ast.literal_eval(str_vars)
                for var in vars:
                    indices = df[df[col] == var].index
                    for idx in indices:
                        list_of_points.append([idx, col])
                    df_copy.loc[indices, col] = base
            df_dtb = df_copy.copy()
            df_dtb.index = df_dtb.index.set_names(["index"])
            df_dtb = df_dtb.reset_index()
            mm_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_dtb.columns],
                style_data_conditional=create_mv_corr_styles(list_of_points),
                page_size=20
            )
            mm_corr_badge = dbc.Badge('Corrected value', color='#50C878', className='me-1')
            updated_df = df_copy.copy()
        elif method == 'convert_mode':
            df_copy = copy.deepcopy(df)
            list_of_points = []
            for col, base, str_vars in mm_list:
                vars = ast.literal_eval(str_vars)
                mode = 0
                mode_var = ""
                all_indices = []
                for var in vars:
                    indices = df[df[col] == var].index
                    for idx in indices:
                        all_indices.append(idx)
                        list_of_points.append([idx, col])
                    amount = len(df[df[col] == var])
                    if amount > mode:
                        mode = amount
                        mode_var = var
                df_copy.loc[all_indices, col] = mode_var
            df_dtb = df_copy.copy()
            df_dtb.index = df_dtb.index.set_names(["index"])
            df_dtb = df_dtb.reset_index()
            mm_corr_dtb = dash_table.DataTable(
                data=df_dtb.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_dtb.columns],
                style_data_conditional=create_mv_corr_styles(list_of_points),
                page_size=20
            )
            mm_corr_badge = dbc.Badge('Corrected value', color='#50C878', className='me-1')
            updated_df = df_copy.copy()
        filepath = 'cached_files/df-no-mms.pkl'
        with open(filepath, 'wb') as h:
            pickle.dump(updated_df, h)
        return html.Div([dcc.Store(id='df-no-mms', data=filepath, storage_type='memory'),
                         html.Hr(),
                         html.H4("Corrections"),
                         html.P(
                             "Your correction technique has been applied on the dataset and the updated dataset is shown below."),
                         mm_corr_badge,
                         mm_corr_dtb])

@app.callback(
    Output('fully-cleaned-dataset', 'children'),
    [Input('stored-df', 'data'),
        Input('df-no-mms', 'data'),
     Input('renamed-cols', 'data'),
     Input('submit-all-corrections', 'n_clicks')
    ]
)
def generate_final_table(dirty_df_fp, clean_df_fp, rename_dct, n_clicks):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'submit-all-corrections' in changed_id and n_clicks > 0:
        dirty_df = fetch_data(dirty_df_fp)
        clean_df = fetch_data(clean_df_fp)
        if rename_dct == None:
            rename_dct = dict()
        final_dtb, removed_rows, removed_cols, imputed_vals = show_final_df(clean_df, dirty_df, rename_dct)
        return html.Div([
            dcc.Store(id='removed-rows', data=removed_rows, storage_type='memory'),
            dcc.Store(id='removed-cols', data=removed_cols, storage_type='memory'),
            dcc.Store(id='imputed-locations', data=imputed_vals, storage_type='memory'),
            final_dtb,
            html.H4("Update and download"),
            html.P("If you are happy with the cleaned version of your dataset, you can update your old version on OpenML and download it for yourself using the buttons below"),
            html.Div([
            dcc.ConfirmDialogProvider(
                children=html.Button("Update", style=button_style),
                id='btn-update-openml',
                message='Are you sure that you want to update your dataset on OpenML?'
            ),
            html.Button("Download", id='btn-download-dataset', style=button_style)], style={'display':'flex', 'gap':'10px'}),
            html.Div(id='update-dataset-output'),
            dcc.Download(id='download-clean-dataset')
        ])
    return html.Div("")

@app.callback(
    [Output('final_table', 'style_data_conditional'),
     Output('final_table', 'style_header_conditional')],
    [Input('final_table', 'page_current'), Input("final_table", 'page_size'),
     Input('stored-df', 'data'), Input('df-no-mms', 'data'), Input("removed-rows",
    'data'), Input("removed-cols", 'data'), Input("imputed-locations", 'data'),
     Input('renamed-cols', 'data')]
)
def update_table_style(page_current, page_size, dirty_df_fp, clean_df_fp, removed_rows, removed_cols, imputed_locations, cryp_mapping):
    if cryp_mapping == None:
        cryp_mapping = dict()
    dirty_df = fetch_data(dirty_df_fp).rename(columns=cryp_mapping).astype(str)
    clean_df = fetch_data(clean_df_fp)
    if page_current == None:
        page_current = 0
    row_styles = create_dup_corr_styles(removed_rows, clean_df)
    col_styles = create_dup_col_corr_styles(dirty_df, removed_cols)
    header_styles = create_cryp_header(cryp_mapping)
    value_styles = []
    for (idx, col) in imputed_locations:
        page_index = idx // page_size
        if page_index == page_current:
            row_index = idx % page_size
            value_styles.append({
                'if':{
                    'row_index':row_index,
                    'column_id':col
                },
                'backgroundColor':'#50C878',
                'color':'white'
            })
    style_data = row_styles + col_styles + value_styles
    return style_data, header_styles

@app.callback(
    Output('update-dataset-output', 'children'),
    [Input('btn-update-openml', 'submit_n_clicks'),
     State('cleaned-df', 'data')],
    prevent_initial_call=True
)
def update_openml_dataset(submit_n_clicks, cleaned_df_fp):
    clean_df = fetch_data(cleaned_df_fp)
    if submit_n_clicks:
        ### hier toevoegen dat de dataset daadwerkelijk wordt geupdate
        return html.P("Your dataset has been updated on OpenML! You can now close this webpage.")

@app.callback(
    Output('download-clean-dataset', 'data'),
    [Input('btn-download-dataset', 'n_clicks'),
     State('cleaned-df', 'data')],
    prevent_initial_call=True
)
def download_dataset(n_clicks, clean_df_fp):
    clean_df = fetch_data(clean_df_fp)
    if n_clicks > 0 :
        return dcc.send_data_frame(clean_df.to_csv, 'cleaned_dataset.csv')

if __name__ == '__main__':
    missing_values = []
    outlier_values = []
    outlier_rows = []
    mislabels = []
    mismatches = []
    app.run_server(debug=True)
