# UK_marginal_rate_dashboard.py

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import product
import time
import pickle
import io
import zipfile
import logging
from logging.handlers import RotatingFileHandler

import dash
from dash import dcc, html, callback_context
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# =====================
# Constants
# =====================

# Modelling constants
STUDENT_LOAN_RATE = 0.09
STUDENT_LOAN_THRESHOLD = 27295  # Plan two, started course between 1 September 2012 and 31 July 2023
MAX_INCOME = 180000
PERTURBATION = 100
DEFAULT_DATASET = "rUK 2024-25"
MAX_KIDS = 6

# System constants
DATASET_FILENAME = 'UK_marginal_tax_datasets.json'
CHART_COLOURS = [
    '#1133AF', '#FF5733', '#33FF57', '#FF33A6', '#A633FF', '#33FFF6', '#F6FF33',
    '#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231', '#911EB4', '#46F0F0', 
    '#F032E6', '#BCF60C', '#FABEBE', '#008080', '#E6BEFF', '#9A6324', '#FFFAC8', 
    '#800000', '#AA6E28', '#808000', '#FFD8B1', '#000075', '#808080', '#FFFFFF', '#000000'
]

# Logging
LOG_FILE = f'marginal_rate_dashboard.log'

# Logging

# Set up logger
logger = logging.getLogger("marginal_rate")
logger.setLevel(logging.DEBUG)

# Create a rotating file handler that writes log messages to LOG_FILE
# and rotates after 1MB (1048576 bytes), keeping the last 3 backups
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1048576, backupCount=3)
file_handler.setLevel(logging.DEBUG)

# Create a stream handler to output to stdout
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# =====================
# Data Loading and Preprocessing
# =====================

def load_tax_data(filename):
    logger.info(f"Loading tax data from {filename}...")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        logger.info("Tax data loaded successfully.")
        return data
    except Exception as e:
        logger.info(f"Error loading tax data: {e}")
        return {}

def generate_childcare_subsidy_options(relevant_datas):
    if not relevant_datas:
        return []
    
    # Extract 'childcare subsidy per child' from each relevant dataset
    subsidies = [data.get("childcare subsidy per child", 0) for data in relevant_datas]
    
    # Find the maximum subsidy value
    max_subsidy = max(subsidies) if subsidies else 0
    
    # Generate four increments up to the maximum subsidy
    increments = 4
    if max_subsidy == 0:
        return [0]  # Only £0 available if max_subsidy is 0
    
    step = max_subsidy / increments
    options = [int(round(i * step)) for i in range(increments + 1)]
    
    # Ensure the last option is exactly the max_subsidy
    options[-1] = int(max_subsidy)
    
    return options

def precalculate_data(tax_data):
    # Pre-generate all combinations of charts
    logger.info("Pre-generating tax calcs for all datasets...")

    start_time = time.time()

    bool_options = [True, False]

    # Initialize a dictionary to store pre-generated figures
    generated_data = {}

    # Now, generate all combinations for each dataset
    for dataset in list(tax_data.keys()):
        relevant_data = tax_data[dataset]
        childcare_options = generate_childcare_subsidy_options([relevant_data])
        for include_student_loan, include_marriage_allowance in product(bool_options, repeat=2):
            for children in range(0, MAX_KIDS + 1):
                for childcare_subsidy_amount in childcare_options:
                    
                    # Generate dataframe and chart for marginal rate
                    df_tax_calcs = calculate_tax_results_for_ruleset(
                        dataset, include_student_loan, childcare_subsidy_amount, include_marriage_allowance, children
                    )
                    
                    # Store the figure in the dictionary
                    key = (dataset, include_student_loan, childcare_subsidy_amount, include_marriage_allowance, children)
                    
                    # Store the figure and max marginal rate data in the dictionary
                    max_marginal_rate = df_tax_calcs['Marginal tax rate'].max()
                    max_gross_income = df_tax_calcs.loc[df_tax_calcs['Marginal tax rate'].idxmax(), 'Gross income (£)']
                    data = {
                        'tax_calcs': df_tax_calcs,
                        'max_marginal_rate': max_marginal_rate,
                        'max_gross_income': max_gross_income,
                    }
                    generated_data[key] = data

    # Calculate total number of versions pre-generated
    total_versions = len(generated_data)

    # Estimate the size of all pre-generated data in MB
    total_size_bytes = sum(len(pickle.dumps(fig)) for fig in generated_data.values())
    total_size_mb = total_size_bytes / (1024 * 1024)

    logger.info(f"Generated: {total_versions} scenarios in {time.time() - start_time:.1f}s, approx size {total_size_mb:.2f} MB")

    return generated_data

# =====================
# Tax Calculation Functions
# =====================

def calculate_tax_and_ni(gross_income, relevant_dataset, tax_type, include_student_loan, childcare_subsidy_amount, include_marriage_allowance, children):
    relevant_data = tax_data[relevant_dataset]
    total_tax = 0

    if tax_type == "income tax":
        # Personal Allowance adjustments
        if gross_income > relevant_data["allowance withdrawal threshold"]:
            modified_personal_allowance = max(
                0,
                relevant_data["statutory personal allowance"] - relevant_data["allowance withdrawal rate"] * (gross_income - relevant_data["allowance withdrawal threshold"])
            )
        elif include_marriage_allowance and gross_income < relevant_data["marriage allowance max earnings"]:
            modified_personal_allowance = relevant_data["statutory personal allowance"] * (1 + relevant_data["marriage allowance"])
        else:
            modified_personal_allowance = relevant_data["statutory personal allowance"]
        
        taxable_net_income = max(0, gross_income - modified_personal_allowance)
        
        # Apply HICBC (High Income Child Benefit Charge)
        if children > 0:
            total_child_benefit = 52 * (relevant_data["child benefit"]["1st"] + relevant_data["child benefit"]["subsequent"] * (children - 1))
            if gross_income < relevant_data["HICBC start"]:
                HICBC = 0
            elif gross_income > relevant_data["HICBC end"]:
                HICBC = total_child_benefit
            else:
                hicbc_step = (relevant_data["HICBC end"] - relevant_data["HICBC start"]) / 100
                number_of_steps = (gross_income - relevant_data["HICBC start"]) / hicbc_step
                HICBC = total_child_benefit * number_of_steps / 100
            total_tax += HICBC
        
        # Student Loan
        if include_student_loan and gross_income > STUDENT_LOAN_THRESHOLD:
            total_tax += (gross_income - STUDENT_LOAN_THRESHOLD) * STUDENT_LOAN_RATE
            
        # Childcare Subsidy (modeled as negative tax)
        if childcare_subsidy_amount > 0 and children > 0:
            if relevant_data["childcare min earnings"] < gross_income < relevant_data["childcare max earnings"]:
                total_tax -= childcare_subsidy_amount * min(children, relevant_data["childcare max children"])

    else:
        # For NI, taxable_net_income is simply the gross_income
        taxable_net_income = gross_income

    last_threshold = 0

    for band in relevant_data[tax_type]:
        threshold = band.get("threshold", 1e12)  # Default to a very high threshold if not specified
        gross_income_in_band = min(taxable_net_income, threshold) - last_threshold
        tax_in_band = gross_income_in_band * band["rate"]
        total_tax += tax_in_band

        last_threshold = threshold

        if taxable_net_income <= threshold:
            break

    return total_tax

def calculate_tax_results_for_ruleset(ruleset, include_student_loan, childcare_subsidy_amount, include_marriage_allowance, children):
    # Create a range of gross incomes
    gross_incomes = np.arange(0, MAX_INCOME  + PERTURBATION, PERTURBATION)


    # Calculate net income and marginal rate
    data = []
    previous_total_tax = 0
    for gross_income in gross_incomes:
        income_tax = calculate_tax_and_ni(gross_income, ruleset, "income tax", include_student_loan, childcare_subsidy_amount, include_marriage_allowance, children)
        employee_ni = calculate_tax_and_ni(gross_income, ruleset, "NI", include_student_loan, childcare_subsidy_amount, include_marriage_allowance, children)
        total_tax_ni = income_tax + employee_ni
        net_income = gross_income - total_tax_ni
        marginal_rate = 0 if gross_income == 0 else ((total_tax_ni - previous_total_tax) / PERTURBATION * 100)
        effective_rate = 0 if gross_income == 0 else ((total_tax_ni / gross_income) * 100)
        data.append([gross_income, net_income, marginal_rate, effective_rate])
        previous_total_tax = total_tax_ni

    # Create DataFrame
    df = pd.DataFrame(data, columns=["Gross income (£)", "Net income", "Marginal tax rate", "Effective tax rate"])
    return df

# =====================
# Chart Creation Functions
# =====================


def create_rate_chart(tax_calcs_list, dataset_names, y_data):
    # Create Plotly figure
    fig = go.Figure()

    for idx, (df, dataset) in enumerate(zip(tax_calcs_list, dataset_names)):
        fig.add_trace(go.Scatter(
            x=df['Gross income (£)'],
            y=df[y_data],
            mode='lines',
            name=dataset,
            line=dict(color=CHART_COLOURS[idx % len(CHART_COLOURS)]),  # Cycle through CHART_COLOURS if datasets exceed color list
            hovertemplate=f"%{{y:.1f}}%",
            showlegend=True
        ))

    fig.update_layout(
        title=f"Gross employment income vs {y_data.lower()}",
        xaxis_title="Gross employment income",
        yaxis_title=y_data,
        template='plotly_white',
        font=dict(family="Poppins"),
        hovermode='x',
        yaxis=dict(range=[0, 90], ticksuffix="%"),
        xaxis=dict(tickprefix="£"),
        title_font=dict(size=24),
        xaxis_title_font=dict(size=18),
        yaxis_title_font=dict(size=18),
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=14),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=16)
        ),
    )

    return fig


def create_gross_net_chart(tax_calcs_list, dataset_names):
    # Create Plotly figure
    fig = go.Figure()

    for idx, (df, dataset) in enumerate(zip(tax_calcs_list, dataset_names)):
        fig.add_trace(go.Scatter(
            x=df['Gross income (£)'],
            y=df['Net income'],
            mode='lines',
            name=dataset,
            hovertemplate='£%{y:,.0f}',
            showlegend=True,
            line=dict(color=CHART_COLOURS[idx % len(CHART_COLOURS)])  # Cycle through CHART_COLOURS if datasets exceed color list
        ))

    fig.update_layout(
        title="Gross employment income vs net income",
        xaxis_title="Gross employment income",
        yaxis_title="Net income",
        template='plotly_white',
        font=dict(family="Poppins"),
        hovermode='x',
        yaxis=dict(tickprefix="£"),
        xaxis=dict(tickprefix="£"),
        title_font=dict(size=24),
        xaxis_title_font=dict(size=18),
        yaxis_title_font=dict(size=18),
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=14),
        showlegend=True,
        legend=dict(
           font=dict(size=16)
        ),
    )

    return fig


# =====================
# Dash App Initialization
# =====================

# Initialize Dash app
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    'https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap',
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "UK Marginal Tax Rates Dashboard"
server = app.server

# Load tax data
tax_data = load_tax_data(DATASET_FILENAME)

# Pre-calculate tax data
pre_generated_tax_calc_data = precalculate_data(tax_data)

# =====================
# Define Dash Layout
# =====================

def define_dash_layout():
    layout = dbc.Container(
        fluid=True,
        children=[
            
            dbc.Row(
                [
                    # "Select year/region" Button
                    dbc.Col(
                        [
                            dbc.Button(
                                "Select year/region",
                                id="open-modal",
                                n_clicks=0,
                                color="primary",
                                className="",
                                style={'marginTop': '15px'},
                            )
                        ],
                        xs=12, sm=12, md=2,  # Responsive widths
                        className="mb-2 mb-md-0"  # Margin bottom on small screens
                    ),
                    
                    # Choice of chart dropdown
                    dbc.Col(
                        [
                            html.Label('Chart type'),
                            dcc.Dropdown(
                                id='chart-type',
                                options=[
                                    {'label': 'Marginal rate', 'value': 'Marginal rate'},
                                    {'label': 'Gross v net', 'value': 'Gross v net'},
                                    {'label': 'Effective rate', 'value': 'Effective rate'}
                                ],
                                value="Marginal rate",
                                clearable=False,
                            )
                        ],
                        xs=12, sm=6, md=2,
                        className="mb-2 mb-md-0"
                    ),
                    
                    # Number of Children Dropdown
                    dbc.Col(
                        [
                            html.Label('Children'),
                            dcc.Dropdown(
                                id='children-dropdown',
                                options=[{'label': str(i), 'value': i} for i in range(0, MAX_KIDS + 1)],
                                value=0,
                                clearable=False,
                            )
                        ],
                        xs=12, sm=6, md=2,
                        className="mb-2 mb-md-0"
                    ),
                    
                    # Childcare Subsidy Dropdown
                    dbc.Col(
                        [
                            html.Label('Childcare'),
                            dcc.Dropdown(
                                id='childcare-dropdown',
                                options=[],  # Will be populated dynamically
                                value=0,
                                clearable=False,
                            )
                        ],
                        xs=12, sm=6, md=2,
                        className="mb-2 mb-md-0"
                    ),
                    
                    # Checkboxes for Options
                    dbc.Col(
                        [
                            html.Label('Options'),
                            dbc.Checklist(
                                options=[
                                    {"label": "Student Loan", "value": 'include_student_loan'},
                                    {"label": "Marriage Allowance", "value": 'include_marriage_allowance'},
                                ],
                                value=[],  # default is all unchecked
                                id="options-checklist",
                                inline=True,
                            ),
                        ],
                        xs=12, sm=12, md=3,
                        className="mb-2 mb-md-0"
                    ),
                    
                    # Download Button
                    dbc.Col(
                        [
                            dbc.Button(
                                "Download",
                                id="download-button",
                                color="success",
                                className="w-100",  # Make button full width on small screens
                                style={'marginTop': '15px'},
                            ),
                            dcc.Download(id="download-component"),
                        ],
                        xs=12, sm=6, md=1,
                    ),
                ],
                className="g-2",  # Adds gutters between columns
                style={'marginTop': '20px'}
            ),
            
            # Modal for selecting multiple datasets
            dbc.Modal(
                [
                    dbc.ModalHeader("Select Year/Region"),
                    dbc.ModalBody(
                        dbc.Checklist(
                            id="dataset-checkboxes",
                            options=[{'label': dataset, 'value': dataset} for dataset in list(tax_data.keys())],
                            value=[],  
                            inline=False
                        )
                    ),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Apply", 
                            id="apply-dataset-selection", 
                            n_clicks=0, 
                            color="primary", 
                            className="ml-auto"
                        )
                    )
                ],
                id="dataset-modal", 
                is_open=False  # Modal starts closed
            ),
            
            # Store component to hold selected datasets
            dcc.Store(id='selected-datasets', data=[]),

            
            # Chart (move this block below the UI elements)
            dbc.Row(
                dbc.Col(
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=[
                            dcc.Graph(
                                id='marginal-tax-rate-chart',
                                style={'height': '80vh'}
                            )
                        ]
                    ),
                    width=12
                )
            ),
            
            # Add a new Row for displaying the maximum marginal rate and logo
            dbc.Row(
                [
                    # Maximum marginal rate text (left column)
                    dbc.Col( 
                        html.Div(
                            id='max-marginal-rate-display',
                            style={'fontSize': '18px', 'fontWeight': 'bold'}
                        ),
                        xs=12, sm=8, md=10,
                        style={'display': 'flex', 'alignItems': 'center'},  # Vertically align
                    ),
                    
                    # Logo image (right column)
                    # Add the "a" tag with the href attribute to make the logo clickable
                    dbc.Col(
                        html.A(
                            html.Img(
                                src='assets/logo_standard_small.jpeg',
                                style={
                                    'height': '50px',
                                    'display': 'block',
                                    'marginLeft': 'auto'
                                }
                            ),
                            href="https://taxpolicy.org.uk/",  # Link to the desired URL
                            target="_blank"  # Open link in a new tab
                        ),
                        xs=12, sm=4, md=2,
                    ),

                ],
                style={
                    'marginTop': '0px',  # Reduced margin to minimize vertical gap
                    'display': 'flex',
                    'alignItems': 'center'  # Vertically centers the text and logo
                } 
            ),
        ],
        style={
            'fontFamily': 'Poppins, sans-serif',
            'padding': '20px'
        }
    )

    return layout

# Assign the layout to the app
app.layout = define_dash_layout()

# =====================
# Define Callbacks
# =====================

# Callback to toggle the modal
@app.callback(
    Output("dataset-modal", "is_open"),
    [
        Input("open-modal", "n_clicks"),
        Input("apply-dataset-selection", "n_clicks"),
    ],
    [State("dataset-modal", "is_open")],
)
def toggle_modal(open_clicks, apply_clicks, is_open):
    ctx = callback_context

    if not ctx.triggered:
        return is_open
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if button_id == "open-modal":
            return not is_open
        elif button_id == "apply-dataset-selection":
            return False
    return is_open

# Callback to update the selected datasets
@app.callback(
    Output('selected-datasets', 'data'),
    [Input('apply-dataset-selection', 'n_clicks')],
    [State('dataset-checkboxes', 'value')]
)
def update_selected_datasets(n_clicks, selected):
    if n_clicks > 0:
        return selected
    return []

# Callback to update the childcare subsidy options based on selected datasets
@app.callback(
    dash.dependencies.Output('childcare-dropdown', 'options'),
    [dash.dependencies.Input('selected-datasets', 'data')]
)
def update_childcare_options(selected_datasets):
    if not selected_datasets:
        selected_datasets = [DEFAULT_DATASET]
    
    # Collect relevant_data for all selected datasets
    relevant_datas = [tax_data[dataset] for dataset in selected_datasets if dataset in tax_data]
    
    # Generate childcare subsidy options based on all selected datasets
    childcare_options = generate_childcare_subsidy_options(relevant_datas)
    
    # Create dropdown options with proper formatting
    options = [{'label': f"£{amount:,}", 'value': amount} for amount in childcare_options]
    
    return options

# Callback to update the childcare subsidy value when options change
@app.callback(
    dash.dependencies.Output('childcare-dropdown', 'value'),
    [dash.dependencies.Input('childcare-dropdown', 'options')]
)
def set_childcare_value(available_options):
    # Set default value to the first option (which should be £0)
    return available_options[0]['value'] if available_options else 0

# Callback to update the number of children if childcare subsidy is selected and children is 0
@app.callback(
    dash.dependencies.Output('children-dropdown', 'value'),
    [
        dash.dependencies.Input('childcare-dropdown', 'value'),
    ],
    [
        dash.dependencies.State('children-dropdown', 'value')
    ]
)
def update_children_value(childcare_subsidy_amount, children):
    childcare_subsidy_amount = childcare_subsidy_amount or 0

    if childcare_subsidy_amount > 0 and children == 0:
        return 1
    else:
        return dash.no_update

# Callback to update the chart based on user inputs and selected datasets
@app.callback(
    [
        dash.dependencies.Output('marginal-tax-rate-chart', 'figure'),
        dash.dependencies.Output('max-marginal-rate-display', 'children'),
    ],
    [
        dash.dependencies.Input('selected-datasets', 'data'),  # New input for selected datasets
        dash.dependencies.Input('children-dropdown', 'value'),
        dash.dependencies.Input('childcare-dropdown', 'value'),
        dash.dependencies.Input('options-checklist', 'value'),
        dash.dependencies.Input('chart-type', 'value'),
    ]
)
def update_figure(selected_datasets, children, childcare_subsidy_amount, options, chart_type):
    if not selected_datasets:
        selected_datasets = [DEFAULT_DATASET]
    
    include_student_loan = 'include_student_loan' in options
    include_marriage_allowance = 'include_marriage_allowance' in options
    childcare_subsidy_amount = childcare_subsidy_amount if childcare_subsidy_amount is not None else 0  # Ensure it's not None
    
    # Collect data for all selected datasets
    datasets_data = []
    for dataset in selected_datasets:
        key = (dataset, include_student_loan, childcare_subsidy_amount, include_marriage_allowance, children)
        data = pre_generated_tax_calc_data.get(key)
        if data:
            datasets_data.append(data)
    
    # Prepare list of dataframes for charting
    tax_calcs_list = [data['tax_calcs'] for data in datasets_data]
    
    # Prepare list of (max_rate, max_income) tuples
    max_rates = [
        (data['max_marginal_rate'], data['max_gross_income']) for data in datasets_data
    ]
    
    # Create the chart based on the selected chart type
    if chart_type == "Marginal rate":
        fig = create_rate_chart(tax_calcs_list, selected_datasets, "Marginal tax rate")
        
    elif chart_type == "Effective rate":
        fig = create_rate_chart(tax_calcs_list, selected_datasets, "Effective tax rate")
        
    else:
        fig = create_gross_net_chart(tax_calcs_list, selected_datasets)
    
    # Prepare display text for maximum marginal rates
    display_text = ""
    for dataset, (max_rate, max_income) in zip(selected_datasets, max_rates):
        display_text += f"{dataset}: Maximum marginal rate: {max_rate:,.1f}% at gross income of £{max_income:,.0f}\n"
    
    return fig, html.Div([html.Span(display_text)], style={'whiteSpace': 'pre-line'})

# Callback to handle the download
@app.callback(
    Output("download-component", "data"),
    [Input("download-button", "n_clicks")],
    [
        State('selected-datasets', 'data'),
        State('children-dropdown', 'value'),
        State('childcare-dropdown', 'value'),
        State('options-checklist', 'value'),
    ],
    prevent_initial_call=True,
)
def generate_download(n_clicks, selected_datasets, children, childcare_subsidy_amount, options):
    if not selected_datasets:
        selected_datasets = [DEFAULT_DATASET]
    
    include_student_loan = 'include_student_loan' in options
    include_marriage_allowance = 'include_marriage_allowance' in options
    
    # Create an in-memory ZIP archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for dataset in selected_datasets:
            key = (dataset, include_student_loan, childcare_subsidy_amount, include_marriage_allowance, children)
            data = pre_generated_tax_calc_data.get(key)
            if data:
                df = data['tax_calcs']
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                # Define a filename for each CSV
                csv_filename = f"{dataset.replace(' ', '_')}.csv"
                zip_file.writestr(csv_filename, csv_buffer.getvalue())
    
    zip_buffer.seek(0)
    return dcc.send_bytes(zip_buffer.read(), "selected_datasets.zip")

# =====================
# Run the App
# =====================

if __name__ == '__main__':
    logger.info("\nStarting UK Marginal Rate Dashboard...")

    logger.info("Running Dash app...") 
    app.run_server(debug=False, host='0.0.0.0')
