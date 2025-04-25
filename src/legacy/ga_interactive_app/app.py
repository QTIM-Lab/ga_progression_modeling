import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import cv2
import base64
import os
from flask import send_file

# Load the data
csv_path = "results/11182024_coris/area_comparisons_af_12202024_model_5_wmetadata_affine.csv"  # Change this to your CSV file path
df = pd.read_csv(csv_path)

# modify registration paths 
df['video'] = df.apply(lambda row: f'/scratch90/veturiy/projects/GA_progression_modeling/results/11182024_coris/powerpoint_af_12202024_model_5_gompertz_affine/metadata/videos_wcontours/{row.PID}_{row.Laterality}_wcontours.mp4', axis=1)

# Register data
df_register = pd.read_csv('src/guis/ga_interactive_app/Nick_Summary_Stats_w_MRNS.csv')
df_register = df_register[['MRN', 'Laterality', 'median_b_over_c']]
df = pd.merge(df, df_register.rename({'MRN': 'PID'}, axis=1), on=['PID', 'Laterality'], how='inner')

# Get registered timepoints
df['ExamDate'] = pd.to_datetime(df['ExamDate'])
# Compute relative date for each patient
df['ExamDate'] = df.groupby(['PID', 'Laterality'])['ExamDate'].transform(lambda x: (x - x.min()).dt.days / 365.25)
# add to median_b_over_c
df['ExamDate'] = df['ExamDate'] - df['median_b_over_c']

# sort values by the time
df.sort_values(by=['PID', 'Laterality', 'ExamDate'], inplace=True)

def get_plot():
    fig = go.Figure()
    
    grouped = df.groupby(['PID', 'Laterality'])
    for (pid, lat), group in grouped:
        fig.add_trace(go.Scatter(
            x=group['ExamDate'], 
            y=group['mm_area'], 
            mode='lines+markers', 
            name=f'{pid} - {lat}',
            line=dict(color='grey', width=1, dash='solid'),
            opacity=0.3,
            hoverinfo='text',
            hovertext=[f'PID: {pid}, Laterality: {lat}']*len(group)
        ))
    
    fig.update_layout(
        title='Disease Lesion Area Over Time',
        xaxis_title='Exam Date',
        yaxis_title='Lesion Area (mm²)',
        template='plotly_white'
    )
    return fig

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='lesion_plot', figure=get_plot(), style={'height': '70vh'})
        ], width=8, className='d-flex align-items-center justify-content-center'),
        dbc.Col([
            html.Div([
                html.H4("Selected Patient Video"),
                html.Video(id='video_display', controls=True, style={'width': '100%'}),
                html.P(id='patient_info', style={'font-weight': 'bold'})
            ], className='d-flex flex-column align-items-center justify-content-center', style={'height': '70vh'})
        ], width=4, className='d-flex align-items-center justify-content-center')
    ], className='vh-100 align-items-center justify-content-center')
])

@app.callback(
    [Output('lesion_plot', 'figure'),
     Output('video_display', 'src'),
     Output('patient_info', 'children')],
    [Input('lesion_plot', 'hoverData')]
)
def highlight_trace(hover_data):
    fig = get_plot()
    video_src = ""
    patient_info = "Hover over a line to see details"
    
    if hover_data:
        point = hover_data['points'][0]
        hover_text = point['hovertext']
        pid, laterality = hover_text.split(', ')[0].split(': ')[1], hover_text.split(', ')[1].split(': ')[1]
        
        selected_group = df[(df['PID'] == int(pid)) & (df['Laterality'] == laterality)]
        
        for trace in fig.data:
            if trace.name == f'{pid} - {laterality}':
                trace.line.color = 'red'
                trace.opacity = 1.0
                trace.line.width = 2
        
        video_path = selected_group['video'].values[0]
        if os.path.exists(video_path):
            video_src = f'data:video/mp4;base64,{base64.b64encode(open(video_path, "rb").read()).decode()}'
        
        patient_info = f'Selected Patient: {pid}, Laterality: {laterality}'
    
    return fig, video_src, patient_info

if __name__ == '__main__':
    app.run_server(debug=True)
