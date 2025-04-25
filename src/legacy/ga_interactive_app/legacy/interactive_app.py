from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_plotly

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

# Load the CSV file
csv_path = "../results/11182024_coris/area_comparisons_af_12202024_model_5_wmetadata_affine.csv"
df = pd.read_csv(csv_path)

with open('../results/11182024_coris/specific_pats_af_12202024_model_5.txt') as f:
    pats = f.read().splitlines()
    pats = [int(pat.split('_')[0]) for pat in pats]
df = df[df.PID.isin(pats)]

# Create a base figure with all traces in grey
base_fig = go.Figure()
trace_map = {}  # Store trace indices for easy updates

for i, (pid, sub_df) in enumerate(df.groupby(["PID", "Laterality"])):
    trace = go.Scatter(
        x=sub_df["ExamDate"], 
        y=sub_df["mm_area"],
        mode="lines+markers",
        line=dict(color="rgba(100,100,100,0.5)"),  # Default: Transparent grey
        name=f"{pid[0]}_{pid[1]}",
        text=[f"MRN: {pid[0]}<br>Laterality: {pid[1]}" for _ in sub_df["ExamDate"]],
        hoverinfo="text",
        visible=True  # Always visible, only color changes
    )
    base_fig.add_trace(trace)
    trace_map[(pid[0], pid[1])] = i  # Store trace index

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize("select_mrn", "Select MRN", df.PID.apply(str).unique().tolist()),
        ui.input_selectize("select_lat", "Select Laterality", ["OD", "OS"])
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Growth Curves Plot"),
            output_widget("progression_plot")
            ),
        ui.card(
            ui.card_header("Growth Video"),
            # ui.output_ui("video_display")
            ),
        col_widths=[7, 5]
    ),
    title="GA Growth Analysis"
)

def server(input, output, session):

    @render_plotly
    def progression_plot():
        """Updates trace colors dynamically based on selection."""
        selected_mrn = input.select_mrn()
        selected_lat = input.select_lat()

        # Copy the base figure
        fig = go.Figure(base_fig)

        # Reset all traces to grey
        for trace in fig.data:
            trace.line.color = "rgba(100,100,100,0.5)"

        # Highlight selected trace in red if it exists
        if (selected_mrn, selected_lat) in trace_map:
            trace_index = trace_map[(selected_mrn, selected_lat)]
            fig.data[trace_index].line.color = "red"

        return fig

    @render.ui
    def video_display():
        selected_mrn = input.select_mrn()
        selected_lat = input.select_lat()
        video_path = df[(df['PID'] == int(selected_mrn)) & (df['Laterality'] == selected_lat)]['video'].iloc[0]
        if os.path.exists(video_path):
            return ui.HTML(f'<video width="100%" controls><source src="videos/video0.mp4" type="video/mp4"></video>')
        else:
            return ui.HTML("<p>Video file not found.</p>")
        return ui.HTML("<p>No video available for the selected patient.</p>")

app = App(app_ui, server)