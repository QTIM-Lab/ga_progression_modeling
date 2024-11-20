import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import base64
from PIL import Image, ImageOps, ImageDraw
import io
import cv2
import numpy as np

# Load the CSV files
data = 'results/07312024/area_comparisons.csv' # 'data/GA_progression_modelling/ga_area_comparison_results_2.csv'
x = 'GA Size Final'
y = 'mm_area' # 'total_area_mm'
df = pd.read_csv(data)
image_column = 'file_path_coris'
ga_seg_column = 'file_path_ga_seg'

df_growth = pd.read_csv('/sddata/data/GA_progression_modelling/ga_growth_rate_comparison_results_2.csv')  # Assuming 'growth_data.csv' has columns 'manual_growth_rate', 'ai_growth_rate', 'image1_path', 'segmentation1_path', 'image2_path', 'segmentation2_path'
x2 = 'manual'
y2 = 'AI'

# Function to overlay binary segmentation onto the original image
def overlay_segmentation(image_path, segmentation_path):
    try:
        # Open the original image (JPEG 2000) using Pillow
        with Image.open(image_path) as img:
            img = img.convert("RGBA")

        # Open the binary segmentation image (PNG) using OpenCV
        seg_img = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)

        # Create a contour from the segmentation image
        contours, _ = cv2.findContours(seg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an overlay for the contours
        overlay = Image.new('RGBA', img.size)
        draw = ImageDraw.Draw(overlay)
        
        for contour in contours:
            points = [tuple(point[0]) for point in contour]
            draw.line(points, fill="red", width=2)

        # Composite the original image with the overlay
        combined = Image.alpha_composite(img, overlay)
        
        # Convert the combined image to RGB before encoding
        combined = combined.convert("RGB")

        # Convert the image to bytes
        buffered = io.BytesIO()
        combined.save(buffered, format="JPEG")
        encoded = base64.b64encode(buffered.getvalue())
        return 'data:image/jpeg;base64,{}'.format(encoded.decode())
    except Exception as e:
        print(f"Error processing image {image_path} with segmentation {segmentation_path}: {e}")
        return ''

# Create a Dash application
app = dash.Dash(__name__)

# Create the original scatter plot using Plotly
fig1 = px.scatter(df, x=x, y=y, hover_data=[image_column, ga_seg_column])
fig1.update_layout(
    title="Manual vs AI-computed GA Area",
    xaxis_title=r"Manual GA Area (mm^2)",
    yaxis_title=r"AI-computed GA Area (mm^2)"
)

# Add the y=x line
line_trace = go.Scatter(
    x=[df[x].min(), df[y].max()],
    y=[df[x].min(), df[y].max()],
    mode='lines',
    line=dict(color='red', width=2),
    name='y=x'
)
fig1.add_trace(line_trace)

# Create the growth rates scatter plot using Plotly
fig2 = px.scatter(df_growth, x=x2, y=y2, hover_data=['file_path_coris_t1', 'file_path_seg_t1', 'file_path_coris_t2', 'file_path_seg_t2'])
fig2.update_layout(
    title="Manual vs AI-computed GA Growth Rate",
    xaxis_title=r"Manual Growth Rate (mm^2/month)",
    yaxis_title=r"AI-computed Growth Rate (mm^2/month)",
)

# Add the y=x line
line_trace = go.Scatter(
    x=[df_growth[x2].min(), df_growth[y2].max()],
    y=[df_growth[x2].min(), df_growth[y2].max()],
    mode='lines',
    line=dict(color='red', width=2),
    name='y=x'
)
fig2.add_trace(line_trace)

# Layout of the Dash application
app.layout = html.Div([
    html.Div([
        dcc.Graph(id='scatter-plot-1', figure=fig1),
        html.Img(id='hover-image-1', style={'width': '300px', 'height': '300px'})
    ], style={'display': 'inline-block', 'width': '48%'}),
    html.Div([
        dcc.Graph(id='scatter-plot-2', figure=fig2),
        html.Div([
            html.Img(id='hover-image-2-1', style={'width': '300px', 'height': '300px'}),
            html.Img(id='hover-image-2-2', style={'width': '300px', 'height': '300px'})
        ])
    ], style={'display': 'inline-block', 'width': '48%'})
])

# Callback to update the image on hover for the first scatter plot
@app.callback(
    Output('hover-image-1', 'src'),
    [Input('scatter-plot-1', 'hoverData')]
)
def display_image_1(hoverData):
    if hoverData is None:
        return ''
    # Get the image path and segmentation path from hover data
    image_path = hoverData['points'][0]['customdata'][0]
    segmentation_path = hoverData['points'][0]['customdata'][1]
    # Create overlay image
    overlaid_image = overlay_segmentation(image_path, segmentation_path)
    if not overlaid_image:
        print(f"Could not create overlay for image {image_path} with segmentation {segmentation_path}")
    return overlaid_image

# Callback to update the image on hover for the second scatter plot
@app.callback(
    [Output('hover-image-2-1', 'src'),
     Output('hover-image-2-2', 'src')],
    [Input('scatter-plot-2', 'hoverData')]
)
def display_image_2(hoverData):
    if hoverData is None:
        return '', ''
    # Get the image paths and segmentation paths from hover data
    image1_path = hoverData['points'][0]['customdata'][0]
    segmentation1_path = hoverData['points'][0]['customdata'][1]
    image2_path = hoverData['points'][0]['customdata'][2]
    segmentation2_path = hoverData['points'][0]['customdata'][3]
    # Create overlay images
    overlaid_image1 = overlay_segmentation(image1_path, segmentation1_path)
    overlaid_image2 = overlay_segmentation(image2_path, segmentation2_path)
    if not overlaid_image1:
        print(f"Could not create overlay for image {image1_path} with segmentation {segmentation1_path}")
    if not overlaid_image2:
        print(f"Could not create overlay for image {image2_path} with segmentation {segmentation2_path}")
    return overlaid_image1, overlaid_image2

# Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True)
