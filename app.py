from __future__ import unicode_literals

from textwrap import dedent
import dash_player
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import glob
from utils import logger
from predict_chords import *
import youtube_dl

save_dir = 'mp3s'
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'outtmpl': '{}/%(title)s.%(ext)s'.format(save_dir),
}



app = dash.Dash(__name__)
server = app.server

app.scripts.config.serve_locally = True

app.layout = html.Div([
    html.Div(
        style={
            'width': '40%',
            'float': 'left',
            'margin': '0% 5% 1% 5%'
        },
        children=[
            dash_player.DashPlayer(
                id='video-player',
                url='static/AtemiBebiBebiBabyBabys0f24HRlH4.mkv',
                controls=True,
                width='100%'
            ),
            html.Div(
                id='div-current-time',
                style={'margin': '10px 0px'}
            ),

            html.Div(
                id='div-method-output',
                style={'margin': '10px 0px'}
            ),
        ]
    ),

    html.Div(
        style={
            'width': '30%',
            'float': 'left'
        },
        children=[
            dcc.Input(
                id='input-url',
                value='static/AtemiBebiBebiBabyBabys0f24HRlH4.mkv'
            ),

            html.Button('Change URL', id='button-update-url'),

            dcc.Checklist(
                id='radio-bool-props',
                options=[{'label': val.capitalize(), 'value': val} for val in [
                    'playing',
                    'loop',
                    'controls',
                    'muted'
                ]],
                value=['controls']
            ),

            html.P("Volume:", style={'margin-top': '10px'}),
            dcc.Slider(
                id='slider-volume',
                min=0,
                max=1,
                step=0.05,
                value=None,
                updatemode='drag',
                marks={0: '0%', 1: '100%'}
            ),

            html.P("Playback Rate:", style={'margin-top': '25px'}),
            dcc.Slider(
                id='slider-playback-rate',
                min=0,
                max=4,
                step=None,
                updatemode='drag',
                marks={i: str(i) + 'x' for i in
                       [0, 0.25, 0.5, 0.75, 1, 2, 3, 4]},
                value=1
            ),

            html.P("Update Interval for Current Time:", style={'margin-top': '30px'}),
            dcc.Slider(
                id='slider-intervalCurrentTime',
                min=40,
                max=1000,
                step=None,
                updatemode='drag',
                marks={i: str(i) for i in [40, 100, 200, 500, 1000]},
                value=100
            ),

            html.P("Update Interval for seconds loaded:", style={'margin-top': '30px'}),
            dcc.Slider(
                id='slider-intervalSecondsLoaded',
                min=200,
                max=2000,
                step=None,
                updatemode='drag',
                marks={i: str(i) for i in [200, 500, 750, 1000, 2000]},
                value=500
            ),

            html.P("Update Interval for duration:",
                   style={'margin-top': '30px'}),
            dcc.Slider(
                id='slider-intervalDuration',
                min=200,
                max=2000,
                step=None,
                updatemode='drag',
                marks={i: str(i) for i in [200, 500, 750, 1000, 2000]},
                value=500
            ),

            html.P("Seek To:", style={'margin-top': '30px'}),
            dcc.Slider(
                id='slider-seek-to',
                min=0,
                max=1,
                step=None,
                updatemode='drag',
                marks={i: str(i * 100) + '%' for i in [0, 0.25, 0.5, 0.75, 1]},
                value=0
            ),
        ]
    ),
])


@app.callback(Output('video-player', 'playing'),
              [Input('radio-bool-props', 'value')])
def update_prop_playing(value):
    return 'playing' in value


@app.callback(Output('video-player', 'loop'),
              [Input('radio-bool-props', 'value')])
def update_prop_loop(value):
    return 'loop' in value


@app.callback(Output('video-player', 'controls'),
              [Input('radio-bool-props', 'value')])
def update_prop_controls(value):
    return 'controls' in value


@app.callback(Output('video-player', 'muted'),
              [Input('radio-bool-props', 'value')])
def update_prop_muted(value):
    return 'muted' in value


@app.callback(Output('video-player', 'volume'),
              [Input('slider-volume', 'value')])
def update_volume(value):
    return value


@app.callback(Output('video-player', 'playbackRate'),
              [Input('slider-playback-rate', 'value')])
def update_playbackRate(value):
    return value


@app.callback(Output('video-player', 'url'),
              [Input('button-update-url', 'n_clicks')],
              [State('input-url', 'value')])
def update_url(n_clicks, value):

    logger.info(value)

    if 'youtube' in value:

        logger.info('is a youtube link')

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            # ydl.download([value])
            info_dict = ydl.extract_info(value, download=True)
            fn = ydl.prepare_filename(info_dict)

        # song = glob.glob('*.mp3')[-1]

        song = os.path.splitext(fn)[0] + '.mp3'

        filename = run(song)

    else:

        logger.info('not a youtube link')

        filename = value    
    
    return filename


# Instance Methods
@app.callback(Output('div-current-time', 'children'),
              [Input('video-player', 'currentTime')])
def update_time(currentTime):
    return 'Current Time: {}'.format(currentTime)


@app.callback(Output('div-method-output', 'children'),
              [Input('video-player', 'secondsLoaded')],
              [State('video-player', 'duration')])
def update_methods(secondsLoaded, duration):
    return 'Second Loaded: {}, Duration: {}'.format(secondsLoaded, duration)


@app.callback(Output('video-player', 'intervalCurrentTime'),
              [Input('slider-intervalCurrentTime', 'value')])
def update_intervalCurrentTime(value):
    return value


@app.callback(Output('video-player', 'intervalSecondsLoaded'),
              [Input('slider-intervalSecondsLoaded', 'value')])
def update_intervalSecondsLoaded(value):
    return value


@app.callback(Output('video-player', 'intervalDuration'),
              [Input('slider-intervalDuration', 'value')])
def update_intervalDuration(value):
    return value


@app.callback(Output('video-player', 'seekTo'),
              [Input('slider-seek-to', 'value')])
def set_seekTo(value):
    return value


if __name__ == '__main__':
    app.run_server(debug=True)