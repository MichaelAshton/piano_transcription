Using Deep Learning to estimate chords in songs

## Overview

This project aims at generating automated piano tutorials from songs. Once a youtube link is provided, a bidirectional transformer model predicts the chords, the predicted chords are converted to midi and a video is generated showing the chords' progression in a synthesia-like format. The uploaded song is trimmed to the first ~40 seconds to constrain the execution time to under 2 minutes. (The project is a Work-In-Progress.)


## Installation (ubuntu)

1. Install anaconda: https://www.anaconda.com/distribution/  (alternative you can install miniconda)


2. Run:  

```bash
sudo apt-get install libsndfile1 ffmpeg libcairo2

git clone https://github.com/MichaelAshton/piano_transcription.git

cd piano_transcription

conda env update -f environment.yml

source activate piano

python app.py
```

open the link provided (i.e http://127.0.0.1:8050)

## TO DO:

 1. Play the original song alongside the generated chords video
 2. Set the chords to start playing after ~2 seconds instead of immediately the video starts
 3. Incorporate prediction of the singing voice to be played using the right hand

