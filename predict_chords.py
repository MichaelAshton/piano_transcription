import os
import glob
import shutil
from utils import logger
from btc_model import *
from utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord
import pandas as pd
from music21 import *
import midi
from midi2audio import FluidSynth
import subprocess
import video

logger.logging_verbosity(1)
# use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

config = HParams.load("run_config.yaml")

def run(mp3_file_path):

    print('run function mp3_file_path argument : {}'.format(mp3_file_path))

    voca = False # True means large vocabulary label type
    if voca == True:
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170
        model_file = 'test/btc_model_large_voca.pt'
        idx_to_chord = idx2voca_chord()
        logger.info("label type: large voca")
    else:
        model_file = 'test/btc_model.pt'
        idx_to_chord = idx2chord
        logger.info("label type: Major and minor")

    model = BTC_model(config=config.model).to(device)

    # Load model
    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file, map_location='cpu')
        mean = checkpoint['mean']
        std = checkpoint['std']
        model.load_state_dict(checkpoint['model'])
        logger.info("restore model")

    # Load mp3
    # example mp3 file is free music audio from http://www.freemusicarchive.org/
    # Monplaisir_-_01_-_Everything_is_true
    # mp3_file_path = glob.glob('./*.mp3')[0]

    base_path, song_name = os.path.split(mp3_file_path)

    
    new_name = "".join(x for x in song_name[:-4] if x.isalnum())
    new_path = os.path.join(base_path, new_name + '.mp3')

    shutil.move(mp3_file_path, new_path)
    
    filename = new_path[:-4]
    print('cleaned filename : {}'.format(filename))
    # filename = 'stereo_hearts'
    # mp3_file_path = 'test/{}.mp3'.format(filename)
    # mp3_file_path = './{}.mp3'.format(filename)
    feature, feature_per_second, song_length_second = audio_file_to_features(new_path, config)
    logger.info("audio file loaded and feature computation success")

    # Majmin type chord recognition
    feature = feature.T
    feature = (feature - mean) / std
    time_unit = feature_per_second
    n_timestep = config.model['timestep']

    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feature.shape[0] // n_timestep

    start_time = 0.0
    lines = []
    with torch.no_grad():
        model.eval()
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        for t in range(num_instance):
            self_attn_output, _ = model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
            prediction, _ = model.output_layer(self_attn_output)
            prediction = prediction.squeeze()
            for i in range(n_timestep):
                if t == 0 and i == 0:
                    prev_chord = prediction[i].item()
                    continue
                if prediction[i].item() != prev_chord:
                    lines.append(
                        '%.6f %.6f %s\n' % (
                            start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                    start_time = time_unit * (n_timestep * t + i)
                    prev_chord = prediction[i].item()
                if t == num_instance - 1 and i + num_pad == n_timestep:
                    if start_time != time_unit * (n_timestep * t + i):
                        lines.append(
                            '%.6f %.6f %s\n' % (
                                start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                    break

    # lab file write
    # test_result_path = 'test/{}.lab'.format(filename)
    test_result_path = './{}.lab'.format(filename)
    with open(test_result_path, 'w') as f:
        for line in lines:
            f.write(line)

    logger.info("label file saved")


    print('read in label file to pandas dataframe')
    df = pd.read_csv('./{}.lab'.format(filename), header=None, delimiter=' ')

    df.columns = ['start', 'stop', 'chord']

    df['duration'] = df.stop - df.start

    df = df.iloc[1:-2].copy(deep=True)

    for index in df.index.values:
    
        chord_ = df.at[index, 'chord']
        
        if chord_ == 'N':
            
            timestamp = df.at[index, 'stop']
            df.at[index-1, 'stop'] = timestamp
            df.drop(index, inplace=True)

    print('start processing the chords into midi')
    try:
        s1 = stream.Stream()
        s1.append(chord.Chord(["C4","G4","E-5"]))
        for index in df.index.values[1:20]:#[1:-2]:
        #     print(index)
            chord_ = df.at[index, 'chord']
            kind = 'major'
            if ':min' in chord_:
                kind = 'minor'
            chord_ = chord_.split(':min')[0]
            duration_ = 2 * df.at[index, 'duration']
            # duration_ = df.at[index, 'duration']
            chord21 = harmony.ChordSymbol(root=chord_, kind = kind, duration=duration_)
            chord21.writeAsChord = True
            s1.append(chord21)

    except Exception as e:
        print(e)

    print('complete')

    print('save midi to disk')
    fp = s1.write('midi', fp='{}.mid'.format(filename))

    sheet = midi.Midi('{}.mid'.format(filename))
    clip = video.midi_videoclip(sheet)
    clip.write_videofile('{}.webm'.format(filename), codec='libvpx', fps=20)

    fs = FluidSynth('/usr/share/sounds/sf2/FluidR3_GM.sf2') # arch
    fs.midi_to_audio('{}.mid'.format(filename), '{}.wav'.format(filename))

    cmd = 'ffmpeg -y -i {}.wav  -r 30 -i {}.webm  -filter:a aresample=async=1 -c:a flac -c:v copy {}.mkv'.format(filename, filename, filename)
    subprocess.call(cmd, shell=True)                                     # "Muxing Done
    print('Muxing Done')

    cmd = 'ffmpeg -i {}.mkv -filter:a "atempo=0.96" -vn -y {}.wav'.format(filename, filename)
    subprocess.call(cmd, shell=True)

    filename_only = os.path.splitext(os.path.basename(filename))[0]

    cmd = 'ffmpeg -y -i {}.wav  -r 30 -i {}.webm  -filter:a aresample=async=1 -c:a flac -c:v copy static/{}.mkv'.format(filename, filename, filename_only)
    subprocess.call(cmd, shell=True)                                     # "Muxing Done
    print('Muxing Done')

    return 'static/{}.mkv'.format(filename_only)


