# Music2Piano

Convert the audio I downloaded from YouTube into piano music.ฅ^•ω•^ฅ

# Requirement

* python 3

# Installation

```bash
pip install yt-dlp demucs basic-pitch pretty_midi mido

sudo apt update
sudo apt install ffmpeg fluidsynth

mkdir -p soundfonts
cd soundfonts

wget https://schristiancollins.com/soundfonts/GeneralUser_GS_1.471.sf2
mv GeneralUser_GS_1.471.sf2 piano.sf2
cd ..
```

# Usage

```bash
git clone https://github.com/hoge/~
python mk_piano.py --links links.txt --soundfont ./soundfonts/piano.sf2 --out output
```

# Note

Please do not try this with any illegally downloaded audio files.
Please be careful about copyright issues. The author assumes no responsibility whatsoever.

# Author

* malva ฅ^•ω•^ฅ


