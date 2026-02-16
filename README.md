# uvr-cli
Ultimate Vocal Remover Command Line Interface and python project that can be imported by other projects.

This project is modified from [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) by [Anjok07](https://github.com/Anjok07) and its dependencies have been stripped down only to the essentials. This has been tested to work on Linux + AMD on Python 3.14. Other platforms probably work but I'm unable to test them.

Currently, the only supported model is [MDX23C-8KFFT-InstVoc_HQ](https://huggingface.co/datasets/SayanoAI/RVC-Studio/resolve/main/karafan/MDX23C-8KFFT-InstVoc_HQ.ckpt). The first time the script is invoked, it will download the model, or you may manually download it [here](https://huggingface.co/datasets/SayanoAI/RVC-Studio/tree/main/karafan) and place it in the directory `models/karafan`.

# Installation
This project uses [pytorch](https://pytorch.org/). Before installing, make sure the proper pytorch version for your system has been installed.

## As a python module

In your active virtual environment:
```
git clone https://github.com/chameleon-ai/uvr-cli.git
cd uvr-cli
pip install -e .
```

Or add this to your project's requirements.txt:
```
git+https://github.com/chameleon-ai/uvr-cli.git
```

## As a stand-alone script
```
git clone https://github.com/chameleon-ai/uvr-cli.git
cd uvr-cli
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Usage
From within a python script:
```
from uvr_cli import uvr_separate
# input is a path to a file and return values are paths to separated stems
vocal_stem, output_instrumental_stem = uvr_separate(input_filename)

# Explicitly specify the model directory
vocal_stem, output_instrumental_stem = uvr_separate(input_filename, model_dir = "./models")
```

or as a CLI:
```
python src/uvr_cli.py input.mp3
```