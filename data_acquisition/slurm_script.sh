#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=8G

pip install -r requirements.txt
python play_by_play_api_script.py 2017-18 out.feather
