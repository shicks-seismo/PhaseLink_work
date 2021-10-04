#PBS -lselect=1:ncpus=16:mem=96gb:ngpus=4:gpu_type=RTX6000
#PBS -lwalltime=00:30:00
source ~/.bashrc
conda activate easyquake
dir=/rds/general/user/shicks17/home/PhaseLink/
cd $dir
python phaselink_train.py params.json
