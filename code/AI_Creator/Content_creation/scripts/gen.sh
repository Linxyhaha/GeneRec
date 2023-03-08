###############
## Arguments ##
###############

## Please change the directories below to your own
export user="XXX"
export project_dir="XXX"
export code_folder="path of code folder" # code folders
export logs_folder="path of log folder" # where to output logs
export data_folder="path of h5 data folder"
export exp_folder="${code_folder}"

export dir="${code_folder}"
cd ${dir}

###############
##  MicroVideo  ##
###############
export config="microVideo_gen"
export data="${data_folder}"
export nfp="4"
export ckpt="0"


export exp="microVideo_cfree_gen-8f-fvd" # group user, nfp=12, other same as exp2
export exp=${exp_folder}/${exp}
export config_mod="data.prob_mask_cond=0.50 model.ngf=288 model.n_head_channels=288 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32 sampling.batch_size=10 sampling.max_data_iter=99999 model.arch=unetmore"
export pretrained="path of pretrained model"

export edit_T="1000"
export denoise_T="1000"
export subsample="100"
export nfp="12"
sh ./scripts/gen_base.sh

