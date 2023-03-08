###############
## Arguments ##
###############

## Please change the directories below to your own
export user="XXX"
export project_dir="XXX"
export code_folder="path of cold folder" # code folders
export logs_folder="path of log folder" # where to output logs
export data_folder="path of h5 data folder"
export exp_folder="${code_folder}"

export dir="${code_folder}"
cd ${dir}

###############
##  Micro Video  ##
###############
export config="microVideo_edit"
export data="${data_folder}"
export nfp="4"

export exp="microVideo_cfree_edit-exp4"
export exp=${exp_folder}/${exp}
export config_mod="data.prob_mask_cond=0.50 model.ngf=288 model.n_head_channels=288 data.num_frames=4 data.num_frames_cond=4 training.batch_size=32 sampling.batch_size=100 sampling.max_data_iter=99999 model.arch=unetmore"
export pretrained="path_of_pretrained_model"

export edit_T="1000"
export denoise_T="1000"
export subsample="100"
export ckpt="100"
sh ./scripts/edit_base.sh


