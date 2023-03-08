## Please change the directories below to your own
export user="XXX"
export project_dir="XXX"
export code_folder="paht of code folder" # code folders
export logs_folder="path of log folder" # where to output logs
export data_folder="paht of h5 data folder"
export exp_folder="${code_folder}"

export dir="${code_folder}"
cd ${dir}

# Video generation
export exp="microVideo_big288_4c4_pmask50_unetm"
export config_mod="model.ngf=288 model.n_head_channels=288 data.prob_mask_cond=0.50 sampling.num_frames_pred=16 data.num_frames=4 data.num_frames_cond=4 training.batch_size=16 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
export config="MicroVideo"
export data="${data_folder}"
export devices="0,5"
export nfp="16"

sh ./scripts/finetune/run.sh