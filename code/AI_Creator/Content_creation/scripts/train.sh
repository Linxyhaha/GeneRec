## Please change the directories below to your own
export user="xylin"
export project_dir="generative_rec"
export code_folder="/storage/${user}/recommendation/${project_dir}/code/mcvd" # code folders
export logs_folder="/storage/${user}/recommendation/${project_dir}/code/logs" # where to output logs
export data_folder="/storage/${user}/recommendation/${project_dir}/data/huawei/huawei_h5/"
export exp_folder="${code_folder}"

export dir="${code_folder}"
cd ${dir}

# Video generation
export exp="huawei_big288_4c4_pmask50_unetm"
export config_mod="model.ngf=288 model.n_head_channels=288 data.prob_mask_cond=0.50 sampling.num_frames_pred=16 data.num_frames=4 data.num_frames_cond=4 training.batch_size=16 sampling.subsample=100 sampling.clip_before=True sampling.batch_size=100 sampling.max_data_iter=1 model.version=DDPM model.arch=unetmore"
export config="huawei"
export data="${data_folder}"
export devices="0,5"
export nfp="16"

sh ./scripts/finetune/run.sh