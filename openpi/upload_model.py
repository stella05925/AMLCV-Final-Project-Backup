from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path="/home/stella/projects/openpi/checkpoints/pi0_libero_low_mem_finetune", repo_id="stellaaaa/Pi0_vggt_libero_spatial_5k", repo_type="model")
