from huggingface_hub import snapshot_download

files = snapshot_download(
    repo_id="cfilt/IITB-IndicMonoDoc",
    repo_type="dataset",
    local_dir="./IITB-IndicMonoDoc",
    allow_patterns="gu/**",
)