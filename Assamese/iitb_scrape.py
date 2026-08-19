from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="cfilt/IITB-IndicMonoDoc",
    repo_type="dataset",
    allow_patterns=[
        "as/*"
    ],
    local_dir="Assamese/data/raw/downloaded/IITB-IndicMonoDoc"
)