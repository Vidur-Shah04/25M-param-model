from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HPLT/HPLT2.0_cleaned",
    repo_type="dataset",
    allow_patterns=[
        "asm_Beng/*"
    ],
    local_dir="Assamese/data/raw/downloaded/HPLT2"
)
