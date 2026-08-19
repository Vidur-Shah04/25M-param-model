from huggingface_hub import hf_hub_download
from pathlib import Path

output_dir = Path(
    r"Assamese\data\raw\downloaded\IndicCorpV2"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

path = hf_hub_download(
    repo_id="ai4bharat/IndicCorpV2",
    filename="data/as.txt",
    repo_type="dataset",
    local_dir=output_dir
)

print("Downloaded:")
print(path)