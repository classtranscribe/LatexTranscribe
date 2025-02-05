from huggingface_hub import snapshot_download

snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir='./', allow_patterns='models/Layout/*')
snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir='./', allow_patterns='models/MFD/*')
snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir='./', allow_patterns='models/MFR/*')
snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir='./', allow_patterns='models/TabRec/*')