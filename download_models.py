from huggingface_hub import snapshot_download

snapshot_download(repo_id='opendatalab/pdf-extract-kit-1.0', local_dir='./',
                  allow_patterns=['models/Layout/*', 'models/MFD/*', 'models/MFR/*', 'models/TabRec/*'])
