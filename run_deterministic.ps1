# Run pipeline with deterministic settings (oneDNN disabled)
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:PYTHONHASHSEED = "42"
$env:TF_DETERMINISTIC_OPS = "1"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

# Run pipeline
python scripts/run_pipeline.py @args
