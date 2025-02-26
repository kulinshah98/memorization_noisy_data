# Clean up Python cache files
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

# Path to model checkpoint
CKPT=logs/00000-ffhq-64x64-300-uncond-ncsnpp-edm-gpus4-batch256-fp32-UuVhK/network-snapshot-000000.pkl

# Directory to save generated samples
OUTDIR=logs/generated_samples

# Path to reference statistics for FID calculation
REF_PATH=fid-refs/ffhq-64x64.npz

# Generate 50k samples using 4 GPUs
torchrun --standalone --nproc_per_node=4 generate.py \
    --outdir=$OUTDIR \
    --seeds=0-50175 \
    --batch=256 \
    --network=$CKPT \
    --steps=40

# Calculate FID score using generated samples
torchrun --standalone --nproc_per_node=1 fid.py calc \
    --images=$OUTDIR \
    --ref=$REF_PATH \
    --num=50000 \
    --batch=64
