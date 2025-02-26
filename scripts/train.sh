OUTDIR=logs/
DATASET=ffhq-64x64-300.zip
NPROC_PER_NODE=4
EXPNAME=test_upload_code

TRAINING_DURATION=30
SIGMA=0.5
nature_noise_fixed=True
LOSS_TYPE=ambient_highnoise_edm_lownoise


# Clean up Python cache files
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

# Launch distributed training with specified number of GPUs
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train.py \
    --outdir=$OUTDIR \                  # Directory to save model checkpoints and logs
    --data=$DATASET \                   # Path to training dataset
    --cond=0 \                          # Unconditional generation (no class labels)
    --arch=ncsnpp \                     # Neural network architecture
    --batch=256 \                       # Total batch size across all GPUs
    --duration=$TRAINING_DURATION \      # Total training duration
    --cres=1,2,2,2 \                    # Channel multipliers for each resolution
    --dropout=0.05 \                    # Dropout probability
    --augment=0.15 \                    # Data augmentation probability  
    --lr=1e-4 \                         # Learning rate
    --sigma=$SIGMA \                    # Nature noise level sigma
    --nature_noise_fixed=$nature_noise_fixed \  # Whether to use nature noise fixed for each example
    --tick=100 \                        # Checkpoint interval
    --loss_type=$LOSS_TYPE \            # Type of training loss (Options: edm (baseline), ambient_highnoise_edm_lownoise)
    --expr_id={$EXPNAME}_ffhq_cp{$CORRUPT_PROB}_sigma{$SIGMA}_dp{$DP}  # Experiment ID
