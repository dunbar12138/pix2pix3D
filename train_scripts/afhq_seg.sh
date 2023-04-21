# Train with AFHQ using 8 GPUs.
NCCL_P2P_DISABLE=1 python train.py --outdir=logs \
                --cfg=afhq --data=data/afhq_v2_train_cat_512.zip \
                --mask_data=data/afhqcat_seg_6c.zip \
                --data_type=seg --semantic_channels=6 \
                --render_mask=True --dis_mask=True \
                --neural_rendering_resolution_initial=128 \
                --resume=ckpts/afhqcats512-128.pkl \
                --gpus=2 --batch=4 --mbstd-group=2 \
                --gamma=5 --gen_pose_cond=True \
                --random_c_prob=0.5 \
                --lambda_d_semantic=0.1 \
                --lambda_lpips=1 \
                --lambda_cross_view=1e-4 \
                --only_raw_recons=True \
                --wandb_log=False
