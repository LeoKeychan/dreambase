CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=/meta_eon_cfs/home/lqc/dreambase \
/meta_eon_cfs/home/szj/miniconda3/envs/dream/bin/python \
  eval_utils/serve_dreambase_policy.py \
  --model-path checkpoints/robotwin_threeview_baseline_16train_full_merged/checkpoint-35000 \
  --device cuda:0 \
  --port 8010 \
  --master-port 29653 \
  --save-video-dir video_pred/ckpt35000 \
  --save-video-fps 10 \
  --persistent-source-cache