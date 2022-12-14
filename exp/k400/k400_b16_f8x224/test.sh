NUM_SHARDS=1
NUM_GPUS=1
BATCH_SIZE=1
BASE_LR=1e-5
work_path="/home/ammara/uniformerv2/UniFormerV2/exp/k400/k400_b16_f8x224"
PYTHONPATH="/home/ammara/uniformerv2/UniFormerV2/slowfast" 
python tools/run_net.py \
  --init_method tcp://localhost:10125 \
  --cfg "/home/ammara/uniformerv2/UniFormerV2/exp/k400/k400_b16_f8x224/config.yaml" \
  --num_shards 1 \
  DATA.PATH_TO_DATA_DIR "/home/ammara/uniformerv2/UniFormerV2/dataset" \
  DATA.PATH_PREFIX you_data_path/k400 \
  DATA.PATH_LABEL_SEPARATOR " " \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 100 \
  TRAIN.BATCH_SIZE 1 \
  TRAIN.SAVE_LATEST False \
  NUM_GPUS 1 \
  NUM_SHARDS 1 \
  SOLVER.MAX_EPOCH 55 \
  SOLVER.BASE_LR 1e-5 \
  SOLVER.BASE_LR_SCALE_NUM_SHARDS False \
  SOLVER.WARMUP_EPOCHS 5. \
  TRAIN.ENABLE False \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.TEST_BEST False \
  TEST.ADD_SOFTMAX True \
  TEST.BATCH_SIZE 1 \
  TEST.CHECKPOINT_FILE_PATH "/home/ammara/uniformerv2/UniFormerV2/pretrained_models/k400_uniformerv2_b16_8x224.pyth" \
  RNG_SEED 6666 \
  OUTPUT_DIR "/home/ammara/uniformerv2/UniFormerV2/exp/k400/k400_b16_f8x224" \
  TENSORBOARD.ENABLE True \
  OUTPUT_DIR "/home/ammara/uniformerv2/UniFormerV2/exp/k400/k400_b16_f8x224" \
  TEST.SAVE_RESULTS_PATH "results.pkl" 