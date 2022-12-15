work_path="/home/omnoai/Desktop/ammara/uniformerv2/uniformerv2/exp/k400/k400_b16_f8x224"
PYTHONPATH="/home/omnoai/Desktop/ammara/uniformerv2/uniformerv2/slowfast" 
python uniformerv2/tools/run_net.py \
  --init_method tcp://localhost:10125 \
  --cfg "/home/omnoai/Desktop/ammara/uniformerv2/uniformerv2/exp/k400/k400_b16_f8x224/config.yaml" \
  --num_shards 1 \
  DATA.PATH_TO_DATA_DIR "/home/omnoai/Desktop/ammara/uniformerv2/uniformerv2/dataset" \
  DATA.PATH_PREFIX you_data_path/k400 \
  DATA.PATH_LABEL_SEPARATOR " " \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 100 \
  TRAIN.BATCH_SIZE 2 \
  TRAIN.SAVE_LATEST False \
  NUM_GPUS 1 \
  NUM_SHARDS 1 \
  SOLVER.MAX_EPOCH 55 \
  SOLVER.BASE_LR 1e-5 \
  SOLVER.BASE_LR_SCALE_NUM_SHARDS False \
  SOLVER.WARMUP_EPOCHS 5. \
  TRAIN.ENABLE True \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.TEST_BEST True \
  TEST.ADD_SOFTMAX True \
  TEST.BATCH_SIZE 3 \
  RNG_SEED 6666 \
  OUTPUT_DIR "/home/omnoai/Desktop/ammara/uniformerv2/uniformerv2/exp/k400/k400_b16_f8x224" \
  TENSORBOARD.ENABLE True \
  TEST.SAVE_RESULTS_PATH "results.pkl" \
  DATA.SAMPLING_RATE 4 \
  DATA.NUM_FRAMES 62 \
  MODEL.NUM_CLASSES 3 \
  DATA.TRAIN_CROP_SIZE 512