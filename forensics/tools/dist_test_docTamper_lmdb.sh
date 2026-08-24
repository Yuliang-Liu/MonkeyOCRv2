CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
while true; do
  PORT=$((RANDOM % 300 + 29500))
  if ! lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    break
  fi
done
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PYTHONWARNINGS="ignore"
export LOGGING_LEVEL=INFO

export compress_pk_file='./data/DocTamperV1/pks/DocTamperV1-FCD_75.pk'
export val_db_path='./data/DocTamperV1/unzip_files/DocTamperV1-FCD'
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --cfg-options \
      visualizer.vis_backends=[] \
      test_dataloader.num_workers=8 \
      test_dataloader.dataset.ann_file="processed/fcd.txt" \
      test_dataloader.batch_size=4

export compress_pk_file='./data/DocTamperV1/pks/DocTamperV1-SCD_75.pk'
export val_db_path='./data/DocTamperV1/unzip_files/DocTamperV1-SCD'
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --cfg-options \
      visualizer.vis_backends=[] \
      test_dataloader.num_workers=8 \
      test_dataloader.dataset.ann_file="processed/scd.txt" \
      test_dataloader.batch_size=4

export compress_pk_file='./data/DocTamperV1/pks/DocTamperV1-TestingSet_75.pk'
export val_db_path='./data/DocTamperV1/unzip_files/DocTamperV1-TestingSet'
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    --cfg-options \
      visualizer.vis_backends=[] \
      test_dataloader.num_workers=8 \
      test_dataloader.dataset.ann_file="processed/val.txt" \
      test_dataloader.batch_size=4