CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

while true; do
  PORT=$((RANDOM % 300 + 29500))
  if ! lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    break
  fi
done

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

CKPT_DIR=$(python -c "from mmengine.config import Config; cfg = Config.fromfile('$CONFIG'); print(cfg.work_dir)")
max_iters=$(python -c "from mmengine.config import Config; cfg = Config.fromfile('$CONFIG'); print(cfg.train_cfg.max_iters)")

if [ -d "$CKPT_DIR" ]; then
  LASTEST_CKPT=$(ls $CKPT_DIR/iter_*.pth | sort -V | tail -n 1)
  BEST_CKPT=$CKPT_DIR/best_checkpoint.pth
else
  LASTEST_CKPT=""
  BEST_CKPT=""
fi

echo "latest checkpoint: $LASTEST_CKPT"
echo "best checkpoint: $BEST_CKPT"

# if latest checkpoint is exist, and iters is equal to max_iters
if [ ! -f "$LASTEST_CKPT" ] || [ ! "$(basename $LASTEST_CKPT | cut -d '_' -f 2 | cut -d '.' -f 1)" -eq $max_iters ]; then
  echo "latest checkpoint is not exist or iters is not equal to max_iters"
      PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
      python -m torch.distributed.launch \
          --nnodes=$NNODES \
          --node_rank=$NODE_RANK \
          --master_addr=$MASTER_ADDR \
          --nproc_per_node=$GPUS \
          --master_port=$PORT \
          $(dirname "$0")/train.py \
          $CONFIG \
          --launcher pytorch --resume ${@:3}
fi

LASTEST_CKPT=$(ls $CKPT_DIR/iter_*.pth | sort -V | tail -n 1)
echo "start to test docTamper"
echo "bash tools/dist_test_docTamper_lmdb.sh $CONFIG $LASTEST_CKPT $GPUS"
bash tools/dist_test_docTamper_lmdb.sh $CONFIG $LASTEST_CKPT $GPUS