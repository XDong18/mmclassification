export CUDA_VISIBLE_DEVICES=4,5,6,7
PORT=29502 ./tools/dist_test.sh configs/resnet/resnet101_b8x4_bdd100k.py \
out/resnet101_bs24/epoch_50.pth 4 --metrics accuracy