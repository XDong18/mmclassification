export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29502 ./tools/dist_test.sh configs/resnet/resnet101_b8x4_bdd100k.py \
out/resnet101_bs24/latest.pth 4 --metrics accuracy