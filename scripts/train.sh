export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29502 ./tools/dist_train.sh configs/resnet/resnet101_b8x4_bdd100k.py 4 --work-dir out/resnet101_bs32