export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29502 ./tools/dist_test.sh configs/dla/dla34_bdd100k.py \
out/dla34_bs48/latest.pth 4 --metrics accuracy