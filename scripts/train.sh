export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29502 ./tools/dist_train.sh configs/dla/dla34_bdd100k.py 4 --work-dir out/dla34_bs48 \
--resume-from out/dla34_bs48/latest.pth