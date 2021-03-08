export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=29502 ./tools/dist_train.sh bdd_mtl/configs/mtl/bdd_dla34up_s.py 4 --work-dir out/dla34up_1x 