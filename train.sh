
VERSION="your version"
# sysu
python train.py --dataset sysu --lr 0.2 --log_path log/sysu/$VERSION/ --model_path save_model/sysu/$VERSION/

# sysu indoor search
python test.py --dataset sysu --log_path log/sysu/$VERSION/ --resume $VERSION/sysu/sysu_p6_n4_lr_0.2_seed_0_best.pth --mode indoor

# sysu all search
python test.py --dataset sysu --log_path log/sysu/$VERSION/ --resume $VERSION/sysu/sysu_p6_n4_lr_0.2_seed_0_best.pth --mode all


# llcm
python train.py --dataset llcm --lr 0.2 --log_path log/llcm/$VERSION/ --model_path save_model/llcm/$VERSION/

# llcm visible to infrared
python test.py --dataset llcm --log_path log/llcm/$VERSION/ --resume $VERSION/llcm/llcm_p6_n4_lr_0.2_seed_0_best.pth --tvsearch 1

# llcm infrared to visible
python test.py --dataset llcm --log_path log/llcm/$VERSION/ --resume $VERSION/llcm/llcm_p6_n4_lr_0.2_seed_0_best.pth --tvsearch 0



# regdb
python train.py --dataset regdb --lr 0.1 --trial 1 --log_path log/regdb/$VERSION/ --model_path save_model/regdb/$VERSION/
python train.py --dataset regdb --lr 0.1 --trial 2 --log_path log/regdb/$VERSION/ --model_path save_model/regdb/$VERSION/
python train.py --dataset regdb --lr 0.1 --trial 3 --log_path log/regdb/$VERSION/ --model_path save_model/regdb/$VERSION/
python train.py --dataset regdb --lr 0.1 --trial 4 --log_path log/regdb/$VERSION/ --model_path save_model/regdb/$VERSION/
python train.py --dataset regdb --lr 0.1 --trial 5 --log_path log/regdb/$VERSION/ --model_path save_model/regdb/$VERSION/
python train.py --dataset regdb --lr 0.1 --trial 6 --log_path log/regdb/$VERSION/ --model_path save_model/regdb/$VERSION/
python train.py --dataset regdb --lr 0.1 --trial 7 --log_path log/regdb/$VERSION/ --model_path save_model/regdb/$VERSION/
python train.py --dataset regdb --lr 0.1 --trial 8 --log_path log/regdb/$VERSION/ --model_path save_model/regdb/$VERSION/
python train.py --dataset regdb --lr 0.1 --trial 9 --log_path log/regdb/$VERSION/ --model_path save_model/regdb/$VERSION/
python train.py --dataset regdb --lr 0.1 --trial 10 --log_path log/regdb/$VERSION/ --model_path save_model/regdb/$VERSION/

# regdb infrared to visible
python test.py --dataset regdb --tvsearch 0 --log_path log/regdb/$VERSION/ --model_path save_model/regdb/$VERSION/

# regdb visible to infrared
python test.py --dataset regdb --tvsearch 1 --log_path log/regdb/$VERSION/ --model_path save_model/regdb/$VERSION/