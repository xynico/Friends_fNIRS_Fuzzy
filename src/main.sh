# PYTHON=/home/lou/Data/Liang_13060835/envs/fNIRS/bin/python
PYTHON=/projects/CIBCIGroup/00DataUploading/Xiaowei/anaconda/dl_pl/dl_pl/bin/python
# export PATH=/home/lou/Data/Liang_13060835/envs/cuda-11.0/bin:$PATH
# export LD_LIBRARY_PATH=/home/lou/Data/Liang_13060835/envs/cuda-11.0/lib64:/home/lou/Data/Liang_13060835/envs/cuda-11.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# config path
# CFG=/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/src/shallowmind/configs/experiment/temp/FuzzyTramsformer-Class/FuzzyTramsformer_num_rules1_num_heads3_dataset_name=PictureClass_label=emotion_base_lr=1.5e-3_depth=3_batch_size=256.py
# CFG=/home/lou/Data/Liang_13060835/projects/fNIRS_fnn/Friends_fNIRS_Fuzzy/src/shallowmind/configs/fNIRS_Emo_tramsformer-AllData_V0_step2.py
# CFG=/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/src/shallowmind/configs/experiment/temp/FuzzyTramsformer-Reversed/1/FuzzyTramsformer_num_rules1_num_heads5_dataset_name=PictureClass_label=emotion_base_lr=3e-2_depth=3_batch_size=256.py
# CFG=/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/src/shallowmind/configs/FNN_exp.py
# CFG=/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/src/shallowmind/configs/Fuzzy_norm.py
CFG=/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/src/shallowmind/configs/ToyFuzzy_config.py
# CFG=/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/src/shallowmind/configs/experiment/temp/FuzzyTramsformer_ALL.py
# CFG=/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/src/shallowmind/configs/experiment/temp/FuzzyTramsformer-Between/FuzzyTramsformer_num_rules1_num_heads3_dataset_name=PictureRating_label=relationship_base_lr=1.5e-2_depth=2_batch_size=128.py
# CFG=/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/src/shallowmind/configs/experiment/temp/FuzzyTramsformer_ALL-Between/FuzzyTramsformer_ALL_num_rules1_num_heads3_dataset_name=PictureRating_label=relationship_base_lr=1.5e-2_depth=2_batch_size=128.py
# CFG=/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/src/shallowmind/configs/experiment/temp/FuzzyTramsformer-Between-Reversed/FuzzyTramsformer_num_rules1_num_heads5_dataset_name=PictureClass_label=relationship_base_lr=1.5e-4_depth=2_batch_size=128.py
# CFG=/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/src/shallowmind/configs/experiment/temp/FuzzyTramsformer_Cross/FuzzyTramsformer_Cross_num_rules1_num_heads3_dataset_name=PictureRating_label=emotion_base_lr=1.5e-2_depth=3_batch_size=128.py
SEED=42

GPU_IDS=1

clear
$PYTHON /projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/src/shallowmind/api/train.py --cfg=$CFG --seed=$SEED --gpu_ids=$GPU_IDS