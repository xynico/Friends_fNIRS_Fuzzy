data_root='/data/xiaowjia/Friends_fNIRS/data'
work_dir='/data/xiaowjia/Friends_fNIRS/work_dir_2'
# data_root='/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/dataset'
# work_dir='/projects/CIBCIGroup/00DataUploading/Liang/fuzzy_fnirs/Friends_fNIRS_Fuzzy/work_dir'
dataset_name='PictureRating'

dataset_para=dict(
    type='fNIRS_Emo_CL_all',
    data_root=data_root+f'/step1_PPCS_0.01_0.2{dataset_name}_data.pkl',
    test_size=0.3,
    dataset_random_state=42,
    temp_save_folder=f"{data_root}/temp",
    label='relationship',
    subject_split='within',

)
# dataset_para.update()

# model settings
base_lr=1.5e-4
batch_size=128
lr=base_lr * batch_size*2 / 256

model_para=dict(
    img_size=(40, 33), # [H, W]
    num_heads=11,
    depth=2,
    cl_embed_dim=64,
    encoder_type='Transformer',
    dropout=0.3,
    model_ckpt=None,
    fixed=False,
    
)

exp_name=f"baseline_TFe-ds={dataset_name}_{dataset_para['type'][-1]}-d={model_para['depth']}-nh={model_para['num_heads']}-bs={batch_size}-ls={base_lr}-label={dataset_para['label']}"

evaluation=dict(metrics=[
    dict(type='TorchMetrics', metric_name='Accuracy', multi_label=False, num_classes=2, prob=False, task='multiclass'),
    dict(type='TorchMetrics', metric_name='Precision', multi_label=False, num_classes=2, prob=False, task='multiclass'),
    dict(type='TorchMetrics', metric_name='Recall', multi_label=False, num_classes=2, prob=False, task='multiclass'),
    dict(type='TorchMetrics', metric_name='F1Score', multi_label=False, num_classes=2, prob=False, task='multiclass'),
])

model=dict(
    # Base Encoder Decoder Architecture (Backbone+Head)
    type='EEGTransClassifer_explainable',
    pretrained=False,
    evaluation=evaluation,
    **model_para,
)

data=dict(
    train_batch_size=batch_size, # for each gpu
    val_batch_size=int(batch_size),
    test_batch_size=int(batch_size),
    num_workers=24,

    train=dict(
        sampler=None,
        tier='train',
        **dataset_para,
        ),

    val=dict(
        sampler='SequentialSampler',
        tier='val',
        **dataset_para,
        ),
    test=dict(
        sampler='SequentialSampler',
        tier='val',
        **dataset_para,
        ),

)

# yapf:enable
# resume from a checkpoint
resume_from=None
cudnn_benchmark=True

# optimization
optimization=dict(
    type='epoch',
    max_iters=800,
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.05, betas=(0.9, 0.95), eps=1e-8),
    scheduler=dict(type='WarmupCosineDecayLR', warmup_epochs=80, last_epoch=-1),
)

# yapf:disable
log=dict(

    project_name='fNIRS_Emo',
    work_dir=work_dir,
    exp_name=exp_name,
    logger_interval=200,
    monitor='val_f1_score',
    logger=[dict(type='comet', key='EOxad0Dbwx7UdmzxjwC2rx38H'), dict(type='csv')],
    checkpoint=dict(type='ModelCheckpoint',
                    top_k=3,
                    mode='max',
                    verbose=False,
                    save_last=True,
                    ),
    earlystopping=dict(
            mode='max',
            strict=False,
            patience=30,
            min_delta=0.0005,
            check_finite=True,
            verbose=True
    )

)