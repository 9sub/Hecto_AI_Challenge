

img_size = 512
batch_size = 64
epochs = 50
lr = 3e-4
seed = 42

ENSEMBLE_MODEL_CONFIGS = [
    ("resnet50_lora", 1),       # FineGrainResNet50LoRA 1개
    ("resnext50_lora", 1),      # FineGrainResNext50LoRA 1개 (새로 추가)
    ("convnext_lora", 1),       # FineGrainConvNextLoRA 1개 (새로 추가)
    ("efficientnet_lora", 1),   # FineGrainEfficientNetLoRA 1개 (기본: efficientnet_b0)
    ("vit_lora", 1),            # FineGrainViTLoRA 1개 (기본: vit_base_patch16_224)
]
# 총 기본 모델 개수 (자동 계산)
NUM_ENSEMBLE_MODELS = sum(count for _, count in ENSEMBLE_MODEL_CONFIGS)

train_root = './data/train'
test_root = './data/test'
crop_root = './data/train_crop'
model_save_dir = "./models/" # 모델 저장 디렉토리

model_name = 'FineGrainResNet50_512_efficient_ensemble_change_label'