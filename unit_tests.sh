python train.py --dataset CUB --method protonet --bn_type=1 --lr=0.0003 \
--model=resnet18 --optimization=Adam --stop_epoch=1 --train_aug=True

python train.py --dataset CUB --method protonet --bn_type=1 --lr=0.0003 \
--model=resnet18 --optimization=Adam --stop_epoch=1 --train_aug=True --semi_sup

python train.py --dataset CUB --method protonet --bn_type=1 --lr=0.0003 \
--model=resnet18 --optimization=Adam --stop_epoch=1 --train_aug=True --dataset_unlabel flowers
