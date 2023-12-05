# python3 implicit-augment.py -r run0 --gpu 2 -nr 0.1 -ir 100 --dataset SVHN --aug-model SHA --los-model NONE --hyper-opt HES
# python3 implicit-augment.py -r run0 --gpu 2 -nr 0.1 -ir 100 --dataset SVHN --aug-model SEP --los-model NONE --hyper-opt HES
# python3 implicit-augment.py -r run0 --gpu 2 -nr 0.1 -ir 100 --dataset SVHN --aug-model SEP --los-model WGHT --hyper-opt HES
# python3 implicit-augment.py -r run0 --gpu 2 -nr 0.1 -ir 100 --dataset SVHN --aug-model SEP --los-model BOTH --hyper-opt HES


# no augmentation
# python3 implicit-augment.py -r run2 --gpu 7 -nr 0.1 -ir 100 --dataset SVHN --aug-model NONE --los-model NONE
python3 implicit-augment.py -r run2 --gpu 7 -nr 0.1 -ir 100 --dataset SVHN --aug-model NONE --los-model NONE --hyper-opt HES
python3 implicit-augment.py -r run2 --gpu 7 -nr 0.1 -ir 100 --dataset SVHN --aug-model NONE --los-model SOFT --hyper-opt HES
python3 implicit-augment.py -r run2 --gpu 7 -nr 0.1 -ir 100 --dataset SVHN --aug-model NONE --los-model WGHT --hyper-opt HES
python3 implicit-augment.py -r run2 --gpu 7 -nr 0.1 -ir 100 --dataset SVHN --aug-model NONE --los-model BOTH --hyper-opt HES