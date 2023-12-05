python3 get-results.py -nr 0.1 -ir 100 --dataset SVHN --aug-model SHA --los-model NONE --hyper-opt HES
python3 get-results.py -nr 0.1 -ir 100 --dataset SVHN --aug-model SEP --los-model SOFT --hyper-opt HES
python3 get-results.py -nr 0.1 -ir 100 --dataset SVHN --aug-model SEP --los-model WGHT --hyper-opt HES
python3 get-results.py -nr 0.1 -ir 100 --dataset SVHN --aug-model SEP --los-model BOTH --hyper-opt HES



# Do not augment
python3 get-results.py -nr 0.1 -ir 100 --dataset SVHN --aug-model NONE --los-model NONE
# python3 get-results.py -nr 0.1 -ir 100 --dataset SVHN --aug-model SHA --los-model NONE --hyper-opt HES
# python3 get-results.py -nr 0.1 -ir 100 --dataset SVHN --aug-model SEP --los-model SOFT --hyper-opt HES
# python3 get-results.py -nr 0.1 -ir 100 --dataset SVHN --aug-model SEP --los-model WGHT --hyper-opt HES
# python3 get-results.py -nr 0.1 -ir 100 --dataset SVHN --aug-model SEP --los-model BOTH --hyper-opt HES