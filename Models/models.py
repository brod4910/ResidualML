# Model Architecture:
# Convolution: ['C', in_channels, out_channels, (kernel), stride, padding, Activation Function]
# Max Pooling: ['M', (kernel), stride, padding]
# Average Pooling: ['A', (kernel), stride, padding]
# Linear Layer: ['L', in_features, out_features, Activation Function]
# Dropout : ['D', probability]
# Alpha Dropout : ['AD', probability]
# Classifying layer: ['FC', in_features, num_classes]
# Possible Activation Fns: 'ReLU', 'PReLU', 'SELU', 'LeakyReLU', 'None'->(Contains no Batch Norm for dimensionality reduction 1x1 kernels)
# srun python main.py --batch-size 64 --epochs 40 --lr 0.001 --momentum .9 --log-interval 100 --arcitecture shallow --root-dir ../ --train-csv ../data_csv/0_3_train.csv --test-csv ../data_csv/0_3_test.csv

feature_layers = {

    'shallow_norm': [['C', 1, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], 
    ['C', 256, 512, (3,3), 1, 1, 'ReLU'], ['C', 512, 256, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0]],

    'residual': [['C', 1, 128, (3,3), 1, 1, 'ReLU'], ['C', 128, 256, (3,3), 1, 1, 'ReLU'], 
    ['C', 384, 512, (3,3), 1, 1, 'ReLU'], ['C', 896, 256, (3,3), 1, 1, 'ReLU'], 
    ['M', (2,2), 2, 0]]
}

classifier_layers = {
    'shallow_norm': [['L', 256 * 16 * 16, 1024, 'ReLU'], ['D', .5], ['FC', 1024, 10]],
    'residual': [['L', 1152 * 16 * 16, 1024, 'ReLU'], ['D', .5], ['FC', 1024, 10]]
}