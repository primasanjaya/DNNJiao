# DNNJiao

# Quick start

## run installation_script.sh
```
./installation_script.sh
```

## Activate environment
```
conda activate tf
```

## create dataset
```
python create_dataset.py --temp-dir '/path/to/muat/data/raw/temp/' --output-path '/path/to/DNNJiao/dataset.csv'
```

## run prediction
```
python dnn.py --fold 1 --input-file '/mnt/g/experiment/DNNJiao/dataset.csv' --model-dir '/path/to/DNNJiao/pcawgSNV/' --output-file '/path/to/DNNJiao/test_pred.csv' --pred
```