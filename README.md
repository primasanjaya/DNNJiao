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

## create dataset SNV
```
python create_dataset.py --temp-dir '/path/to/muat/data/raw/temp/' --output-path '/path/to/DNNJiao/dataset.csv'
```

## create dataset SNV Pos
```
python create_datasetSNVPos.py --temp-dir '/path/to/muat/data/raw/temp/' --output-path '/path/to/DNNJiao/dataset.csv'
```


## run prediction SNV
```
python dnn.py --fold 1 --input-file '/path/to/DNNJiao/DNNJiao/dataset.csv' --model-dir '/path/to/DNNJiao/pcawgSNV/' --output-file '/path/to/DNNJiao/test_pred.csv' --pred
```

## run prediction SNV Pos
```
python dnn.py --fold 1 --input-file '/path/to/DNNJiao/dataset.csv' --model-dir '/path/to/DNNJiao/pcawgSNVPos/'
 --output-file '/path/to/DNNJiao/test_pred.csv' --pred
```