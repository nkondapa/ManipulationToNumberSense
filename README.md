# A number sense as an emergent property of the manipulating brain

[[Paper]](https://www.nature.com/articles/s41598-024-56828-2) [[ArXiv]](https://arxiv.org/abs/2012.04132) [[`BibTeX`](#Citing)]


## Environment setup
1) pip install -r requirements.txt
2) In global_variables.py set ROOT to the path to the folder on your computer.

## Create Test Datasets
```python /data_creation/generate_test_sets.py```

** The way I generate datasets is memory inefficient, I save everything as numpy arrays (not a compressed format like .jpg).
Feel free to modify that if needed.


## Train a model
``` python ./training/model_datasetX-Y.py```

X = 1 -> same size and contrast <br>
X = 2 -> random size and contrast <br>
Y : number of objects

## Recreate all experiments
```python /training/execute.py``` (this will take some time depending on the GPU you have available)


## Analysis & Plotting
The analysis scripts are in the /analysis folder and have the word "script" in their name. <br>
```python /analysis/analyze_model_script{number}.py``` <br>
Each scripts analyzes a different experiment.

The plotting scripts are in the /figures_creation folder. <br>
```python /figures_creation/{figure_name}.py``` <br>

## Citing
```
@article{nkondapa2024emergent,
  title={A number sense as an emergent property of the manipulating brain},
  author={Kondapaneni, Neehar and Perona, Pietro},
  journal={Scientific reports},
  volume={14},
  number={6858},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```