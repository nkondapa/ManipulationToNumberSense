There's a requirements.txt; you can setup a virtual/conda environment with pip install -r requirements.txt

0) set ROOT to path to the folder on your computer
1) run /data_creation/generate_test_sets.py

** The way I generate datasets is memory inefficient, I save everything as numpy arrays (not a compressed format like .jpg).
Feel free to modify that and whatever else might be necessary.


To train a specifc model run /training/model_datasetX-Y
X = 1 -> same size and contrast
X = 2 -> random size and contrast
Y : number of objects

This is also a template to specify your own model.

To recreate everything (this will take some time depending on the GPU you have available)
1) run /training/execute.py


analysis, figures, aux_figures, figures_creation, plotting are analysis/visualization code.
These are a bit harder to parse since some of it hasn't been cleaned up.

model/ - has the neural network models.
I have a suspicion that pre-training might not actually be needed,
so if you want to see what happens without that feel free.