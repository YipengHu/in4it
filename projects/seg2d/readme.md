

Create a conda env
```bash
conda create -n in4it numpy h5py tensorflow
conda activate in4it
pip install nibabel matplotlib  # only for data download and visualisation
```

Optionally, use GPUs
```bash
conda install tensorflow-gpu
```

Download the data set (hosted at in4it-data)
```bash
python data.py
```

Train the model
```bash
python train.py
```

Predict using a trained model (inference)
```bash
python predict.py
```
