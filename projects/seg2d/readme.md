

Create a conda env
```bash
conda create -n in4it numpy h5py tensorflow
conda activate in4it
pip install nibabel matplotlib  # only for creating data loader and visualisation
```

Optionally, using GPUs
```bash
conda install tensorflow-gpu
```

Training the model
```bash
python train.py
```

Predicting using a trained model (inference)
```bash
python predict.py
```
