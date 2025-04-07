# create virtual environment

```
python3.11 -m venv ~/py311_env  && source ~/py311_env/bin/activate
```

# install dependencies

```
pip install --upgrade pip      
pip install tensorflow==2.12.0
pip install tf2onnx
pip install opencv-python
pip install onnxruntime-silicon
pip install onnxruntime   
pip install numpy==1.26.4     
```

# Convert tensorflow model to onnx

```
python -m tf2onnx.convert \
--saved-model /Users/<username>/.cache/kagglehub/models/google/movenet/tensorFlow2/singlepose-thunder \
--output singlepose-thunder.onnx --opset 13
```

# Contenido de pip list


Package                      Version
---------------------------- ---------
absl-py                      2.2.2
astunparse                   1.6.3
certifi                      2025.1.31
charset-normalizer           3.4.1
coloredlogs                  15.0.1
flatbuffers                  25.2.10
gast                         0.6.0
google-pasta                 0.2.0
grpcio                       1.71.0
h5py                         3.13.0
humanfriendly                10.0
idna                         3.10
keras                        3.9.2
libclang                     18.1.1
Markdown                     3.7
markdown-it-py               3.0.0
MarkupSafe                   3.0.2
mdurl                        0.1.2
ml_dtypes                    0.5.1
mpmath                       1.3.0
namex                        0.0.8
numpy                        1.26.4
onnx                         1.17.0
onnxruntime                  1.21.0
onnxruntime-silicon          1.16.3
opencv-python                4.11.0.86
opt_einsum                   3.4.0
optree                       0.15.0
packaging                    24.2
pip                          25.0.1
protobuf                     3.20.3
Pygments                     2.19.1
requests                     2.32.3
rich                         14.0.0
setuptools                   75.6.0
six                          1.17.0
sympy                        1.13.3
tensorboard                  2.19.0
tensorboard-data-server      0.7.2
tensorflow                   2.19.0
tensorflow-io-gcs-filesystem 0.37.1
termcolor                    3.0.1
tf2onnx                      1.16.1
typing_extensions            4.13.1
urllib3                      2.3.0
Werkzeug                     3.1.3
wheel                        0.45.1
wrapt                        1.17.2
