import kagglehub
# Download latest version
path = kagglehub.model_download("google/movenet/tensorFlow2/multipose-lightning")
path = kagglehub.model_download("google/movenet/tensorFlow2/singlepose-thunder")
print("Path to model files:", path)