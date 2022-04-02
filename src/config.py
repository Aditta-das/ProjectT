import torch
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

df_path = '/home/aditta/Desktop/ProjectT/input/'
json_data_path = "/home/aditta/Desktop/ProjectT/input/json_data"
LR = 0.01