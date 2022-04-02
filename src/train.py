from statistics import mode
import torch
import torch.nn as nn
import os
import config
from dataset import SpeechDataset, collate_fn_padding
from model import SpeechModel

def train_fn(data_loader, model, optimizer, epoch, device=config.device):
    model.train()
    for batch_idx, (data, target, input_len, spec_len) in enumerate(data_loader):
        data = data.to(config.device)
        # print(data.shape)
        target = target.to(config.device)
        # print(target.shape)
        input_len = input_len
        # print(len(input_len))
        spec_len = spec_len
        # print(len(spec_len))
        model.zero_grad()
        hidden = model._init_hidden(32)
        hn, c0 = hidden[0].to(config.device), hidden[1].to(config.device)
        output, _ = model(data, (hn, c0))
        output = nn.functional.log_softmax(output, dim=2)
        print(output.shape)
        loss = nn.CTCLoss(blank=28, zero_infinity=True)(output, target, input_len, spec_len)

        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    train_dataset = SpeechDataset(
        json_path=os.path.join(config.json_data_path, "train.json")
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn_padding,
        drop_last=True
    )

    # for (d,c,v,a) in train_loader:
    #     print(c.shape)
    #     print(v, a)
    #     break

    model = SpeechModel(hidden_size=1024, num_layers=1, n_feats=81, dropout=0.3, num_classes=29).to(config.device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.LR)

    for epoch in range(100):
        train_fn(
            train_loader, 
            model, 
            optim, 
            epoch
        )