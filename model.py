# model.py
import torch.nn as nn

class GestureRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(GestureRecognitionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.lstm = nn.LSTM(input_size=64 * 56 * 56, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x): 
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        x = self.cnn(x)
        x = x.view(b, t, -1)  
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n.squeeze(0)) 
        return out

