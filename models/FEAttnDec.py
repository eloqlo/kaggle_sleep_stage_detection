class FeatureExtractor(nn.Module):
    def __init__(self, in_channel, d_model, kernel_sizes):
        super().__init__()
        self.d_model=d_model
        layers=[nn.Sequential(
            nn.Conv1d(in_channel,d_model//2, kernel_size=11, stride=1, padding=(11-1)//2),
            nn.BatchNorm1d(d_model//2, eps=1e-6),
            nn.ReLU(),
            nn.Conv1d(d_model//2,d_model, kernel_size=11, stride=1, padding=(11-1)//2),
            nn.BatchNorm1d(d_model, eps=1e-6),
            nn.ReLU(),
        )]
        for ks in kernel_sizes:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(d_model,d_model, kernel_size=ks, stride=1, padding=(ks-1)//2),
                    nn.BatchNorm1d(d_model, eps=1e-6),
                    nn.ReLU()
                )
            )
        self.layerlist = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layerlist:
            x=layer(x)
        return x


class DownSampler(nn.Module):
    def __init__(self, d_model, kernel_sizes):
        super().__init__()
        self.activation = nn.ReLU()
        self.stride = 2 if len(kernel_sizes)>3 else 4
        self.kernel_sizes = kernel_sizes
        layers=[]
        for ks in kernel_sizes:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(d_model,d_model, kernel_size=ks, stride=self.stride, padding=(ks-self.stride)//2),
                    nn.BatchNorm1d(d_model, eps=1e-6),
                    nn.ReLU()
                )
            )
        self.layerlist = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layerlist:
            x = layer(x)
        return x 
    
    
class UpSampler(nn.Module):
    def __init__(self, scale, d_model,  kernel_sizes):
        super().__init__()
        self.activation = nn.ReLU()
        self.scale = scale
        layers=[]
        for ks in kernel_sizes:
            layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=self.scale),
                    nn.Conv1d(in_channels=d_model,
                              out_channels=d_model, 
                              kernel_size=ks, 
                              stride=1, 
                              padding=(ks-1)//2),
                    nn.BatchNorm1d(d_model, eps=1e-6),
                    nn.ReLU()
                )
            )
        self.layerlist = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layerlist:
            x = layer(x)
        return x


class MyAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.scale = d_model ** 0.5
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=7, stride=3, padding=2)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=7, stride=3, padding=2)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=7, stride=3, padding=2)
        self.upsample = nn.Upsample(scale_factor=3)
        self.conv_attn = nn.Conv1d(d_model, d_model, kernel_size=5, stride=1, padding=2)  # N C L

        self.attn_dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(d_model, eps=1e-6)
        self.batch_norm2 = nn.BatchNorm1d(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        
    def forward(self, x):
        # (B d L)
        residual = x.clone()  # ?
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        
        attn = torch.matmul((k/self.scale).transpose(-1,-2), q) # (Seq/2,d)*(d,Seq/2)
        attn = self.attn_dropout(F.softmax(attn, dim=1))
        x = torch.matmul(attn, v.transpose(-1,-2))  # B L d
        x = self.dropout1(x).transpose(-1,-2) # N C L
        x = self.batch_norm1(x)
        
        x = self.upsample(x)
        x = F.relu(self.batch_norm2(self.conv_attn(x)))
        x = x + residual

        return x
    

class FFNN(nn.Module):

    def __init__(self, d_model, d_hid, residual=True, dropout=0.1):
        super().__init__()
        self.residual=residual
        self.w_1 = nn.Linear(d_model, d_hid)  # params= 16384
        self.w_2 = nn.Linear(d_hid, d_model)  # params= 16384
        self.batch_norm1 = nn.BatchNorm1d(d_hid, eps=1e-6)
        self.batch_norm2 = nn.BatchNorm1d(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        # B d S
        if self.residual:
            residual = x
        
        x = F.relu(self.batch_norm1(self.dropout1(self.w_1(x.transpose(-1,-2))).transpose(-1,-2)))
        x = F.relu(self.batch_norm2(self.dropout2(self.w_2(x.transpose(-1,-2)).transpose(-1,-2))))
        
        if self.residual:
            x =x+ residual
        
        # B d S
        return x

    

class DecoderCONV(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.conv_blks1 = nn.Sequential(
            nn.Conv1d(d_model,d_model,kernel_size=31, stride=1, padding=(31-1)//2),
            nn.BatchNorm1d(d_model, eps=1e-6),
            nn.ReLU(),
            nn.Conv1d(d_model,d_model,kernel_size=31, stride=1, padding=(31-1)//2),
            nn.BatchNorm1d(d_model, eps=1e-6),
            nn.ReLU(),
            nn.Conv1d(d_model,d_model,kernel_size=31, stride=1, padding=(31-1)//2),
            nn.BatchNorm1d(d_model, eps=1e-6),
            nn.ReLU(),
        )
        self.ffnn = FFNN(d_model, d_model*2, False, dropout)
    
    def forward(self, x):
        # B d L
        x = self.conv_blks1(x)
        x = self.ffnn(x)
        # B d S
        return x


    
class FEAttnDec(nn.Module):
    def __init__(self,in_c, out_c, up_scale, d_model, d_hid, n_blks, \
                 fe_sizes ,down_sizes, up_sizes, late_dropout=0.5, dropout=0.1):
        super().__init__()
#         self.fc_input = nn.Linear(in_c, d_model)
        self.feature_extractor = FeatureExtractor(in_c, d_model, fe_sizes)
        self.downsampler = DownSampler(d_model, down_sizes)
        self.upsampler = UpSampler(up_scale, d_model, up_sizes)
        self.enc_blks = nn.ModuleList([
            nn.Sequential(MyAttention(d_model, dropout),
                          MyAttention(d_model, dropout),
                          FFNN(d_model, d_hid),)
            for _ in range(n_blks)]
        )
        self.decoder = DecoderCONV(d_model, dropout)
        self.late_dropout = nn.Dropout(late_dropout)
        self.fc_output = nn.Linear(d_model, out_c)

    
    def forward(self, x):
        
        #out: b d seq
        x = x.transpose(-1,-2)
        x = self.feature_extractor(x)
        
        x = self.downsampler(x)
        for enc_blk in self.enc_blks:
            x = enc_blk(x)
        x = self.upsampler(x)
        x = self.decoder(x)
        x = self.late_dropout(x)
        x = self.fc_output(x.transpose(-1,-2))  # there are sigmoid in F.BCE !! no need to normalize of something.
#         x = F.sigmoid(x)  # for MSE !!!
        return x
