# loss function - regression, outliers represent anomalies that should be detected. -> use MSE type of regression loss
class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, seize_average=True, alpha=1., gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # here already contains sigmoid.
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
        
        return focal_loss.mean()
