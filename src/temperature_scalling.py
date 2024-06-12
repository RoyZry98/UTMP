from sys import flags
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, flag, model=None):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.adder = nn.Parameter(torch.zeros(1))
        self.flag = flag
        self.nll = None

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)
    
    def reset_temperature(self, temp):
        self.temperature = nn.Parameter(torch.ones(1) * temp)

    def get_temperature(self):
        return self.temperature
    
    def update_model(self,model):
        self.model = model

    def get_flag(self):
        return self.flag
    
    def set_flag(self,flag):
        self.flag = flag

    def get_uncertainty(self):
        # return torch.sigmoid(1 / self.temperature)
        return  self.confidence
        # return torch.tensor(1./self.nll)

    def set_confidence(self,logits):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        self.confidence = torch.mean(confidences)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature + self.adder
    
    def ccf(self, result,label):
        test_preds = result.view(-1).cpu().detach().numpy()
        test_truth = label.view(-1).cpu().detach().numpy() 
        ccf = np.corrcoef(test_preds, test_truth)[0][1]
        return ccf
    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        # nll_criterion = nn.SmoothL1Loss().cuda()
        ece_criterion = _ECELoss().cuda()
        # ece_criterion = nn.L1Loss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for i_batch, ( batch_X, label, batch_META) in enumerate(valid_loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = label.squeeze(-1)
                # if self.args.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()

                # input = input.cuda()
                logits = self.model(text, audio, vision, self.flag, 0)
                # logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list,0).cuda()
            labels = torch.cat(labels_list).cuda()
            # labels = labels.reshape([labels.shape[1],labels.shape[0]])
            # logits = logits.reshape([logits.shape[1],logits.shape[0]])
            # labels = labels.flatten()
            # if self.args.dataset == 'iemocap':
            logits = logits.view(-1, 2)
            labels = labels.view(-1)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels.long()).item()
        self.nll = before_temperature_nll
        before_temperature_ece = ece_criterion(logits, labels.long()).item()
        # before_ccf = self.ccf(logits, labels)
        # print('Before temperature - NLL: %.3f, ECE: %.3f, adder%.3f, temperature%.3f, ccf%.8f' % (before_temperature_nll, before_temperature_ece,self.adder,self.temperature,0))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature,self.adder], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        # for i in range(20):
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        self.set_confidence(self.temperature_scale(logits))
        # after_ccf = self.ccf(self.temperature_scale(logits), labels)
        # print('Optimal temperature: %.3f' % self.temperature.item())
        # print('After temperature - NLL: %.3f, ECE: %.3f, adder%.3f, temperature%.3f, ccf%.8f' % (after_temperature_nll, after_temperature_ece,self.adder,self.temperature,0))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece