import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
import torch.nn.modules.activation as activation
import math

class ExpActivation(nn.Module):
    """
    Exponential activation function from Koo & Ploenzke, 2021 (PMID: 34322657)
    """
    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, x):
        return torch.exp(x)

class Unsqueeze(torch.nn.Module):
    """
    Unsqueeze for sequential models
    """
    def forward(self, x):
        return x.unsqueeze(-1)
    
# =============================================================================
# ExplaiNN
# =============================================================================

class ExplaiNN(nn.Module):
    """
    The ExplaiNN model (PMID: 37370113)
    """
    def __init__(self, num_cnns, input_length, num_classes, filter_size=19, num_fc=2, pool_size=7, pool_stride=7,
                 weight_path=None):
        """
        :param num_cnns: int, number of independent cnn units
        :param input_length: int, input sequence length
        :param num_classes: int, number of outputs
        :param filter_size: int, size of the unit's filter, default=19
        :param num_fc: int, number of FC layers in the unit, default=2
        :param pool_size: int, size of the unit's maxpooling layer, default=7
        :param pool_stride: int, stride of the unit's maxpooling layer, default=7
        :param weight_path: string, path to the file with model weights
        """
        super(ExplaiNN, self).__init__()

        self._options = {
            "num_cnns": num_cnns,
            "input_length": input_length,
            "num_classes": num_classes,
            "filter_size": filter_size,
            "num_fc": num_fc,
            "pool_size": pool_size,
            "pool_stride": pool_stride,
            "weight_path": weight_path
        }

        if num_fc == 0:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(input_length - (filter_size-1)),
                nn.Flatten())
        elif num_fc == 1:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Flatten())
        elif num_fc == 2:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(in_channels=100 * num_cnns,
                          out_channels=1 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Flatten())
        else:
            self.linears = nn.Sequential(
                nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                          groups=num_cnns),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                          out_channels=100 * num_cnns, kernel_size=1,
                          groups=num_cnns),
                nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU())

            self.linears_bg = nn.ModuleList([nn.Sequential(nn.Dropout(0.3),
                                                           nn.Conv1d(in_channels=100 * num_cnns,
                                                                     out_channels=100 * num_cnns, kernel_size=1,
                                                                     groups=num_cnns),
                                                           nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                                                           nn.ReLU()) for i in range(num_fc - 2)])

            self.last_linear = nn.Sequential(nn.Dropout(0.3),
                                             nn.Conv1d(in_channels=100 * num_cnns, out_channels=1 * num_cnns,
                                                       kernel_size=1,
                                                       groups=num_cnns),
                                             nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                                             nn.ReLU(),
                                             nn.Flatten())

        self.final = nn.Linear(num_cnns, num_classes)

        if weight_path:
            self.load_state_dict(torch.load(weight_path))

    def forward(self, x):
        x = x.repeat(1, self._options["num_cnns"], 1)
        if self._options["num_fc"] <= 2:
            outs = self.linears(x)
        else:
            outs = self.linears(x)
            for i in range(len(self.linears_bg)):
                outs = self.linears_bg[i](outs)
            outs = self.last_linear(outs)
        out = self.final(outs)
        return out

# =============================================================================
# ConvNetDeep
# =============================================================================
class ConvNetDeep(nn.Module):
    """
    CNN with 3 convolutional layers adapted from Basset (PMID: 27197224)
    designed for input sequences of length 200 bp
    """
    def __init__(self, num_classes, drop_out = 0.3, weight_path=None):
        """
        :param num_classes: int, number of outputs
        :param weight_path: string, path to the file with model weights
        """
        super(ConvNetDeep, self).__init__()
        # Block 1 :
        self.c1 = nn.Conv1d(4, 100, 19)
        self.bn1 = nn.BatchNorm1d(100)
        self.rl1 = activation.ReLU()
        self.mp1 = nn.MaxPool1d(3, 3)

        # Block 2 :
        self.c2 = nn.Conv1d(100, 200, 11)
        self.bn2 = nn.BatchNorm1d(200)
        self.rl2 = activation.ReLU()
        self.mp2 = nn.MaxPool1d(4, 4)

        # Block 3 :
        self.c3 = nn.Conv1d(200, 200, 7)
        self.bn3 = nn.BatchNorm1d(200)
        self.rl3 = activation.ReLU()
        self.mp3 = nn.MaxPool1d(4, 4)

        # Block 4 : Fully Connected 1 :
        self.d4 = nn.Linear(2000, 1000)  # 1000 for 608 input size
        self.bn4 = nn.BatchNorm1d(1000, 1e-05, 0.2, True)
        self.rl4 = activation.ReLU()
        self.dr4 = nn.Dropout(drop_out)

        # Block 5 : Fully Connected 2 :
        self.d5 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000, 1e-05, 0.2, True)
        self.rl5 = activation.ReLU()
        self.dr5 = nn.Dropout(drop_out)

        # Block 6 :4Fully connected 3
        self.d6 = nn.Linear(1000, num_classes)
        # self.sig = activation.Sigmoid()

        if weight_path:
            self.load_state_dict(torch.load(weight_path))

    def forward(self, x, embeddings=False):
        """
        :param: embeddings: bool, if `True`, forward return embeddings
                            along with the output
        """
        x = self.rl1(self.bn1(self.c1(x)))
        activations = x
        x = self.mp1(x)
        x = self.mp2(self.rl2(self.bn2(self.c2(x))))
        em = self.mp3(self.rl3(self.bn3(self.c3(x))))
        o = torch.flatten(em, start_dim=1)
        o = self.dr4(self.rl4(self.bn4(self.d4(o))))
        o = self.dr5(self.rl5(self.bn5(self.d5(o))))
        o = self.d6(o)
        activations, act_index = torch.max(activations, dim=2)
        if embeddings: return (o, activations, act_index, em)
        return o
# =============================================================================
# DanQ
# =============================================================================
class DanQ(nn.Module):
    """
    PyTorch implementation of DanQ (PMID: 27084946)
    """
    def __init__(self, input_length, num_classes, weight_path=None):
        """
        :param input_length: int, input sequence length
        :param num_classes: int, number of output classes
        :param weight_path: string, path to the file with model weights
        """
        super(DanQ, self).__init__()

        self._options = {
            "input_length": input_length,
            "num_classes": num_classes,
            "weight_path": weight_path
        }

        self.conv1 = nn.Conv1d(4, 320, kernel_size=26)
        self.act1 = nn.ReLU()
        self.maxp1 = nn.MaxPool1d(kernel_size=13, stride=13)

        self.bi_lstm_layer = nn.LSTM(320, 320, num_layers=1,
                                     batch_first=True, bidirectional=True)

        self._in_features_L1 = math.floor((input_length - 25) / 13.) * 640

        self.linear = nn.Sequential(
            nn.Linear(self._in_features_L1, 925),
            nn.ReLU(),
            nn.Linear(925, num_classes),
        )

        if weight_path:
            self.load_state_dict(torch.load(weight_path))

    def forward(self, input):      
        x = self.act1(self.conv1(input))
        x = nn.Dropout(0.2)(self.maxp1(x))
        x = x.transpose(1, 2)
        x, _ = self.bi_lstm_layer(x)
        x = x.contiguous().view(-1, self._in_features_L1)
        x = nn.Dropout(0.5)(x)
        x = self.linear(x)
        return x
#################################################################################
# =============================================================================
# ConvNetDeep2
# =============================================================================
class ConvNetDeep2(nn.Module):
    """
    CNN with 3 convolutional layers adapted from Basset (PMID: 27197224)
    designed for input sequences of length 200 bp
    """
    def __init__(self, num_classes, drop_out = 0.3, weight_path=None):
        """
        :param num_classes: int, number of outputs
        :param weight_path: string, path to the file with model weights
        """
        super(ConvNetDeep2, self).__init__()
        # Block 1 :
        self.c1 = nn.Conv1d(4, 100, 19)
        self.bn1 = nn.BatchNorm1d(100)
        self.rl1 = activation.ReLU()
        self.mp1 = nn.MaxPool1d(3, 3)

        # Block 2 :
        self.c2 = nn.Conv1d(100, 200, 11)
        self.bn2 = nn.BatchNorm1d(200)
        self.rl2 = activation.ReLU()
        self.mp2 = nn.MaxPool1d(3, 3)

        # Block 3 :
        self.c3 = nn.Conv1d(200, 200, 7)
        self.bn3 = nn.BatchNorm1d(200)
        self.rl3 = activation.ReLU()
        self.mp3 = nn.MaxPool1d(4, 4)

        # Block 3.5 :
        self.c7 = nn.Conv1d(200, 400, 4)
        self.bn7 = nn.BatchNorm1d(400)
        self.rl7 = activation.ReLU()
        self.mp7 = nn.MaxPool1d(4, 4) 

        # Block 4 : Fully Connected 1 :
        self.d4 = nn.Linear(800, 800)  # 2000 for 608 input size
        self.bn4 = nn.BatchNorm1d(800, 1e-05, 0.9, True)
        self.rl4 = activation.ReLU()
        self.dr4 = nn.Dropout(drop_out)

        # Block 5 : Fully Connected 2 :
        self.d5 = nn.Linear(800, 800)
        self.bn5 = nn.BatchNorm1d(800, 1e-05, 0.9, True)
        self.rl5 = activation.ReLU()
        self.dr5 = nn.Dropout(drop_out)

        # Block 6 :4Fully connected 3
        self.d6 = nn.Linear(800, num_classes)
        # self.sig = activation.Sigmoid()

        if weight_path:
            self.load_state_dict(torch.load(weight_path))

    def forward(self, x, embeddings=False):
        """
        :param: embeddings: bool, if `True`, forward return embeddings
                            along with the output
        """
        x = self.rl1(self.bn1(self.c1(x)))
        activations = x
        x = self.mp1(x)
        x = self.mp2(self.rl2(self.bn2(self.c2(x))))
        x = self.mp3(self.rl3(self.bn3(self.c3(x))))
        em = self.mp7(self.rl7(self.bn7(self.c7(x))))

        o = torch.flatten(em, start_dim=1)
        o = self.dr4(self.rl4(self.bn4(self.d4(o))))
        o = self.dr5(self.rl5(self.bn5(self.d5(o))))
        o = self.d6(o)
        activations, act_index = torch.max(activations, dim=2)
        if embeddings: return (o, activations, act_index, em)
        return o

# =============================================================================
# ExplaiNN2
# =============================================================================

class ExplaiNN2(nn.Module):
    def __init__(self, num_cnns, input_length, num_classes, 
                 filter_size = 19, num_fc=2, pool_size=7, pool_stride=7, 
                 fc_filter1 = 20, fc_filter2 = 1, drop_out = 0.3, weight_path = None):
        super(ExplaiNN2, self).__init__()
        self._options = {
            "num_cnns": num_cnns,
            "input_length": input_length,
            "num_classes": num_classes,
            "filter_size": filter_size,
            "num_fc": num_fc,
            "pool_size": pool_size,
            "pool_stride": pool_stride,
            "weight_path": weight_path
        }
        self.linears = nn.Sequential(
            # Convolution Layer
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=filter_size, padding = 'same'),
            nn.BatchNorm1d(1),
            ExpActivation(),
            nn.MaxPool1d(pool_size, pool_stride), # pool_size=7, pool_stride=7
            nn.Flatten(),
            # Linear Layer 1
            nn.Linear(((input_length-pool_size)//pool_stride) + 1, fc_filter1),
            nn.BatchNorm1d(fc_filter1),
            ExpActivation(),
            nn.Dropout(p=drop_out),
            # Linear Layer 2
            nn.Linear(fc_filter1, fc_filter2),
            nn.BatchNorm1d(fc_filter2),
            ExpActivation(),
            nn.Flatten()
        )
        self.final = nn.Linear(num_cnns*fc_filter2, num_classes)

        if weight_path:
            self.load_state_dict(torch.load(weight_path))
        
    def forward(self, x):
        encoder = []
        for i in range(self._options['num_cnns']):
            xnn = self.linears(x)
            encoder.append(xnn)
        encoder = torch.cat(encoder, dim=-1)
        results = self.final(encoder)
        return results


class ExplaiNN3(nn.Module):
    """
    [B,4,608] -> replicate [B, 4 * n_cnn, 608] -> Conv1d(4 * n_cnn, n_cnn, kernel = 19)
    -> [B, n_cnn, 590] -> ExpAct -> MaxPool (7,7) -> [B, n_cnn, 84] -> flat ->
    [B, n_cnn * 84] -> Unsqueeze -> [B, n_cnn * 84, 1] -> Conv1d (n_cnn * 84, n_cnn * 100, kernel = 1) ->
    [B, n_cnn * 100, 1] -> Conv1d (n_cnn * 100, n_cnn * 1, kernel = 1) -> [B, n_cnn, 1]
    -> flat -> [B, n_cnn] -> Linear (n_cnn, num_classes) -> [B, num_classes]
    """
    def __init__(self, num_cnns, input_length, num_classes, 
                 filter_size = 19, num_fc=2, pool_size=7, pool_stride=7, 
                 drop_out = 0.3, weight_path = None):
        super(ExplaiNN3, self).__init__()
        self._options = {
            "num_cnns": num_cnns,
            "input_length": input_length,
            "num_classes": num_classes,
            "filter_size": filter_size,
            "num_fc": num_fc,
            "pool_size": pool_size,
            "pool_stride": pool_stride,
            "weight_path": weight_path
        }
        self.linears = nn.Sequential(
            nn.Conv1d(in_channels=4 * num_cnns, out_channels=1 * num_cnns, kernel_size=filter_size,
                        groups=num_cnns),
            nn.BatchNorm1d(num_cnns),
            ExpActivation(),
            nn.MaxPool1d(pool_size, pool_stride),
            nn.Flatten(),
            Unsqueeze(),
            nn.Conv1d(in_channels=int(((input_length - (filter_size-1)) - (pool_size-1)-1)/pool_stride + 1) * num_cnns,
                        out_channels=100 * num_cnns, kernel_size=1,
                        groups=num_cnns),
            nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Conv1d(in_channels=100 * num_cnns,
                        out_channels=1 * num_cnns, kernel_size=1,
                        groups=num_cnns),
            nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
            nn.ReLU(),
            nn.Flatten()
            )
        self.final = nn.Linear(num_cnns, num_classes)

        if weight_path:
            self.load_state_dict(torch.load(weight_path))
        
    def forward(self, x):
        x = x.repeat(1, self._options["num_cnns"], 1)
        outs = self.linears(x)
        results = self.final(outs)
        return results
    
class DeepSTARR(nn.Module):
    """
    PyTorch implementation of DeepSTARR (PMID: 35551305)
    [B,4,608] ==(Block1)==> [B, 256, 608] => [B, 256, 304]
            ==(Block2)==> [B, 60, 304] => [B, 60, 152]
            ==(Block3)==> [B, 60, 152] => [B, 60, 76]
            ==(Block4)==> [B, 120, 76] => [B, 120, 38]
            ==(Flatten)==> [B, 4560]
            ==(Linear1)==> [B, 256]
            ==(Linear1)==> [B, 256]
            ==(Linear1)==> [B, num_classes]
    """
    def __init__(self, weight_path=None, num_classes = 2):
        super(DeepSTARR, self).__init__()
        self.convol = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=256, kernel_size=7,
                      padding=int((7 - 1) / 2)),  # same padding
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=256, out_channels=60, kernel_size=3,
                      padding=int((3 - 1) / 2)),  # same padding
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=60, out_channels=60, kernel_size=5,
                      padding=int((5 - 1) / 2)),  # same padding
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=60, out_channels=120, kernel_size=3,
                      padding=int((3 - 1) / 2)),  # same padding
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Flatten()
        )

        self.linear = nn.Sequential(
            nn.Linear(4560, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        if weight_path:
            self.load_state_dict(torch.load(weight_path))

    def forward(self, x):
        o = self.linear(self.convol(x))
        return o