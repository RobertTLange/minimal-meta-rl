import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


class MetaLSTM(nn.Module):
    """ A2C LSTM - Inputs: In Dim, Hidden Dim , Num Layers, Out Dim """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,
                 linear_embed=False, linear_dim=None,
                 bidirectional=False, learn_hidden_init=False,
                 recurrent_init=None, forget_bias_init=False):
        super(MetaLSTM, self).__init__()
        """ Define net AC architecture w. LSTM core as in Wang et al. (16') """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 1 + bidirectional

        self.linear_embed = linear_embed
        if linear_embed and linear_dim is not None:
            self.embed_in = nn.Linear(input_dim, linear_dim)
            self.lstm = nn.LSTM(linear_dim, hidden_dim, num_layers,
                                batch_first=True, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True, bidirectional=bidirectional)

        self.hidden_2_value = nn.Linear(self.num_directions*hidden_dim, 1)
        self.hidden_2_policy_logits = nn.Linear(hidden_dim, output_dim)
        self.softmax_policy = nn.Softmax(dim=2)

        # Set whether to use orthogonal init
        self.init_lstm_weights(recurrent_init, forget_bias_init)

        # Set whether to learn initial hidden states - else zeros
        self.learn_hidden_init = learn_hidden_init
        if self.learn_hidden_init:
            self.init_hidden_h = nn.Parameter(
                                 torch.randn(self.num_layers*self.num_directions,
                                 1, self.hidden_dim),
                                 requires_grad=True)
            self.init_hidden_c = nn.Parameter(torch.randn(
                                 self.num_layers*self.num_directions,
                                 1, self.hidden_dim),
                                 requires_grad=True)

    def forward(self, state_input, hidden_in):
        """ Forward pass - Out: Policy-Distr - Cat, Value Est, Hidden State """
        # Use linear embedding of state input if desired
        if self.linear_embed:
            lstm_input = F.relu(self.embed_in(state_input))
        else:
            lstm_input = state_input

        lstm_out, hidden_out = self.lstm(lstm_input, hidden_in)
        policy_logits = self.hidden_2_policy_logits(lstm_out)
        policy_out = self.softmax_policy(policy_logits)
        value_out = self.hidden_2_value(lstm_out)
        return Categorical(policy_out), value_out, hidden_out

    def init_hidden(self, device, batch_size=1):
        """ Init the hidden state of the network - batch-size: num_threads """
        if not self.learn_hidden_init:
            return (torch.zeros(self.num_layers*self.num_directions, batch_size,
                                self.hidden_dim).to(device),
                    torch.zeros(self.num_layers*self.num_directions, batch_size,
                                self.hidden_dim).to(device))
        else:
            batch_h = torch.cat(batch_size*[self.init_hidden_h], 1)
            batch_c = torch.cat(batch_size*[self.init_hidden_c], 1)
            return (batch_h, batch_c)

    def init_lstm_weights(self, recurrent_init=None, forget_bias_init=False):
        """ Special Initialization of the network parameters. """
        # Option for orthogonal init of rec weights - following Saxe et al. 13'
        if recurrent_init == "orthogonal":
            for name, param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

        # Option for gaussian init of rec weights
        elif recurrent_init == "gaussian":
            for name, param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.normal_(param.data, std=1./param.data.size(1))
                elif 'bias' in name:
                    param.data.fill_(0)

        # Option to set initial bias for forget gate to 1 - longer time horizon
        if forget_bias_init:
            for name, param in self.lstm.named_parameters():
                if 'bias' in name:
                    param.data.fill_(1)


def load_net_params(network, ckpth_path):
    """ Load in the parameters of the network from a checkpoint"""
    checkpoint = torch.load(ckpth_path, map_location='cpu')
    network.load_state_dict(checkpoint)
