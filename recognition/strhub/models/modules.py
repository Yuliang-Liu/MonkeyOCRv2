r"""Shared modules used by CRNN and TRBA"""
from torch import nn


class BidirectionalLSTM(nn.Module):
    """Ref: https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/modules/sequence_modeling.py"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size], T = num_steps.
        output : contextual feature [batch_size x T x output_size]
        """
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
