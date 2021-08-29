import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor, cluster_set=None):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            if cluster_set is not None:
                output = input_tensor.sum(dim=self.dim, keepdim=True)
                cluster_set_shape = cluster_set.size()
                cluster_set = cluster_set[:, :, None]
                cluster_set = cluster_set.expand(cluster_set_shape + output.size()[2:])
                # print('avg cluster set size', cluster_set.shape)
                # print('avg output size', output.shape)
                output = torch.div(output, cluster_set)
            else:
                output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input, cluster_set=None):
        return SegmentConsensus(self.consensus_type, self.dim)(input, cluster_set=cluster_set)
