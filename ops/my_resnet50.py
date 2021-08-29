import torch
import numpy as np
import random

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering


class MyResNet50(ResNet):
    def __init__(self):
        super(MyResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])

    def forward(self, x, full_segments=[32, 16, 8], n_segments=[16, 8, 4], merge=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if merge:
            x = self.sample_merge(x, 18, 8)
        x = self.layer2(x)
        # if merge:
        #     x = self.sample_merge(x, full_segments[1], n_segments[1])
        x = self.layer3(x)
        # if merge:
        #     x = self.sample_merge(x, full_segments[2], n_segments[2])
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

    def sample_merge(self, x, full_segment, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // full_segment
        x = x.view(n_batch, full_segment, -1)
        # print(x[0].size())
        merged_x = self.accumulated_selection(x[0], n_segment)[None,:,:]
        for batch in range(1, n_batch):
            tmp = self.accumulated_selection(x[batch], n_segment)
            merged_x = torch.cat([merged_x, tmp[None,:,:]], dim=0)
        # print('merged_x size:', merged_x.size())
        merged_nt = n_batch * n_segment

        return merged_x.view(merged_nt, c, h, w)

    def accumulated_selection(self, x, n_segment, print_cluster=False):
        numpy_x = x.clone()
        numpy_x = numpy_x.detach().cpu().numpy()
        hamming = distance.hamming(np.sign(numpy_x[0]), np.sign(numpy_x[1]))
        accum_dis = [0, hamming]
        sample_num = numpy_x.shape[0]
        for j in range(1, sample_num - 1):
            hamming = distance.hamming(np.sign(numpy_x[j]), np.sign(numpy_x[j + 1]))
            accum_dis.append(hamming + accum_dis[-1])
        dis_index = accum_dis[-1] / n_segment

        if dis_index == 0:
            return x[:n_segment]

        cnt = 1
        clus = []
        new_x = None
        for k in range(sample_num):
            if accum_dis[k] <= dis_index * cnt:
                clus.append(k)
            else:
                if new_x is None:
                    # if len(clus) > 4:
                    #     mid = len(clus) // 4
                    #     new_x = torch.sum(x[clus[mid-1]:clus[mid+2] + 1], 0, keepdim=True) / 4
                    # else:
                    new_x = torch.sum(x[clus[0]:clus[-1] + 1], 0, keepdim=True) / len(clus)
                else:
                    # if len(clus) > 4:
                    #     mid = len(clus) // 4
                    #     tmp = torch.sum(x[clus[mid-1]:clus[mid+2] + 1], 0, keepdim=True) / 4
                    # else:
                    tmp = torch.sum(x[clus[0]:clus[-1] + 1], 0, keepdim=True) / len(clus)
                    new_x = torch.cat((new_x, tmp), dim=0)
                if print_cluster:
                    print(clus)
                clus = []
                cnt += 1
                clus.append(k)
        while new_x.size(0) < n_segment:
            # tmp = torch.sum(x[clus[0]:clus[-1] + 1], 0, keepdim=True) / len(clus)
            new_x = torch.cat((new_x, tmp), dim=0)

        return new_x

    def slope_selection(self, x, n_segment, print_cluster=False):
        if n_segment == 1:
            return torch.sum(x[:], 0, keepdim=True) / 32
        numpy_x = x.clone()
        numpy_x = numpy_x.detach().cpu().numpy()
        slope = []
        sample_num = numpy_x.shape[0]
        for j in range(sample_num-1):
            hamming = distance.hamming(np.sign(numpy_x[j]), np.sign(numpy_x[j+1]))
            slope.append(hamming)
        # print('slope:', slope)
        partition = sorted(range(len(slope)), key=lambda i: slope[i])[-(n_segment-1):]
        partition.sort()
        if print_cluster:
            print('partition:', partition)
        # if partition[0] > 2:
        #     mid = partition[0] // 2
        #     new_x = torch.sum(x[mid-1:mid+1], 0, keepdim=True) / 2
        # else:
        new_x = torch.sum(x[0:partition[0]+1], 0, keepdim=True) / (partition[0]+1)
        # print([0, partition[0]+1])
        for c in range(n_segment-2):
            # if (partition[c+1] - partition[c]) > 2:
            #     mid = (partition[c+1]+partition[c]) // 2
            #     tmp = torch.sum(x[mid-1:mid + 1], 0, keepdim=True) / 2
            #     new_x = torch.cat((new_x, tmp), dim=0)
            # else:
            tmp = torch.sum(x[partition[c]+1:partition[c+1]+1], 0, keepdim=True) / (partition[c+1]-partition[c])
            new_x = torch.cat((new_x, tmp), dim=0)
        # if x[partition[(n_segment-1)-1]+1:].size(0) > 2:
        #     mid = partition[(n_segment-1)-1] + 1 + x[partition[(n_segment-1)-1]+1:].size(0) // 2
        #     tmp = torch.sum(x[mid-1:mid+1], 0, keepdim=True) / 2
        # else:
        tmp = torch.sum(x[partition[n_segment-2]+1:], 0, keepdim=True) / x[partition[(n_segment-1)-1]+1:].size(0)
        new_x = torch.cat((new_x, tmp), dim=0)
        # print(torch.from_numpy(new_x).float().cuda().size())
        return new_x

    def even_segments(self, x, n_segments):
        numpy_x = x.clone()
        numpy_x = numpy_x.detach().cpu().numpy()
        sample_num = numpy_x.shape[0]
        per_cluster_frames = sample_num // n_segments
        # print('frames per cluster', per_cluster_frames)
        # if per_cluster_frames > 4:
        #     mid = per_cluster_frames // 4
        #     new_x = torch.sum(x[mid-1:mid+3], 0, keepdim=True) / 4
        #     for i in range(1, n_segments):
        #         tmp = torch.sum(x[i * per_cluster_frames+mid-1:i * per_cluster_frames + mid +3], 0,
        #                         keepdim=True) / 4
        #         new_x = torch.cat((new_x, tmp), dim=0)
        # else:
        new_x = torch.sum(x[0:per_cluster_frames], 0, keepdim=True) / per_cluster_frames
        for i in range(1, n_segments):
            tmp = torch.sum(x[i*per_cluster_frames:i*per_cluster_frames+per_cluster_frames], 0, keepdim=True) / per_cluster_frames
            new_x = torch.cat((new_x, tmp), dim=0)
        return new_x

    def uneven_segments(self, x, n_segments):
        numpy_x = x.clone()
        numpy_x = numpy_x.detach().cpu().numpy()
        sample_num = numpy_x.shape[0]
        offsets = sorted(random.sample(range(1, sample_num-1), n_segments-1))
        # print('uneven segments:', offsets)
        # if offsets[0] > 2:
        #     mid = offsets[0] // 2
        #     new_x = torch.sum(x[mid-1:mid+1], 0, keepdim=True) / 2
        # else:
        new_x = torch.sum(x[0:offsets[0]], 0, keepdim=True) / offsets[0]
        for i in range(1, 7):
            # if (offsets[i]-offsets[i-1]) > 2:
            #     mid = (offsets[i]+offsets[i-1]) // 2
            #     tmp = torch.sum(x[mid-1:mid+1], 0, keepdim=True) / 2
            # else:
            tmp = torch.sum(x[offsets[i-1]:offsets[i]], 0, keepdim=True) / (offsets[i]-offsets[i-1])
            new_x = torch.cat((new_x, tmp), dim=0)
        # if x[offsets[i]:].size(0) > 2:
        #     mid = offsets[i] + x[offsets[i]:].size(0) // 2
        #     tmp = torch.sum(x[mid-1:mid+1], 0, keepdim=True) / 2
        # else:
        tmp = torch.sum(x[offsets[i]:], 0, keepdim=True) / x[offsets[i]:].size(0)
        # print('last tmp size:', tmp.size())
        new_x = torch.cat((new_x, tmp), dim=0)
        return new_x
