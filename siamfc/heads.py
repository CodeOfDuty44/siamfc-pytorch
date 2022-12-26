from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = ['SiamFC']


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
    
    def forward(self, z, x):
        x0 = x[0].unsqueeze(0)
        x1 = x[1].unsqueeze(0)
        x2 = x[2].unsqueeze(0)
        res0 = self.pixel_corr_mat(x0, z).sum(dim = 1, keepdim = True)
        res1 = self.pixel_corr_mat(x1, z).sum(dim = 1, keepdim = True)
        res2 = self.pixel_corr_mat(x2, z).sum(dim = 1, keepdim = True)
        res = torch.cat([res0,res1,res2], dim = 0)
        return res * self.out_scale

        # res = self._fast_xcorr(z, x) * self.out_scale
        # return res
    
    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out

    def ccor_normed_depthwise(self, x,kernel):
        """depthwise normalized cross correlation  
        x, k: tensors of shape (batch,channel, ,h, w)
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        kernelHeigth, kernelWidth = kernel.size(2), kernel.size(3)
        kernelArea = kernelHeigth * kernelWidth
        outHeigth = x.shape[2] - kernel.shape[2] + 1
        outWidth = x.shape[3] - kernel.shape[3] + 1
        out = torch.zeros((batch, channel, outHeigth, outWidth ))
        if x.is_cuda: out = out.to(device = "cuda")
        epsilon = 10**(-12) #for nan values
        kernelSqr = torch.sum( kernel*kernel, (2,3) ) #for normalization
        
        for i in range(outHeigth):
            for j in range(outWidth):
                window = x[:,:, i: i+kernelHeigth, j: j+kernelWidth] 
                result = window * kernel
                result = torch.sum(result, (2,3))

                #normalize
                windowSqr = torch.sum( window*window, (2,3) )
                normalizer = torch.sqrt(windowSqr * kernelSqr)
                result = result / normalizer
                result = torch.nan_to_num(result, nan=epsilon ) #set nan values to epsilon
                out[:,:,i,j] = result

        return out

    def ccoeff_depthwise(self, x,kernel):
        """depthwise cross correlation coefficient
        x, k: tensors of shape (batch,channel, ,h, w)
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        kernelHeigth, kernelWidth = kernel.size(2), kernel.size(3)
        kernelArea = kernelHeigth * kernelWidth
        #calculate T'
        kernelprime = kernel - ( torch.sum(kernel, (2,3), keepdim = True) / kernelArea )
        outHeigth = x.shape[2] - kernel.shape[2] + 1
        outWidth = x.shape[3] - kernel.shape[3] + 1
        out = torch.zeros((batch, channel, outHeigth, outWidth ))
        if x.is_cuda: out = out.to(device = "cuda")

        for i in range(outHeigth):
            for j in range(outWidth):
                window = x[:,:, i: i+kernelHeigth, j: j+kernelWidth]
                windowSum = torch.sum(window, (2,3), keepdim = True)
                #calculate I'
                windowprime = window - windowSum / kernelArea 
                result = windowprime * kernelprime
                result = torch.sum(result, (2,3))
                out[:,:,i,j] = result

        return out

    def pixel_corr_mat(self, x, z):
        """Pixel-wise correlation (implementation by matrix multiplication)
        The speed is faster because the computation is vectorized"""
        b, c, h, w = x.size()
        z_mat = z.view((b, c, -1)).transpose(1, 2)  # (b, hz * wz, c)
        x_mat = x.view((b, c, -1))  # (b, c, hx * wx)
        return torch.matmul(z_mat, x_mat).view((b, z.shape[2]*z.shape[3], h, w))  # (b, hz * wz, hx * wx) --> (b, hz * wz, hx, wx)
