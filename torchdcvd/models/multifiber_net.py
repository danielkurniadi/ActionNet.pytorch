from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MultiFiberNet3d',]


class MultiFiberNet3d(nn.Module):
	""" Multi-Fiber network implementation for action net classification

    Arguments
    ------------------------------
		.. num_classes (int):
				number of classification group
	"""

	def __init__(self, num_classes):
		super().__init__()

		self.num_classes = num_classes

		self.conv1 = nn.Sequential(OrderedDict([
			('conv', nn.Conv3d(3, 16, kernel_size=(3,5,5), padding=(1,2,2), stride=(1,2,2))),
			('bn', nn.BatchNorm3d(16)),
		]))

		self.max_pool1 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

		self.conv2 = nn.Sequential(OrderedDict([
			('B01', MultiFiberUnit(16, 96, 96, (2,1,1), is_first=True)),
			('B02', MultiFiberUnit(96, 96, 96, (1,1,1))),
			('B03', MultiFiberUnit(96, 96, 96, (1,1,1))),
		]))

		self.conv3 = nn.Sequential(OrderedDict([
			('B01', MultiFiberUnit(96, 192, 192, (1,2,2), is_first=True)),
			('B02', MultiFiberUnit(192, 192, 192, (1,1,1))),
			('B03', MultiFiberUnit(192, 192, 192, (1,1,1))),
			('B04', MultiFiberUnit(192, 192, 192, (1,1,1)))
		]))

		self.conv4 = nn.Sequential(OrderedDict([
			('B01', MultiFiberUnit(192, 384, 384, (1,2,2), is_first=True)),
			('B02', MultiFiberUnit(384, 384, 384, (1,1,1))),
			('B03', MultiFiberUnit(384, 384, 384, (1,1,1))),
			('B04', MultiFiberUnit(384, 384, 384, (1,1,1))),
			('B05', MultiFiberUnit(384, 384, 384, (1,1,1))),
			('B06', MultiFiberUnit(384, 384, 384, (1,1,1)))
		]))

		self.conv5 = nn.Sequential(OrderedDict([
			('B01', MultiFiberUnit(384, 768, 768, (1,2,2), is_first=True)),
			('B02', MultiFiberUnit(768, 768, 768, (1,1,1))),
			('B03', MultiFiberUnit(768, 768, 768, (1,1,1))),
		]))

		self.bn_1 = nn.BatchNorm3d(768)
		self.global_pool1 = nn.AvgPool3d(kernel_size=(8,7,7), stride=(1,1,1))

		self.classifier = nn.Linear(768, num_classes)

	def forward(self, x):
		"""
		Arguments
		------------------------------
			.. x (torch.Tensor) input of shape [N x C x T x W x H] which matches 
				input dimension to Conv3d: https://pytorch.org/docs/stable/nn.html#conv3d
		"""
		assert (len(x.shape) == 5) and (x.shape[2] == 16)

		h = F.relu(self.conv1(x))
		h = self.max_pool1(h)
		h = self.conv2(h)
		h = self.conv3(h)
		h = self.conv4(h)
		h = self.conv5(h)

		h = F.relu(self.bn_1(h))
		h = self.global_pool1(h)

		h = h.view(h.shape[0], -1)
		h = self.classifier(x)

		return h


class MultiFiberUnit(nn.Module):
	""" Unit component of Multi Fiber Network. Contains Multiplexer and Adapter
	Arguments
	------------------------------
		.. in_ch (int)
		.. mux_ch (int)
		.. out_ch (int)
		.. main_stride (tuple[int])
		.. is_first (bool)
	"""

	def __init__(self, in_ch, mux_ch, out_ch, main_stride, is_first=False):
		super().__init__()

		ix_ch = mux_ch // 4

		self.is_first = is_first

		self.conv_i1 = _conv_fiber_unit(in_ch, ix_ch, kernel_size=(1,1,1), padding=(0,0,0), stride=(1,1,1), groups=1)
		self.conv_i2 = _conv_fiber_unit(ix_ch, in_ch, kernel_size=(1,1,1), padding=(0,0,0), stride=(1,1,1), groups=1)
		self.conv_m1 = _conv_fiber_unit(in_ch, mux_ch, kernel_size=(3,3,3), padding=(1,1,1), stride=main_stride, groups=16)  # multiplexer

		if is_first:
			self.conv_m2 = _conv_fiber_unit(mux_ch, out_ch, kernel_size=(1,1,1), padding=(0,0,0), stride=(1,1,1), groups=1)
			self.conv_w1 = _conv_fiber_unit(in_ch, out_ch, kernel_size=(1,1,1), padding=(0,0,0), stride=main_stride, groups=1)  # adapter
		else:
			self.conv_m2 = _conv_fiber_unit(mux_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1), stride=(1,1,1), groups=16)

	def forward(self, x):
		"""
		Arguments
		------------------------------
			.. x (torch.Tensor) input of shape [N x C x T x W x H] which matches 
				input dimension to Conv3d: https://pytorch.org/docs/stable/nn.html#conv3d
		"""
		assert (len(x.shape) == 5) and (x.shape[2] == 16)

		h = self.conv_i1(x)
		x_in = x + self.conv_i2(h)

		h = self.conv_m1(x_in)
		h = self.conv_m2(h)

		if self.is_first:
			x = self.conv_w1(x)

		return h + x


def _conv_fiber_unit(in_channels, out_channels, kernel_size, bias=False,
					padding=(0,0,0), stride=(1,1,1), groups=1):
	""" Helper to contruct convolution layer of BN + Relu + Conv

	Arguments
	------------------------------
		.. in_channels
		.. out_channels
		.. kernel_size
		.. bias
		.. padding
		.. stride
		.. groups
	"""
	return nn.Sequential(OrderedDict([
		('bn', nn.BatchNorm3d(in_channels)),
		('relu', nn.ReLU(inplace=True)),
		('conv', nn.Conv3d(**locals()))
	]))
