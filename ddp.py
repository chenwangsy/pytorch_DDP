import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

#DDP需要编写的内容
#1.  用torch.utils.data.distributed.DistributedSampler()来包装原有的dataset，
#    其返回值作为一个特别的sampler用于dataloader的sampler参数
#    from torch.utils.data.distributed import DistributedSampler

#2.  需要用argparse获取local_rank这个参数的值
#    DDP要求不同的进程跑相同的代码，但是不同的进程完成的工作又有些差异
#    因此，每个进程一定要清楚自己的“定位”。local_rank代表着一个进程在一个机子中的序号，是进程的一个身份标识。
#    因此DDP需要local_rank作为一个变量被进程捕获，在程序的很多位置，这个变量可以用来标识进程编号，同时也是对应的GPU编号。
#    我们将args.local_rank的取值作为torch.cuda.set_device()的参数。那么后面每当我们调用XX.cuda()或是XX.to('cuda')时，
#    就都将XX放到了序号为local_rank的GPU上。因此，虽然每个进程运行同一份代码，但由于每个进程local_rank不同，每个进程的操作也不同了

	# parser = argparse.ArgumentParser()
	# parser.add_argument("--local_rank", default=-1, type=int)
	# args = parser.parse_args()
	#
	# torch.cuda.set_device(args.local_rank)

#3.   初始化进程组, 如果用的是GPU平台，直接选'nccl'就好了
#       dist.init_process_group(backend='nccl')

#4.   生产DDP model。只需要用DDP“包一下”原有model就可以了，device_ids这个参数需要用列表列出此进程对应的GPU，用local_rank来代表
#     model = DDP(model, device_ids=[args.local_rank])

#5.   trainloader.sampler.set_epoch(epoch)
#     DistributedSampler需要保证各个进程打乱后的结果是相同的（方便分配任务），并且保证各个进程分到的数据是不重叠的
#      各个进程对完整数据集打乱的时候，只有采用相同的随机种子，才能保证打乱的结果相同。
#      这句话的含义就是将epoch值作为所有进程的随机种子，来统一所有进程的shuffle结果

#6.   启动方式 python -m torch.distributed.launch --nproc_per_node 2 main.py



#================================================================================================================================================
#代码示例
################
## main.py文件
import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


### 1. 基础模块 ###
# 假设我们的模型是这个，与DDP无关
class ToyModel(nn.Module):
	def __init__(self):
		super(ToyModel, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


# 假设我们的数据是这个
def get_dataset():
	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                           download=True, transform=transform)
	# DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。
	#      用，就完事儿！sampler的原理，第二篇中有介绍。
	train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
	# DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。
	#      也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
	trainloader = torch.utils.data.DataLoader(my_trainset,
	                                          batch_size=16, num_workers=2, sampler=train_sampler)
	return trainloader


### 2. 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 准备数据，要在DDP初始化之后进行
trainloader = get_dataset()

# 构造模型
model = ToyModel().to(local_rank)
# DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
	model.load_state_dict(torch.load(ckpt_path))
# DDP: 构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 假设我们的loss是这个
loss_func = nn.CrossEntropyLoss().to(local_rank)

### 3. 网络训练  ###
model.train()
iterator = tqdm(range(100))
for epoch in iterator:
	# DDP：设置sampler的epoch，
	# DistributedSampler需要这个来指定shuffle方式，
	# 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
	trainloader.sampler.set_epoch(epoch)
	# 后面这部分，则与原来完全一致了。
	for data, label in trainloader:
		data, label = data.to(local_rank), label.to(local_rank)
		optimizer.zero_grad()
		prediction = model(data)
		loss = loss_func(prediction, label)
		loss.backward()
		iterator.desc = "loss = %0.3f" % loss
		optimizer.step()
	# DDP:
	# 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
	#    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
	# 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
	if dist.get_rank() == 0:
		torch.save(model.module.state_dict(), "%d.ckpt" % epoch)

################
## Bash运行
# DDP: 使用torch.distributed.launch启动DDP模式
# 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 main.py
