import sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributed as dist
import kornia
import math
from utils import *
from custom_models.utils import *
from custom_transforms import plot_debug_images
from tqdm import tqdm
from typing import List, Sequence, Union, Tuple, Optional, Dict

_SOFTPLUS_UNITY_ = 1.4427
_SIGMOID_UNITY_ = 2.0
_EPS_ = 1e-8 # small regularization constant

__all__ = ['innerTest', 'innerTrain', 'classTrain', 'vizStat',
            'LossModel', 'AugmentModelNONE', 'AugmentModel', 'hyperHesTrain']


def metricCE(logit, target):
    return F.cross_entropy(logit, target)


class MemoryBankModule(torch.nn.Module):
    """Memory bank implementation

    This is a parent class to all loss functions implemented by the lightly
    Python package. This way, any loss can be used with a memory bank if
    desired.

    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.

    Examples:
        >>> class MyLossFunction(MemoryBankModule):
        >>>
        >>>     def __init__(self, memory_bank_size: int = 2 ** 16):
        >>>         super(MyLossFunction, self).__init__(memory_bank_size)
        >>>
        >>>     def forward(self, output: Tensor,
        >>>                 labels: Tensor = None):
        >>>
        >>>         output, negatives = super(
        >>>             MyLossFunction, self).forward(output)
        >>>
        >>>         if negatives is not None:
        >>>             # evaluate loss with negative samples
        >>>         else:
        >>>             # evaluate loss without negative samples

    """

    def __init__(self, size: int = 2**16):
        super(MemoryBankModule, self).__init__()

        if size < 0:
            msg = f"Illegal memory bank size {size}, must be non-negative."
            raise ValueError(msg)

        self.size = size
        self.register_buffer(
            "bank", tensor=torch.empty(0, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "bank_ptr", tensor=torch.empty(0, dtype=torch.long), persistent=False
        )

    @torch.no_grad()
    def _init_memory_bank(self, dim: int) -> None:
        """Initialize the memory bank if it's empty

        Args:
            dim:
                The dimension of the which are stored in the bank.

        """
        # create memory bank
        # we could use register buffers like in the moco repo
        # https://github.com/facebookresearch/moco but we don't
        # want to pollute our checkpoints
        bank: Tensor = torch.randn(dim, self.size).type_as(self.bank)
        self.bank: Tensor = torch.nn.functional.normalize(bank, dim=0)
        self.bank_ptr: Tensor = torch.zeros(1).type_as(self.bank_ptr)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: Tensor) -> None:
        """Dequeue the oldest batch and add the latest one

        Args:
            batch:
                The latest batch of keys to add to the memory bank.

        """
        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)

        if ptr + batch_size >= self.size:
            self.bank[:, ptr:] = batch[: self.size - ptr].T.detach()
            self.bank_ptr[0] = 0
        else:
            self.bank[:, ptr : ptr + batch_size] = batch.T.detach()
            self.bank_ptr[0] = ptr + batch_size

    def forward(
        self,
        output: Tensor,
        labels: Optional[Tensor] = None,
        update: bool = False,
    ) -> Union[Tuple[Tensor, Optional[Tensor]], Tensor]:
        """Query memory bank for additional negative samples

        Args:
            output:
                The output of the model.
            labels:
                Should always be None, will be ignored.

        Returns:
            The output if the memory bank is of size 0, otherwise the output
            and the entries from the memory bank.

        """

        # no memory bank, return the output
        if self.size == 0:
            return output, None

        _, dim = output.shape

        # initialize the memory bank if it is not already done
        if self.bank.nelement() == 0:
            self._init_memory_bank(dim)

        # query and update memory bank
        bank = self.bank.clone().detach()

        # only update memory bank if we later do backward pass (gradient)
        if update:
            self._dequeue_and_enqueue(output)

        return output, bank

class NTXentLoss(MemoryBankModule):
    """Implementation of the Contrastive Cross Entropy Loss.

    This implementation follows the SimCLR[0] paper. If you enable the memory
    bank by setting the `memory_bank_size` value > 0 the loss behaves like
    the one described in the MoCo[1] paper.

    - [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    - [1] MoCo, 2020, https://arxiv.org/abs/1911.05722

    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        memory_bank_size:
            Number of negative samples to store in the memory bank.
            Use 0 for SimCLR. For MoCo we typically use numbers like 4096 or 65536.
        gather_distributed:
            If True then negatives from all gpus are gathered before the
            loss calculation. This flag has no effect if memory_bank_size > 0.

    Raises:
        ValueError: If abs(temperature) < 1e-8 to prevent divide by zero.

    Examples:

        >>> # initialize loss function without memory bank
        >>> loss_fn = NTXentLoss(memory_bank_size=0)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(output)

    """

    def __init__(
        self,
        temperature: float = 0.5,
        memory_bank_size: int = 0,
        gather_distributed: bool = False,
    ):
        super(NTXentLoss, self).__init__(size=memory_bank_size)
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8

        if abs(self.temperature) < self.eps:
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(self, out0: torch.Tensor, out1: torch.Tensor):
        """Forward pass through Contrastive Cross-Entropy Loss.

        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as
        negative samples.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            Contrastive Cross Entropy Loss value.

        """

        device = out0.device
        batch_size, _ = out0.shape

        # normalize the output to length 1
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        # ask memory bank for negative samples and extend it with out1 if
        # out1 requires a gradient, otherwise keep the same vectors in the
        # memory bank (this allows for keeping the memory bank constant e.g.
        # for evaluating the loss on the test set)
        # out1: shape: (batch_size, embedding_size)
        # negatives: shape: (embedding_size, memory_bank_size)
        out1, negatives = super(NTXentLoss, self).forward(
            out1, update=out0.requires_grad
        )

        # We use the cosine similarity, which is a dot product (einsum) here,
        # as all vectors are already normalized to unit length.
        # Notation in einsum: n = batch_size, c = embedding_size and k = memory_bank_size.

        if negatives is not None:
            # use negatives from memory bank
            negatives = negatives.to(device)

            # sim_pos is of shape (batch_size, 1) and sim_pos[i] denotes the similarity
            # of the i-th sample in the batch to its positive pair
            sim_pos = torch.einsum("nc,nc->n", out0, out1).unsqueeze(-1)

            # sim_neg is of shape (batch_size, memory_bank_size) and sim_neg[i,j] denotes the similarity
            # of the i-th sample to the j-th negative sample
            sim_neg = torch.einsum("nc,ck->nk", out0, negatives)

            # set the labels to the first "class", i.e. sim_pos,
            # so that it is maximized in relation to sim_neg
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)

        else:
            # user other samples from batch as negatives
            # and create diagonal mask that only selects similarities between
            # views of the same image
            if self.gather_distributed and dist.world_size() > 1:
                # gather hidden representations from other processes
                out0_large = torch.cat(dist.gather(out0), 0)
                out1_large = torch.cat(dist.gather(out1), 0)
                diag_mask = dist.eye_rank(batch_size, device=out0.device)
            else:
                # single process
                out0_large = out0
                out1_large = out1
                diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)

            # calculate similiarities
            # here n = batch_size and m = batch_size * world_size
            # the resulting vectors have shape (n, m)
            logits_00 = torch.einsum("nc,mc->nm", out0, out0_large) / self.temperature
            logits_01 = torch.einsum("nc,mc->nm", out0, out1_large) / self.temperature
            logits_10 = torch.einsum("nc,mc->nm", out1, out0_large) / self.temperature
            logits_11 = torch.einsum("nc,mc->nm", out1, out1_large) / self.temperature

            # remove simliarities between same views of the same image
            logits_00 = logits_00[~diag_mask].view(batch_size, -1)
            logits_11 = logits_11[~diag_mask].view(batch_size, -1)

            # concatenate logits
            # the logits tensor in the end has shape (2*n, 2*m-1)
            logits_0100 = torch.cat([logits_01, logits_00], dim=1)
            logits_1011 = torch.cat([logits_10, logits_11], dim=1)
            logits = torch.cat([logits_0100, logits_1011], dim=0)

            # create labels
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            if self.gather_distributed:
                labels = labels + dist.rank() * batch_size
            labels = labels.repeat(2)

        loss = self.cross_entropy(logits, labels)

        return loss
      
class ValLossModel(nn.Module):
    def __init__(self, N, C, init_targets, apply, model, grad, sym, device):
        super().__init__()

        dim_mlp = 10 #TODO change 
        fc = nn.Linear(dim_mlp, 4)
        if True:
            fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), fc)
        self.head = fc

    def forward(self, idx, logit, target):
        criterion = nn.CrossEntropyLoss().cuda()
        return criterion(logit, target)

class AugmentModelNONE(nn.Module):
    def __init__(self):
        super(AugmentModelNONE, self).__init__()

    def forward(self, idx, x):
        return x


class AugmentModel(nn.Module):
    def __init__(self, N, magn, apply, mode, grad, device):
        super(AugmentModel, self).__init__()
        # enable/disable manual augmentation
        self.N = N
        self.apply = apply
        self.device = device
        self.mode = mode # mode
        self.K = 7 # number of affine magnitude params
        self.J = 0 # number of color magnitude params
        K = self.K
        J = self.J
        magnNorm = torch.ones(1)*magn/10.0 # normalize to 10 like in RandAugment
        probNorm = torch.ones(1)*1/(K-2) # 1/(K-2) probability
        magnLogit = torch.log(magnNorm/(1-magnNorm)) # convert to logit
        probLogit = torch.log(probNorm/(1-probNorm)) # convert to logit
        # affine transforms (mid, range)
        self.angle = [0.0, 30.0] # [-30.0:30.0] rotation angle
        self.trans = [0.0, 0.45] # [-0.45:0.45] X/Y translate
        self.shear = [0.0, 0.30] # [-0.30:0.30] X/Y shear
        self.scale = [1.0, 0.50] # [ 0.50:1.50] X/Y scale
        # color transforms (mid, range)
        #self.bri = [0.0, 0.9] # [-0.9:0.9] brightness
        #self.con = [1.0, 0.9] # [0.1:1.9] contrast
        #self.sat = [0.1, 1.9] # [-0.30:0.30] saturation
        #self.hue = [1.0, 0.50] # [ 0.70:1.30] hue
        #self.gam = [1.0, 0.50] # [ 0.70:1.30] gamma
        #
        self.actP = nn.Sigmoid()
        self.actM = nn.Sigmoid()
        self.paramP = nn.Parameter(probLogit*torch.ones(K+J,N), requires_grad=grad)
        self.paramM = nn.Parameter(magnLogit*torch.ones(K+J,N), requires_grad=grad)

    def forward(self, idx, x):
        B,C,H,W = x.shape
        device = self.device
        mode = self.mode
        if self.apply:
            K = self.K
            J = self.J
            # learnable hyperparameters
            if self.N == 1:
                paramPos = torch.log(    self.actP(self.paramP)).repeat(1,B) # [-Inf:0]
                paramNeg = torch.log(1.0-self.actP(self.paramP)).repeat(1,B) # [-Inf:0]
                paramM = self.actM(self.paramM).repeat(1,B) # (K+J)xB [0:1], default=magn
            else:
                paramPos = torch.log(    self.actP(self.paramP[:,idx])) # [-Inf:0]
                paramNeg = torch.log(1.0-self.actP(self.paramP[:,idx])) # [-Inf:0]
                paramM = self.actM(self.paramM[:,idx]) # (K+J)xB [0:1], default=magn
            paramP = torch.cat([paramPos.view(-1,1), paramNeg.view(-1,1)], dim=1) # B*(K+J)x2
            # reparametrize probabilities and magnitudes
            sampleP = F.gumbel_softmax(paramP, tau=1.0, hard=True).to(device) # B*(K+J)x2
            sampleP = sampleP[:,0]
            sampleP = sampleP.reshape(K,B)
            # reparametrize magnitudes
            #sampleM = paramM[:K] * torch.rand(K,B).to(device) # KxB, prior: U[0,1]
            sampleM = paramM[:K] * torch.randn(K,B).to(device) # KxB, prior: N(0,1)
            # affine augmentations
            R: torch.tensor = torch.zeros(B,3,3).to(device) + torch.eye(3).to(device)
            # define the rotation angle
            ANG: torch.tensor = sampleP[0] * sampleM[0] * self.angle[1] # B(0/1)*U[0,1]*M/10
            # define the rotation center
            CTR: torch.tensor = torch.cat((W*torch.ones(B).to(device)//2, H*torch.ones(B).to(device)//2)).view(-1,2)
            # define the scale factor
            SCL: torch.tensor = torch.zeros_like(CTR).to(device)
            SCL[:,0] = self.scale[0] + sampleP[5] * sampleM[5] * self.scale[1] # mid + B(0/1)*U[0,1]*M/10
            SCL[:,1] = self.scale[0] + sampleP[6] * sampleM[6] * self.scale[1] # mid + B(0/1)*U[0,1]*M/10
            R[:,0:2] = kornia.get_rotation_matrix2d(CTR, ANG, SCL)
            # translation: border not defined yet
            T: torch.tensor = torch.zeros_like(R) + torch.eye(3).to(device)
            T[:,0,2] = W * sampleP[1] * sampleM[1] * self.trans[1]
            T[:,1,2] = H * sampleP[2] * sampleM[2] * self.trans[1]
            # shear: check this
            S: torch.tensor = torch.zeros_like(R) + torch.eye(3).to(device)
            S[:,0,1] = sampleP[3] * sampleM[3] * self.shear[1]
            S[:,1,0] = sampleP[4] * sampleM[4] * self.shear[1]
            # apply the transformation to original image
            M: torch.tensor = torch.bmm(torch.bmm(S,T),R)
            if mode == 0: #upscale
                x = kornia.geometry.resize(x, (4*H, 4*W))
                x_warped: torch.tensor = kornia.warp_perspective(x, M, dsize=(4*H,4*W), border_mode='border')
                x_warped = kornia.geometry.resize(x_warped, (H,W))
            else:
                x_warped: torch.tensor = kornia.warp_perspective(x, M, dsize=(H,W), border_mode='border')
            ## color augmentations
            #if mode == 1:
            #    BRI: torch.tensor = self.bri[0] + sampleP[6] * sampleC[0] * self.bri[1] # mid + B(0/1)*U[0,1]*M/10
            #    CON: torch.tensor = self.con[0] + sampleP[7] * sampleC[1] * self.con[1]
            #    x_color = kornia.adjust_brightness(kornia.adjust_contrast(x_warped, CON), BRI)
            #else:
            #    x_color = x_warped
            
            return x_warped

        else: # process val to compensate for Kornia artifacts!
            if mode == 0: #upscale
                M: torch.tensor = torch.zeros(B,3,3).to(device) + torch.eye(3).to(device)
                x = kornia.geometry.resize(x, (4*H, 4*W))
                x_warped: torch.tensor = kornia.warp_perspective(x, M, dsize=(4*H,4*W), border_mode='border')
                x_warped = kornia.geometry.resize(x_warped, (H,W))
            else:
                x_warped = x #torch.tensor = kornia.warp_perspective(x, M, dsize=(H,W), border_mode='border')
            ## color augmentations
            #if mode == 1:
            #    BRI: torch.tensor = torch.zeros(B).to(device)
            #    CON: torch.tensor = torch.ones(B).to(device)
            #    x_color = kornia.adjust_brightness(kornia.adjust_contrast(x_warped, CON), BRI)
            #else:
            #    x_color = x_warped
            
            return x_warped


def hyperHesTrain(args, encoder, decoder, optimizer, device, valid_loader, train_loader, epoch, start,
        trainLosModel, trainAugModel, validLosModel, validAugModel, hyperOptimizer):
    print(encoder)
    print("HYPERHESTRAIN start")
    encoder.eval()
    decoder.eval()
    validLosModel.eval()
    validAugModel.eval()
    trainLosModel.train()
    trainAugModel.train()
    M = len(valid_loader.dataset)
    N = len(train_loader.dataset)
    B = len(train_loader) # number of batches
    v_loader_iterator = iter(valid_loader)
    t_loader_iterator = iter(train_loader)
    dDivs = 4*[0.0]
    #
    for param_group in optimizer.param_groups:
        task_lr = param_group['lr']
    hyperParams = list()
    for n,p in trainAugModel.named_parameters():
        if p.requires_grad:
            hyperParams.append(p)
    for n,p in trainLosModel.named_parameters():
        if p.requires_grad:
            hyperParams.append(p)
    theta = list()
    # _theta = list()
    for n,p in decoder.named_parameters():
        if (len(args.hyper_theta) == 0) or (any([e in n for e in args.hyper_theta]) and ('.weight' in n)):
            theta.append(p)
    
    # _decoder = validLosModel.head
    # for n,p in _decoder.named_parameters(): 
    #     _theta.append(p)

    
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    v0Norms    = AverageMeter()
    v1Norms    = AverageMeter()
    mvpNorms   = AverageMeter()
    end = time.time()
    ######## start train hyper model ########
    for batch_idx in range(0,1*B):
        # sample val batch
        try:
            vData, vTarget, vIndex = next(v_loader_iterator)
        except StopIteration:
            v_loader_iterator = iter(valid_loader)
            vData, vTarget, vIndex = next(v_loader_iterator)
        # sample train batch
        try:
            tData, tTarget, tIndex = next(t_loader_iterator)
        except StopIteration:
            t_loader_iterator = iter(train_loader)
            tData, tTarget, tIndex = next(t_loader_iterator)
        #
        tData  = tData.to(device)
        tTarget= tTarget.to(device)
        tIndex = tIndex.to(device)

        print(f'tData.shape: {tData.shape}, tTarget.shape: {tTarget.shape}, tIndex.shape: {tIndex.shape}')
        print(f'vData.shape: {vData.shape}, vTarget.shape: {vTarget.shape}, vIndex.shape: {vIndex.shape}')
        
        vData  = vData.to(device)
        vTarget= vTarget.to(device)
        vIndex = vIndex.to(device)
        
        
        n_rot_images = 4*vData.shape[0]
        use_images = vData
        nimages = vData.shape[0]
        rotated_images = torch.zeros([n_rot_images, use_images.shape[1], use_images.shape[2], use_images.shape[3]]).cuda()
        rot_classes = torch.zeros([n_rot_images]).long().cuda()

        rotated_images[:nimages] = use_images
        # rotate 90
        rotated_images[nimages:2*nimages] = use_images.flip(3).transpose(2,3)
        rot_classes[nimages:2*nimages] = 1
        # rotate 180
        rotated_images[2*nimages:3*nimages] = use_images.flip(3).flip(2)
        rot_classes[2*nimages:3*nimages] = 2
        # rotate 270
        rotated_images[3*nimages:4*nimages] = use_images.transpose(2,3).flip(3)
        rot_classes[3*nimages:4*nimages] = 3

        vTarget = rot_classes.to(device)
        # output = model(head="rotnet", im_q=rotated_images)
        # rot_loss = criterion(output, target)
        
        
        # measure data loading time
        data_time.update(time.time() - end)
        # warm-up learning rate
        hyper_lr = hyper_warmup_learning_rate(args, epoch-start, batch_idx, B, hyperOptimizer)
        # v1 = dL_v / dTheta: Lx1
        optimizer.zero_grad()
        # vData = validAugModel(vIndex, vData)
        vEncode = encoder(rotated_images)
        vOutput = decoder(vEncode)
        print(f'vEncode.shape: {vEncode.shape}')
    
        # print(f'vOutput.shape: {vOutput[0].shape}')
        # validLosModel.set_fuck(vEncode.shape[1])

        vOutput = validLosModel.head(vOutput[0])
        vLoss = validLosModel(vIndex, vOutput, vTarget)
        print(vLoss)
        

        ## raise prev part

        g1 = torch.autograd.grad(vLoss, theta)
        v1 = [e.detach().clone() for e in g1]
        # v0 = dL_t / dTheta: Lx1
        optimizer.zero_grad()
        tData = trainAugModel(tIndex, tData)
        tEncode = encoder(tData)
        tOutput = decoder(tEncode)
        tLoss = trainLosModel(tIndex, tOutput, tTarget)
        g0 = torch.autograd.grad(tLoss, theta, create_graph=True)
        v0 = [e.detach().clone() for e in g0]
        # v2 = H^-1 * v0: Lx1
        v2 = [-e.detach().clone() for e in v1]
        if args.hyper_iters > 0: # Neumann series
            for j in range(0, args.hyper_iters):
                ns = torch.autograd.grad(g0, theta, grad_outputs=v1, create_graph=True)
                v1 = [v1[l] - args.hyper_alpha*e for l,e in enumerate(ns)]
                v2 = [v2[l] - e.detach().clone() for l,e in enumerate(v1)]
        # MVP compute
        v0Norm = torch.sum(torch.cat([t.detach().clone().view(-1)*v.detach().clone().view(-1) for t,v in zip(v0,v0)])) # gLt*gLt
        v1Norm = torch.sum(torch.cat([t.detach().clone().view(-1)*v.detach().clone().view(-1) for t,v in zip(v1,v1)])) # gLv*gLv
        v2Norm = torch.sum(torch.cat([t.detach().clone().view(-1)*v.detach().clone().view(-1) for t,v in zip(v0,v1)])) # gLv*gLt
        mmd = v0Norm + v1Norm -2.0*v2Norm
        bDivs = list([v0Norm, v1Norm, -2.0*v2Norm, mmd])
        dDivs = [e1+e2 for e1,e2 in zip(dDivs, bDivs)]
        mvpNorm = mmd
        #
        v0Norms.update(v0Norm.item())
        v1Norms.update(v1Norm.item())
        mvpNorms.update(mvpNorm.item())
        # v3 = (dL_t / dLambda) * v2: Px1
        hyperOptimizer.zero_grad()
        torch.autograd.backward(g0, grad_tensors=v2)
        hyperOptimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # logger
        if batch_idx % 200 == 0: #args.log_interval == 0:
            print('hyperTrain batch {:.0f}% ({}/{}), task_lr={:.6f}, hyper_lr={:.6f}\t'
                #'gLtNorm {v0Norm.val:.4f} ({v0Norm.avg:.4f})\t'
                #'gLvNorm {v1Norm.val:.4f} ({v1Norm.avg:.4f})\t'
                #'mvpNorm {mvpNorm.val:.4f} ({mvpNorm.avg:.4f})\n'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                100.0*batch_idx/B, batch_idx, B, task_lr, hyper_lr,
                #v0Norm=v0Norms, v1Norm=v1Norms, mvpNorm=mvpNorms,
                batch_time=batch_time, data_time=data_time))
    #
    dDivs = [e/B for e in dDivs]
    #print('Epoch: {}\t Divergence: {:.4f}'.format(epoch, dDivs[-1]))
    return dDivs


def innerTrain(args, encoder, decoder, optimizer, device, loader, epoch, augModel):
    encoder.train() # train encoder model
    decoder.train() # train decoder model
    augModel.eval() # use fixed hyperModel parameters
    #
    N = len(loader.dataset) # dataset size
    B = len(loader) # number of batches
    train_loss = 0.0
    #
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    criterion = NTXentLoss()
    #
    for batch_idx, data in tqdm(enumerate(loader), total=len(loader)):
        # print(f'batch_idx: {batch_idx}')
        image = data[0].to(device)
        target= data[1].to(device)
        index = data[2].to(device)
        # measure data loading time
        data_time.update(time.time() - end)
        # warm-up learning rate
        lr = warmup_learning_rate(args, epoch, batch_idx, B, optimizer)
        # model+loss
        aug_image = augModel(index, image)
        encode0 = encoder(image)
        encode1 = encoder(aug_image)
        output0 = decoder(encode0)
        output1 = decoder(encode1)
        loss = criterion(output0, output1)
        losses.update(loss.item())
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot images from first batch for debugging
        if args.plot_debug and (epoch == 0) and (batch_idx < 64):
            fname = 'ori_train_batch_{}_{}.png'.format(batch_idx, args.dataset)
            plot_debug_images(args, rows=2, cols=2, imgs=image, fname=fname)
            fname = 'aug_train_batch_{}_{}.png'.format(batch_idx, args.dataset)
            plot_debug_images(args, rows=2, cols=2, imgs=aug_image, fname=fname)
        # logger
        if batch_idx % args.log_interval == 0:
            print('innerTrain batch {:.0f}% ({}/{}), lr={:.6f}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                100.0*batch_idx/B, batch_idx, B, lr,
                loss=losses, batch_time=batch_time, data_time=data_time))
    #
    train_loss /= B
    print('Epoch: {}\t Inner Train loss: {:.4f}'.format(epoch, train_loss))
    return train_loss


def classTrain(args, encoder, decoder, optimizer, device, loader, epoch, losModel, augModel):
    encoder.eval() # eval encoder
    decoder.train() # train decoder
    losModel.eval() # use fixed hyperModel parameters
    augModel.eval() # use fixed hyperModel parameters
    #
    N = len(loader.dataset) # dataset size
    B = len(loader) # number of batches
    train_loss = 0.0
    #
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    #
    for batch_idx, data in enumerate(loader):
        image = data[0].to(device)
        target= data[1].to(device)
        index = data[2].to(device)
        # measure data loading time
        data_time.update(time.time() - end)
        # warm-up learning rate
        lr = warmup_learning_rate(args, epoch, batch_idx, B, optimizer)
        # augment+encoder
        with torch.no_grad():
            aug_image = augModel(index, image)
            encode = encoder(aug_image)
        # classifier
        output = decoder(encode.detach())
        loss = losModel(index, output, target)
        losses.update(loss.item())
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot images from first batch for debugging
        if args.plot_debug and (epoch == 0) and (batch_idx < 64):
            fname = 'ori_train_batch_{}_{}.png'.format(batch_idx, args.dataset)
            plot_debug_images(args, rows=2, cols=2, imgs=image, fname=fname)
            fname = 'aug_train_batch_{}_{}.png'.format(batch_idx, args.dataset)
            plot_debug_images(args, rows=2, cols=2, imgs=aug_image, fname=fname)
        # logger
        if batch_idx % args.log_interval == 0:
            print('classTrain batch {:.0f}% ({}/{}), lr={:.6f}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                100.0*batch_idx/B, batch_idx, B, lr,
                loss=losses, batch_time=batch_time, data_time=data_time))
    #
    train_loss /= B
    print('Epoch: {}\t Class Train loss: {:.4f}'.format(epoch, train_loss))
    return train_loss


def innerTest(args, encoder, decoder, device, loader, epoch):
    encoder.eval() # eval encoder
    decoder.eval() # eval decoder
    M = len(loader.dataset) # dataset size
    B = len(loader) # number of batches
    test_loss = 0.0
    correct = 0
    miss_indices = list()
    #
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    #
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(loader), total=len(loader)):
            image = data[0].to(device)
            target= data[1].to(device)
            index = data[2].to(device)
            # plot images for debugging
            if args.plot_debug and (epoch == 0) and (batch_idx < 64):
                fname = 'test_batch_{}_{}.png'.format(batch_idx, args.dataset)
                plot_debug_images(args, rows=2, cols=2, imgs=image, fname=fname)
            #
            encode = encoder(image)
            output = decoder(encode)
            loss = metricCE(output[0], target)
            losses.update(loss)
            test_loss += loss
            pred = output[0].max(1, keepdim=True)[1] # get the index of the max probability
            match = pred.eq(target.view_as(pred))
            correct += match.sum().item()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # logger
            if batch_idx % args.log_interval == 0:
                print('innerTest batch {:.0f}% ({}/{})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(100.0*batch_idx/B, batch_idx, B,
                    loss=losses, batch_time=batch_time))
    #
    test_loss /= B
    acc = 100.0 * correct / M
    print('Epoch: {}\t Test loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(epoch, test_loss, correct, M, acc))

    return acc, test_loss, miss_indices


def vizStat(args, encoder, decoder, device, loader, T, F):
    encoder.eval() # eval encoder
    decoder.eval() # eval decoder
    total_loss = 0.0
    correct = 0
    #
    M = len(loader.dataset) # dataset size
    B = len(loader) # number of batches
    #
    fv = torch.zeros(T, F).to(device)
    gt = torch.zeros(T, dtype=torch.long).to(device)
    pr = torch.zeros(T, dtype=torch.long).to(device)
    mi = torch.tensor([], dtype=torch.long).to(device)
    #
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    #
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            image  = data[0].to(device)
            target = data[1].to(device)
            index  = data[2].to(device)
            #
            encode = encoder(image)
            output = decoder(encode)
            loss = metricCE(output[0], target)
            losses.update(loss)
            total_loss += loss
            pred = output[0].max(1, keepdim=True)[1] # get the index of the max probability
            match = pred.eq(target.view_as(pred))
            mask = match.le(0)
            fv[index] = encode
            gt[index] = target.view(-1)
            pr[index] = pred.view(-1)
            mi = torch.cat([mi, torch.masked_select(index.view(-1, 1), mask)])
            correct += match.sum().item()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # logger
            if 1: #batch_idx % args.log_interval == 0:
                print('vizTest batch {:.0f}% ({}/{})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(100.0*batch_idx/B, batch_idx, B,
                    loss=losses, batch_time=batch_time))
    #
    total_loss /= B
    acc = 100.0 * correct / M
    print('Loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(total_loss, correct, M, acc))

    return fv, gt, pr, mi
