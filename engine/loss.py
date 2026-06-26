import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


class ContourLoss(torch.nn.Module):
    def __init__(self):
        super(ContourLoss, self).__init__()

    def forward(self, pred, target, weight=10):
        '''
        target, pred: tensor of shape (B, C, H, W), where target[:,:,region_in_contour] == 1,
                        target[:,:,region_out_contour] == 0.
        weight: scalar, length term weight.
        '''
        # length term
        delta_r = pred[:,:,1:,:] - pred[:,:,:-1,:] # horizontal gradient (B, C, H-1, W) 
        delta_c = pred[:,:,:,1:] - pred[:,:,:,:-1] # vertical gradient   (B, C, H,   W-1)

        delta_r    = delta_r[:,:,1:,:-2]**2  # (B, C, H-2, W-2)
        delta_c    = delta_c[:,:,:-2,1:]**2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c) 

        epsilon = 1e-8 # where is a parameter to avoid square root is zero in practice.
        length = torch.mean(torch.sqrt(delta_pred + epsilon)) # eq.(11) in the paper, mean is used instead of sum.

        c_in  = torch.ones_like(pred)
        c_out = torch.zeros_like(pred)

        region_in  = torch.mean( pred     * (target - c_in )**2 ) # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.mean( (1-pred) * (target - c_out)**2 ) 
        region = region_in + region_out

        loss =  weight * length + region

        return loss


class IoULoss(torch.nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1
            # IoU loss is (1-IoU1)
            IoU = IoU + (1-IoU1)
        # return IoU/b
        return IoU


class PatchIoULoss(torch.nn.Module):
    def __init__(self):
        super(PatchIoULoss, self).__init__()
        self.iou_loss = IoULoss()

    def forward(self, pred, target):
        win_y, win_x = 64, 64
        iou_loss = 0.
        for anchor_y in range(0, target.shape[0], win_y):
            for anchor_x in range(0, target.shape[1], win_y):
                patch_pred = pred[:, :, anchor_y:anchor_y+win_y, anchor_x:anchor_x+win_x]
                patch_target = target[:, :, anchor_y:anchor_y+win_y, anchor_x:anchor_x+win_x]
                patch_iou_loss = self.iou_loss(patch_pred, patch_target)
                iou_loss += patch_iou_loss
        return iou_loss


class ThrReg_loss(torch.nn.Module):
    def __init__(self):
        super(ThrReg_loss, self).__init__()

    def forward(self, pred, gt=None):
        return torch.mean(1 - ((pred - 0) ** 2 + (pred - 1) ** 2))


def weighted_seg_loss(pred, target, weight, eps=1e-6):
    """Pixel-wise confidence weighted BCE + soft IoU loss.

    Shape:
        pred: [B, 1, H, W], logit or probability.
        target: [B, 1, H, W], soft pseudo-label in [0, 1].
        weight: [B, 1, H, W], detached pixel-wise confidence or boosted confidence.
    """
    if not torch.is_tensor(pred) or not torch.is_tensor(target) or not torch.is_tensor(weight):
        raise TypeError("pred, target, and weight must be torch.Tensor")

    prob = _prediction_to_probability(pred, eps=eps)
    target = _prepare_loss_map(target, prob, mode='bilinear').clamp(0.0, 1.0)
    weight = _prepare_loss_map(weight.detach(), prob, mode='bilinear').clamp_min(0.0)
    weight = torch.nan_to_num(weight, nan=0.0, posinf=1.0, neginf=0.0)

    if target.size(1) == 1 and prob.size(1) != 1:
        target = target.expand(-1, prob.size(1), -1, -1)
    if weight.size(1) == 1 and prob.size(1) != 1:
        weight = weight.expand(-1, prob.size(1), -1, -1)

    structure_weight = 1.0 + 5.0 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
    pixel_weight = weight * structure_weight

    wbce = F.binary_cross_entropy(prob, target, reduction='none')
    wbce = (wbce * pixel_weight).sum() / (pixel_weight.sum() + eps)

    inter = (prob * target * pixel_weight).sum(dim=(2, 3))
    union = ((prob + target) * pixel_weight).sum(dim=(2, 3))
    wiou = 1.0 - (inter + eps) / (union - inter + eps)
    wiou = torch.nan_to_num(wiou, nan=0.0, posinf=0.0, neginf=0.0).mean()
    return wbce + wiou


def _prediction_to_probability(pred, eps=1e-6):
    with torch.no_grad():
        finite = torch.isfinite(pred.detach())
        if finite.any():
            pred_min = float(pred.detach()[finite].min().item())
            pred_max = float(pred.detach()[finite].max().item())
            is_probability = pred_min >= -eps and pred_max <= 1.0 + eps
        else:
            is_probability = False
    prob = pred if is_probability else pred.sigmoid()
    return torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0).clamp(eps, 1.0 - eps)


def _prepare_loss_map(value, ref, mode='bilinear'):
    if value.dim() == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.dim() == 3:
        value = value.unsqueeze(1)
    elif value.dim() != 4:
        raise ValueError("loss map must have shape [H,W], [B,H,W], or [B,C,H,W]")
    value = value.to(device=ref.device, dtype=ref.dtype)
    if value.size(0) != ref.size(0):
        if value.size(0) == 1:
            value = value.expand(ref.size(0), -1, -1, -1)
        else:
            raise ValueError("loss map batch size must match prediction batch size")
    if tuple(value.shape[-2:]) != tuple(ref.shape[-2:]):
        value = F.interpolate(value, size=ref.shape[-2:], mode=mode, align_corners=False)
    return value



class PixLoss(nn.Module):
    """
    Pixel loss for each refined map output.
    """
    def __init__(self, config):
        super(PixLoss, self).__init__()
        self.config = config
        self.lambdas_pix_last = self.config.lambdas_pix_last

        self.criterions_last = {}
        if 'bce' in self.lambdas_pix_last and self.lambdas_pix_last['bce']:
            self.criterions_last['bce'] = nn.BCELoss()
        if 'iou' in self.lambdas_pix_last and self.lambdas_pix_last['iou']:
            self.criterions_last['iou'] = IoULoss()
        if 'iou_patch' in self.lambdas_pix_last and self.lambdas_pix_last['iou_patch']:
            self.criterions_last['iou_patch'] = PatchIoULoss()
        if 'ssim' in self.lambdas_pix_last and self.lambdas_pix_last['ssim']:
            self.criterions_last['ssim'] = SSIMLoss()
        if 'mse' in self.lambdas_pix_last and self.lambdas_pix_last['mse']:
            self.criterions_last['mse'] = nn.MSELoss()
        if 'reg' in self.lambdas_pix_last and self.lambdas_pix_last['reg']:
            self.criterions_last['reg'] = ThrReg_loss()
        if 'cnt' in self.lambdas_pix_last and self.lambdas_pix_last['cnt']:
            self.criterions_last['cnt'] = ContourLoss()

    def forward(self, scaled_preds, gt):
        loss = 0.
        for _, pred_lvl in enumerate(scaled_preds):
            if pred_lvl.shape != gt.shape:
                pred_lvl = nn.functional.interpolate(pred_lvl, size=gt.shape[2:], mode='bilinear', align_corners=True)
            pred_lvl = pred_lvl.sigmoid()
            for criterion_name, criterion in self.criterions_last.items():
                _loss = criterion(pred_lvl, gt) * self.lambdas_pix_last[criterion_name]
                loss += _loss
                # print(criterion_name, _loss.item())
        return loss


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def saliency_structure_consistency(x, y):
    ssim = torch.mean(SSIM(x,y))
    return ssim

def __sinetv2_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def sinetv2_loss(preds, mask):
    return __sinetv2_loss(preds[0], mask) + __sinetv2_loss(preds[1], mask) + __sinetv2_loss(preds[2], mask) + __sinetv2_loss(preds[3], mask)
