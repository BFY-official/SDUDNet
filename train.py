import os
import itertools
import torch
import torchvision
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from net import Generator, DNN, EDNet
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn import init
from math import exp
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LEARNING_RATE1 = 1e-4  
LEARNING_RATE2 = 1e-4  
BATCH_SIZE = 1
IMAGE_SIZE = 128
CHANNELS_IMG = 2  # 1
Z_DIM = 1
NUM_EPOCHS = 30
FEATURES_DISC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01
LAMBDA_GP = 10


mean = 1
std = 0.3
shape = (BATCH_SIZE, Z_DIM, IMAGE_SIZE, IMAGE_SIZE)

transform = transforms.Compose([
    transforms.RandomCrop(IMAGE_SIZE),
    # transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.5],
    #     std=[0.5]
    # )
])

noisy_transform = transforms.Compose([
    transforms.RandomCrop(IMAGE_SIZE),  
    transforms.ToTensor(),
])

clean_transform = transforms.Compose([
    transforms.RandomCrop(IMAGE_SIZE),  
    # transforms.Resize(IMAGE_SIZE),   
    transforms.ToTensor(),
])


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m): 
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


train_path = './train'


class Mydataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.clean_images = os.listdir(os.path.join(root_dir, "final"))
        self.noisy_images = os.listdir(os.path.join(root_dir, "real_sar"))

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean_img_name = os.path.join(self.root_dir, "final", self.clean_images[idx])
        noisy_img_name = os.path.join(self.root_dir, "real_sar", self.noisy_images[idx])

        clean_image = Image.open(clean_img_name)
        noisy_image = Image.open(noisy_img_name)


        clean_image = clean_transform(clean_image)
        noisy_image = noisy_transform(noisy_image)

        return noisy_image, clean_image


dataset = Mydataset(root_dir='train')
dataLoader = DataLoader(
    dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

def total_variation(image_in):

    tv_h = torch.sum(torch.abs(image_in[ :, :-1] - image_in[ :, 1:]))
    tv_w = torch.sum(torch.abs(image_in[ :-1, :] - image_in[ 1:, :]))
    tv_loss = tv_h + tv_w

    return tv_loss 

def TV_loss(im_batch, weight):
    TV_L = 0.0

    for tv_idx in range(len(im_batch)):
        TV_L = TV_L + total_variation(im_batch[tv_idx,0,:,:])

    TV_L = TV_L/len(im_batch)

    return weight*TV_L



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class log_SSIM_loss(nn.Module):
    def __init__(self, window_size=11, channel=1, is_cuda=True, size_average=True):
        super(log_SSIM_loss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.window = create_window(window_size, channel)
        if is_cuda:
            self.window = self.window.cuda()


    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return -torch.log10(ssim_map.mean())


o2s = Generator.Generator().to(device)
sar = EDNet.UNet().to(device)
optical = EDNet.UNet().to(device)
s2o = DNN.DNN().to(device)

optimizer_G = torch.optim.Adam(itertools.chain(o2s.parameters(), s2o.parameters()), lr=LEARNING_RATE1, betas=(0.9, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(sar.parameters(), optical.parameters()), lr=LEARNING_RATE2,
                               betas=(0.9, 0.999))

loss_fn = nn.MSELoss().cuda()
loss_idt = nn.L1Loss().cuda()
loss_dn = nn.L1Loss().cuda()
loss_ssim = log_SSIM_loss().cuda()

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
writer_dnn = SummaryWriter(f"logs/dnn")
writer_gt = SummaryWriter(f"logs/gt")
writer_d_sar = SummaryWriter(f"logs/d_sar")

init_weights(o2s, init_type='kaiming', init_gain=0.02)
init_weights(sar, init_type='kaiming', init_gain=0.02)
init_weights(optical, init_type='kaiming', init_gain=0.02)
init_weights(s2o, init_type='kaiming', init_gain=0.02)

step = 0

for epoch in range(NUM_EPOCHS):

    epoch_loss_o2s = 0.0
    epoch_loss_s2o = 0.0
    epoch_loss_sar = 0.0
    epoch_loss_optical = 0.0
    epoch_loss_cycle = 0.0

    o2s.train()
    s2o.train()
    sar.train()
    optical.train()

    for batch_idx, (real_sar, real_optical) in enumerate(dataLoader):
        real_sar = real_sar.to(device)
        real_optical = real_optical.to(device)
        cur_batch_size = real_sar.shape[0]

        optimizer_D.zero_grad()


        "sar2optical"
        # train Discriminator with real data
        output_real_optical_local, output_real_optical_global = optical(real_optical)
        # output_real_optical_local = output_real_optical_local.reshape(-1)
        # output_real_optical_global = output_real_optical_global.reshape(-1)
        lossD_optical_real_local = loss_fn(output_real_optical_local, torch.ones_like(output_real_optical_local))
        lossD_optical_real_global = loss_fn(output_real_optical_global, torch.ones_like(output_real_optical_global))
        lossD_optical_real = lossD_optical_real_local + lossD_optical_real_global*0.1


        # train Discriminator with fake data
        # noise = torch.randn(size=(cur_batch_size, Z_DIM, 64, 64), device=device)
        fake_optical, speckle = s2o(real_sar)
        output_fake_optical_local, output_fake_optical_global = optical(fake_optical.detach())
        # output_fake_optical_local = output_fake_optical_local.reshape(-1)
        # output_fake_optical_global = output_fake_optical_global.reshape(-1)
        lossD_optical_fake_local = loss_fn(output_fake_optical_local, torch.zeros_like(output_fake_optical_local))
        lossD_optical_fake_global = loss_fn(output_fake_optical_global, torch.zeros_like(output_fake_optical_global))
        lossD_optical_fake = lossD_optical_fake_local + lossD_optical_fake_global*0.1


        lossD_optical = (lossD_optical_real + 0.1*lossD_optical_fake) * 0.5
        lossD_optical.backward()



        "optical2sar"
        # train Discriminator with real data
        output_real_sar_local, output_real_sar_global = sar(real_sar)
        # output_real_sar_local = output_real_sar_local.reshape(-1)
        # output_real_sar_global = output_real_sar_global.reshape(-1)
        lossD_sar_real_local = loss_fn(output_real_sar_local, torch.ones_like(output_real_sar_local))
        lossD_sar_real_global = loss_fn(output_real_sar_global, torch.ones_like(output_real_sar_global))
        lossD_sar_real = lossD_sar_real_local + lossD_sar_real_global*0.1


        # train Discriminator with fake data
        # speckle = torch.randn(size=(cur_batch_size, Z_DIM, IMAGE_SIZE, IMAGE_SIZE), device=device)
        # speckle = torch.normal(mean= mean, std = std, size=shape, device=device)
        fake_sar = o2s(real_optical, speckle)
        output_fake_sar_local, output_fake_sar_global = sar(fake_sar.detach())
        # output_fake_sar_local = output_fake_sar_local.reshape(-1)
        # output_fake_sar_global = output_fake_sar_global.reshape(-1)
        lossD_sar_fake_local = loss_fn(output_fake_sar_local, torch.zeros_like(output_fake_sar_local))
        lossD_sar_fake_global = loss_fn(output_fake_sar_global, torch.zeros_like(output_fake_sar_global))
        lossD_sar_fake = lossD_sar_fake_local + lossD_sar_fake_global*0.1

        lossD_sar = (lossD_sar_real + lossD_sar_fake) * 0.5
        lossD_sar.backward()



        optimizer_D.step()

        epoch_loss_sar += lossD_sar
        epoch_loss_optical += lossD_optical

        # < train Generator > ########################
        for i in range(3):
            optimizer_G.zero_grad()


            "sar2optical"

            fake_optical, speckle = s2o(real_sar)
            output_fake_optical_local, output_fake_optical_global = optical(fake_optical)
            # output_fake_optical_local = output_fake_optical_local.reshape(-1)
            # output_fake_optical_global = output_fake_optical_global.reshape(-1)
            lossG_optical_local = loss_fn(output_fake_optical_local, torch.ones_like(output_fake_optical_local))
            lossG_optical_global = loss_fn(output_fake_optical_global, torch.ones_like(output_fake_optical_global))
            lossG_optical = lossG_optical_local + lossG_optical_global*0.1

            idt_optical, temp = s2o(real_optical)
            lossG_idt_optical = loss_idt(idt_optical, real_optical)




            "optical2sar"
            # speckle = torch.randn(size=(cur_batch_size, Z_DIM, IMAGE_SIZE, IMAGE_SIZE), device=device)
            # speckle = torch.normal(mean= mean, std = std, size=shape, device=device)
            fake_sar = o2s(real_optical, speckle)
            output_fake_sar_local,  output_fake_sar_global= sar(fake_sar)
            # output_fake_sar_local = output_fake_sar_local.reshape(-1)
            # output_fake_sar_global = output_fake_sar_global.reshape(-1)
            lossG_sar_local = loss_fn(output_fake_sar_local, torch.ones_like(output_fake_sar_local))
            lossG_sar_global = loss_fn(output_fake_sar_global, torch.ones_like(output_fake_sar_global))
            lossG_sar = lossG_sar_local + lossG_sar_global*0.1


            idt_sar= o2s(real_sar, speckle)
            lossG_idt_sar = loss_idt(idt_sar, real_sar)



            "cycle"
            ## |G_y(G_x(y)) - y|
            cycle_optical, speckle = s2o(fake_sar)
            lossC_optical = loss_dn(cycle_optical, real_optical)

            # lossC_sar = loss_dn(o2s(fake_optical, speckle), real_sar)
            # loss_tv = TV_loss(cycle_optical, 0.000005)
            los_ssim = loss_ssim(cycle_optical, real_optical)

            loss_cycle =  lossC_optical
            loss_identity = (lossG_idt_sar + lossG_idt_optical)*0.5

            loss_G = 2*loss_identity + 6*loss_cycle + lossG_optical + 3*lossG_sar + 3*los_ssim
            loss_G.backward()
            optimizer_G.step()

        epoch_loss_o2s += lossG_sar
        epoch_loss_s2o += lossG_optical
        epoch_loss_cycle += loss_cycle



        if batch_idx % 100 == 0 and batch_idx > 0:
            o2s.eval()
            s2o.eval()
            sar.eval()
            optical.eval()
            print(
                f"Epoch[{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataLoader)}\
                Loss D: {lossD_sar:.6f},Loss G: {lossG_sar:.6f},Loss DNN: {lossD_optical:.6f}"
            )
            with torch.no_grad():
                fake_img = o2s(real_optical, speckle)
                desp_img, temp = s2o(fake_img)
                desp_sar, temp = s2o(real_sar)
                # DrawGen(gen, epoch, fixed_noise, con)
                img_grid_real = torchvision.utils.make_grid(
                    real_sar, normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake_img, normalize=True
                )
                img_grid_dnn = torchvision.utils.make_grid(
                    desp_img, normalize=True
                )
                img_grid_gt = torchvision.utils.make_grid(
                    real_optical, normalize=True
                )
                img_grid_d_sar = torchvision.utils.make_grid(
                    desp_sar, normalize=True
                )

                writer_real.add_image("RealImg", img_grid_real, global_step=step)
                writer_fake.add_image("fakeImg", img_grid_fake, global_step=step)
                writer_dnn.add_image("despImg", img_grid_dnn, global_step=step)
                writer_gt.add_image("GT_Img", img_grid_gt, global_step=step)
                writer_d_sar.add_image("Desp_SAR", img_grid_d_sar, global_step=step)

            step += 1
            o2s.train()
            sar.train()
            s2o.train()
            optical.train()

    
    data_len = len(dataset)
    epoch_loss_o2s /= data_len
    epoch_loss_s2o /= data_len
    epoch_loss_sar /= data_len
    epoch_loss_optical /= data_len
    epoch_loss_cycle /= data_len

    print(f"--------Epoch[{epoch}/{NUM_EPOCHS}] epoch_loss_o2s: {epoch_loss_o2s:.6f},epoch_loss_s2o: {epoch_loss_s2o:.6f}, epoch_loss_sar: {epoch_loss_sar:.6f}, epoch_loss_optical: {epoch_loss_optical:.6f}, epoch_loss_cycle: {epoch_loss_cycle:.6f}--------")
        
    
    torch.save(s2o.state_dict(), f"models/share_final_ssim3_s2o_{epoch}.pth")
    torch.save(o2s.state_dict(), f"models/share_final_ssim3_o2s_{epoch}.pth")
    torch.save(sar.state_dict(), f"models/share_final_ssim3_sar_{epoch}.pth")
    torch.save(optical.state_dict(), f"models/share_final_ssim3_optical_{epoch}.pth")


    torch.save(optimizer_D.state_dict(), f"models/share_final_optimizer_D_{epoch}.pth")
    torch.save(optimizer_G.state_dict(), f"models/share_final_optimizer_G_{epoch}.pth")

    # torch.save(obj=s2o, f=f'models/mix_ssim2_s2o_{epoch}.pth')
    # torch.save(obj=o2s, f=f'models/mix_ssim2_o2s_{epoch}.pth')
    # torch.save(obj=sar, f=f'models/mix_ssim2_sar_{epoch}.pth')
    # torch.save(obj=optical, f=f'models/mix_ssim2_optical_{epoch}.pth')


torch.save(obj=s2o, f='models/s2o.pth')
torch.save(obj=o2s, f='models/o2s.pth')
torch.save(obj=sar, f='models/sar.pth')
torch.save(obj=optical, f='models/optical.pth')
