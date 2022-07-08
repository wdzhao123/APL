from torch.autograd import Variable
from models.our_model import Create_nets
from datasets import *
from options import TrainOptions
from optimizer import *
import numpy as np

args = TrainOptions().parse()

# Initialize generator and discriminator
DBD, generator, discriminator = Create_nets(args)
# Loss functions
criterion_GAN, criterion_pixelwise, criterion_pixelwise_mae, criterion_pixelwise_bce, criterion_pixelwise_cel = Get_loss_func(args)
# Optimizers
optimizer_DBD, optimizer_G, optimizer_D = Get_optimizers(args, DBD, generator, discriminator)
# Configure dataloaders
path_input=args.data_root
dataloader = Get_dataloader_train(path_input, args.batch_size)

for epoch in range(0, args.epoch_num):
    for i, sample in enumerate(dataloader):
        image, LBP, clear = sample['image'], sample['mask'], sample['clear']
        input_image=Variable(image).cuda()
        LBP_image = Variable(LBP).cuda()
        Dis_C1_image1 = Variable(clear).cuda()

        patch=(1,1,1)
        valid = Variable(torch.FloatTensor(np.ones((input_image.size(0),*patch))).cuda(), requires_grad=False)
        fake = Variable(torch.FloatTensor(np.zeros((input_image.size(0),*patch))).cuda(), requires_grad=False)

        optimizer_DBD.zero_grad()
        requires_grad(DBD, True)
        requires_grad(generator, False)
        requires_grad(discriminator, False)

        # optimizer_G.zero_grad()
        # requires_grad(DBD, False)
        # requires_grad(generator, True)
        # requires_grad(discriminator, False)

        dbd_result = DBD(input_image)
        syn_image = generator(input_image, dbd_result)
        pred_fake = discriminator(syn_image)
        loss_GAN1 = criterion_GAN(pred_fake, valid)
        loss_mae = criterion_pixelwise_bce(dbd_result, LBP_image)

        loss_DBD = loss_GAN1 + loss_mae
        loss_DBD.backward()
        optimizer_DBD.step()

        # loss_GAN1.backward()
        # optimizer_G.step()



        optimizer_D.zero_grad()
        requires_grad(DBD, False)
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        # Real loss
        pred_real = discriminator(Dis_C1_image1)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(syn_image.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        if i%args.sample_interval==0:
            print(
                "\r[Epoch%d]-[Batch%d]-[loss_DBD:%f]-[loss_Dis:%f]" %
                (epoch, i, loss_DBD.data.cpu(), loss_D.data.cpu()))

    # Save model checkpoints
    if epoch == args.epoch_num:
        torch.save(DBD.state_dict(), 'log/%s-%s/%s/DBD_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,epoch))
        torch.save(generator.state_dict(), 'log/%s-%s/%s/generator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,epoch))
        torch.save(discriminator.state_dict(), 'log/%s-%s/%s/discriminator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir, epoch))