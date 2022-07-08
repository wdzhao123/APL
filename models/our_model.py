import torch
from models.DeblurNet import CNN_for_Generator, Discriminator
from models.DBDNet import CNN_for_DBD

def Create_nets(args):
    DBD = CNN_for_DBD()
    generator = CNN_for_Generator()
    discriminator = Discriminator(args)

    if torch.cuda.is_available():
        DBD = DBD.cuda()
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    if args.epoch_start != 0:
        # Load pretrained models
        DBD.load_state_dict(torch.load('log/%s-%s/%s/DBD_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,args.epoch_start)))
        generator.load_state_dict(torch.load('log/%s-%s/%s/generator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,args.epoch_start)))
        discriminator.load_state_dict(torch.load('log/%s-%s/%s/discriminator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,args.epoch_start)))

    return DBD, generator, discriminator