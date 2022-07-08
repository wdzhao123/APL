import torch
# Optimizers
def Get_optimizers(args, DBD, generator, discriminator):

    # SGD for Discriminator, Adam for Generator
    optimizer_DBD = torch.optim.SGD(
        DBD.parameters(),
        lr=args.lr, momentum=0.5)
    optimizer_G = torch.optim.SGD(
        generator.parameters(),
        lr=args.lr, momentum=0.5)
    optimizer_D = torch.optim.SGD(
        discriminator.parameters(),
        lr=args.lr, momentum=0.5)

    return optimizer_DBD, optimizer_G, optimizer_D

# Loss functions
def Get_loss_func(args):
    criterion_GAN = torch.nn.BCELoss()
    criterion_pixelwise = torch.nn.MSELoss()
    criterion_pixelwise_bce = torch.nn.BCEWithLogitsLoss()
    criterion_pixelwise_mae = torch.nn.L1Loss()
    criterion_pixelwise_cel = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()
        criterion_pixelwise_bce.cuda()
        criterion_pixelwise_mae.cuda()
        criterion_pixelwise_cel.cuda()
    return criterion_GAN, criterion_pixelwise, criterion_pixelwise_mae, criterion_pixelwise_bce, criterion_pixelwise_cel

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag