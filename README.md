# APL in PyTorch
Implementation of "United Defocus Blur Detection and Deblurring via Adversarial Promoting Learning" in PyTorch

# Datasets
* `train_data`:
   * `1204source`: Contains 604 training images of CUHK Dataset and 600 training images of DUT Dataset.
   * `FC`: 500 natural full clear images.
* `test_data of DBD`: 
   * `CUHK`: Contains 100 testing images of CUHK Dataset and it's GT.
   * `DUT`: Contains 500 testing images of DUT Dataset and it's GT.

* `test_data of Deblurring`:
   * `CUHK`: Contains 100 testing images.
   * `DUT`: Contains 500 testing images.
   * `DP`: Contains 76 testing images of DP Dataset and it's GT.

Download and unzip datasets from https://pan.baidu.com/s/1_3Z7w9IrlqOU25QbVdhytQ, password: aktv to "./dataset".


# Test
You can use the following command to test：

>python test.py --image_path TEST_DATA_PATH --result_save_path RESULT_IMAGE_PATH

You can use the following model to output results directly.Here is our parameters:
baidu link: https://pan.baidu.com/s/1sAbhPioPCLrsAid1W8UMAg?pwd=t8hq passward: t8hq

Put "DBD.pth" and "deblur.pth" in "./saved_models".

# Train
You can use the following command to train：

>python train.py --data_root TRAIN_DATA_PATH

* `train.py`: the entry point for training.
* `models/our_model.py`: the whole model of APL.
* `models/DBDNet.py`: defines the architecture of the DBD Generator models.
* `models/DeblurNet.py`: defines the architecture of the Deblur Generator models and Discriminator models.
* `options.py`: creates option lists using argparse package. More individuals are dynamically added in other files as well.
* `datasets.py`: process the dataset before passing to the network.
* `optimizer.py`: defines the optimization and losses used in APL.

# Eval
### DBD
If you want to use Fmax and MAE to evaluate the results, you can run the following code in MATLAB. It shows the PR curve and F-measure curve at the same time.

>./evaluate_dbd/evaluate.m

### Deblurring
If you want to use PSNR, SSIM and MAE to evaluate the result, use the following code:

>python evaluate.py --image_save_path RESULT_IMAGE_PATH --test_gt_path GT_PATH
