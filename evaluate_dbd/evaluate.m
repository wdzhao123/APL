clc
clear

imgpath = './result/Ours-DUT/';
GTpath = './dataset/test/dut500-gt/';

ImgEnum=dir([GTpath '*.bmp']);
ImgNum=length(ImgEnum);
ids={ImgEnum.name};
n = ImgNum;
MAE1 = [];
tic;
for i= 1 :n
    imname_GT = [ GTpath ImgEnum(i).name ];
    GT = imread(imname_GT);
    GT = imresize(GT,[160,160]);
    [ height,width ] = size(GT);
    for m = 1:height
        for n = 1:width
            if GT(m,n)<125
                GT(m,n)=255;
            else
                GT(m,n)=0;
            end
        end
    end
    GT = double(GT);
    PixNum = height*width;  
    imname1 = [ imgpath ImgEnum(i).name(1:end-4) '.bmp'];
    Img1 = double( imread( imname1 ) );
    Img1 = imresize(Img1,[160,160]);
    for m = 1:height
        for n = 1:width
            Img1(m,n)=255-Img1(m,n);
        end
    end
    error1 = abs(Img1-GT(:,:,1))/255;
    error1_sum = sum(error1(:))/PixNum;
    MAE1 = [MAE1,error1_sum];
end
basedir = GTpath;
gt_dir = { 'best', basedir };
alg_dir = ...
{
        {'Ours', imgpath, [], '', 'bmp'}    
};
addpath ('./subcode/');
alg_dir_FF = candidateAlgStructure( alg_dir );
dataset = datasetStructure( gt_dir(1), gt_dir(2) );

[ mPre, mRecall, mFmeasure, mFmeasureWF , mFalseAlarm, AUC ] = ...
performCalcu(dataset,alg_dir_FF);

save( [ basedir 'base_', gt_dir{1} ], 'mPre', 'mRecall', 'mFmeasure', 'mFmeasureWF', 'mFalseAlarm', 'AUC' );
curve_bar_plot_b( basedir, gt_dir, alg_dir, mPre, mRecall, mFmeasure, mFmeasureWF , mFalseAlarm);
MAE_11 = sum(MAE1)/ ImgNum
disp(['FMeasure=']);
disp([mFmeasure(3,:)]);
toc;