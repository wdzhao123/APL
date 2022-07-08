%%
function [ mPre, mRecall, mFmeasure, mFmeasureWF , mFalseAlarm, AUC ] = ...
    performCalcu(datasetStruct,algStructArray)

evaluateSal = @(sMap,thresholds,gtMap) thresholdBased_HR_FR(sMap,thresholds,gtMap);

fprintf(['\nEvaluating dataset: ' datasetStruct.datasetName '\n']);
% thresholds = 0.95:-.05:0.05;
thresholds = 1:-.05:0;
GTfiles=dir([datasetStruct.GTdir '/*.png']);
GTfiles = [GTfiles; dir([datasetStruct.GTdir '/*.jpg'])];
GTfiles = [GTfiles; dir([datasetStruct.GTdir '/*.bmp'])];

numOfFiles = size(GTfiles,1);
numOfAlgos = length(algStructArray);  

[hitRate, falseAlarm] = deal(zeros(numOfFiles,length(thresholds),numOfAlgos)); 
[Pre, Recall] = deal(zeros(numOfFiles,length(thresholds),numOfAlgos));     
[Fmeasure] = deal(zeros(numOfFiles,3,numOfAlgos));                  

%Iterate over images
totalNum = numOfFiles* ones(numOfAlgos,1); %100

for imIndx=1:numOfFiles
    
    fprintf('Processing image %i out of %i\n',imIndx,numOfFiles);
    [~,base_name,ext] = fileparts(GTfiles(imIndx).name);
    
    gtMap = im2double(imread([datasetStruct.GTdir base_name ext]));
    gtSize = size(gtMap);   
    if (length(gtSize) == 3)
        gtMap = rgb2gray(gtMap);
        gtSize(3)= [];
    end
    gtMap = logical(gtMap>=0.1);   
    totalNum = numOfFiles* ones(numOfAlgos,1);  
    for algIdx = 1:numOfAlgos
        sMap = readSaliencyMap(algStructArray{algIdx},base_name,gtSize);
        if sum(sum(sMap)) == 0
            totalNum(algIdx) = totalNum(algIdx) - 1;
        end
        
        [Pre(imIndx,:,algIdx), Recall(imIndx,:,algIdx), ...
            hitRate(imIndx,:,algIdx), falseAlarm(imIndx,:,algIdx)] ...
            = evaluateSal(sMap,thresholds,gtMap);
        [Fmeasure(imIndx,:,algIdx)] = Fmeasure_calu(sMap,gtMap,gtSize);
        
    end
    
end %End of image loop

%Average across images -
mmHitRate = permute( sum(hitRate,1),[2 3 1] );
mmFalseAlarm = permute( sum(falseAlarm,1),[2 3 1]);
mmPre = permute( sum(Pre,1),[2 3 1]);
mmRecall = permute( sum(Recall,1),[2 3 1]);
mmFmeasure = permute( sum(Fmeasure,1),[2 3 1]);

for j=1:numOfAlgos
    mmHitRate(:,j) = mmHitRate(:,j)./totalNum(j);
    mmFalseAlarm(:,j) = mmFalseAlarm(:,j)./totalNum(j);
    mmPre(:,j) = mmPre(:,j)./totalNum(j);
    mmRecall(:,j) = mmRecall(:,j)./totalNum(j);
    mmFmeasure(:,j) = mmFmeasure(:,j)./totalNum(j);
end
mHitRate = mmHitRate;
mFalseAlarm = mmFalseAlarm;
mPre = mmPre;
mRecall = mmRecall;
mFmeasure = mmFmeasure;
mFmeasureWF1 = mPre .* mRecall;
mFmeasureWF = ( ( 1.3 * mFmeasureWF1 ) ./ ( .3 * mPre + mRecall ) );
% mFmeasureWF = mFmeasureWF(:, 21);
AUC = nan(1,size(mFalseAlarm,2));
for algIdx=1:numOfAlgos
    AUC(algIdx) = trapz(mFalseAlarm(:,algIdx),mHitRate(:,algIdx));
end

end


% Read and resize map
function sMap = readSaliencyMap(algStruct,base_name,gtSize)
file_name = fullfile(algStruct.dir,[algStruct.prefix base_name algStruct.postfix '.' algStruct.ext]);
sMap = imresize(im2double(imread(file_name)),gtSize(1:2));
if (size(sMap,3)==3)
    sMap = rgb2gray(sMap);
end
sMap(sMap<0)=0;
maxnum = max(sMap(:));
if maxnum==0
    sMap = zeros(gtSize(1:2));
else
    sMap = sMap./maxnum;
end

end


function [Pre, Recall, hitRate, falseAlarm] ...
    = thresholdBased_HR_FR(sMap,thresholds,gtMap)
numOfThreshs = length(thresholds);
[Pre, Recall, hitRate, falseAlarm] = deal(zeros(1,numOfThreshs));
for threshIdx=1:numOfThreshs
    cThrsh=thresholds(threshIdx);
    %     if mean(gtMap(:))==0
    %         if cThrsh==0
    %             cThrsh=0.01;
    %         end
    %     elseif mean(gtMap(:))==1
    %         if cThrsh==1
    %             cThrsh=0.99;
    %         end
    %     end
    [Pre(threshIdx), Recall(threshIdx), hitRate(threshIdx), falseAlarm(threshIdx)] ...
        = Pre_Recall_hitRate(sMap,gtMap,cThrsh);
end
end

