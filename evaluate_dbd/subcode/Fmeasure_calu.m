%%
function Fmeasure = Fmeasure_calu(sMap,gtMap,gtsize)
max_Fmeasure = 0;
max_PreFtem = 0;
max_RecallFtem = 0;
% FmeasureF = 0;
%     sumLabel =  1.5*mean(sMap(:)) %1.5
%     if ( sumLabel > 1 )
%         sumLabel = 1;
%     end
for i = 1:255
    sumLabel = i*0.00392157;
    Label3 = zeros( gtsize );           %320*320
    Label3( sMap>=sumLabel ) = 1;
    if sum(sum(gtMap)) == 0             %真值为全0
        gtMap = ones(gtsize);
        Label3 = ones(gtsize)-Label3;
    end
    NumRec = length( find( Label3==1 ) ); %全部预测为正的
    LabelAnd = Label3 & gtMap;  %全部预测正确
%     LabelAnd = Label3 & gtMap;
    NumAnd = length( find ( LabelAnd==1 ) );%正确预测为正
    num_obj = sum(sum(gtMap));%真值为正

    if NumAnd == 0
        PreFtem = 0;
        RecallFtem = 0;
        FmeasureF = 0;
    else
        PreFtem = NumAnd/NumRec;
        RecallFtem = NumAnd/num_obj;
        FmeasureF = ( ( 1.3 * PreFtem * RecallFtem ) / ( .3 * PreFtem + RecallFtem ) );

    end
    if FmeasureF > max_Fmeasure
        max_Fmeasure = FmeasureF;
        max_PreFtem = PreFtem;
        max_RecallFtem = RecallFtem;
    end
    FmeasureF = max_Fmeasure;
    PreFtem = max_PreFtem;
    RecallFtem = max_RecallFtem;
end
% if mean(sMap(:))>0.95
%     sumLabel = 0.95;
% elseif mean(sMap(:))<0.05
%     sumLabel = 0.05;
% else
%     sumLabel = 1.5*mean(sMap(:));
% end
% Label3 = zeros( gtsize );           %320*320
% Label3( sMap>=sumLabel ) = 1;
% if sum(sum(gtMap)) == 0             %真值为全0
%     gtMap = ones(gtsize);
%     Label3 = ones(gtsize)-Label3;
% end
% NumRec = length( find( Label3==1 ) ); %全部预测为正的
% LabelAnd = Label3 & gtMap;  %全部预测正确
% NumAnd = length( find ( LabelAnd==1 ) );%正确预测为正
% num_obj = sum(sum(gtMap));%真值为正
% 
% if NumAnd == 0
%     PreFtem = 0;
%     RecallFtem = 0;
%     FmeasureF = 0;
% else
%     PreFtem = NumAnd/NumRec;
%     RecallFtem = NumAnd/num_obj;
%     FmeasureF = ( ( 1.3 * PreFtem * RecallFtem ) / ( .3 * PreFtem + RecallFtem ) );
%     
% end

Fmeasure = [PreFtem, RecallFtem, FmeasureF];

