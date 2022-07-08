function [Pre, Recall, hitRate , falseAlarm] = Pre_Recall_hitRate(sMap,gtMap,cThrsh)
gtsize = size(gtMap);
if sum(sum(gtMap)) == 0  
    gtMap = ones(gtsize)-gtMap;
    sMap = ones(gtsize)-sMap;
end

testMap = zeros( gtsize );
sumLabel = cThrsh;
testMap( sMap>=sumLabel ) = 1;
neg_gtMap = ~gtMap;
neg_testMap = ~testMap;
hitCount = sum(sum(testMap.*gtMap));
% trueAvoidCount = sum(sum(neg_testMap.*neg_gtMap));
missCount = sum(sum(testMap.*neg_gtMap));
falseAvoidCount = sum(sum(neg_testMap.*gtMap));

if hitCount == 0
    Pre = 0;
    Recall = 0;
else
    Pre = hitCount/(hitCount + missCount );
    Recall = hitCount/(hitCount + falseAvoidCount);
end
falseAlarm = 1;
hitRate = 1;


% gtsize = size(gtMap);
% sumLabel = cThrsh;
% % if cThrsh == 0
% %     if sum(sum(gtMap)) == 0
% %         sumLabel = 0.01;
% %     end
% % end
% % if cThrsh == 1
% %     if sum(sum(gtMap)) == 0
% %         sumLabel = 0.99;
% %     end
% % end
% Label3 = zeros( gtsize );           %320*320
% Label3( sMap>=sumLabel ) = 1;
% 
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
%     Pre = 0;
%     Recall = 0;
% else
%     Pre = NumAnd/NumRec;
%     Recall = NumAnd/num_obj;
% end
% falseAlarm = 1;
% hitRate = 1;


% neg_gtMap = ~gtMap;
% neg_testMap = ~testMap;
% 
% hitCount = sum(sum(testMap.*gtMap));
% trueAvoidCount = sum(sum(neg_testMap.*neg_gtMap));
% missCount = sum(sum(testMap.*neg_gtMap));
% falseAvoidCount = sum(sum(neg_testMap.*gtMap));
% 
% if hitCount==0
%     Pre = 0;
%     Recall = 0;
% else
%     Pre = hitCount/(hitCount + missCount );
%     Recall = hitCount/(hitCount + falseAvoidCount);
% end
% 
% falseAlarm = 1 - trueAvoidCount / (eps+trueAvoidCount + missCount);
% hitRate = hitCount / (eps+ hitCount + falseAvoidCount);
end





