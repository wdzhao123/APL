function curve_bar_plot_b( basedir, gt_dir, alg_dir, mPre, mRecall, mFmeasure, mFmeasureWF , mFalseAlarm)
%%
dataname = gt_dir{1}; 

len = length(alg_dir);   
method2 = cell(len,1);   
for j = 1:len
    method2(j)=alg_dir{j}(1);
end
method_col = { '[1 0.19 0.19]', '[0.65 0.50 0.72]', '[1 0.85 0.4]', '[0.40 0.67 0.62]', '[0.73 0.82 0.47]', '[1 0.71 0.44]', '[0.95 0.58 0.76]', '[1.00 0.80 0.80]', '[0.8 0.9 1]', '[0.8 0.92 0.50]', '[0.77 0.34 0.44]', '--c', '--k', '--g', '--m', '--c'};

figure(4);
hold on;
for j=1:len
     plot(mRecall((2:21),j),mPre((2:21),j),'Color',method_col{j},'LineWidth',2);
     axis([0 1 0.3 1]);
     xticks(0:0.2:1);
end
%  for j=1:2
%      plot(mRecall(:,j),mPre(:,j),method_col{j});
%  end

grid on;
hold off;
xlabel('Recall');     ylabel('Precision');
legend( method2 );
% saveas( figure(4), [basedir, dataname, '_pr.fig']);

figure(5);
hold on;

for j=1:len
    Xbiao = 1:-.05:0;
%     mFmeasureWF = squeeze(mFmeasureWF)
    plot(Xbiao, mFmeasureWF(:, j),'Color',method_col{j},'LineWidth',2);
    axis([0 1 0 1]);
    xticks(0:0.2:1);
end

grid on;
hold off;
xlabel('Threshold');     ylabel('F-measure');
legend( method2 );
% saveas( figure(5), [basedir, dataname, '_pr.fig']);



% 
% figure(5);
% barMsra = mFmeasure';
% bar( barMsra);
% b = bar(barMsra);
% set(b(1),'facecolor','[0.49 0.65 0.88]');
% set(b(2),'facecolor','[0.65 0.50 0.72]');
% set(b(3),'facecolor','[1 0.71 0.44]');
% if len == 1
%     grid on;
%     set( gca ,'xticklabels',{'Precision','Recall','Fmeasure'}, 'fontsize', 8 );
%     legend(method2);
% else
%     set( gca, 'xtick', 1:1:len ),
%     grid on;
%     set( gca ,'xticklabels',  method2 , 'fontsize', 8 );
%     legend('Precision','Recall','Fmeasure');
% 
% end
% saveas( figure(5), [ basedir, dataname, '_bar.fig'] );
