i = 20;
j = 25;
%Create new neural network
net=newff(TrainData,TrainDataTargets,[i j]); % dyo krymmena epipeda L1:10 ,L2: 20


%Define data parameters
net.divideParam.trainRatio=0.8;
net.divideParam.valRatio=0.2;
net.divideParam.testRatio=0;

%Define train function and epochs
net.trainFcn = 'traingd';
net.trainParam.epochs= 10000;
cnt = 1;
%Training multiple nn with different learning ratios
for i = 0.05:0.05:0.4
    
    net.trainParam.lr = i;
    %Train neural network
    [net,tr_l]=train(net,TrainData,TrainDataTargets);
    TestDataOutput=sim(net,TestData);
    a1=tr_l.epoch(end);
    [accuracy_gd(cnt),p,r]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
    %a = mean(accuracy,2);
    
    
%     f1score(cnt) = mean( 2 * (p .* r) ./ (p + r));
    cnt = cnt + 1;
end

%save('f1score6egdx2l','f1score')
%% other learining
clearvars net
i = 20;
j = 5;
%Create new neural network
net=newff(TrainData,TrainDataTargets,[i j]); % dyo krymmena epipeda L1:10 ,L2: 20


%Define data parameters
net.divideParam.trainRatio=0.8;
net.divideParam.valRatio=0.2;
net.divideParam.testRatio=0;

%Define train function and epochs
net.trainFcn = 'traingdx';
net.trainParam.epochs= 1000;
cnt = 1;
%Training multiple nn with different learning ratios
%for i = 0.05:0.05:0.4
    
    %net.trainParam.lr = i;
    %Train neural network
    [net,tr_l1]=train(net,TrainData,TrainDataTargets);
    TestDataOutput=sim(net,TestData);
    a2=tr_l1.epoch(end);

    [accuracy_gdx(cnt),p,r]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
    %a = mean(accuracy,2);
    
    
%     f1score(cnt) = mean( 2 * (p .* r) ./ (p + r));
  %  cnt = cnt + 1;
%end
i = 0.05:0.05:0.4;
% figure()
% title('Variable learning ratios with train function traingd');
% 
% ylabel('Accuracy');
% xlabel('Learning Rate');
% hold on;
%save('f1score6egdx2l','f1score')
colors={'r' 'b' 'g' 'k' 'c' 'y' };
i = 0.05:0.05:0.4;
figure()
title('Variable learning ratios with train functions traingd triangdx');

ylabel('Accuracy');
xlabel('Learning Rate');

plot(i,accuracy_gd,'color',colors{1});
legendInfo{1} = ['train function = traingd ' ];
hold on;  
plot(i,accuracy_gdx,'color',colors{2});

legendInfo{2} = ['train function = traingdx' ];
h=legend(legendInfo);
set(h, 'Location', 'best')
hold off;
figure()
plot(a1,'color','g')
hold on;
plot(a2,'color','b')