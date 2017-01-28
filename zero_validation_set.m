i = 30
j = 5

%Create new neural network
net=newff(TrainData,TrainDataTargets,[i j]); % dyo krymmena epipeda L1:10 ,L2: 20


%Define data parameters
net.divideParam.trainRatio=0.8;
net.divideParam.valRatio=0.2; %zero validetion set for this step
net.divideParam.testRatio=0;
%Define train function and epochs
net.trainFcn = 'trainlm';
cnt=1;
for i=20000:5000:100000
%Train neural network
    net.trainParam.epochs=i;
    [net,tr_lm]=train(net,TrainData,TrainDataTargets);
    ep(1,cnt)=tr_lm.epoch(end);
    TestDataOutput=sim(net,TestData);
    [accuracy(cnt),p,r]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
    cnt=cnt+1;
end
a = mean(accuracy,2);
figure()
i=20000:5000:100000;
plot(i,accuracy);
title('Variable epochs with set trainlm');

ylabel('Accuracy');
xlabel('Epochs');
% f1score = mean( 2 * (p .* r) ./ (p + r))
i = 30
j = 5

%Create new neural network
net=newff(TrainData,TrainDataTargets,[i j]); % dyo krymmena epipeda L1:10 ,L2: 20


%Define data parameters
net.divideParam.trainRatio=0.8;
net.divideParam.valRatio=0.2; %zero validetion set for this step
net.divideParam.testRatio=0;
%Define train function and epochs
net.trainFcn = 'traingd';
cnt=1;
for i=20000:5000:100000
%Train neural network
    net.trainParam.epochs=i;
    [net,tr_d]=train(net,TrainData,TrainDataTargets);
    ep(2,cnt)=tr_d.epoch(end);
    TestDataOutput=sim(net,TestData);
    [accuracy(cnt),p,r]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
    cnt=cnt+1;
end
a = mean(accuracy,2);
figure()
i=20000:5000:100000;
plot(i,accuracy);
title('Variable epochs with zero validation set traingd');

ylabel('Accuracy');
xlabel('Epochs');
i = 30
j = 5

%Create new neural network
net=newff(TrainData,TrainDataTargets,[i j]); % dyo krymmena epipeda L1:10 ,L2: 20


%Define data parameters
net.divideParam.trainRatio=0.8;
net.divideParam.valRatio=0.2; %zero validetion set for this step
net.divideParam.testRatio=0;
%Define train function and epochs
net.trainFcn = 'traingdx';
cnt=1;
for i=20000:5000:100000
%Train neural network
    net.trainParam.epochs=i;
    [net,tr_dx]=train(net,TrainData,TrainDataTargets);
    ep(3,cnt)=tr_dx.epoch(end);
    TestDataOutput=sim(net,TestData);
    [accuracy(cnt),p,r]= eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
    cnt=cnt+1;
end
a = mean(accuracy,2);
figure()
i=20000:5000:100000;
plot(i,accuracy);
title('Variable epochs with set traingdx');

ylabel('Accuracy');
xlabel('Epochs');
funct={'trainlm' 'traingd' 'traingdx'}
col={'r' 'g' 'b'};
figure()
j=20000:5000:100000;
for i=1:3
    plot(ep(i,:),'color',col{i})
    hold on;
    legendinf{i}=[funct{i}];
end    
    legend(legendinf);
    ylabel('Epochs')
    xlabel('limit of epochs(20000:5000:100000)')