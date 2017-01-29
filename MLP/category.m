%% categories results

          net = newff(TrainData, TrainDataTargets, [30 20], {'tansig' 'tansig' 'purelin'} , 'trainlm');
            
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0;
            net.trainParam.epochs = 1000;
            
            net = train(net, TrainData, TrainDataTargets);
            TestDataOutput = sim(net, TestData);
            [Acc,p,r] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);
a = mean(Acc,2);
p = mean(p,2);

r = mean(r,2);

f1score = 2 * (p .* r) ./ (p + r)

index = find(f1score >= max(f1score)) * 5
res = max(f1score)


net = newff(TrainData, TrainDataTargets, [30 20], {'tansig' 'tansig' 'purelin'} , 'traingd');
            
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0;
            net.trainParam.epochs = 1000;
            
            net = train(net, TrainData, TrainDataTargets);
            TestDataOutput = sim(net, TestData);
            [Acc,p,r] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);
a = mean(Acc,2);
p = mean(p,2);

r = mean(r,2);

f1score_traingd = 2 * (p .* r) ./ (p + r)

index = find(f1score >= max(f1score)) * 5
%nevrwnes1 = [(5* (fix(index/5))) , (5 * (rem(index,5)+1))]
res = max(f1score)

          net = newff(TrainData, TrainDataTargets, [30 20], {'tansig' 'tansig' 'purelin'} , 'traingdx');
            
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0;
            net.trainParam.epochs = 1000;
            
            net = train(net, TrainData, TrainDataTargets);
            TestDataOutput = sim(net, TestData);
            [Acc,p,r] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);
a = mean(Acc,2);
p = mean(p,2);

r = mean(r,2);

f1score_traingdx = 2 * (p .* r) ./ (p + r)

index = find(f1score >= max(f1score)) * 5
res = max(f1score)

net = newff(TrainData, TrainDataTargets, [30 20], {'tansig' 'tansig' 'purelin'} , 'traingda');
            
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0;
            net.trainParam.epochs = 1000;
            
            net = train(net, TrainData, TrainDataTargets);
            TestDataOutput = sim(net, TestData);
            [Acc,p,r] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);
a = mean(Acc,2);
p = mean(p,2);

r = mean(r,2);

f1score_traingda = 2 * (p .* r) ./ (p + r)

index = find(f1score >= max(f1score)) * 5
res = max(f1score)


figure('Name','Different trains different categories')
f=[f1score f1score_traingd f1score_traingdx f1score_traingda];
bar(f)
h=legend('trainlm','traingd','traingdx','traingda')
set(h,'Location','South')