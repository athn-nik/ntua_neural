LearnFunct = {'learngd', 'learngdm'};
for k =1:2
            net = newff(TrainData, TrainDataTargets,[20 15],{'tansig' 'purelin'}, 'traingd',char(LearnFunct{k}) );
            if strcmp(char(LearnFunct{k}),'learngd')
                net.trainParam.lr = 0.05;
            end
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0;
            net.trainParam.epochs = 100000;
            % epochs accuracy 
            [net,tr] = train(net, TrainData, TrainDataTargets);
            epochs_learn(k)=tr.epoch(end);
            TestDataOutput = sim(net, TestData);
            [Acc_learn(k),p1,p2] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);
             
end
figure();

title('Different learning functions');

x=1:2;
bar(x,Acc_learn)
%type above each bar the exact number


str = {'learngd','learngdm'};

%type above each bar the exact number
for i=1:numel(Acc_learn)
    text(x(i),Acc_learn(i),num2str(Acc_learn(i),'%0.3f'),...
               'HorizontalAlignment','center',...
               'VerticalAlignment','bottom')
end
set(gca, 'XTickLabel',str, 'XTick',1:numel(str))
figure();

title('Different learning functions different epochs');

x=1:2;
bar(x,epochs_learn,.4)
%type above each bar the exact number


str = {'learngd','learngdm'};

%type above each bar the exact number
for i=1:numel(LearnFunct)
    text(x(i),epochs_learn(i),num2str(epochs_learn(i),'%0.0f'),...
               'HorizontalAlignment','center',...
               'VerticalAlignment','bottom')
end
set(gca, 'XTickLabel',str, 'XTick',1:numel(str))