clear all; close all;
%% Step 1
% load data
load('dataSet.mat');
%% no process
trainFunctions = {'traingdx', 'trainlm', 'traingd', 'traingda'};
%% early stopping
%% One hidden Layer
% evaled_DATA_Single_nopr = zeros(4,6);
% for k = 1:4,
%     for i = 1:6,
%         net = newff(TrainData, TrainDataTargets, 5*i, {'tansig' 'purelin'} , char(trainFunctions(k)));
%         net.divideParam.trainRatio = 0.8;
%         net.divideParam.valRatio = 0.2;
%         net.divideParam.testRatio = 0;
%             net.trainParam.epochs = 100000;
%         
%         [net,tr] = train(net, TrainData, TrainDataTargets);
%         e_no(k,i)=tr.epoch(end);
%         TestDataOutput = sim(net, TestData);
%         [Acc,~,~] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);
%         evaled_DATA_Single_nopr(k,i) = Acc;
%     end
% end
% figure();
% title('epochs no proccess')
% plot()
% colors={'r' 'b' 'g' 'k' 'c' 'y' };
% y=1:5:30
% for j=1:4
%     help=evaled_DATA_Single_nopr(j,:);
%     plot(y,e_no(j,:),'color',colors{j});
%     hold on;
%     legendInfo{j} = ['train function = ' char(trainFunctions{j})];
% end
% h=legend(legendInfo);
% set(h, 'Location', 'best')
% ylabel('Epochs');
% xlabel('Neurons');
% [~ , ind1] = max(evaled_DATA_Single_nopr(:));
% [bFunc1, bFL1] = ind2sub(size(evaled_DATA_Single_nopr), ind1);
% fprintf('\n \n Best method is %s \n first layer: %d \n with Accuracy: %f\n', char(trainFunctions(bFunc1)), bFL1*5,evaled_DATA_Single(bFunc1,bFL1));
% 
% 
% figure();
% title('(no proccess)Variable train functions for different combinations of 1 hidden layers no of neurons(5:5:30)');
% colors={'r' 'b' 'g' 'k' 'c' 'y' };
% y=5:5:30
% for j=1:4
%     help=evaled_DATA_Single_nopr(j,:);
%     plot(y,help,'color',colors{j});
%     hold on;
%     legendInfo{j} = ['train function = ' char(trainFunctions{j})];
% end
% h=legend(legendInfo);
% set(h, 'Location', 'best')
% ylabel('Accuracy');
% 
%% Preproccesing
%sum adds column-wise so use dimension 2
mpares = sum(TrainDataTargets,2);
figure();
x=1:5;
bar(x,mpares);
%type above each bar the exact number
for i=1:numel(mpares)
    text(x(i),mpares(i),num2str(mpares(i),'%0.0f'),...
               'HorizontalAlignment','center',...
               'VerticalAlignment','bottom')
end
ylabel('# of image segments');
xlabel('Categories');
%how many we want from each category
no_el = min(mpares);

for i=1:5
    index(i,:) = find(TrainDataTargets(i, :),no_el);
end
%% Step 2
%as in part 3
indexes = [index(1,:) index(2,:) index(3,:) index(4,:) index(5,:)];
permutation = randperm(size(indexes,2));
indexes = indexes(permutation);
TrainDataTargets = TrainDataTargets(:,indexes);
%TrainData=TrainData(:,indexes);
%perform mmapstd before 
[TrainData,settings] = mapstd(TrainData);
TestData = mapstd('apply',TestData,settings);

%remove constant rows of matrices 
[TrainData, ps] = removeconstantrows(TrainData(:,indexes));

TestData = removeconstantrows('apply', TestData,ps);
% reduce dimensions

[TrainData,ps] = processpca(TrainData, 0.01);
TestData = processpca('apply',TestData, ps);
