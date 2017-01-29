trainFunctions = {'traingdx', 'trainlm', 'traingd', 'traingda'};
%% early stopping
%% One hidden Layer
evaled_DATA_Single = zeros(4,6);
for k = 1:4,
    for i = 1:6,
        net = newff(TrainData, TrainDataTargets, 5*i, {'tansig' 'purelin'} , char(trainFunctions(k)));
        net.divideParam.trainRatio = 1;
        net.divideParam.valRatio = 0;
        net.divideParam.testRatio = 0;
        
                net.trainParam.epochs = 100000;
        [net,tr] = train(net, TrainData, TrainDataTargets);
        s_ep(k,i)=tr.epoch(end);
        TestDataOutput = sim(net, TestData);
        [Acc,~,~] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);
        evaled_DATA_Single(k,i) = Acc;
    end
end
% save evaled_DATA_Single;
% 
% load evaled_DATA_Single;
[~ , ind1] = max(evaled_DATA_Single(:));
[bFunc1, bFL1] = ind2sub(size(evaled_DATA_Single), ind1);
fprintf('\n \n Best method is %s \n first layer: %d \n with Accuracy: %f\n', char(trainFunctions(bFunc1)), bFL1*5,evaled_DATA_Single(bFunc1,bFL1));


figure();
title('Variable train functions for different combinations of 2 hidden layers no of neurons(5:5:30)');
colors={'r' 'b' 'g' 'k' 'c' 'y' };
y=5:5:30
for j=1:4
    help=evaled_DATA_Single(j,:);
    plot(y,help,'color',colors{j});
    hold on;
    legendInfo{j} = ['train function = ' char(trainFunctions{j})];
end
h=legend(legendInfo);
set(h, 'Location', 'best')
ylabel('Accuracy');

%% One hidden Layer
evaled_DATA_Single = zeros(4,6);
for k = 1:4,
    for i = 1:6,
        net = newff(TrainData, TrainDataTargets, 5*i, {'tansig' 'purelin'} , char(trainFunctions(k)));
        net.divideParam.trainRatio = 0.8;
        net.divideParam.valRatio = 0.2;
        net.divideParam.testRatio = 0;
            net.trainParam.epochs = 100000;
        
        
        [net,tr] = train(net, TrainData, TrainDataTargets);
        s_ep(k,i)=tr.epoch(end);
        TestDataOutput = sim(net, TestData);
        [Acc,~,~] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);
        evaled_DATA_Single(k,i) = Acc;
    end
end
% save evaled_DATA_Single;
% 
% load evaled_DATA_Single;
[~ , ind1] = max(evaled_DATA_Single(:));
[bFunc1, bFL1] = ind2sub(size(evaled_DATA_Single), ind1);
fprintf('\n \n Best method is %s \n first layer: %d \n with Accuracy: %f\n', char(trainFunctions(bFunc1)), bFL1*5,evaled_DATA_Single(bFunc1,bFL1));
figure();
title('Epochs to conerge for Variable train functions for different combinations of 1 hidden layers no of neurons(5:5:30)');
colors={'r' 'b' 'g' 'k' 'c' 'y' };
for j=1:4
    help=s_ep(j,:);
    plot(help,'color',colors{j});
    hold on;
    legendInfo{j} = ['train function = ' char(trainFunctions{j})];
end
h=legend(legendInfo);
set(h, 'Location', 'best')
ylabel('Epochs');
xlabel('Neurons');
%% Two Hidden Layers
evaled_DATA_Double = zeros(4,6,6);
for k = 1:4,
    for i = 1:6,
        for j = 1:6,
            net = newff(TrainData, TrainDataTargets, [5*i 5*j], {'tansig' 'tansig' 'purelin'} , char(trainFunctions(k)));
            
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0;
            net.trainParam.epochs = 100000;
            m_ep(k,i,j)=tr.epoch(end);
            net = train(net, TrainData, TrainDataTargets);
            TestDataOutput = sim(net, TestData);
            [Acc,~,~] = eval_Accuracy_Precision_Recall(TestDataOutput, TestDataTargets);
            evaled_DATA_Double(k,i,j) = Acc;
            
        end
    end
end

% save evaled_DATA_Double;
% 
% load evaled_DATA_Double;
[~ , ind2] = max(evaled_DATA_Double(:));
[bFunc2, bFL2, bSL2] = ind2sub(size(evaled_DATA_Double), ind2);
fprintf('\n \n Best method is %s \n first layer: %d snd layer: %d \n with Accuracy: %f\n', char(trainFunctions(bFunc2)), bFL2*5, bSL2*5,evaled_DATA_Double(bFunc2,bFL2,bSL2));
figure();
title('Variable train functions for different combinations of 1 hidden layers no of neurons(5:5:30)');
colors={'r' 'b' 'g' 'k' 'c' 'y' };
y=5:5:30
for j=1:4
    help=evaled_DATA_Single(j,:);
    plot(y,help,'color',colors{j});
    hold on;
    legendInfo{j} = ['train function = ' char(trainFunctions{j})];
end
h=legend(legendInfo);
set(h, 'Location', 'best')
ylabel('Accuracy');
for i=1:4
 evaled_DATA_double_f_plot(i,:)=reshape( evaled_DATA_Double(i,:,:),1,36) ;

end
figure();
title('Variable train functions for different combinations of 2 hidden layers no of neurons(5:5:30)');
colors={'r' 'b' 'g' 'k' 'c' 'y' };
x=1:5;
for j=1:4
    help=evaled_DATA_double_f_plot(j,:);
    plot(help,'color',colors{j});
    hold on;
    legendInfo{j} = ['train function = ' char(trainFunctions{j})];
end
h=legend(legendInfo);
set(h, 'Location', 'best')
ylabel('Accuracy');

set(gca,'XTick',1:36,'XTickLabel','') 
% Define the labels 
labb={'(5,5)' '(5,10)' '(5,15)' '(5,20)' '(5,25)' '(5,30)'...
    '(10,5)' '(10,10)' '(10,15)' '(10,20)' '(10,25)' '(10,30)' ...
    '(15,5)' '(15,10)' '(15,15)' '(15,20)' '(15,25)' '(15,30)' ...
    '(20,5)' '(20,10)' '(20,15)' '(20,20)' '(20,25)' '(20,30)'...
     '(25,5)' '(25,10)' '(25,15)' '(25,20)' '(25,25)' '(25,30)'...
     '(30,5)' '(30,10)' '(30,15)' '(30,20)' '(30,25)' '(30,30)'};
 angle = 90; 
 axis_label = 'Neurons combinations (First layer,Second layer)';

% Reduce the size of the axis so that all the labels fit in the figure.
pos = get(gca,'Position');
set(gca,'Position',[pos(1), .2, pos(3) .65])

ax = axis; % Current axis limits
axis(axis); % Fix the axis limits
Yl = ax(3:4); % Y-axis limits

%set(gca, 'xtick', 1:length(tick_labels));
set(gca, 'xtick', 0.7:1:length(labb));
Xt = get(gca, 'xtick');

% Place the text labels
t = text(Xt,Yl(1)*ones(1,length(Xt)),labb);
set(t,'HorizontalAlignment','right','VerticalAlignment','top', 'Rotation', 90);

% Remove the default labels
set(gca,'XTickLabel','')

% Get the Extent of each text object. This
% loop is unavoidable.
for i = 1:length(t)
  ext(i,:) = get(t(i),'Extent');
end

% Determine the lowest point. The X-label will be
% placed so that the top is aligned with this point.
LowYPoint = min(ext(:,2));

% Place the axis label at this point
if ~isempty(axis_label)
  Xl = get(gca, 'Xlim');
  XMidPoint = Xl(1)+abs(diff(Xl))/2;
  tl = text(XMidPoint,LowYPoint, axis_label, 'VerticalAlignment','top', ...
'HorizontalAlignment','center');
end

