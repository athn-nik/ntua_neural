% Part 4: Pruning

%We choose 1 hidden layer with 30 neurons, as the task commands.
%We also choose purelin for transfer function of output layer. See quest3.m

net = newff(TrainData, TrainDataTargets, [30], {'tansig' 'tansig' 'purelin'},'traingd', 'learngd');

net.divideParam.trainRatio=1;
net.divideParam.valRatio=0;
net.divideParam.testRatio=0;
net.trainParam.epochs=1;
net.trainParam.lr = 0.4;

d = 0.1;
l = 0.001;
performance = [];
non_zero_weights = [];
cnt=0;
% cmap = colormap(parula(10));
figure('Name','Multiple ë');
title('Variable ë for constant no of iterations = 5');
colors={'r' 'b' 'g' 'k' 'c' 'y' };
for j=0.1:0.05:0.3
    performance=[];
    cnt=cnt+1;
    for i=1:5
        old_w = getwb(net);
        [net,tr] = train(net,TrainData,TrainDataTargets);   
        new_w_changed = getwb(net) - j*old_w;
        performance = [performance tr.perf(2)];
        idxs = find(abs(new_w_changed) < d);
        weights=getwb(net);
        weights(idxs) = 0;
        n_zero_weights = [non_zero_weights (length(new_w_changed)-length(idxs))];
        net = setwb(net,weights);
    end
    plot(performance,'color',colors{cnt});
    hold on;
    legendInfo{cnt} = ['ë = ' num2str(j)];
end
legend(legendInfo);
performance=[];
non_zero_weights = [];
net = newff(TrainData, TrainDataTargets, [30], {'tansig' 'tansig' 'purelin'},'traingd', 'learngd');

net.divideParam.trainRatio=1;
net.divideParam.valRatio=0;
net.divideParam.testRatio=0;
net.trainParam.epochs=1;
net.trainParam.lr = 0.4;

d = 0.1;
l = 0.0001;
for i=1:1000
old_w = getwb(net);
        [net,tr] = train(net,TrainData,TrainDataTargets);   
        new_w_changed = getwb(net) - l*old_w;
        performance = [performance tr.perf(2)];
        idxs = find(abs(new_w_changed) < d);
        weights=getwb(net);
        weights(idxs) = 0;
        n_zero_weights = [n_zero_weights (length(new_w_changed)-length(idxs))];
        net = setwb(net,weights);
end

figure('Name','Weights Decay(Non Zero Weights)','NumberTitle','off');
plot(n_zero_weights,'r--');
grid on; title('Weights Decay(Non Zero Weights)');
xlabel('#Iterations');
ylabel('Non Zero Weights');

figure('Name','Weights Decay(Train Error)','NumberTitle','off');
plot(performance,'b');
grid on; title('Weights Decay(Train Error)');
xlabel('#Iterations');
ylabel('Train Error');


