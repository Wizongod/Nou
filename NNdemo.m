close all
clear all
clc

NN = ConstructNN(2,1,[2 2]); % constructs a 2-2-2-1 neural network
fprintf('Constructed a neural network of structure [');
fprintf('%d ',NN.structure);
fprintf(']\n\n');

% Let's do an XOR gate. Each row in these matrices represents one set of
% training data.
input_set = [1 0; 0 1; 1 1; 0 0];
output_set = [1; 1; 0; 0];

% Do not go higher than 0.01 in general. Too high learning rates cause
% divergence, i.e. you'll get NaN outputs.
% Also, in some cases, it may converge and then diverge, or converge too
% slowly. If that's the case, a decaying learning rate may be needed (e.g.
% decay learn rate by 0.1% each epoch) or a more adaptive algorithm that
% decays the learn rate if divergence is detected, and increases the learn
% rate if convergence is detected.
learn_rate = 0.01;

epochs = 10000;

fprintf('Begin training network over %d epochs...\n',epochs);

figure(1)
hold on
xlabel('No. of epochs')
ylabel('Average squared error')

for cycle = 1:epochs
    fprintf('Epoch: %d\n',cycle);
    NN = TrainNN(NN,input_set,output_set,learn_rate);
    if mod(cycle,100) == 0
        avg_error = TestNN(NN,input_set,output_set);
        plot(cycle,avg_error, '.k')
        drawnow
        if avg_error < 1e-4
            fprintf('\nFully converged!\n');
            break
        end
    end
end
fprintf('\nTraining Complete!\n\n')


TestNN(NN,input_set,output_set,'v+');

% pause(0.1)
%learn_rate = learn_rate * 0.98;