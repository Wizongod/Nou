function NN = TrainNNCPU(NN,input_set,output_set,learn_rate,weight_decay)
% DEPRECATED

% Trains the network using the CPU through one epoch.
% Input_set and output_sets are matrices.
% Each row of input_set is one set of inputs and the corresponding row in
% the output_set should contain the desired network output.

%% Start training

w_new = NN.w*0;
b_new = NN.b*0;

sizeNNw1 = size(NN.w,1);
sizeNNw2 = sizeNNw1; % = size(NN.w,2)

sizeIN = size(input_set,1);

prev_toc = 0;
tic
for cycle = 1:sizeIN
    
    I = input_set(cycle,:);
    O = output_set(cycle,:);
    
    NN = RunNN(NN,I);
    
    % Cost function is 0.5*sum(O-NN.output)^2
    derr_gb = O - NN.output;
    
    if size(derr_gb,2) ~= sizeNNw2
        derr_gb(size(derr_gb,2)+1:sizeNNw2) = 0;
    end
    
    for L = NN.layers-1:-1:1
        dy_dxwb = dReLU(NN.x(:,:,L),NN.w(:,:,L),NN.b(:,:,L));
        dy_dw = NN.x(:,:,L)'*dy_dxwb;
        w_new(:,:,L) = NN.w(:,:,L) + learn_rate*(dy_dw).*repmat(derr_gb,sizeNNw1,1) ...
                        - learn_rate*weight_decay*NN.w(:,:,L)./sizeIN;
        b_new(:,:,L) = NN.b(:,:,L) + learn_rate*(dy_dxwb).*derr_gb;
        derr_gb = (NN.w(:,:,L)*(derr_gb.*dy_dxwb)')';
    end
    
    NN.w = w_new;
    NN.b = b_new;
    
    if toc - prev_toc >= 1
        prev_toc = toc;
        fprintf('%d samples have been trained in this epoch\n',cycle)
    end

end

end