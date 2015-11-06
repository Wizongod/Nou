function NN = TrainNN(NN,input_set,output_set,learn_rate)
% trains the network for 1 epoch (i.e. across one complete set of input and
% outputs)
% input_set and output_sets are matrices.
% Each row of input_set is one set of inputs and the corresponding row in
% the output_set should contain the desired network output.

%% Start training

w_new = NN.w*0;
b_new = NN.b*0;

prev_toc = 0;
tic
for cycle = 1:size(input_set,1)
    
    I = input_set(cycle,:);
    O = output_set(cycle,:);
    
    NN = RunNN(NN,I);
    
    derr_gb = O - NN.output;
    
    if size(derr_gb,2) ~= size(NN.w,2)
        derr_gb(size(derr_gb,2)+1:size(NN.w,2)) = 0;
    end
    
    for L = NN.layers-1:-1:1
        dy_dxwb = dReLU(NN.x(:,:,L),NN.w(:,:,L),NN.b(:,:,L));
        %dy_dxwb = dy_dxwb.*([ones(1,NN.structure(L+1)) zeros(1,max(NN.structure)-NN.structure(L+1))]);
        dy_dw = NN.x(:,:,L)'*dy_dxwb;
        w_new(:,:,L) = NN.w(:,:,L) + learn_rate*(dy_dw).*repmat(derr_gb,size(NN.w(:,:,L),1),1);
        b_new(:,:,L) = NN.b(:,:,L) + learn_rate*(dy_dxwb).*derr_gb;
        derr_gb = derr_gb(1)*dy_dxwb(1)*NN.w(:,1,L)';
    end
    
    NN.w = w_new;
    NN.b = b_new;
    
    if toc - prev_toc >= 1
        prev_toc = toc;
        fprintf('%.2fsecs : %d cycles have passed in this batch\n',toc,cycle)
    end
    
    %%%%%% Old code %%%%%%

%     for j = size(NN.structure,2):-1:2
%         for i = NN.structure(j):-1:1
%             if j == size(NN.structure,2)
%                 glob_grad = NN_output(i) - O(i);
%             end
% 
%             for k = 1:NN.structure(j-1)
%                 grad = glob_grad * dReLU(x(:,j-1),[],[]) * x(k,(j-1));
%                 w_new(i,j,k) = w(i,j,k) - learn_rate * grad;
%             end
%             
%             grad = glob_grad * dReLU(x(:,j-1),[],[]);
%             b_new(i,j) = b(i,j) - learn_rate * grad;
%         end
%         glob_grad = glob_grad * dReLU(x(:,j-1),[],[]) * w(1,j,1);
%     end

end

end