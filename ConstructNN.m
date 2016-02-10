function NN = ConstructNN(nodes_in,nodes_out,hidden_struct)
% Neural Network constructor. Returns a Neural Network Object.
% Specify the hidden structure as a row
% vector e.g. [5 4] creates two hidden layers of width 5 -> 4

% Return values:

% NN.structure is a row matrix containing the width of your network for
% each layer

% NN.layers tells you the number of layers in your network (incl. in/out)

% NN.x is a number of row vectors which contain the outputs of each node.
% Format: NN.x(1,node_no,layer_no)
% E.g. NN.x(1,:,1) contains the outputs of layer 1, with NN.x(1,2,3)
% specifying the output from node 2 of layer 3.

% NN.b is a number of row vectors which contain the biases added on to NN.x
% for the NEXT layer.
% Format: NN.b(1,node_no,from_layer_no)
% E.g. NN.b(1,:,1) contains all biases added on to the nodes of the NEXT
% layer before passing through the activation function, with NN.b(1,2,3)
% specifying the bias applied to node 2 on layer 4 (3+1) from all nodes in
% layer 3.

% NN.w is a number of 2D matrices which contains the weights as follows:
% NN.w(nodes_from,into_node,from_layer)
% e.g. NN.w(:,1,2) contains the weights for each node from layer 2 going
% into node 1 on layer 3, with NN.w(5,1,2) specifying from node 5 of layer
% 2 going into node 1 on layer 3.


%% Define Neural Network Properties

NN.inputs = nodes_in;
NN.outputs = nodes_out;
NN.hidden_nodes = hidden_struct; % hidden layers

NN.structure = [NN.inputs NN.hidden_nodes NN.outputs];
NN.layers = size(NN.structure,2);

%% Prepare the relevant matrices

NN.x = zeros(1,max(NN.structure),NN.layers); % output matrix of each node



NN.b = zeros(1,max(NN.structure),NN.layers-1); % bias matrix of each node
% get rid of biases which do not actually exist (if this is already zero
% then this chunk isn't necessary)
% for L = 1:NN.layers-1
%     i = max(NN.structure);
%     while i > NN.structure(L+1)
%         NN.b(:,i,L) = 0;
%         i = i - 1;
%     end
% end



NN.w = normrnd(0,1,max(NN.structure),max(NN.structure),NN.layers-1);

% normalise (L2) the weights entering each node
for L = 1:NN.layers-1
    NN.w(:,:,L) = NN.w(:,:,L)./sqrt(ones(size(NN.w(:,:,L),1),size(NN.w(:,:,L),2))*(NN.w(:,:,L).^2));
end

% get rid of weights which do not actually exist
for L = 1:NN.layers-1
    i = max(NN.structure);
    while i > NN.structure(L+1)
        NN.w(:,i,L) = 0;
        i = i - 1;
    end
end
for L = 1:NN.layers-1
    i = max(NN.structure);
    while i > NN.structure(L)
        NN.w(i,:,L) = 0;
        i = i - 1;
    end
end



NN.output = NN.x(1,1:NN.outputs,NN.layers);

end