function class_score = TestNNclass(NN,input_set,output_set)

% Tests the classification accuracy of the network on input_set and output_set

score = 0;
for i = 1:size(input_set,1)
    
    I = input_set(i,:);
    O = output_set(i,:);
    
    NN = RunNN(NN,I);
    
    NN.output = NN.output > 0.5;
    
    if NN.output == O
        score = score + 1;
    end
    
end

% Mutliply score by 100 to get percentage
class_score = score/size(output_set,1);