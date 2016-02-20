function avg_error = TestNN(NN,input_set,output_set,costF,verbosity)
% Tests a neural network based on the input_set and output_set given.
% Returns the average average^2 error.
% input_set is a matrix where each row is the NN input vector and the
% output_set corresponding row is the desired NN output vector.

% verbosity can be empty (nothing printed), 'v' (some stuff),
% 'v+' (alot of stuff. Not recommended for long input and/or output vectors)

if ~exist('verbosity','var')
    verbosity = '';
end 

if strcmp(verbosity,'v') || strcmp(verbosity,'v+')
    fprintf('--------Test cases:--------\n\n')
end
avg_error = 0;
for i = 1:size(input_set,1)
    
    I = input_set(i,:);
    O = output_set(i,:);

    NN = RunNN(NN,I);
    
    if strcmp(verbosity,'v') || strcmp(verbosity,'v+')
        fprintf('--Test case %d--\n',i)
        if strcmp(verbosity,'v+')
            fprintf('Input:           ')
            fprintf('%9.3f',I)
            fprintf('\n')
            fprintf('Output required:     ')
            fprintf('%2.3f    ',O)
            fprintf('\n')
            fprintf('Output of NN:        ')
            fprintf('%2.4f   ',NN.output)
            fprintf('\n')
        end
        fprintf('Squared error: ')
        fprintf('%.4f',costF(O,NN.output))
        fprintf('\n\n')
    end
    
    avg_error = avg_error + costF(O,NN.output);

end

avg_error = avg_error/size(input_set,1);

if strcmp(verbosity,'v') || strcmp(verbosity,'v+')
    fprintf('Average error: %.6f\n\n',avg_error)
    fprintf('------------------------------\n\n')
end

end