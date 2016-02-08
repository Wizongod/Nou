%% To demonstrate the usage of GPU for matrix operations

% Notice that if the random array is 100x100, the CPU is faster, but if the
% array is changed to 1000x1000, the GPU is faster.

GPU_array = rand(100);

an_array = GPU_array;

g_card = gpuDevice(1);
reset(g_card); % resetting the card before a run can help prevent weird slowdown problems

GPU_array = gpuArray(single(GPU_array));

tic
for i = 1:1000
    GPU_array*GPU_array;
end
toc

GPU_array = gather(GPU_array);

tic
for i = 1:1000
    an_array*an_array;
end
toc