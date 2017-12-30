function [cp, param_grad] = CeNN_net(params, layers, data, labels)
I = length(layers);
batch_size = layers{1}.batch_size;
assert(strcmp(layers{1}.type, 'DATA') == 1, 'first layer must be data layer');

% prepare the output of data layer
output{1}.data = data;
output{1}.height = layers{1}.height;
output{1}.width = layers{1}.width;
%output{1}.channel = layers{1}.channel;
output{1}.batch_size = layers{1}.batch_size;
output{1}.diff = 0;

% forward
for i = 2:l-1
    switch layers{i}.type
        case 'CONV'
            % forward of conv layer
            output{i} = CeNN_conv_forward(output{i-1}, layers{i}, params{i-1});
        case 'POOLING'
            % forward of pooling layer
            output{i} = CeNN_pooling_forward(output{i-1}, layers{i});
        case 'IP'
            % forward of inner product layer
            output{i} = CeNN_innerproduct_forward(output{i-1}, layers{i}, params{i-1});
        case 'RELU'
            % forward of relu layer
            output{i} = CeNN_relu_forward(output{i-1}, layers{i});
    end
end

% forward and backward of loss layer
wb = [params{i-1}.w(:); params{i-1}.b(:)];
[cost, grad, input_od, percent] = mlrloss(wb, output{i-1}.data, labels, layers{i}.num, 0, 1);

if nargout >= 2
    param_grad{i-1}.w = reshape(grad(1:length(params{i-1}.w(:))), size(params{i-1}.w));
    param_grad{i-1}.b = reshape(grad(end - length(params{i-1}.b(:)) + 1 : end), size(params{i-1}.b));
    param_grad{i-1}.w = param_grad{i-1}.w / batch_size;
    param_grad{i-1}.b = param_grad{i-1}.b /batch_size;
end

cp.cost = cost/batch_size;
cp.percent = percent;


end
