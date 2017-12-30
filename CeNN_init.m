function params = CeNN_init(layers)


% Initialize the parameters of all layers
h = layers{1}.height; %Image Height
w = layers{1}.width;  %Image Width
c = layers{1}.channel;%Image channel
for i = 2:length(layers)
    switch layers{i}.type

      case 'CONV'
            % Initialize the parameter of conv layers
            params{i-1}.A = 0;
            params{i-1}.B = rand(layers{i}.k, layers{i}.k) ;
            params{i-1}.z = 0;
            h = h - 1;
            w = w - 1;
            %c = layers{i}.num;

        case 'RELU'
              % The templates below are used for ReLU operation, leaves all
              % positive values unchanged, and thresholds all values below 0
              params{i-1}.A = 0;
              params{i-1}.B = [0 0 0; 0 1 0; 0 0 0];
              params{i-1}.z1 = -1;
              params{i-1}.z2 = 1;
              h = h - 1;
              w = w - 1;

        case 'POOLING'
              params{i-1}.A = 0;
              params{i-1}.B = [0 -1 0; 0 0 0; 0 0 0];
              params{i-1}.z = 0;

        case 'IP'
                % Initialize the parameter of inner product layer
                switch layers{i}.init_type
                    % Gaussian initialization
                    case 'gaussian'
                        scale = sqrt(3/(h*w*c)); % (h*w*c, num)
                        params{i-1}.w = scale*randn(h*w*c, layers{i}.num);
                        params{i-1}.b = zeros(1, layers{i}.num);
                    % Uniform initialization
                    case 'uniform'
                        scale = sqrt(3/(h*w*c));
                        params{i-1}.w = 2*scale*rand(h*w*c, layers{i}.num) - scale;
                        params{i-1}.b = zeros(1, layers{i}.num);
                end
                h = 1;
                w = 1;
                c = layers{i}.num;

         case 'LOSS'
            % Initialize the parameter of inner product layer
            scale = sqrt(3/(h*w*c)); % (h*w*c, num)
            % last layer is K-1
            params{i-1}.w = 2*scale*rand(h*w*c, layers{i}.num - 1) - scale;
            params{i-1}.w = params{i-1}.w';
            params{i-1}.b = zeros(1, layers{i}.num - 1);
            params{i-1}.b = params{i-1}.b';
            h = 1;
            w = 1;
            c = layers{i}.num;

    end
end

end
