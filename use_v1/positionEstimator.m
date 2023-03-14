function [x, y, newModelParameters] = positionEstimator(testData, modelParameters)

% - test_data:
%     test_data(m).trialID
%         unique trial ID
%     test_data(m).startHandPos
%         2x1 vector giving the [x y] position of the hand at the start
%         of the trial
%     test_data(m).decodedHandPos
%         [2xN] vector giving the hand position estimated by your
%         algorithm during the previous iterations. In this case, N is
%         the number of times your function has been called previously on
%         the same data sequence.
%     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
%     in this case, t goes from 1 to the current time in steps of 20

%Time length of spikes data
time_max = length(testData.spikes);

% Set parameters of SVM model
svm_s=0.07;
svm_c=1;

%Apply trained parameters on classification
classificationParameters=modelParameters.classificationParameters;

if time_max==320 || time_max==400 || time_max==480 || time_max==560

    %Reorganized Test Data
    dt = 80;
    t_ind=time_max/dt-3;

    T=dt:dt:time_max;
    Xt = zeros([98*length(T),1]);
    for ind=1:length(T)
    t1=dt*(ind-1)+1;
    t2=dt*ind;
    for i=1:98
        Xt(i+(ind-1)*98,1)=sum(testData.spikes(i,t1:t2))/dt;
    end
    end

    %project features into optimal plane for SVM
    Wopt=classificationParameters(t_ind).Wopt_SVM;
    mx=classificationParameters(t_ind).mx_SVM;
    Wmean=classificationParameters(t_ind).Wmean_SVM;
    Wt=Wopt'*(Xt-mx);
    pred_dir=do_SVM(Wt,Wmean,svm_s,svm_c);


else
    % if there is no time for updating, just use previous test results
    pred_dir=modelParameters.test_label;
end

% Use PCR to estimate positions based on PCA results
[x,y] = do_PCR(time_max, testData,modelParameters.coeffs_gc,pred_dir);

modelParameters.test_label = pred_dir;

newModelParameters = modelParameters;

end

function [x,y] = do_PCR(T_end, testData,coeffs_gc,pred_dir)

%reorganized test data
dt = 20;
T = (320:dt:560);
T = dt:dt:min(T_end,T(end));
features = zeros([98*length(T),1]);
for ind=1:length(T)
    t1=dt*(ind-1)+1;
    t2=dt*ind;
    for i=1:98
        features(i+(ind-1)*98,1)=sum(testData.spikes(i,t1:t2))/dt;
    end
end

%calculate the coefficient of x and y axis
c_x = coeffs_gc(length(features)/98 - 15, pred_dir).values(:,1);
c_y = coeffs_gc(length(features)/98 - 15, pred_dir).values(:,2);

mm = coeffs_gc(length(features)/98 - 15, pred_dir).mean_feature;

%calculate the mean positions in x and y axis
mean_posx = coeffs_gc(length(features)/98 - 15, pred_dir).mean_hand_pos_x;
mean_posy = coeffs_gc(length(features)/98 - 15, pred_dir).mean_hand_pos_y;

x = (features-mm)'*c_x+mean_posx;
y = (features-mm)'*c_y+mean_posy;

end

%Predicte top or lower 
function [labels,error] = pred(ord,test_data,train_data,train_lab,NN_vec)

%assign initial matrices
labels = zeros(length(NN_vec) , size(test_data,2));
error = zeros(length(NN_vec) , size(test_data,2));

for n=1:size(test_data,2)
    test_data_vec = test_data(:,n);
    [err1,ind1] = mink(sum((abs(test_data_vec-train_data)).^ord),max(NN_vec));
   
    for index_NN = 1:length(NN_vec)
        NN = NN_vec(index_NN);
        ind = ind1(1:NN);
        err = err1(1:NN);
        train_lab1 = train_lab(ind);
        [~,~,temp] = mode(train_lab1);

        err = err(ismember(train_lab1,temp{1}'));
        train_lab1 = train_lab1(ismember(train_lab1,temp{1}'));
        [temp1,temp] = min(err);
        labels(index_NN,n) = train_lab1(temp);
        error(index_NN,n) = temp1;
    end
end

end

%SVM algorithm
function p = do_SVM(Xt,Xmean,s2,c)

model1 = svmTrain(Xmean', [0 0 1 1 1 1 0 0]', c, @(x1, x2) gaussianKernel(x1, x2, s2));
model2 = svmTrain(Xmean(:,[1 2 7 8])', [1 1 0 0]', c, @(x1, x2) gaussianKernel(x1, x2, s2));
model3 = svmTrain(Xmean(:,3:6)', [1 1 0 0]', c, @(x1, x2) gaussianKernel(x1, x2, s2));
p=zeros(1,size(Xt,2));

for n=1:size(Xt,2)
    pred1 = svmPredict(model1,Xt(:,n)');
    if pred1==0 % right direction
        pred2=svmPredict(model2,Xt(:,n)');
        if pred2==0 %lower right
            p(n)= pred(2,Xt(:,n),Xmean(:,7:8),7:8,1);
        else %top right           
            p(n)= pred(2,Xt(:,n),Xmean(:,1:2),1:2,1); 
        end
    else %left direction
        pred2=svmPredict(model3,Xt(:,n)');
        if pred2==0 % top left or lower left          
            %lower left
            p(n)= pred(2,Xt(:,n),Xmean(:,5:6),5:6,1);
        else
             %top left
            p(n)= pred(2,Xt(:,n),Xmean(:,3:4),3:4,1);
        end

    end
end
end

%SVM train,taken from lecture notes
function [model] = svmTrain(X, Y, C, kernelFunction, ...
    tol, max_passes)
%SVMTRAIN Trains an SVM classifier using a simplified version of the SMO
%algorithm.
%   [model] = SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) trains an
%   SVM classifier and returns trained model. X is the matrix of training
%   examples.  Each row is a training example, and the jth column holds the
%   jth feature.  Y is a column matrix containing 1 for positive examples
%   and 0 for negative examples.  C is the standard SVM regularization
%   parameter.  tol is a tolerance value used for determining equality of
%   floating point numbers. max_passes controls the number of iterations
%   over the dataset (without changes to alpha) before the algorithm quits.
%
% Note: This is a simplified version of the SMO algorithm for training
%       SVMs. In practice, if you want to train an SVM classifier, we
%       recommend using an optimized package such as:
%
%           LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
%           SVMLight (http://svmlight.joachims.org/)
%
%

if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-3;
end

if ~exist('max_passes', 'var') || isempty(max_passes)
    max_passes = 5;
end

% Data parameters
m = size(X, 1);
n = size(X, 2);

% Map 0 to -1
Y(Y==0) = -1;

% Variables
alphas = zeros(m, 1);
b = 0;
E = zeros(m, 1);
passes = 0;
eta = 0;
L = 0;
H = 0;

% Pre-compute the Kernel Matrix since our dataset is small
% (in practice, optimized SVM packages that handle large datasets
%  gracefully will _not_ do this)
%
% We have implemented optimized vectorized version of the Kernels here so
% that the svm training will run faster.
if strcmp(func2str(kernelFunction), 'linearKernel')
    % Vectorized computation for the Linear Kernel
    % This is equivalent to computing the kernel on every pair of examples
    K = X*X';
elseif contains(func2str(kernelFunction), 'gaussianKernel')
    % Vectorized RBF Kernel
    % This is equivalent to computing the kernel on every pair of examples
    X2 = sum(X.^2, 2);
    K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
    K = kernelFunction(1, 0) .^ K;
else
    % Pre-compute the Kernel Matrix
    % The following can be slow due to the lack of vectorization
    K = zeros(m);
    for i = 1:m
        for j = i:m
            K(i,j) = kernelFunction(X(i,:)', X(j,:)');
            K(j,i) = K(i,j); %the matrix is symmetric
        end
    end
end

% Train
% fprintf('\nTraining ...');
dots = 12;
while passes < max_passes
    
    num_changed_alphas = 0;
    for i = 1:m
        
        % Calculate Ei = f(x(i)) - y(i) using (2).
        % E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
        E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);
        
        if ((Y(i)*E(i) < -tol && alphas(i) < C) || (Y(i)*E(i) > tol && alphas(i) > 0))
            
            % In practice, there are many heuristics one can use to select
            % the i and j. In this simplified code, we select them randomly.
            j = ceil(m * rand());
            while j == i  % Make sure i \neq j
                j = ceil(m * rand());
            end
            
            % Calculate Ej = f(x(j)) - y(j) using (2).
            E(j) = b + sum (alphas.*Y.*K(:,j)) - Y(j);
            
            % Save old alphas
            alpha_i_old = alphas(i);
            alpha_j_old = alphas(j);
            
            % Compute L and H by (10) or (11).
            if (Y(i) == Y(j))
                L = max(0, alphas(j) + alphas(i) - C);
                H = min(C, alphas(j) + alphas(i));
            else
                L = max(0, alphas(j) - alphas(i));
                H = min(C, C + alphas(j) - alphas(i));
            end
            
            if (L == H)
                % continue to next i.
                continue;
            end
            
            % Compute eta by (14).
            eta = 2 * K(i,j) - K(i,i) - K(j,j);
            if (eta >= 0)
                % continue to next i.
                continue;
            end
            
            % Compute and clip new value for alpha j using (12) and (15).
            alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;
            
            % Clip
            alphas(j) = min (H, alphas(j));
            alphas(j) = max (L, alphas(j));
            
            % Check if change in alpha is significant
            if (abs(alphas(j) - alpha_j_old) < tol)
                % continue to next i.
                % replace anyway
                alphas(j) = alpha_j_old;
                continue;
            end
            
            % Determine value for alpha i using (16).
            alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));
            
            % Compute b1 and b2 using (17) and (18) respectively.
            b1 = b - E(i) ...
                - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
            b2 = b - E(j) ...
                - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';
            
            % Compute b by (19).
            if (0 < alphas(i) && alphas(i) < C)
                b = b1;
            elseif (0 < alphas(j) && alphas(j) < C)
                b = b2;
            else
                b = (b1+b2)/2;
            end
            
            num_changed_alphas = num_changed_alphas + 1;
            
        end
        
    end
    
    if (num_changed_alphas == 0)
        passes = passes + 1;
    else
        passes = 0;
    end
    
    %     fprintf('.');
    dots = dots + 1;
    if dots > 78
        dots = 0;
        fprintf('\n');
    end
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
end
% fprintf(' Done! \n\n');

% Save the model
idx = alphas > 0;
model.X= X(idx,:);
model.y= Y(idx);
model.kernelFunction = kernelFunction;
model.b= b;
model.alphas= alphas(idx);
model.w = ((alphas.*Y)'*X)';
end

%gaussian kernel, taken from lecture notes
function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

sim = exp(-(norm(x1 - x2) ^ 2) / (2 * (sigma ^ 2)));

end

%SVM predict, taken from lecture notes
function pred = svmPredict(model, X)
%SVMPREDICT returns a vector of predictions using a trained SVM model
%(svmTrain).
%   pred = SVMPREDICT(model, X) returns a vector of predictions using a
%   trained SVM model (svmTrain). X is a mxn matrix where there each
%   example is a row. model is a svm model returned from svmTrain.
%   predictions pred is a m x 1 column of predictions of {0, 1} values.
%

% Check if we are getting a column vector, if so, then assume that we only
% need to do prediction for a single example
if (size(X, 2) == 1)
    % Examples should be in rows
    X = X';
end

% Dataset
m = size(X, 1);
p = zeros(m, 1);
pred = zeros(m, 1);

if strcmp(func2str(model.kernelFunction), 'linearKernel')
    % We can use the weights and bias directly if working with the
    % linear kernel
    p = X * model.w + model.b;
elseif contains(func2str(model.kernelFunction), 'gaussianKernel')
    % Vectorized RBF Kernel
    % This is equivalent to computing the kernel on every pair of examples
    X1 = sum(X.^2, 2);
    X2 = sum(model.X.^2, 2)';
    K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));
    K = model.kernelFunction(1, 0) .^ K;
    K = bsxfun(@times, model.y', K);
    K = bsxfun(@times, model.alphas', K);
    p = sum(K, 2);
else
    % Other Non-linear kernel
    for i = 1:m
        prediction = 0;
        for j = 1:size(model.X, 1)
            prediction = prediction + ...
                model.alphas(j) * model.y(j) * ...
                model.kernelFunction(X(i,:)', model.X(j,:)');
        end
        p(i) = prediction + model.b;
    end
end

% Convert predictions into 0 / 1
pred(p >= 0) =  1;
pred(p <  0) =  0;

end
