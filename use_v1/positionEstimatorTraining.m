function  [modelParameters] = positionEstimatorTraining(trainingData)
r=size(trainingData,1)-1; %Number of Principal Components for PCR

t_train = 320:80:560; %times at which angle classification is updated
classificationParameters = struct;
[F,l,t]=organize_data(trainingData,80,t_train(end)); % create feature vectors for training data dt=80
for t_ind = 1:length(t_train)
    T = t_train(t_ind);
    X = F(t<=T,:);
    % Mutiple LDA values were tested. larger LDA value will increase decoding
    % time; smaller LDA will reduce the decoding time and RMSE.
    % LDA = supervised learning algorithm
    % PCA = unsupervised learning algorithm
    LDA = 5;
    PCA_SVM = 180;
    [~,mean_x,WPCA,~] = eigenmodel(X,size(X,2));
   
    %Compute scatter matrices
    c = unique(l);
    mc = zeros(size(X,1),length(c));
    mean_x = mean(X,2);
    for im_id=1:length(c)
        mc(:,im_id)=mean(X(:,l==c(im_id)),2);
    end
    sb=(mc-mean_x)*(mc-mean_x)';
    st=(X-mean_x)*(X-mean_x)';
    sw=st-sb;
    
    % Optimal Mpca for SVM, M_lda = 7
    M_pca=PCA_SVM;
    %create optimal matrix
    [Wlda,L] = eig((WPCA(:,1:M_pca)'*sw*WPCA(:,1:M_pca))^-1*WPCA(:,1:M_pca)'*sb*WPCA(:,1:M_pca));
    [~,ind]=maxk(diag(L),LDA);
    Wopt=WPCA(:,1:M_pca)*Wlda(:,ind);

    W=Wopt'*(X-mean_x);
    Wmean=zeros([size(W,1) 8]);
    for k=1:8
        Wmean(:,k)=mean(W(:,l==k),2);
    end
    %save model parameters
    classificationParameters(t_ind).Wopt_SVM=Wopt;
    classificationParameters(t_ind).mx_SVM=mean_x;
    classificationParameters(t_ind).Wmean_SVM=Wmean;
   
end

%create feature vectors for training data (dt=20)
T_end = 560;
dt=20;
T = (320:dt:T_end);
[feat,l,t]=organize_data(trainingData,dt,T_end);

%get hand positions from data
x_mat=[];
y_mat=[];
for k=1:8
    for n=1:size(trainingData,1)
        x=trainingData(n,k).handPos(1,:);
        while length(x)>size(x_mat,2) && size(x_mat,1)>0
            x_mat=[x_mat x_mat(:,end)];
        end
        while size(x_mat,2)>length(x) && size(x_mat,1)>0
            x=[x x(end)];
        end
        x_mat=[x_mat;x];
        y=trainingData(n,k).handPos(2,:);
        while length(y)>size(y_mat,2) && size(y_mat,1)>0
            y_mat=[y_mat y_mat(:,end)];
        end
        while size(y_mat,2)>length(y) && size(y_mat,1)>0
            y=[y y(end)];
        end
        y_mat=[y_mat;y];
    end
end
x_mean=[];
y_mean=[];
x=zeros([size(x_mat,1)/8 size(x_mat,2) 8]);
y=zeros([size(y_mat,1)/8 size(y_mat,2) 8]);
for k=1:8
    x(:,:,k)=x_mat(((k-1)*n+1):(k*n),:);
    y(:,:,k)=y_mat(((k-1)*n+1):(k*n),:);
    x_mean=[x_mean;mean(x_mat(((k-1)*n+1):(k*n),:))];
    y_mean=[y_mean;mean(y_mat(((k-1)*n+1):(k*n),:))];
end

%sample at the values of time we care about
x_resampled = x(:,T,:);
y_resampled = y(:,T,:);

%create a regression model for each angle at each time
%Computing PCR
angle =1:8;
coeffs_gc = struct;
for angle_index =1:length(angle)
    handPos_x_for_regression = x_resampled(:,:,angle_index);
    handPos_y_for_regression = y_resampled(:,:,angle_index);
    for time_index = 1:length(T)
        features_temp = feat(t<=T(time_index),l==angle_index);
        coeffs_gc(time_index, angle_index).mean_hand_pos_x = mean(handPos_x_for_regression(:,time_index));
        coeffs_gc(time_index, angle_index).mean_hand_pos_y = mean(handPos_y_for_regression(:,time_index));
        
        hand_pos_temp_x = handPos_x_for_regression(:,time_index)-coeffs_gc(time_index, angle_index).mean_hand_pos_x;
        hand_pos_temp_y = handPos_y_for_regression(:,time_index)-coeffs_gc(time_index, angle_index).mean_hand_pos_y;
        
        [~,mean_x,U,~]=eigenmodel(features_temp,r); 
        W=U'*(features_temp-mean_x);
        coeffs_gc(time_index, angle_index).mean_feature=mean_x;
        bx=U*(W*W')^(-1)*W* hand_pos_temp_x;
        by=U*(W*W')^(-1)*W* hand_pos_temp_y;
        
        coeffs_gc(time_index, angle_index).values = [bx , by];
    end
end
% Calculate the average trajectories
classes = length(trainingData(1,:));
max_len_spikes = 0; % initialise the longest trial to 0
for c = 1:classes
    for i=1:length(trainingData(:,1))
        % extract the length of the spike trains for each trial
        len_spikes = length(trainingData(i,c).spikes(1,:));
        % check if this length of the spike is the largest 
        if len_spikes > max_len_spikes
            max_len_spikes = len_spikes;
        end
    end
end

% make sure all lengths of the trajectories are the same, otherwise we cannot
% compute the average (number of data points must be the same across 
% trajectories of the same class)
for c=1:classes
    for i=1:length(trainingData(:,1))
        for j=length(trainingData(i,c).spikes(1,:))+1:max_len_spikes
            % pad shorter handPos with the last value until 
            % max_len_spikes. do not pad with zeros as you would compute the
            % wrong avg, do not cut to shortest trajectory as you would lose
            % information
            trainingData(i,c).handPos = [trainingData(i,c).handPos trainingData(i,c).handPos(:, end)];
        end
    end
end

% compute the average trajectory 
average_traj(classes).handPos = [];
for c=1:classes
    trajectories = zeros(2,max_len_spikes); % two rows, x and y pos
    for i=1:length(trainingData(:,1))
        for j=1:length(trainingData(i,c).handPos(1,:))
            % insert x and y handPos in the trajectory arrays
            trajectories(:,j) = trajectories(:,j) + trainingData(i,c).handPos(1:2,j);
        end
    end
    % compute the average trajectory for each class
    average_traj(c).handPos = trajectories(:,:)/length(trainingData(:,1));
end

modelParameters.trajectory = average_traj;
modelParameters.classificationParameters = classificationParameters;
modelParameters.coeffs_gc = coeffs_gc;
modelParameters.r=r;
modelParameters.M_lda=LDA;
modelParameters.M_pca_SVM=PCA_SVM;
end

%Extracts unsuperivised learning algorithm model parameters
function [int,mean_x,U,L]=eigenmodel(x,p)
% int is integer equal to the number of columns in 'x'
int=size(x,2);
mean_x=mean(x,2);
A=x-mean_x;
% Sample covariance matrix
cov_m=A'*A/int; 
[U,L]=eig(cov_m);
p=min(p,size(U,2));
[~,ind]=maxk(diag(L),p);
U=A*U(:,ind);
U=U./sqrt(sum(U.^2));
L=L(ind,ind);
end

%create feature vectors (f)
function [X,l,t]=organize_data(data,dt,T_end)
T=dt:dt:T_end;
X0=zeros([98,size(data,1),size(data,2),length(T)]);

for ind=1:length(T)
    t1=dt*(ind-1)+1;
    t2=dt*ind;
    for k=1:size(data,2)
        for n=1:size(data,1)
            for i=1:98
                X0(i,n,k,ind)=sum(data(n,k).spikes(i,t1:t2))/dt;
            end
        end
    end
end
X1=zeros([size(X0,1)*floor(T(end)/dt) size(X0,2) size(X0,3)]);
t=zeros([1 size(X0,1)*floor(T(end)/dt)]);
for ind=1:floor(T(end)/dt)
    X1(((ind-1)*98+1):((ind-1+1)*98),:,:)=X0(:,:,:,ind);
    t(1,((ind-1)*98+1):((ind-1+1)*98))=T(ind);
end
X=zeros([size(X1,1) size(X1,2)*size(data,2)]);
l=zeros([1 size(X1,2)*size(data,2)]);
for k=1:size(data,2)
    X(:,(k-1)*size(X1,2)+(1:size(X1,2)))=X1(:,:,k);
    l(:,(k-1)*size(X1,2)+(1:size(X1,2)))=k;
end
end
