function  [modelParameters] = positionEstimatorTraining(trainingData)
% - trainingData:
%     trainingData(n,k)              (n = trial id,  k = reaching angle)
%     trainingData(n,k).trialId      unique number of the trial
%     trainingData(n,k).spikes(i,t)  (i = neuron id, t = time)
%     trainingData(n,k).handPos(d,t) (d = dimension [1-3], t = time)

r=size(trainingData,1)-1; %Number of Principal Components for PCR

t_train = 320:80:560; %times at which angle classification is updated
classificationParameters = struct;
[F,l,t]=organize_data(trainingData,80,t_train(end)); % create feature vectors for training data dt=80
for t_ind=1:length(t_train)
    T=t_train(t_ind);
    X=F(t<=T,:);
    M_lda=7;
    M_pca_kNN=170;
    M_pca_SVM=180;
    M_pca_bayes=31;
    % DO PCA using max number of bases (N), the fact that we pre-do it makes the
    % code faster
    [~,mx,Wpca,~]=eigenmodel(X,size(X,2));
    [sb,sw]=make_sb_sw(X,l); %Compute scatter matrices
    M_pca=M_pca_kNN;
    % PCA-LDA for the optimal Mpca for kNN, Mlda=7
    Wopt=make_Wopt(Wpca,M_pca,M_lda,sb,sw);
    W=Wopt'*(X-mx);
    Wmean=zeros([size(W,1) 8]);
    for k=1:8
        Wmean(:,k)=mean(W(:,l==k),2);
    end
    %save model parameters
    classificationParameters(t_ind).Wopt_kNN=Wopt;
    classificationParameters(t_ind).mx_kNN=mx;
    classificationParameters(t_ind).Wmean_kNN=Wmean;
    
    % PCA-LDA for the optimal Mpca for SVM, Mlda=7
    M_pca=M_pca_SVM;
    Wopt=make_Wopt(Wpca,M_pca,M_lda,sb,sw);
    W=Wopt'*(X-mx);
    Wmean=zeros([size(W,1) 8]);
    for k=1:8
        Wmean(:,k)=mean(W(:,l==k),2);
    end
    %save model parameters
    classificationParameters(t_ind).Wopt_SVM=Wopt;
    classificationParameters(t_ind).mx_SVM=mx;
    classificationParameters(t_ind).Wmean_SVM=Wmean;
    
    %PCA-LDA for optimal Mpca for bayes, Mlda=7
    M_pca=M_pca_bayes;
    Wopt=make_Wopt(Wpca,M_pca,M_lda,sb,sw);
    W=Wopt'*(X-mx);
    %find the cov-mtrix and mean of the gaussian fitted in data
    A_tot=[];
    C_1_tot=[];
    m_tot=[];
    for k=1:8
        [A,C_1,m]=get_gauss_params(W(:,l==k));
        A_tot=cat(3,A_tot,A);
        C_1_tot=cat(3,C_1_tot,C_1);
        m_tot=cat(3,m_tot,m);
    end
    %save model parameters
    classificationParameters(t_ind).A_bayes=A_tot;
    classificationParameters(t_ind).C_bayes=C_1_tot;
    classificationParameters(t_ind).m_bayes=m_tot;
    classificationParameters(t_ind).Wopt_bayes=Wopt;
    classificationParameters(t_ind).mx_bayes=mx;
    classificationParameters(t_ind).Wmean_bayes=Wmean;
end

%create feature vectors for training data (dt=20)
T_end = 560;
dt=20;
T = (320:dt:T_end);
[feat,l,t]=organize_data(trainingData,dt,T_end);
%get hand positions from data
[~,~,x,y,~,~]=get_all_handPos(trainingData);
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
        
        [~,mx,U,~]=eigenmodel(features_temp,r); 
        W=U'*(features_temp-mx);
        coeffs_gc(time_index, angle_index).mean_feature=mx;
        bx=U*(W*W')^(-1)*W* hand_pos_temp_x;
        by=U*(W*W')^(-1)*W* hand_pos_temp_y;
        
        coeffs_gc(time_index, angle_index).values = [bx , by];
    end
end
average_traj = calculate_avg_traj(trainingData);

modelParameters.trajectory = average_traj;
modelParameters.classificationParameters = classificationParameters;
modelParameters.coeffs_gc = coeffs_gc;
modelParameters.r=r;
modelParameters.M_lda=M_lda;
modelParameters.M_pca_kNN=M_pca_kNN;
modelParameters.M_pca_SVM=M_pca_SVM;
modelParameters.M_pca_bayes=M_pca_bayes;
end

% this function calculates the average trajectory
function average_traj = calculate_avg_traj(trainingData)

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
end

%this function extracts PCA model parameters
function [N,mx,U,L]=eigenmodel(x,p)
N=size(x,2);
mx=mean(x,2);
A=x-mx;
S=A'*A/N;
[U,L]=eig(S);
p=min(p,size(U,2));
[~,ind]=maxk(diag(L),p);
U=A*U(:,ind);
U=U./sqrt(sum(U.^2));
L=L(ind,ind);
end

%this function makes the within-class/between class scatter matrices
function [sb,sw]=make_sb_sw(X,l)
c=unique(l);
mc=zeros(size(X,1),length(c));
mx=mean(X,2);
for im_id=1:length(c)
    mc(:,im_id)=mean(X(:,l==c(im_id)),2);
end
sb=(mc-mx)*(mc-mx)';
st=(X-mx)*(X-mx)';
sw=st-sb;
end

%create optimal PCA-LDA matrix
function [Wopt]=make_Wopt(Wpca,M_pca,M_lda,sb,sw)
[Wlda,L] = eig((Wpca(:,1:M_pca)'*sw*Wpca(:,1:M_pca))^-1*Wpca(:,1:M_pca)'*sb*Wpca(:,1:M_pca));
[~,ind]=maxk(diag(L),M_lda);
Wopt=Wpca(:,1:M_pca)*Wlda(:,ind);
end

%get scaling factor, covariance and mean for Bayesian
function [A,C_1,m]=get_gauss_params(X)
C=cov(X');
[~,S,~] = svd(C);
temp=S(S~=0);
a=10^(-sum(log10(temp))/length(temp));
C_1=a*(a*C)^-1;
A=-(size(C,1)/2)*log(2*pi/a)-0.5*log(det(a*C));
m=mean(X,2);
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

%Extracting hand positions
function[mx,my,x,y,l,in_data]=get_all_handPos(data)
%mx, my = average trajectory
%x,y - row is trials, column is time, 3rd dimension is angle
%l (matrix) - row is length of trial, column is corresponding angle

x_mat=[];
y_mat=[];
for k=1:8
    for n=1:size(data,1)
        x=data(n,k).handPos(1,:);
        l(n,k)=length(x);
        while length(x)>size(x_mat,2) && size(x_mat,1)>0
            x_mat=[x_mat x_mat(:,end)];
        end
        while size(x_mat,2)>length(x) && size(x_mat,1)>0
            x=[x x(end)];
        end
        x_mat=[x_mat;x];
        y=data(n,k).handPos(2,:);
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
mx=zeros([1 size(x_mean,2) size(x_mean,1)]);
my=zeros([1 size(y_mean,2) size(y_mean,1)]);
for k=1:8
    mx(1,:,k)=x_mean(k,:);
    my(1,:,k)=y_mean(k,:);
end
in_data=zeros(size(x));
for k=1:8
    for n=1:size(in_data,1)
        in_data(n,1:l(n,k),k)=ones(1,l(n,k));
    end
end      
end