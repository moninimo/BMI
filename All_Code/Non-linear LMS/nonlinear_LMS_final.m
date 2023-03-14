%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Nonlinear LMS Algorithm an Adaptive Nonlinearity  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Function Definition
function [y,error,w,lambda,s] = nonlinear_LMS_final(x,d,mu,N,w_init)
%----Inputs----- 
%x = input feature vector, d = mean hand position, mu = step-size, N=number of trials used 

%----Outputs-----
% y = output signal, error = error vector, h and g = matrices of weight evolution  

    filter_order = length(x);
    %Initialising variables
    y = zeros(N,1);
    error=zeros(N-1,1);
    w = zeros(filter_order,N+1); %Stores weight time-evolution
    %Using pre-training weights
    w(:,1)=w_init;
    lambda = ones(N+1,1);
    s=zeros(N,1);

    for n=1:N
        s(n) = w(:,n)'*x(:,n);
        y(n) = lambda(n)*tanh(s(n));
        error(n)=d(n)-y(n); %Error calculation
        % weights update rule
        w(:,n+1)=w(:,n)+(mu*error(n)*x(:,n)*lambda(n)*(1-(tanh(s(n))^2)));
        %Scale update rule - From Mandic book
%         lambda(n+1) =  lambda(n) + rho*error(n)*tanh(s(n));
    end
    w = w(:,2:end); 
end