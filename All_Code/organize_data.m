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