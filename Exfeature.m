function [X,l,t,mX] = Exfeature(data)
T = 320;
dt = 320;
datasize = size(data,1);
X0 = zeros([98,datasize,8,length(T)]);

for index=1:length(T)

    t1=dt*(index-1)+1;
    t2=dt*index;

    for k=1:8
        for n=1:size(data,1)
            for i=1:98
                X0(i,n,k,index)=sum(data(n,k).spikes(i,t1:t2))/dt;
            end
        end
    end
end

X1=zeros([size(X0,1)*floor(T(end)/dt) size(X0,2) size(X0,3)]);
t=zeros([1 size(X0,1)*floor(T(end)/dt)]);

for index=1:floor(T(end)/dt)
    X1(((index-1)*98+1):((index-1+1)*98),:,:)=X0(:,:,:,index);
    t(1,((index-1)*98+1):((index-1+1)*98))=T(index);
end
X=zeros([size(X1,1) size(X1,2)*8]);
l=zeros([1 size(X1,2)*8]);
for k=1:8
    X(:,(k-1)*size(X1,2)+(1:size(X1,2)))=X1(:,:,k);
    l(:,(k-1)*size(X1,2)+(1:size(X1,2)))=k;
end
mX=reshape(mean(X1,2),[size(X1,1) size(X1,3)]);
end

