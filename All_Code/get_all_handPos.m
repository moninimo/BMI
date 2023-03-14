function[mx,my,x,y,l,in_data]=get_all_handPos(data)
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
