function pred = do_SVM(Xt,Xmean,s2,c)
model1 = svmTrain(Xmean', [0 0 1 1 1 1 0 0]', c, @(x1, x2) gaussianKernel(x1, x2, s2));
model2 = svmTrain(Xmean(:,3:6)', [1 1 0 0]', c, @(x1, x2) gaussianKernel(x1, x2, s2));
model3 = svmTrain(Xmean(:,[1 2 7 8])', [1 1 0 0]', c, @(x1, x2) gaussianKernel(x1, x2, s2));
pred=zeros(1,size(Xt,2));
for n=1:size(Xt,2)
    p1=svmPredict(model1,Xt(:,n)');
    if p1==1 %3-6
        p2=svmPredict(model2,Xt(:,n)');
        if p2==1 %3-4
            pred(n)=do_kNN_fast(2,Xt(:,n),Xmean(:,3:4),3:4,1);
        else
            pred(n)=do_kNN_fast(2,Xt(:,n),Xmean(:,5:6),5:6,1);
        end
    else % 1 2 7 8
        p2=svmPredict(model3,Xt(:,n)');
        if p2==1 %1-2
            pred(n)=do_kNN_fast(2,Xt(:,n),Xmean(:,1:2),1:2,1);
        else
            pred(n)=do_kNN_fast(2,Xt(:,n),Xmean(:,7:8),7:8,1);
        end
    end
end