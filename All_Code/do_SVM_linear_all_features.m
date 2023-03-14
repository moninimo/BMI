function pred = do_SVM_linear_all_features(Xt,X,l,c)

%LAYER 1:
%3,4,5,6 ->1 vs 1,2,7,8->0
model1 = svmTrain(X', 1*[(l>2)&(l<7)]', c, @(x1, x2) linearKernel(x1, x2));


%LAYER 2
l2a=l([(l>2)&(l<7)]);
%2a)3,4->1 VS 5,6->0
model2a = svmTrain(X(:,[(l>2)&(l<7)])', 1*[l2a<5]', c, @(x1, x2) linearKernel(x1, x2));
%2b)1,2->1 vs 7,8->0
l2b=l(~[(l>2)&(l<7)]);
model2b = svmTrain(X(:,~[(l>2)&(l<7)])', 1*[l2b<3]', c, @(x1, x2) linearKernel(x1, x2));

%LAYER 3
%2a1)3 vs 4
l2a1=l((l==3) | (l==4));
model2a1 = svmTrain(X(:,[(l==3)|(l==4)])', 1*(l2a1==3)', c, @(x1, x2) linearKernel(x1, x2));

%2a2)5 vs 6
l2a2=l((l==5) | (l==6));
model2a2 = svmTrain(X(:,[(l==5)|(l==6)])', 1*(l2a2==5)', c, @(x1, x2) linearKernel(x1, x2));

%2b1)1 vs 2
l2b1=l((l==1) | (l==2));
model2b1 = svmTrain(X(:,[(l==1) | (l==2)])', 1*(l2b1==1)', c, @(x1, x2) linearKernel(x1, x2));

%2b2)7 vs 8
l2b2=l((l==7) | (l==8));
model2b2 = svmTrain(X(:,[(l==7) | (l==8)])', 1*(l2b2==7)', c, @(x1, x2) linearKernel(x1, x2));


pred=zeros(1,size(Xt,2));
for n=1:size(Xt,2)
    p1=svmPredict(model1,Xt(:,n)');
    if p1==1 %3-6
        p2=svmPredict(model2a,Xt(:,n)');
        if p2==1 %3-4
            p3 = svmPredict(model2a1,Xt(:,n)');
            if p3==1
                pred(n)= 3;
            else
                pred(n)= 4;
            end
        else
            pred(n)= svmPredict(model2a2,Xt(:,n)');
            if p3==1
                pred(n)= 5;
            else
                pred(n)= 6;
            end
        end
    else % 1 2 7 8
        p2=svmPredict(model2b,Xt(:,n)');
        if p2==1 %1-2
            p3=svmPredict(model2b1,Xt(:,n)');
            if p3==1
                pred(n)= 1;
            else
                pred(n)= 2;
            end
        else
            p3= svmPredict(model2b2,Xt(:,n)');
            if p3==1
                pred(n)= 7;
            else
                pred(n)= 8;
            end
        end
    end
%     [p1 p2 p3 pred(n)]
end