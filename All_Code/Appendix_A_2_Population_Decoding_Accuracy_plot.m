clc;clear;close all;
acc_mat=[];
N_iter=20;
for repetitions=1:N_iter
    tic
    repetitions
load("monkeydata_training.mat")

%Random permutations of 100 trials
ix = randperm(length(trial));
split = 70; %As percentage

% Select training and testing data (you can choose to split your data in a different way if you wish)
train = trial(ix(1:split),:);
test = trial(ix(split+1:end),:);

N=size(train,1);
angles=[30 70 110 150 190 230 310 350];%*pi/180;

index_mat=[];
for T_end=320:20:560
    T_end;
    firing_rate_mean_for_all=[];
    firing_rate_std_for_all=[];
    for i=1:98
        firing_rate_mean=zeros(1,8);
        firing_rate_std=zeros(1,8);
        for k=1:8
            lengths=zeros(1,N);
            firing_rates=zeros(1,N);
            for n=1:N
                temp=trial(n,k).spikes(i,1:T_end);
                lengths(n)=length(temp);
                firing_rates(n)=sum(temp);
            end
            firing_rates=firing_rates./lengths;
            firing_rate_mean(k)=mean(firing_rates);
            firing_rate_std(k)=std(firing_rates);
        end
        %plot(angles,firing_rate_mean,'o-')
        firing_rate_mean_for_all=[firing_rate_mean_for_all;firing_rate_mean];
        firing_rate_std_for_all=[firing_rate_std_for_all;firing_rate_std];
        %xticks(angles)
        %hold on
    end
    min_firing=min(firing_rate_mean_for_all');
    max_firing=max(firing_rate_mean_for_all');
    [peak_val,peak_ind]=max(firing_rate_mean_for_all');
    std_at_peak=zeros(1,length(peak_ind));
    for ind=1:length(peak_ind)
        std_at_peak(ind)=firing_rate_std_for_all(ind,peak_ind(ind));
    end
    all_index=[];
    for k=1:8
        dif1=[];
        dif2=[];
        indexes=[];
        for ind=1:98
            lab=std_at_peak(ind)/peak_val(ind);
            if peak_ind(ind)==k && lab<0.5
                temp=firing_rate_mean_for_all(ind,:);
                temp=(temp-min(temp))/(max(temp)-min(temp));
                [vals,sorted]=sort(temp,'descend');
                dif1=[dif1 temp(sorted(1))-temp(sorted(2))];
                dif2=[dif2 temp(sorted(2))-temp(sorted(3))];
                indexes=[indexes ind];
            end
        end
        indexes;
        [~,best_index1]=max(dif1);
        dif2(best_index1)=0;
        [~,best_index2]=max(dif2);
        all_index=[all_index indexes([best_index1, best_index2])];
    end
    index_mat=[index_mat;all_index];
end

%%

%
% firing_rate_mean_for_all=[];
% firing_rate_std_for_all=[];
% for i=1:98
%     firing_rate_mean=zeros(1,8);
%     firing_rate_std=zeros(1,8);
%     for k=1:8
%         lengths=zeros(1,N);
%         firing_rates=zeros(1,N);
%         for n=1:N
%             temp=train(n,k).spikes(i,:);
%             temp=temp(1:end);
%             lengths(n)=length(temp);
%             firing_rates(n)=sum(temp);
%         end
%         firing_rates=firing_rates./lengths;
%         firing_rate_mean(k)=mean(firing_rates);
%         firing_rate_std(k)=std(firing_rates);
%     end
%     %plot(angles,firing_rate_mean,'o-')
%     firing_rate_mean_for_all=[firing_rate_mean_for_all;firing_rate_mean];
%     firing_rate_std_for_all=[firing_rate_std_for_all;firing_rate_std];
%     %xticks(angles)
%     %hold on
% end
% average_firing=mean(firing_rate_mean_for_all');
% min_firing=min(firing_rate_mean_for_all');
% max_firing=max(firing_rate_mean_for_all');
% [peak_val,peak_ind]=max(firing_rate_mean_for_all');


%%
% neuron_index=1:98;
% thr=0.5;
%
% std_at_peak=zeros(1,length(peak_ind));
% for ind=1:length(peak_ind)
%     std_at_peak(ind)=firing_rate_std_for_all(ind,peak_ind(ind));
% end
% %plot(std_at_peak./peak_val);
% useful_peak_ind=peak_ind((std_at_peak./peak_val)<thr);
% neuron_index=neuron_index((std_at_peak./peak_val)<thr);
angles_x=cos(angles*pi/180);
angles_y=sin(angles*pi/180);
for time_index=1:13
    for kappa=1:8
        kappa;
        %figure(time_index)
        
%         neuron_index=[18 43 4 84 40 44 22 98 94 2 90 91 81 87];
        neuron_index=index_mat(time_index,:);
        useful_peak_ind=[1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8];
        ind=1;
        m=zeros(size(neuron_index));
        F_x=m;
        F_y=m;
        subplot(4,2,kappa)
        for n=1:30
            for k=kappa
                for index=1:length(neuron_index)
                    i=neuron_index(index);
                    temp=test(n,k).spikes(i,:);
                    temp=temp(1:end);
                    m(index)=((sum(temp)/length(temp))-min_firing(i))/max_firing(i);
                    if m(index)<0
                        m(index)=0;
                    end
                    F_x(index)=angles_x(useful_peak_ind(index));%/sum(useful_peak_ind==useful_peak_ind(index));
                    F_y(index)=angles_y(useful_peak_ind(index));%/sum(useful_peak_ind==useful_peak_ind(index));
                end
                Fx=sum(m.*F_x);%/sum(m);
                Fy=sum(m.*F_y);%/sum(m);
                pred(kappa,n)=atan2(Fy,Fx)*180/pi;
                plot([0 angles_x(k) Fx./sqrt(Fx.^2+Fy.^2)],[0 angles_y(k) Fy./sqrt(Fx.^2+Fy.^2)],'-o')
                hold on
                axis square
            end
            x=-1:0.01:1;
            plot(x,sqrt(1-x.^2),'k');
            plot(x,-sqrt(1-x.^2),'k');
            title(num2str(time_index));
        end
        hold off
    end
    pred_ang=pred+(pred<0)*360;
    for n=1:30
        for k=1:8
            temp=pred_ang(k,n);
            if temp>=10 && temp<50
                pred_dir(n,k)=1;
            elseif temp>=50 && temp<90
                pred_dir(n,k)=2;
            elseif temp>=90 && temp<130
                pred_dir(n,k)=3;
            elseif temp>=130 && temp<170
                pred_dir(n,k)=4;
            elseif temp>=170 && temp<210
                pred_dir(n,k)=5;
            elseif temp>=210 && temp<250
                pred_dir(n,k)=6;
            elseif temp>=250 && temp<290
                pred_dir(n,k)=0;
            elseif temp>=290 && temp<330
                pred_dir(n,k)=7;
            else
                pred_dir(n,k)=8;
            end
        end
    end
    acc(time_index)=sum(sum(pred_dir==(ones(30,1)*(1:8))))/240;
end
acc
acc_mat=[acc_mat;acc];
toc
end
%%
mean_acc=mean(acc_mat);
std_acc=std(acc_mat);
figure
plot(320:20:560,100*mean_acc)
xlabel('Classification time (ms)')
ylabel('Accuracy (%)')
