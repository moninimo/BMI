clc;clear;close all;
load("monkeydata_training.mat")
% n=Trial, k=angle (one of eight), i=neuron_id
% N=number of trials, number of angles =8, number of neurons =98

angles=[30 70 110 150 190 230 310 350];%*pi/180;
N=100;
index_mat=[];
for T_end=320:20:560
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
    firing_rate_mean_for_all=[firing_rate_mean_for_all;firing_rate_mean];
    firing_rate_std_for_all=[firing_rate_std_for_all;firing_rate_std];
end

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
        if peak_ind(ind)==k
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
    [~,best_index2]=max(dif2);
    all_index=[all_index indexes([best_index1, best_index2])];
end
index_mat=[index_mat;all_index];
end
%%

for k=1:8
    figure(k)
    count=1;
    for ind=1:98
        lab=std_at_peak(ind)/peak_val(ind);
        if peak_ind(ind)==k
            subplot(5,5,count)
            temp=firing_rate_mean_for_all(ind,:);
            [~,index]=max(temp);
            plot([angles-360 angles angles+360],[temp temp temp],'o-')
            xlim([-180 180]+angles(index));
            title([num2str(ind)])
            count=count+1;
        end
    end
end

