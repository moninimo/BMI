function [part1,part2]=split_data(data, part1_percentage)
N1=round(part1_percentage*size(data,1));
temp=randperm(size(data,1));
train_index=temp(1:N1);
part1=data(train_index,:);
if N1<size(data,1)
    test_index=temp((N1+1):end);
    part2=data(test_index,:);
else
    part2=[];
end
end


