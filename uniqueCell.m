function [ uniqueElements, uniqueLabels ] = uniqueCell( a )

uniqueElements=[];
uniqueArray=[];
n=length(a);

uniqueElements=horzcat(uniqueElements, a{1});
uniqueArray=horzcat(uniqueArray, 1);

for i=2:n
    m1=length(a{i});
    for j=1:m1
        if ~ismember(a{i}(j), uniqueElements)
            uniqueElements=horzcat(uniqueElements, a{i}(j));
        end
    end
    m2=length(uniqueArray);
    flag=0;
    for j=1:m2
        if isequal(a{i}, a{uniqueArray(j)})
            flag=1;
        end
    end
    if flag==0
        uniqueArray=horzcat(uniqueArray, i);
    end
end
uniqueLabels=cell(1, length(uniqueArray));
for i=1:length(uniqueArray)
    uniqueLabels{i}=a{uniqueArray(i)};
end

end

