% Ground work 
rng('shuffle');
[allClasses, allLabels]=uniqueCell(labels);
cvFold=5;

% labelsTemp will be used for cross validation.
labelsTemp=zeros(length(labels), 1);
for i=1:length(labels)
    for j=1:length(allLabels)
        if isequal(labels{i}, allLabels{j})
            labelsTemp(i)=j;
        end
    end
end

obtainedLabels=cell(cvFold, 1);
accuracySave=zeros(cvFold, 1);
originalLabels=cell(cvFold, 1);

%CV partition
CVO = cvpartition(labelsTemp,'k',cvFold);
for cv=1:cvFold
    trIdx = CVO.training(cv);
    teIdx = CVO.test(cv);
    trainS=dataset(trIdx, :);
    testS=dataset(teIdx, :);
    labelTrPoints=find(trIdx);
    labelTPoints=find(teIdx);
    originalLabels{cv}=labelTPoints;
    labelTr=cell(1, length(labelTrPoints));
    labelT=cell(1, length(labelTPoints));
    for i=1:length(labelTrPoints)
        labelTr{i}=labels{labelTrPoints(i)};
    end
    for i=1:length(labelTPoints)
        labelT{i}=labels{labelTPoints(i)};
    end

    %Initialisation 
    p=-20; % power used in generalised mean.
    eata=-0.5; % learning rate.
    maxIteration=20000;
    inputDim=size(dataset, 2);
    hiddenDim=10;
    outputDim=length(allClasses);
    wInputHidden=2*rand(inputDim, hiddenDim)-1;
    wHiddenOutput=2*rand(hiddenDim, outputDim)-1;
    biasInput=2*rand(1, hiddenDim)-1;
    biasHidden=2*rand(1, outputDim)-1;

    n=size(trainS, 1);
    m=size(testS, 1);
    errorStore=[];
    for iteration=1:maxIteration 

        disp(iteration);
        errorSave=0;

        % Cleaning old changes 
        clear avgCngWInputHidden;
        clear avgCngWHiddenOutput;
        clear avgCngBiasInput;
        clear avgCngBiasHidden;

        % Initializing average changes
        avgCngWInputHidden=zeros(inputDim, hiddenDim);
        avgCngWHiddenOutput=zeros(hiddenDim, outputDim);
        avgCngBiasInput=zeros(1, hiddenDim);
        avgCngBiasHidden=zeros(1, outputDim);
            
        for i=1:n

            % Cleaning weight changes
            clear changeBiasInput;
            clear changeBiasHidden;
            clear changeWInputHidden;
            clear changeWHiddenOutput;

            % Reinitializing weight changes 
            changeBiasInput=zeros(1, hiddenDim);
            changeBiasHidden=zeros(1, outputDim);
            changeWInputHidden=zeros(inputDim, hiddenDim);
            changeWHiddenOutput=zeros(hiddenDim, outputDim);

            % Feed foroward
            clear hiddenOut;
            clear outputOut;

            hiddenIn=trainS(i, :)*wInputHidden + biasInput;
            hiddenOut=1./(1+exp(-hiddenIn));
            outputIn=hiddenOut*wHiddenOutput + biasHidden;
            outputOut=1./(1+exp(-outputIn));
           
            % Backpropagation 

            error=0;
            for j=1:length(labelTr{i})

                target=zeros(1, outputDim);
                target(labelTr{i}(j))=1;
                temp1=(sum((outputOut-target).^2)*0.5)^(p-1);
                error=error+(sum((outputOut-target).^2)*0.5)^p;
                for k1=1:outputDim
                    for k2=1:hiddenDim
                        temp2=(outputOut(k1)-target(k1))*outputOut(k1)*(1-outputOut(k1))*hiddenOut(k2);
                        changeWHiddenOutput(k2, k1)=changeWHiddenOutput(k2, k1)+(temp1*temp2);
                    end
                    temp3=(outputOut(k1)-target(k1))*outputOut(k1)*(1-outputOut(k1));
                    changeBiasHidden(k1)=changeBiasHidden(k1)+(temp1*temp3);
                end

                for k1=1:hiddenDim
                    temp4=((outputOut-target).*outputOut.*(1-outputOut))*(wHiddenOutput(k1, :)');
                    for k2=1:inputDim
                        temp5=hiddenOut(k1)*(1-hiddenOut(k1))*trainS(i, k2);
                        changeWInputHidden(k2, k1)=changeWInputHidden(k2, k1)+(temp1*temp4*temp5);
                    end
                    temp6=hiddenOut(k1)*(1-hiddenOut(k1));
                    changeBiasInput(k1)=changeBiasInput(k1)+(temp1*temp4*temp6);
                end
                               
            end

            error = error/length(labelTr{i});
            temp7 = error^((1/p)-1);
            temp8=eata*temp7*(1/length(labelTr{i}));
            avgCngWHiddenOutput=avgCngWHiddenOutput+(changeWHiddenOutput.*temp8);
            avgCngWInputHidden=avgCngWInputHidden+(changeWInputHidden.*temp8);
            avgCngBiasHidden=avgCngBiasHidden+(changeBiasHidden.*temp8);
            avgCngBiasInput=avgCngBiasInput+(changeBiasInput.*temp8);
            errorSave=errorSave+(error^(1/p));

        end

        % Effective changes 
        avgCngWHiddenOutput=avgCngWHiddenOutput./n;
        avgCngWInputHidden=avgCngWInputHidden./n;
        avgCngBiasHidden=avgCngBiasHidden./n;
        avgCngBiasInput=avgCngBiasInput./n;
        errorSave=errorSave/n;
        errorStore=vertcat(errorStore, errorSave);

        wHiddenOutput=wHiddenOutput+avgCngWHiddenOutput;
        wInputHidden=wInputHidden+avgCngWInputHidden;
        biasInput=biasInput+avgCngBiasInput;
        biasHidden=biasHidden+avgCngBiasHidden;

        if errorSave<=0.01
            display('Terminating after convergence');
            break;
        end
    end
    display('Training complete ...');

	% Testing Phase
    p1=zeros(1, m);
    accuracy=0;
    for i=1:m

        % Feed foroward
        clear hiddenOut;
        clear outputOut;

        hiddenIn=testS(i, :)*wInputHidden + biasInput;
        hiddenOut=1./(1+exp(-hiddenIn));
        outputIn=hiddenOut*wHiddenOutput + biasHidden;
        outputOut=1./(1+exp(-outputIn));

        [maxOut, maxLoc]=max(outputOut);
        p1(i)=maxLoc; % selecting the class label with maximum outoput.
        if ismember(p1(i), labelT{i})
            accuracy=accuracy+1;
        end
        
    end
    accuracySave(cv)=accuracy/m;
    obtainedLabels{cv}=p1;
    display('Testing complete...');
    
end

meanAcc=mean(accuracySave);
stdAcc=std(accuracySave);

        
            
        
        
        
    
    
    
    
    
    
    
    