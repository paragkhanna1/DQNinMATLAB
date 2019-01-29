%Initialize the environment:
clear all
close all
L=2;%Distance from end of robot to castor wheel ahead
b=2;%distance between 2 actuated wheels(back wheels,
%above is 2*dis btw wheel and centre of robot, defined in stateUpdate
X=[10;10;pi/4];
%figure('Position',[0 0 631 600]) %size to get 489x489 image saved from fig
figure('Position',[0 0 316 300]) %size to get 245x245 image saved from fig
F = getframe;    
[I,Map] = frame2im(F);
while(1) %forcing to get a image of 245x245
close all;
figure('Position',[0 0 330 315]);

F = getframe;    
[I,Map] = frame2im(F);
    if(size(I,1)==256 && size(I,2)==256)
        break;
    end
end
%figure('Position',[0 0 526 500])
%figure('Position',[0 0 316 300])
rectangle('Position',[0,0,50,50]);
rectangle('Position',[23,23,4,4],'EdgeColor','k'); %GOAL always centrred at 25

%rectangle('Position',[10,40,2,2],'Curvature',[1 1]);
rectangle('Position',[20,30,2,2],'Curvature',[1 1]);
%rectangle('Position',[20,5,2,2],'Curvature',[1 1]);
%rectangle('Position',[40,30,2,2],'Curvature',[1 1]);
%plot(X(1),X(2));

hold on
[~,tri]=stateUpdate_getRoboTriangle_DQN(X,0);
%p = plot(X(1),X(2),'k*');
p=plot(tri(1,:), tri(2,:), 'linewidth', 1,'color',[0 0 0]);


%p=plot(X(1),X(2),'r*');
hold off
%%
% Perform random actions to generate Get Data for Training the CNN first
% time, This is analogous to Initializing the CNN with random weights.
for i=1:200
    action=choose_random_action();
     %  X=[10;10;pi/4]; %can be randomized later
                X=[randi([4 46]);randi([4 46]);randi([1 6])*rand()]; %randomized state
    F = getframe;    
    [I,Map] = frame2im(F);
    states(:,:,:,i)=rgb2gray(I);
            
    [X,tri]=stateUpdate_getRoboTriangle_DQN(X,action);%timestep of 1 unit
    reward(:,:,:,i)=get_reward(X,action);%reward must be a(1x1x4) output here, for 4 actions, (as required by CNN-MATLAB)
    %rewards act as random targets!
     
    set(p,'XData',tri(1,:));
    set(p,'YData',tri(2,:));
         %p.Xdata = X(1); %MAY HAVE TO USE THESE ones IN SOME MATLAB VERSION               
         %p.Ydata = X(2);
    drawnow;
    %pause(0.1)
end

%below is done also for creating and populating Replay Memory!
% So the memory size is determined the number of iterations of this loop
for i=1:50 %getting Replay DATA. Keep size>20 for Priority Memory (Check add_in_memory_priority)
    action=choose_random_action();
        X=[randi([4 46]);randi([4 46]);randi([1 6])*rand()]; %randomized state
    [X,tri]=stateUpdate_getRoboTriangle_DQN(X,action);%timestep of 1 unit
    F = getframe;    
    [I,Map] = frame2im(F);
    states2(:,:,:,i)=rgb2gray(I);

    [reward2(:,:,:,i),terminals(i)]=get_reward(X,action);%reward must be a(1x1x4) output here, for 4 actions, (as required by CNN-MATLAB)
    
    %rewards act as random targets!
    

    set(p,'XData',tri(1,:));
    set(p,'YData',tri(2,:));
    drawnow;
        F = getframe;    
        [I,Map] = frame2im(F);
        states2_new=rgb2gray(I);
        %    rectangle('Positi  on',[X(1)-1,X(2)-1,2,2]);
  
Replay_memory(i,:)={states2(:,:,:,i),action,reward2(:,:,action,i),terminals(i),states2_new};

end
Rp_mem_Size=size(Replay_memory,1);
Mini_Batch_Size=5;
%%
%Define the convolutional neural network architecture.

layers = [
    %imageInputLayer([489 489 1],'Name', 'input')
    imageInputLayer([256 256 1],'Name', 'input')
    convolution2dLayer(16,4,'Padding',1,'Name', 'conv1') %layer = convolution2dLayer(filterSize,numFilters)
    batchNormalizationLayer('Name', 'Batch_N')
    reluLayer('Name', 'relu1')
    
    %maxPooling2dLayer(2,'Stride',2,'Name', 'maxPool1')
    
%     convolution2dLayer(8,4,'Padding',1,'Name', 'conv2')
%     batchNormalizationLayer('Name', 'Batch_N2')
%     reluLayer('Name', 'relu2')
    
    %maxPooling2dLayer(2,'Stride',2,'Name', 'maxPool2')
%     
%     convolution2dLayer(4,4,'Padding',1,'Name', 'conv3')
%     batchNormalizationLayer('Name', 'Batch_N3')
%     reluLayer('Name', 'relu3')
%     
   % fullyConnectedLayer(15,'Name', 'FullyC')% size is eual to number of actions
  
        fullyConnectedLayer(8,'Name', 'FullyC2')% size is eual to number of actions
  regressionLayer('Name','Output')];

options = trainingOptions('sgdm', ...
    'Momentum',0.3, ...
    'MaxEpochs',4, ...
    'InitialLearnRate',0.00001, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(states,reward,layers,options);
%layersTransfer = net.Layers(1:end);
%net = trainNetwork(states2,reward2,layersTransfer,options);

%used previously generated random States,Rewarrd to train,
%thus the network is initialized and ready for DQN
%%
options = trainingOptions('sgdm',  ...
'Momentum',0.3, ...
'MaxEpochs',3, ...
'InitialLearnRate',0.0001, ...
'Verbose',false, ...
 'Plots','none'); %to supress plot occuring at each step when CNN is trained

Max_episodes=1;
Max_steps=100;
gamma_learning=0.5; % for the target equation :target_st=reward_st+gamma*maxQ_new; 
epsilon=0.7; %e-greedy algorithm factor t choose action
for ep=1:Max_episodes
    %set intitial position 
    %X=[10;10;randi([1 6])*rand()]; %can be randomized later
    if(ep<10)
        X=[randi([4 46]);randi([4 46]);randi([1 6])*rand()];
    %X=[10;10;pi/4];
    else
       % X=[randi([4 46]);randi([4 46]);randi([1 6])*rand()];
    X=[40;40;pi];
    end    
    [~,tri]=stateUpdate_getRoboTriangle_DQN(X,0); % GIVES the Triangle at this X with no change in X

    set(p,'XData',tri(1,:));
    set(p,'YData',tri(2,:));
    drawnow;
         
    drawnow;
    Net_reward(ep)=0;
    for st=1:Max_steps
        %acquire current state image,
        F = getframe;    
        [I,Map] = frame2im(F);
        state(:,:,:,1)=rgb2gray(I); 
%can use below created state_new variable instead of getting frame again,
%will se later!!!!
       
        
        if ep>25
            epsi=5*epsilon/(ep-epsilon*(ep)-5);
        else %Random Start State and intial 5 eps of fixedStart given fixed epsi
            epsi=epsilon;
        end
        if rand()<abs(epsi) %e-greedy algo, reducing epsilon
            act_st = randi([1 8],1);
        else
            %do a full forward pass through CNN.
            Q_st=predict(net,state(:,:,:,1));% acquire Q values for all acitons
            Q_st(isnan(Q_st)) = -1; %Removeing NAN values

            % choose action to do
            [~,act_st]=max(Q_st);%index of max Q value gives the desired action to do!
        end
        %get new state
    [X,tri]=stateUpdate_getRoboTriangle_DQN(X,act_st);%timestep of 1 unit
    [reward_st,term]=get_reward_simple(X);%reward must be a single output here
    %Update Figure 
    set(p,'XData',tri(1,:));
    set(p,'YData',tri(2,:));
    drawnow;
    %p.Xdata = X(1); %MAY HAVE TO USE THESE ones IN SOME MATLAB VERSION               
         %p.Ydata = X(2);
        F = getframe;    
        [I,Map] = frame2im(F);
        state_new(:,:,:,1)=rgb2gray(I); 
        
        Replay_memory=add_in_memory_priority(Replay_memory,state(:,:,:,1),act_st,reward_st,term,state_new(:,:,:,1));
        
        %sample RANDOMLYa Mini Batch from Replay Memory
        indices = randperm(size(Replay_memory,1));
        indices = indices(1:Mini_Batch_Size);
        Mini_Batch=Replay_memory(indices,:);
        
        %prepare Mini-Batch to train CNN
        for i=1:Mini_Batch_Size
            states_Mini_Batch(:,:,:,i)=cell2mat(Mini_Batch(i,1));
            actions_Mini_Batch(i)=cell2mat(Mini_Batch(i,2));
            rewards_Mini_Batch(i)=cell2mat(Mini_Batch(i,3));
            terminals_Mini_Batch(i)=cell2mat(Mini_Batch(i,4));
            states_new_Mini_Batch(:,:,:,i)=cell2mat(Mini_Batch(i,5));
        end

        Q_st_Mini_Batch=predict(net,states_Mini_Batch);% acquire Q values for all acitons
         NAN_trigger=0;
        if(isnan(Q_st_Mini_Batch))
        NAN_trigger=1
        end
        Q_st_Mini_Batch(isnan(Q_st_Mini_Batch)) = -1;
       

        %do a full forward pass through CNN to get Q values for new states
        Q_st_new_Mini_batch=predict(net,states_new_Mini_Batch);% acquire Q values for all acitons
        Q_st_new_Mini_batch(isnan(Q_st_new_Mini_batch)) = -1;
        
        if(NAN_trigger == 1 || max(abs(Q_st_new_Mini_batch(:))) > 10000 || max(abs(Q_st_Mini_Batch(:))) > 10000 )
            target_Mini_batch=target_Mini_batch_old;%for cases when Q_vals go large or NAN
            layersTransfer=old_layers;
        else
        
       % [maxQ_new,~]=max(Q_st_new);%is the max possible Q_value for next state
        
        %set Targets for CNN:
            for i=1:Mini_Batch_Size
                target_Mini_batch(:,:,1,i)= Q_st_Mini_Batch(i,1);target_Mini_batch(:,:,2,i)= Q_st_Mini_Batch(i,2);
                target_Mini_batch(:,:,3,i)= Q_st_Mini_Batch(i,3);target_Mini_batch(:,:,4,i)= Q_st_Mini_Batch(i,4);
                target_Mini_batch(:,:,5,i)= Q_st_Mini_Batch(i,5);target_Mini_batch(:,:,6,i)= Q_st_Mini_Batch(i,6);
                target_Mini_batch(:,:,7,i)= Q_st_Mini_Batch(i,7);target_Mini_batch(:,:,8,i)= Q_st_Mini_Batch(i,8);
                 
                %for all actions, default target is the Q value predicted, so loss
                %is zero(computed inside the CNN), hence no weights updated
                %target for taken action:
                if (terminals_Mini_Batch(i)==1) %TERMiNAL STATE REACHED, To end episode
                    target_st=rewards_Mini_Batch(i);
                else %non terminal state, Target is given by the update term from BELLMAN equation
                    target_st=rewards_Mini_Batch(i)+gamma_learning*max(Q_st_new_Mini_batch(i,:)); %gamma set as parameter before
                end
            %At this state, the target value for the action performed is got by
            %above
                target_Mini_batch(:,:,actions_Mini_Batch(i),i)=target_st;
      
            end
      
            target_Mini_batch_old=target_Mini_batch;%for cases when Q_vals go large or NAN
            targets=reshape(target_Mini_batch, [8 Mini_Batch_Size])'; %just to display out
            %TRAIN the Netwrok with above got (states,targets) pairs
            layersTransfer = net.Layers(1:end); %transferring all previous layers(with weights)
       
        end
        old_layers=layersTransfer;
      
        net = trainNetwork(states_Mini_Batch,target_Mini_batch,layersTransfer,options);
        
        % feed out number of steps
        steps(ep)=st;
        Net_reward(ep)=Net_reward(ep)+reward_st;

        if(term==1)%terminal reached
            break;
        end
%        disp(st);
    end
show=['Episode:',num2str(ep),' Num Stpes:',num2str(steps(ep)),' Total Reward:',num2str(Net_reward(ep))];
disp(show);
disp('Q_st_new_Mini_batch')
disp(Q_st_new_Mini_batch)
disp('targets_Mini_batch reshaped* ')
disp(targets)
end

%%
%evaluATION
X=[10;40;pi/4];
%figure('Position',[0 0 631 600]) %size to get 489x489 image saved from fig
figure('Position',[0 0 316 300]) %size to get 245x245 image saved from fig
F = getframe;    
[I,Map] = frame2im(F);
while(1) %forcing to get a image of 489x489
close all;
figure('Position',[0 0 330 315]);

F = getframe;    
[I,Map] = frame2im(F);
    if(size(I,1)==256 && size(I,2)==256)
        break;
    end
end
%figure('Position',[0 0 526 500])
%figure('Position',[0 0 316 300])
rectangle('Position',[0,0,50,50]);
rectangle('Position',[23,23,4,4],'EdgeColor','k');

%rectangle('Position',[10,40,2,2],'Curvature',[1 1]);
rectangle('Position',[20,30,2,2],'Curvature',[1 1]);
%rectangle('Position',[20,5,2,2],'Curvature',[1 1]);
%rectangle('Position',[40,30,2,2],'Curvature',[1 1]);
%plot(X(1),X(2));

hold on
[~,tri]=stateUpdate_getRoboTriangle_DQN(X,0);

p=plot(tri(1,:), tri(2,:), 'linewidth', 1,'color',[0 0 0]);

%p=plot(X(1),X(2),'r*');
hold off

Max_episodes=20;
Max_steps=100;
gamma_learning=1; % for the target equation :target_st=reward_st+gamma*maxQ_new; 
epsilon=0.4; %e-greedy algorithm factor t choose action
for ep=1:Max_episodes
    predictionError=0;
    %set intitial position 
            %X=[randi([4 46]);randi([4 46]);randi([1 6])*rand()]; %randomized state
X=[40;40;pi];

%    X=[45;10;pi/4]; %can be randomized later
     [~,tri]=stateUpdate_getRoboTriangle_DQN(X,0);

    set(p,'XData',tri(1,:));
    set(p,'YData',tri(2,:));
    drawnow;
  
  Net_reward_validation(ep)=0;
    for st=1:Max_steps
        %acquire current state image,
        F = getframe;    
        [I,Map] = frame2im(F);
        state(:,:,:,1)=rgb2gray(I); 
%can use below created state_new variable instead of getting frame again,
%will se later!!!!
       
        %do a full forward pass through CNN.
        Q_st=predict(net,state(:,:,:,1));% acquire Q values for all acitons
        
        % choose action to do
        [~,act_st]=max(Q_st);%index of max Q value gives the desired action to do!
%         
%         if rand()<(epsilon/(1*ep)) %e-greedy algo, reducing epsilon,
%             act_st = randi([1 4],1);
%         end

        [X,tri]=stateUpdate_getRoboTriangle_DQN(X,act_st);%timestep of 1 unit
        [reward_st,term]=get_reward_simple(X);%reward must be a single output here
        
        %get new state

    set(p,'XData',tri(1,:));
    set(p,'YData',tri(2,:));
    drawnow;
     %p.Xdata = X(1); %MAY HAVE TO USE THESE ones IN SOME MATLAB VERSION               
         %p.Ydata = X(2);
      
        F = getframe;    
        [I,Map] = frame2im(F);
        state_new(:,:,:,1)=rgb2gray(I); 
        
        %do a full forward pass through CNN.
        Q_st_new=predict(net,state_new);% acquire Q values for all acitons
        [maxQ_new,~]=max(Q_st_new);%is the max possible Q_value for next state
        
       
        %target for taken action:
        if (term==1) %TERMiNAL STATE REACHED, To end episode
            target_st=reward_st;
        else %non terminal state, Target is given by the update term from BELLMAN equation
            target_st=reward_st+gamma_learning*maxQ_new; %gamma set as parameter before
        end
        %At this state, the target value for the action performed is got by
        %above

        predictionError = predictionError+target_st - Q_st(act_st);
        
        % feed out number of steps
        steps_validation(ep)=st;
        Net_reward_validation(ep)=Net_reward_validation(ep)+reward_st;

        if(term==1)%terminal reached
            break;
        end
%        disp(st);

    end
    rmse(ep)=predictionError*predictionError;
show=['Validation: Episode:',num2str(ep),' Num Stpes:',num2str(steps_validation(ep)),' Total Reward:',num2str(Net_reward_validation(ep)),' RMSE: ',num2str(rmse(ep))];
disp(show);
end
   