%Initialize the environment:
clear all
close all
X=[20;25;pi/4];
%figure('Position',[0 0 631 600]) %size to get 489x489 image saved from fig
figure('Position',[0 0 316 300]) %size to get 245x245 image saved from fig
F = getframe;    
[I,Map] = frame2im(F);
while(1) %forcing to get a image of 489x489
close all;
figure('Position',[0 0 316 300]);

F = getframe;    
[I,Map] = frame2im(F);
    if(size(I,1)==245 && size(I,2)==245)
        break;
    end
end
%figure('Position',[0 0 526 500])
%figure('Position',[0 0 316 300])
rectangle('Position',[0,0,50,50]);
rectangle('Position',[25,25,2,2],'EdgeColor','k');

rectangle('Position',[10,40,2,2],'Curvature',[1 1]);
rectangle('Position',[20,30,2,2],'Curvature',[1 1]);
rectangle('Position',[20,5,2,2],'Curvature',[1 1]);
rectangle('Position',[40,30,2,2],'Curvature',[1 1]);
%plot(X(1),X(2));

hold on

p = plot(X(1),X(2),'k*');

%p=plot(X(1),X(2),'r*');
hold off
%%
% Perform random actions to generate Get Data for Training the CNN first
% time, This is analogous to Initializing the CNN with random weights.
for i=1:10
    action=choose_random_action_old();
       % X=[10;10;pi/4]; %can be randomized later
                X=[randi([4 46]);randi([4 46]);randi([1 6])*rand()]; %randomized state

    X=stateUpdate_DQN(X,action);%timestep of 1 unit
    reward(:,:,:,i)=get_reward_old(X,action);%reward must be a(1x1x4) output here, for 4 actions, (as required by CNN-MATLAB)
    %rewards act as random targets!
    set(p,'XData',X(1));
    set(p,'YData',X(2));
         %p.Xdata = X(1); %MAY HAVE TO USE THESE ones IN SOME MATLAB VERSION               
         %p.Ydata = X(2);
    drawnow;
%    rectangle('Position',[X(1)-1,X(2)-1,2,2]);
F = getframe;    
[I,Map] = frame2im(F);
states(:,:,:,i)=rgb2gray(I);
   % pause(0.1);
end
for i=1:50 %getting more DATA
    action=choose_random_action_old();
        X=[randi([4 46]);randi([4 46]);randi([1 6])*rand()]; %randomized state
    X=stateUpdate_DQN(X,action);%timestep of 1 unit
    reward2(:,:,:,i)=get_reward_old(X,action);%reward must be a(1x1x4) output here, for 4 actions, (as required by CNN-MATLAB)
    %rewards act as random targets!
    set(p,'XData',X(1));
    set(p,'YData',X(2));
    drawnow;
F = getframe;    
[I,Map] = frame2im(F);
states2(:,:,:,i)=rgb2gray(I);
   % pause(0.1);
end

%%
%Define the convolutional neural network architecture.

layers = [
    %imageInputLayer([489 489 1],'Name', 'input')
    imageInputLayer([245 245 1],'Name', 'input')
    convolution2dLayer(8,8,'Padding',1,'Name', 'conv1')
    batchNormalizationLayer('Name', 'Batch_N')
    reluLayer('Name', 'relu1')
    
    %maxPooling2dLayer(2,'Stride',2,'Name', 'maxPool1')
    
    convolution2dLayer(4,4,'Padding',1,'Name', 'conv2')
    batchNormalizationLayer('Name', 'Batch_N2')
    reluLayer('Name', 'relu2')
    
    %maxPooling2dLayer(2,'Stride',2,'Name', 'maxPool2')
    
    convolution2dLayer(4,4,'Padding',1,'Name', 'conv3')
    batchNormalizationLayer('Name', 'Batch_N3')
    reluLayer('Name', 'relu3')
    
    %fullyConnectedLayer(512,'Name', 'FullyC')% size is eual to number of actions
  
        fullyConnectedLayer(4,'Name', 'FullyC2')% size is eual to number of actions
  regressionLayer('Name','Output')];

options = trainingOptions('sgdm', ...
    'MaxEpochs',1, ...
    'InitialLearnRate',0.001, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(states,reward,layers,options);
layersTransfer = net.Layers(1:end);
net = trainNetwork(states,reward,layersTransfer,options);

%used previously generated random States,Rewarrd to train,
%thus the network is initialized and ready for DQN
%% TRAINING
options = trainingOptions('sgdm', ...
'MaxEpochs',1, ...
'InitialLearnRate',0.01, ...
'Verbose',false, ...
'Plots','none'); %to supress plot occuring at each step when CNN is trained

Max_episodes=20;
Max_steps=100;
gamma_learning=0.002; % for the target equation :target_st=reward_st+gamma*maxQ_new; 
epsilon=0.8; %e-greedy algorithm factor t choose action
for ep=1:Max_episodes
    %set intitial position 
    %X=[10;10;randi([1 6])*rand()]; %can be randomized later
    if(ep<10)
        X=[randi([4 46]);randi([4 46]);randi([1 6])*rand()];
    
    else
        X=[randi([4 46]);randi([4 46]);randi([1 6])*rand()];
    end    
    set(p,'XData',X(1));
    set(p,'YData',X(2));
         
    drawnow;
    Net_reward(ep)=0;
    for st=1:Max_steps
        %acquire current state image,
        F = getframe;    
        [I,Map] = frame2im(F);
        state(:,:,:,1)=rgb2gray(I); 
%can use below created state_new variable instead of getting frame again,
%will se later!!!!
       
        %do a full forward pass through CNN.
        Q_st=predict(net,state(:,:,:,1));% acquire Q values for all acitons
        Q_st(isnan(Q_st)) = 10; %Removeing NAN values

        %state(:,:,:,2)=zeros(245,245);
        %Q_st_extra=predict(net,state(:,:,:,2));%Targets forthe extra layer, 
        %adding extra layer of zeros to get 255*255*1*4 dim'n forCNN-MATLAB
         %WHY ZERSOS/ONES!!,, THIS CAN CAUSE WHOLE LOT OF TROUBLE IN WEIGHTS
        %BETTER DUPLICATE ABOVE LAYER ITSELF!
        state(:,:,:,2)=rgb2gray(I);
        Q_st_extra=predict(net,state(:,:,:,2));
        Q_st_extra(isnan(Q_st_extra)) = 10;

        % choose action to do
        [~,act_st]=max(Q_st);%index of max Q value gives the desired action to do!
        
        if rand()<(epsilon/(ep-epsilon*(ep))) %e-greedy algo, reducing epsilon
            act_st = randi([1 4],1);
        end
        X=stateUpdate_DQN(X,act_st);%timestep of 1 unit
        reward_st=get_reward_simple(X);%reward must be a single output here
        
        %get new state
        set(p,'XData',X(1));
        set(p,'YData',X(2));
         %p.Xdata = X(1); %MAY HAVE TO USE THESE ones IN SOME MATLAB VERSION               
         %p.Ydata = X(2);
        drawnow;
        F = getframe;    
        [I,Map] = frame2im(F);
        state_new(:,:,:,1)=rgb2gray(I); 
        
        %do a full forward pass through CNN.
        Q_st_new=predict(net,state_new);% acquire Q values for all acitons
        Q_st_new(isnan(Q_st_new)) = 10;

        [maxQ_new,~]=max(Q_st_new);%is the max possible Q_value for next state
        
        %set Targetx for CNN
        %for all actions, default target is the Q value predicted, so loss
        %is zero(computed inside the CNN), hence no weights updated
        target(:,:,1:4,1:2)=0; %extra 2nd layer embedded for getting dim'n as 1*1*4*2
        target(:,:,1,1)= Q_st(1);target(:,:,2,1)= Q_st(2);
        target(:,:,3,1)= Q_st(3);target(:,:,4,1)= Q_st(4);
        target(:,:,1,2)= Q_st_extra(1);target(:,:,2,2)= Q_st_extra(2);
        target(:,:,3,2)= Q_st_extra(3);target(:,:,4,2)= Q_st_extra(4);
       
        %target for taken action:
        if reward_st==1000||reward_st==-1000 %TERMiNAL STATE REACHED, To end episode
            target_st=reward_st;
        else %non terminal state, Target is given by the update term from BELLMAN equation
            target_st=reward_st+gamma_learning*maxQ_new; %gamma set as parameter before
        end
        %At this state, the target value for the action performed is got by
        %above
        target(:,:,act_st,1)=target_st;
        target(:,:,act_st,2)=target_st;
        %TRAIN the Netwrok with above (state,target) pair
        layersTransfer = net.Layers(1:end); %transferring all previous layers(with weights)
        
        net = trainNetwork(state,target,layersTransfer,options);
        
        % feed out number of steps
        steps(ep)=st;
        Net_reward(ep)=Net_reward(ep)+reward_st;

        if(reward_st==1000 || reward_st==-1000)%terminal reached
            break;
        end
%        disp(st);
    end
show=['Episode:',num2str(ep),' Num Stpes:',num2str(steps(ep)),' Total Reward:',num2str(Net_reward(ep))];
disp(show);
end

%%
%evaluATION
X=[10;10;pi/4];
%figure('Position',[0 0 631 600]) %size to get 489x489 image saved from fig
figure('Position',[0 0 316 300]) %size to get 245x245 image saved from fig
F = getframe;    
[I,Map] = frame2im(F);
while(1) %forcing to get a image of 489x489
close all;
figure('Position',[0 0 316 300]);

F = getframe;    
[I,Map] = frame2im(F);
    if(size(I,1)==245 && size(I,2)==245)
        break;
    end
end
%figure('Position',[0 0 526 500])
%figure('Position',[0 0 316 300])
rectangle('Position',[0,0,50,50]);
rectangle('Position',[25,25,2,2],'EdgeColor','k');

rectangle('Position',[10,40,2,2],'Curvature',[1 1]);
rectangle('Position',[20,30,2,2],'Curvature',[1 1]);
rectangle('Position',[20,5,2,2],'Curvature',[1 1]);
rectangle('Position',[40,30,2,2],'Curvature',[1 1]);
%plot(X(1),X(2));

hold on

p = plot(X(1),X(2),'k*');

%p=plot(X(1),X(2),'r*');
hold off

Max_episodes=10;
Max_steps=100;
gamma_learning=0.002; % for the target equation :target_st=reward_st+gamma*maxQ_new; 
epsilon=0.4; %e-greedy algorithm factor t choose action
for ep=1:Max_episodes
    predictionError=0;
    %set intitial position 
            X=[randi([4 46]);randi([4 46]);randi([1 6])*rand()]; %randomized state

%    X=[45;10;pi/4]; %can be randomized later
    set(p,'XData',X(1));
    set(p,'YData',X(2));
         
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

        X=stateUpdate_DQN(X,act_st);%timestep of 1 unit
        reward_st=get_reward_simple(X);%reward must be a single output here
        
        %get new state
        set(p,'XData',X(1));
        set(p,'YData',X(2));
         %p.Xdata = X(1); %MAY HAVE TO USE THESE ones IN SOME MATLAB VERSION               
         %p.Ydata = X(2);
        drawnow;
        F = getframe;    
        [I,Map] = frame2im(F);
        state_new(:,:,:,1)=rgb2gray(I); 
        
        %do a full forward pass through CNN.
        Q_st_new=predict(net,state_new);% acquire Q values for all acitons
        [maxQ_new,~]=max(Q_st_new);%is the max possible Q_value for next state
        
       
        %target for taken action:
        if reward_st==1000||reward_st==-1000 %TERMiNAL STATE REACHED, To end episode
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

        if(reward_st==1000 || reward_st==-1000)%terminal reached
            break;
        end
%        disp(st);

    end
    rmse(ep)=predictionError*predictionError;
show=['Validation: Episode:',num2str(ep),' Num Stpes:',num2str(steps_validation(ep)),' Total Reward:',num2str(Net_reward_validation(ep)),' RMSE: ',num2str(rmse(ep))];
disp(show);
end
   