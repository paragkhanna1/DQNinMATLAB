function [reward,terminal]=get_reward(state,action)
%GOAL AT: 26,26 %Goal Radius=2
%miNES AT: [11,41];[21,31];[21,6];[41,31] %Mine Radius=2

if(sqrt((state(1)-25)^2 +(state(2)-25)^2)<4)
    r=250; %GOAL
    terminal=1;
% elseif(sqrt((state(1)-11)^2 +(state(2)-41)^2)<2)
% r=-1000;%MINE
% elseif(sqrt((state(1)-41)^2 +(state(2)-31)^2)<2)
% r=-1000;%MINE
% elseif(sqrt((state(1)-21)^2 +(state(2)-6)^2)<2)
% r=-1000;%MINE
elseif(sqrt((state(1)-21)^2 +(state(2)-31)^2)<2)
    r=-250;%MINE
    terminal=1;
else
    r=-1; terminal=0;%any other position
end
reward(:,:,1)=-1;reward(:,:,2)=-1;reward(:,:,3)=-1;reward(:,:,4)=-1;
reward(:,:,5)=-1;reward(:,:,6)=-1;reward(:,:,7)=-1;reward(:,:,8)=-1;
reward(:,:,action)=r;

