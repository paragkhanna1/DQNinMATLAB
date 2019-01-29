function [X_out]=stateUpdate_DQN(X_in,action)
%parameter
wheel_rad=1;
wheel_dis=1;%distance between robot centre and wheel
if action==3
    wheel_dot(1)=3;
    wheel_dot(2)=0;
elseif action==4
    wheel_dot(1)=0;
    wheel_dot(2)=3;
elseif action==2
    wheel_dot(1)=3;
    wheel_dot(2)=3;
elseif action==1
    wheel_dot(1)=-3;
    wheel_dot(2)=-3;
elseif action==5
    wheel_dot(1)=2;
    wheel_dot(2)=2;
elseif action==6
    wheel_dot(1)=-1;
    wheel_dot(2)=2;
elseif action==7
    wheel_dot(1)=3;
    wheel_dot(2)=-3;
elseif action==8
    wheel_dot(1)=2;
    wheel_dot(2)=-1;
end
R_theta_inv=[cos(X_in(3)) -sin(X_in(3)) 0
                 sin(X_in(3)) cos(X_in(3))  0
                    0 0 1];

    twist_robot_dot=0.5*(wheel_rad/wheel_dis)*[wheel_dot(1)+wheel_dot(2);
                                                 0;
                                               wheel_dot(1)-wheel_dot(2)];
    twist_world_dot=R_theta_inv*twist_robot_dot;
    X_out(1)=X_in(1)+twist_world_dot(1)*1;%timestep of 1 unit
    X_out(2)=X_in(2)+twist_world_dot(2)*1;
    X_out(3)=X_in(3)+twist_world_dot(3)*1;
    if X_out(1)<1 || X_out(1)>49
       X_out=X_in;
       % disp('X position going Out of bounds, cannot move')
    end
    if X_out(2)<1 || X_out(2)>49
       X_out=X_in;
        %disp('Y position going Out of bounds cannot move')
    end
    
        