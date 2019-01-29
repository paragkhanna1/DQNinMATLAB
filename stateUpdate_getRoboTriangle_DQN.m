function [X_out,triangle]=stateUpdate_getRoboTriangle_DQN(X_in,action)
%parameter
wheel_rad=0.5;
wheel_dis=1;%distance between robot centre and wheel
    L=3;%Distance from end of robot to castor wheel ahead
e=2*wheel_dis;%distance between 2 actuated wheels(back wheels ,

if action==1
    wheel_dot(1)=3;
    wheel_dot(2)=0;
elseif action==2
    wheel_dot(1)=0;
    wheel_dot(2)=3;
elseif action==3
    wheel_dot(1)=3;
    wheel_dot(2)=3;
elseif action==4
    wheel_dot(1)=-3;
    wheel_dot(2)=-3;
elseif action==5
    wheel_dot(1)=2;
    wheel_dot(2)=2;
elseif action==6
    wheel_dot(1)=-1;
    wheel_dot(2)=3;
elseif action==7
    wheel_dot(1)=3;
    wheel_dot(2)=-3;
elseif action==8
    wheel_dot(1)=3;
    wheel_dot(2)=-1;
elseif action==0 %%NOT INVCLUDED IN ACTION SPACE, JUST USED INITIALLY!
    wheel_dot(1)=0;
    wheel_dot(2)=0;
end
R_theta_inv=[cos(X_in(3)) -sin(X_in(3)) 0
                 sin(X_in(3)) cos(X_in(3))  0
                    0 0 1];

    twist_robot_dot=0.5*(wheel_rad/wheel_dis)*[wheel_dot(1)+wheel_dot(2);
                                                 0;
                                               wheel_dot(1)-wheel_dot(2)];
    twist_world_dot=R_theta_inv*twist_robot_dot;
    X_out(1)=X_in(1)+twist_world_dot(1)*1;%timestep of 1 unit
    X_out(2)=X_in(2)+twist_world_dot(2)*1;%timestep of 1 unit
    X_out(3)=X_in(3)+twist_world_dot(3)*1;
    if X_out(1)<1 || X_out(1)>49
       X_out=X_in;
       % disp('X position going Out of bounds, cannot move')
    end
    if X_out(2)<1 || X_out(2)>49
       X_out=X_in;
        %disp('Y position going Out of bounds cannot move')
    end
%above is 2*dis btw wheel and centre of robot, defined in stateUpdate
oTm=[cos(X_out(3)) -sin(X_out(3)) X_out(1);
     sin(X_out(3)) cos(X_out(3))  X_out(2);
        0       0           1];
V1=oTm*[ 1 0 2*L/3;
         0 1 0;
         0 0 1];
V2=oTm*[ 1 0 -L/3;
         0 1 e/2;
         0 0 1];
V3=oTm*[ 1 0 -L/3;
         0 1 -e/2;
         0 0 1];
triangle=[V1(1,3) V2(1,3) V3(1,3) V1(1,3);V1(2,3) V2(2,3) V3(2,3) V1(2,3)];

  if V1(1,3)<0.5 || V2(1,3)<0.5 || V3(1,3)<0.5 || V1(2,3)>49.5 || V2(2,3)>49.5 || V3(2,3)>49.5 
       X_out=X_in;
        oTm=[cos(X_out(3)) -sin(X_out(3)) X_out(1);
             sin(X_out(3)) cos(X_out(3))  X_out(2);
                    0       0           1];
        V1=oTm*[ 1 0 2*L/3;
                 0 1 0;
                 0 0 1];
        V2=oTm*[ 1 0 -L/3;
                 0 1 e/2;
                 0 0 1];
        V3=oTm*[ 1 0 -L/3;
                 0 1 -e/2;
                 0 0 1];
        triangle=[V1(1,3) V2(1,3) V3(1,3) V1(1,3);V1(2,3) V2(2,3) V3(2,3) V1(2,3)];

       % disp('position going Out of bounds, cannot move')
  end
     
        