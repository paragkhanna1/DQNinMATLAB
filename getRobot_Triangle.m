function [triangle]=getRobot_Triangle(X_in)
L=3;%Distance from end of robot to castor wheel ahead
b=2;%distance between 2 actuated wheels(back wheels,
%above is 2*dis btw wheel and centre of robot, defined in stateUpdate
oTm=[cos(X_in(3)) -sin(X_in(3)) X_in(1);
     sin(X_in(3)) cos(X_in(3))  X_in(2);
        0       0           1];
V1=oTm*[ 1 0 2*L/3;
         0 1 0;
         0 0 1];
V2=oTm*[ 1 0 -L/3;
         0 1 b/2;
         0 0 1];
V3=oTm*[ 1 0 -L/3;
         0 1 -b/2;
         0 0 1];
triangle=[V1(1,3) V2(1,3) V3(1,3) V1(1,3);V1(2,3) V2(2,3) V3(2,3) V1(2,3)];