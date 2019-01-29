function Replay_mem_updated=add_in_memory_priority(Rp_memory,state_current,action,reward,terminal,state_new)
newelement={state_current,action,reward,terminal,state_new};
priority=20;%prioritizing the terminal states
priority_memory=Rp_memory(1:priority,:);    
Rp_memory_without_prio = Rp_memory((priority+1):end,:);
    if terminal==1
        priority_memory = priority_memory(2:end,:); %copy 2nd to last element, thus Fst n oldest element gets deleted
        priority_memory((size(priority_memory,1)+1),:) = newelement; %add the newest element as last element
    else
        Rp_memory_without_prio = Rp_memory_without_prio(2:end,:); %copy 2nd to last element, thus Fst n oldest element gets deleted
        Rp_memory_without_prio((size(Rp_memory_without_prio,1)+1),:) = newelement;%add the newest element as last element
    end
        
Rp_memory(1:priority,:)= priority_memory; 
Rp_memory((priority+1):end,:) = Rp_memory_without_prio; 

Replay_mem_updated=Rp_memory;
end
