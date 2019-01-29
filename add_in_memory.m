function Replay_mem_updated=add_in_memory(Rp_memory,state_current,action,reward,terminal,state_new)
newelement={state_current,action,reward,terminal,state_new};
Rp_memory = Rp_memory(2:end,:);
Rp_memory((size(Rp_memory,1)+1),:) = newelement;
Replay_mem_updated=Rp_memory;
end
