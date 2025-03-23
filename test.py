import ao_core as ao

# building an agent
arch  = ao.Arch(arch_i=[1,1,1], arch_z=[1], connector_function="full_conn")
agent = ao.Agent(arch, notes="my 1st AO agent", save_meta=True)

# training and inference from the agent
agent.next_state(INPUT=[0,0,0], LABEL=[1])  
agent.next_state(INPUT=[1,1,1])
    # to trigger learning, include a LABEL
    # for automatic (self-triggered) learning, define some INSTINCTS

# querying the agent's state history (goodbye, blackbox!)
state = agent.state
history = agent.story[0:state, :]
metahistory = agent.metastory[0:state, :]