import ao_core as ao

# For visualization
import matplotlib.pyplot as plt
import networkx as nx



def visualize_nn():


    # Retrieve current state and story      
    print(arch.datamatrix)
            # 5 rows, as follows:
            #0 Type
            #1 Input Connections
            #2 Neighbor Connections
            #3 C Connections
            #4 Dominant Connection
            #    ** note; the dominant c



    current_story = Agent.story[Agent.state - 1]

    # Extract neuron values from the story
    num_i = sum(arch_i)
    num_q = sum(arch_i)
    num_z = sum(arch_z)

    I_values = current_story[0:num_i ]
    print("I values:", I_values)
    Q_values = current_story[num_i:2 * num_i]
    print("Q values:", Q_values)
    Z_values = current_story[2 * num_i:2 * num_i + num_z]
    print("Z values:", Z_values)

    # Visualization using NetworkX with I, Q, and Z layers
    G = nx.DiGraph()
        
    # Add input nodes with value labels
    for i, node in enumerate(arch.datamatrix[0][arch.I__flat]):
        unique_node = f"{node}_{i}"
        print("node: ", unique_node)
        G.add_node(unique_node)
    # # Add hidden Q nodes with value labels
    for i, node in enumerate(arch.datamatrix[0][arch.Q__flat]):
        unique_node = f"{node}_{i}"
        print("node: ", unique_node)
        G.add_node(unique_node)
    for i, node in enumerate(arch.datamatrix[0][arch.Z__flat]):
        unique_node = f"{node}_{i}"
        print("node: ", unique_node)
        G.add_node(unique_node)



    for idx, (node, q_connections) in enumerate(zip(arch.datamatrix[0][arch.Q__flat], arch.datamatrix[1][arch.Q__flat])):
        if q_connections !=0:
            for i, connection in enumerate(q_connections):
                print(f"{node}_{idx}, {arch.datamatrix[0][connection]}_{connection}")
                G.add_edge(f"{node}_{idx}" , f"{arch.datamatrix[0][connection]}_{connection}")

    for idx, (node, z_connections) in enumerate(zip(arch.datamatrix[0][arch.Z__flat], arch.datamatrix[1][arch.Z__flat])):
        if z_connections !=0:
            for i, connection in enumerate(z_connections):
                print(f"{node}_{idx}, {arch.datamatrix[0][connection]}_{connection}")
                G.add_edge(f"{node}_{idx}" , f"{arch.datamatrix[0][connection]}_{i}")

    
    # Set positions: three columns for I (x=0), Q (x=1), and Z (x=2)
    pos = {}
    # Distribute nodes vertically (using negative y for downward arrangement)
    for i, node in enumerate(arch.datamatrix[0][arch.I__flat]):
        unique_node = f"{node}_{i}"
        pos[unique_node] = (0, -i)
    for i, node in enumerate(arch.datamatrix[0][arch.Q__flat]):
        unique_node = f"{node}_{i}"
        pos[unique_node] = (1, -i)
    for i, node in enumerate(arch.datamatrix[0][arch.Z__flat]):
        unique_node = f"{node}_{i}"
        pos[unique_node] = (2, -i)
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos=pos)
    plt.title("Neural Network Architecture with I, Q, and Z Layers")
    plt.show()

# Call the visualization function
arch_i=[1,1,1]
arch_z=[1]
arch  = ao.Arch(arch_i=[1,1,1], arch_z=[1], connector_function="forward_full_conn")
Agent = ao.Agent(arch)

# training and inference from the agent
Agent.next_state(INPUT=[0,0,0], LABEL=[1], INSTINCTS=False)  
Agent.next_state(INPUT=[1,1,1], LABEL=[0], INSTINCTS=False)

Agent.next_state(INPUT=[1,1,1])

visualize_nn()
