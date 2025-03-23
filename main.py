import ao_core as ao

# For visualization
import matplotlib.pyplot as plt
import networkx as nx



def visualize_nn():


    # Retrieve current state and story      
    print(arch.datamatrix)
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
    
    # Create labels for nodes including their values
    input_layer_nodes = [f"I{i}" for i in range(num_i)]
    q_layer_nodes = [f"Q{i}" for i in range(num_i)]  # Hidden layer (size same as input)
    output_layer_nodes = [f"Z{i}" for i in range(num_z)]
    
    # Add input nodes with value labels
    for i, node in enumerate(input_layer_nodes):
        G.add_node(node, label=f"I{i}: {I_values[i]}")
    # Add hidden Q nodes with value labels
    for i, node in enumerate(q_layer_nodes):
        G.add_node(node, label=f"Q{i}: {Q_values[i]}")
    # Add output nodes with value labels
    for i, node in enumerate(output_layer_nodes):
        G.add_node(node, label=f"Z{i}: {Z_values[i]}")
    
    # Fully connect Input layer to Q layer
    for i_node in input_layer_nodes:
        for q_node in q_layer_nodes:
            G.add_edge(i_node, q_node)
    # Fully connect Q layer to Output layer
    for q_node in q_layer_nodes:
        for o_node in output_layer_nodes:
            G.add_edge(q_node, o_node)
    
    # Set positions: three columns for I (x=0), Q (x=1), and Z (x=2)
    pos = {}
    # Distribute nodes vertically (using negative y for downward arrangement)
    for i, node in enumerate(input_layer_nodes):
        pos[node] = (0, -i)
    for i, node in enumerate(q_layer_nodes):
        pos[node] = (1, -i)
    for i, node in enumerate(output_layer_nodes):
        pos[node] = (2, -i)
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True,
            labels=nx.get_node_attributes(G, 'label'),
            node_size=2000, node_color="lightblue", edge_color="gray",
            arrows=True)
    plt.title("Neural Network Architecture with I, Q, and Z Layers")
    plt.show()

# Call the visualization function
arch_i=[1,1,1]
arch_z=[1]
arch  = ao.Arch(arch_i=[1,1,1], arch_z=[1], connector_function="full_conn")
Agent = ao.Agent(arch)

# training and inference from the agent
Agent.next_state(INPUT=[0,0,0], LABEL=[1], INSTINCTS=False)  
Agent.next_state(INPUT=[1,1,1], LABEL=[0], INSTINCTS=False)

Agent.next_state(INPUT=[1,1,1])

visualize_nn()
