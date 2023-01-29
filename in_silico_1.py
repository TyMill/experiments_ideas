import numpy as np

# Define the simulation environment
class SimpleModel:
    def __init__(self, initial_state):
        self.state = initial_state
    
    def update(self, input_data):
        self.state += input_data
        
    def get_state(self):
        return self.state

# Generate the input data for the simulation
input_data = np.random.randn(100)

# Run the simulation
model = SimpleModel(0)
for i in range(100):
    model.update(input_data[i])

# Analyze the results of the simulation
final_state = model.get_state()
print("Final state:", final_state)
