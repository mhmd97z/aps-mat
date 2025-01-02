import random
import uuid
import hashlib
import networkx as nx

class UeApNetwork:
    def __init__(self):
        self.graph = nx.Graph()
        self.ap_count = 0

    def generate_unique_id(self, prefix):
        """Generates a unique random ID with the given prefix."""
        new_id = f"{prefix}_{hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:5]}"
        return new_id

    def add_ap(self):
        """Adds a new AP (Access Point) with a unique ID."""
        ap_id = f"AP_{self.ap_count}"
        self.graph.add_node(ap_id, type="AP")
        self.ap_count += 1
        return ap_id

    def add_ue(self):
        """Adds a new UE (User Equipment) with a unique ID and connects it to all APs."""
        ue_id = self.generate_unique_id("UE")
        self.graph.add_node(ue_id, type="UE")

        # Connect UE to all APs and assign random channel gains
        for node in self.graph.nodes:
            if self.graph.nodes[node]['type'] == 'AP':
                self.graph.add_edge(ue_entity, node, channel_gain=None, connected=None, measurment=None)

        return ue_id

    def update_channel_gain(self, ue_id):
        """Updates the channel gains of a given UE to all connected APs."""
        if ue_id in self.graph and self.graph.nodes[ue_id]['type'] == 'UE':
            for neighbor in self.graph.neighbors(ue_id):
                if self.graph.nodes[neighbor]['type'] == 'AP':
                    new_gain = random.uniform(0, 1)  # Simulating a new channel gain
                    self.graph[ue_id][neighbor]['channel_gain'] = new_gain
                    print(f"Updated channel gain between {ue_id} and {neighbor}: {new_gain:.4f}")
        else:
            print(f"UE {ue_id} not found in the network or is not a valid UE.")

    def get_channel_gains(self, ue_id):
        """Returns the channel gains of a given UE to all APs."""
        gains = {}
        if ue_id in self.graph and self.graph.nodes[ue_id]['type'] == 'UE':
            for neighbor in self.graph.neighbors(ue_id):
                if self.graph.nodes[neighbor]['type'] == 'AP':
                    gains[neighbor] = self.graph[ue_id][neighbor]['channel_gain']
        return gains

if __name__ == "__main__":
    # Example usage
    network = UeApNetwork()

    # Add APs
    ap1 = network.add_ap()
    ap2 = network.add_ap()

    # Add UEs
    ue1 = network.add_ue()
    ue2 = network.add_ue()

    # Update channel gains for a specific UE
    network.update_channel_gain(ue1)

    # Remove a UE
    network.remove_ue(ue1)

    # Get channel gains of a UE
    gains = network.get_channel_gains(ue2)
    print("Channel gains for UE2:", gains)
