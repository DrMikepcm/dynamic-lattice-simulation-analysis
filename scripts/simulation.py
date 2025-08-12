import random
import os
import time
import math

# --- Simulation Settings ---
GRID_SIZE = 50
NUM_PARTICLES = 500
NUM_STEPS = 500
PARTICLE_SYMBOL = '🟤'
# A list to store the masses of each particle, indexed by particle number
particle_masses = []

# --- Simulation Variables ---
grid = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
particle_positions = []
density_data = []

# --- Simulation Variable ---
# The base strength of the repulsion force. Higher values mean more repulsion.
BASE_REPULSION_STRENGTH = 0.5

# --- Simulation Variable ---
# The factor by which the lattice curvature decays each step (e.g., 0.95 means a 5% decay per step)
CURVATURE_DECAY = 0.95

# --- Simulation Variable ---
# The strength of random, thermal movement. This will be scaled by local density.
BASE_THERMAL_ENERGY = 0.2

# --- Simulation Variable ---
# The probability (0 to 1) that a particle will decay each step.
PARTICLE_DECAY_RATE = 0.01

# --- New Simulation Variable ---
# The number of new particles to spawn each step.
PARTICLE_SPAWN_RATE = 2

def initialize_grid():
    """Initializes the grid with a single, central curvature well."""
    global grid
    grid = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    # Place a deep curvature well in the center
    center_x, center_y = GRID_SIZE // 2, GRID_SIZE // 2
    grid[center_x][center_y] = 20.0

def initialize_particles():
    """Initializes particles at random positions with varying masses."""
    global particle_positions, particle_masses
    particle_positions = []
    particle_masses = []
    for _ in range(NUM_PARTICLES):
        particle_positions.append([
            random.randint(0, GRID_SIZE - 1),
            random.randint(0, GRID_SIZE - 1)
        ])
        # Assign a random mass (e.g., 1, 2, or 3) to each particle
        particle_masses.append(random.choice([1, 2, 3]))

def display_grid():
    """Clears the console and prints the grid with particles."""
    os.system('cls' if os.name == 'nt' else 'clear')
    grid_with_particles = [row[:] for row in grid]
    
    # Place particles on the grid
    for x, y in particle_positions:
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            grid_with_particles[x][y] = PARTICLE_SYMBOL
            
    print("Simulation Model: Dynamic Lattice with Multiple Masses, Repulsion, Decay, and Dynamic Thermal Energy")
    print("-" * GRID_SIZE * 2)
    for row in grid_with_particles:
        line = ""
        for cell in row:
            if cell == PARTICLE_SYMBOL:
                line += f"{PARTICLE_SYMBOL} "
            elif isinstance(cell, (int, float)) and cell > 10:
                line += "🔴 "  # High curvature
            elif isinstance(cell, (int, float)) and cell > 0:
                line += "🟠 "  # Low curvature
            else:
                line += "⚪ "
        print(line)
    print("-" * GRID_SIZE * 2)

def get_local_density(x, y):
    """Calculates the density of particles in the immediate vicinity of a given point."""
    count = 0
    for i in range(len(particle_positions)):
        px, py = particle_positions[i]
        distance = math.sqrt((x - px)**2 + (y - py)**2)
        if distance < 3:  # Check a small radius around the point
            count += 1
    return count / (math.pi * 3**2)  # Return normalized density

def move_particles():
    """Moves all particles based on a combination of curvature attraction, particle repulsion, and thermal energy."""
    global particle_positions
    new_particle_positions = []
    for i in range(len(particle_positions)):
        x, y = particle_positions[i]
        
        # Calculate net force for each neighbor
        best_move_score = -float('inf')
        best_move = [x, y]
        
        # Check all 8 neighboring squares
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                    # Calculate attraction from curvature (gravity)
                    attraction = grid[new_x][new_y]
                    
                    # Optimized repulsion calculation based on local density
                    repulsion = BASE_REPULSION_STRENGTH * get_local_density(new_x, new_y)
                                
                    # Calculate dynamic thermal energy based on local density
                    local_density_at_current_pos = get_local_density(x, y)
                    thermal_energy = BASE_THERMAL_ENERGY * local_density_at_current_pos
                    
                    # The move score is attraction minus repulsion, plus a random thermal component
                    move_score = attraction - repulsion + random.uniform(-thermal_energy, thermal_energy)
                    
                    if move_score > best_move_score:
                        best_move_score = move_score
                        best_move = [new_x, new_y]
        
        new_particle_positions.append(best_move)
    particle_positions = new_particle_positions

def update_dynamic_lattice():
    """Updates the curvature based on particle density and mass."""
    global grid
    # Each particle adds a small amount of curvature proportional to its mass
    for i, (x, y) in enumerate(particle_positions):
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            grid[x][y] += 0.05 * particle_masses[i]

def decay_curvature():
    """Gradually reduces the curvature of the entire grid."""
    global grid
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            grid[x][y] *= CURVATURE_DECAY

def decay_and_spawn_particles():
    """Decays some particles and replaces them with new ones."""
    global particle_positions, particle_masses
    
    # Identify particles to remove
    to_remove = []
    for i in range(len(particle_positions)):
        if random.random() < PARTICLE_DECAY_RATE:
            to_remove.append(i)
    
    # Remove decayed particles
    new_particle_positions = []
    new_particle_masses = []
    for i in range(len(particle_positions)):
        if i not in to_remove:
            new_particle_positions.append(particle_positions[i])
            new_particle_masses.append(particle_masses[i])
    
    # Add new particles to maintain the total count
    num_to_spawn = len(to_remove)
    for _ in range(num_to_spawn):
        new_particle_positions.append([
            random.randint(0, GRID_SIZE - 1),
            random.randint(0, GRID_SIZE - 1)
        ])
        new_particle_masses.append(random.choice([1, 2, 3]))
        
    # --- New Feature: Constant Spawning ---
    # Add new particles to the simulation at a constant rate
    for _ in range(PARTICLE_SPAWN_RATE):
        new_particle_positions.append([
            random.randint(0, GRID_SIZE - 1),
            random.randint(0, GRID_SIZE - 1)
        ])
        new_particle_masses.append(random.choice([1, 2, 3]))
        
    particle_positions = new_particle_positions
    particle_masses = new_particle_masses
            
def calculate_density():
    """Calculates the density of particles in concentric rings from the center."""
    center_x, center_y = GRID_SIZE // 2, GRID_SIZE // 2
    density_profile = {}
    for x, y in particle_positions:
        distance = int(((x - center_x)**2 + (y - center_y)**2)**0.5)
        if distance not in density_profile:
            density_profile[distance] = 0
        density_profile[distance] += 1
    
    normalized_profile = {}
    for distance, count in density_profile.items():
        area = (distance + 1)**2 - distance**2
        if area > 0:
            normalized_profile[distance] = count / area
    
    return normalized_profile

if __name__ == "__main__":
    
    print("--- Running Dynamic Lattice Simulation with Multiple Masses and Repulsion (with Decay and Dynamic Thermal Energy) ---")
    
    initialize_grid()
    initialize_particles()
    
    for step in range(NUM_STEPS):
        move_particles()
        update_dynamic_lattice()
        decay_curvature()
        decay_and_spawn_particles()
        
        if step % 50 == 0:
            current_density = calculate_density()
            print(f"\nStep {step} Density Profile:")
            for dist, density in sorted(current_density.items()):
                print(f"Distance: {dist}, Density: {density:.4f}")
            print("-" * 30)

    final_density_profile = calculate_density()
    print("\n--- Final Density Profile ---")
    for dist, density in sorted(final_density_profile.items()):
        print(f"Distance: {dist}, Density: {density:.4f}")
    
    print("\nSimulation complete. Analyze the output to see the formation of a core.")
