import pandas as pd
import matplotlib.pyplot as plt
import re
import io

# Raw data from the conversation, combined into a single string
raw_data = """
--- Running Dynamic Lattice Simulation with Multiple Masses and Repulsion (with Decay and Dynamic Thermal Energy) ---

Step 0 Density Profile:
Distance: 0, Density: 2.0000
Distance: 1, Density: 0.3333
Distance: 2, Density: 0.2000
Distance: 3, Density: 1.0000
Distance: 4, Density: 0.4444
Distance: 5, Density: 0.9091
Distance: 6, Density: 0.4615
Distance: 7, Density: 0.7333
Distance: 8, Density: 0.6471
Distance: 9, Density: 0.5263
Distance: 10, Density: 0.6667
Distance: 11, Density: 0.6957
Distance: 12, Density: 0.4400
Distance: 13, Density: 0.3704
Distance: 14, Density: 0.6552
Distance: 15, Density: 1.0000
Distance: 16, Density: 0.5455
Distance: 17, Density: 0.5143
Distance: 18, Density: 0.5135
Distance: 19, Density: 1.0513
Distance: 20, Density: 0.5366
Distance: 21, Density: 0.6977
Distance: 22, Density: 0.4444
Distance: 23, Density: 0.4255
Distance: 24, Density: 0.6735
Distance: 25, Density: 0.5882
Distance: 26, Density: 0.3962
Distance: 27, Density: 0.2364
Distance: 28, Density: 0.2807
Distance: 29, Density: 0.2373
Distance: 30, Density: 0.1311
Distance: 31, Density: 0.0635
Distance: 32, Density: 0.0923
Distance: 33, Density: 0.0597
Distance: 34, Density: 0.0145
------------------------------

Step 50 Density Profile:
Distance: 1, Density: 0.3333
Distance: 2, Density: 0.4000
Distance: 3, Density: 1.1429
Distance: 4, Density: 0.3333
Distance: 5, Density: 1.7273
Distance: 6, Density: 0.3077
Distance: 7, Density: 0.5333
Distance: 8, Density: 1.0000
Distance: 9, Density: 0.3158
Distance: 10, Density: 0.7619
Distance: 11, Density: 1.0870
Distance: 12, Density: 0.5600
Distance: 13, Density: 0.4444
Distance: 14, Density: 1.1379
Distance: 15, Density: 0.9677
Distance: 16, Density: 0.3939
Distance: 17, Density: 0.6000
Distance: 18, Density: 0.8378
Distance: 19, Density: 1.2051
Distance: 20, Density: 0.7317
Distance: 21, Density: 0.9070
Distance: 22, Density: 0.7556
Distance: 23, Density: 0.5319
Distance: 24, Density: 0.7551
Distance: 25, Density: 0.8235
Distance: 26, Density: 0.3208
Distance: 27, Density: 0.1273
Distance: 28, Density: 0.3333
Distance: 29, Density: 0.3390
Distance: 30, Density: 0.1475
Distance: 31, Density: 0.1111
Distance: 32, Density: 0.0462
Distance: 33, Density: 0.0448
------------------------------

Step 100 Density Profile:
Distance: 0, Density: 2.0000
Distance: 1, Density: 1.0000
Distance: 2, Density: 0.4000
Distance: 3, Density: 0.7143
Distance: 4, Density: 0.3333
Distance: 5, Density: 1.6364
Distance: 6, Density: 0.5385
Distance: 7, Density: 0.7333
Distance: 8, Density: 1.4118
Distance: 9, Density: 0.5263
Distance: 10, Density: 0.6190
Distance: 11, Density: 1.0435
Distance: 12, Density: 0.4800
Distance: 13, Density: 0.4074
Distance: 14, Density: 1.5517
Distance: 15, Density: 0.9355
Distance: 16, Density: 0.3939
Distance: 17, Density: 0.7429
Distance: 18, Density: 0.9189
Distance: 19, Density: 1.4103
Distance: 20, Density: 0.9756
Distance: 21, Density: 1.2326
Distance: 22, Density: 0.8000
Distance: 23, Density: 0.5106
Distance: 24, Density: 0.8776
Distance: 25, Density: 0.9412
Distance: 26, Density: 0.5660
Distance: 27, Density: 0.1091
Distance: 28, Density: 0.3860
Distance: 29, Density: 0.2542
Distance: 30, Density: 0.2131
Distance: 31, Density: 0.1746
Distance: 32, Density: 0.1077
Distance: 33, Density: 0.0746
Distance: 34, Density: 0.0290
------------------------------

Step 150 Density Profile:
Distance: 0, Density: 1.0000
Distance: 1, Density: 1.0000
Distance: 2, Density: 1.0000
Distance: 3, Density: 1.2857
Distance: 4, Density: 0.3333
Distance: 5, Density: 2.4545
Distance: 6, Density: 0.2308
Distance: 7, Density: 0.8667
Distance: 8, Density: 1.2941
Distance: 9, Density: 0.8947
Distance: 10, Density: 0.8571
Distance: 11, Density: 1.0435
Distance: 12, Density: 0.8400
Distance: 13, Density: 0.5556
Distance: 14, Density: 1.5862
Distance: 15, Density: 1.3548
Distance: 16, Density: 0.3939
Distance: 17, Density: 1.0571
Distance: 18, Density: 1.2432
Distance: 19, Density: 1.3333
Distance: 20, Density: 1.0000
Distance: 21, Density: 1.0698
Distance: 22, Density: 0.8889
Distance: 23, Density: 0.6809
Distance: 24, Density: 1.0204
Distance: 25, Density: 0.8824
Distance: 26, Density: 0.8302
Distance: 27, Density: 0.0545
Distance: 28, Density: 0.3333
Distance: 29, Density: 0.3559
Distance: 30, Density: 0.2459
Distance: 31, Density: 0.1905
Distance: 32, Density: 0.0769
Distance: 33, Density: 0.1343
Distance: 34, Density: 0.0435
------------------------------

Step 200 Density Profile:
Distance: 0, Density: 2.0000
Distance: 1, Density: 1.3333
Distance: 2, Density: 1.4000
Distance: 3, Density: 1.7143
Distance: 4, Density: 0.4444
Distance: 5, Density: 3.0000
Distance: 6, Density: 0.3846
Distance: 7, Density: 0.9333
Distance: 8, Density: 1.9412
Distance: 9, Density: 1.2105
Distance: 10, Density: 1.2381
Distance: 11, Density: 1.1739
Distance: 12, Density: 0.9200
Distance: 13, Density: 0.5926
Distance: 14, Density: 1.8621
Distance: 15, Density: 1.4194
Distance: 16, Density: 0.3939
Distance: 17, Density: 1.0000
Distance: 18, Density: 1.2162
Distance: 19, Density: 1.4872
Distance: 20, Density: 1.0976
Distance: 21, Density: 1.3023
Distance: 22, Density: 0.8889
Distance: 23, Density: 0.8298
Distance: 24, Density: 1.1224
Distance: 25, Density: 1.0196
Distance: 26, Density: 0.6604
Distance: 27, Density: 0.1091
Distance: 28, Density: 0.4561
Distance: 29, Density: 0.4576
Distance: 30, Density: 0.2295
Distance: 31, Density: 0.1905
Distance: 32, Density: 0.0923
Distance: 33, Density: 0.1343
Distance: 34, Density: 0.0290
------------------------------

Step 250 Density Profile:
Distance: 0, Density: 2.0000
Distance: 1, Density: 1.3333
Distance: 2, Density: 2.0000
Distance: 3, Density: 1.8571
Distance: 4, Density: 0.4444
Distance: 5, Density: 3.0000
Distance: 6, Density: 0.5385
Distance: 7, Density: 0.6667
Distance: 8, Density: 1.8824
Distance: 9, Density: 1.1053
Distance: 10, Density: 1.3810
Distance: 11, Density: 1.1739
Distance: 12, Density: 1.2000
Distance: 13, Density: 0.6667
Distance: 14, Density: 2.0690
Distance: 15, Density: 1.5484
Distance: 16, Density: 0.7273
Distance: 17, Density: 1.0286
Distance: 18, Density: 1.1081
Distance: 19, Density: 1.6154
Distance: 20, Density: 1.1951
Distance: 21, Density: 1.6047
Distance: 22, Density: 0.9556
Distance: 23, Density: 0.9149
Distance: 24, Density: 1.3061
Distance: 25, Density: 1.1176
Distance: 26, Density: 0.8679
Distance: 27, Density: 0.1818
Distance: 28, Density: 0.5965
Distance: 29, Density: 0.5254
Distance: 30, Density: 0.2295
Distance: 31, Density: 0.1905
Distance: 32, Density: 0.0615
Distance: 33, Density: 0.1493
Distance: 34, Density: 0.0580
------------------------------

Step 300 Density Profile:
Distance: 0, Density: 3.0000
Distance: 1, Density: 1.6667
Distance: 2, Density: 2.0000
Distance: 3, Density: 2.2857
Distance: 4, Density: 0.6667
Distance: 5, Density: 3.0909
Distance: 6, Density: 0.3077
Distance: 7, Density: 1.1333
Distance: 8, Density: 1.2353
Distance: 9, Density: 1.4737
Distance: 10, Density: 1.5238
Distance: 11, Density: 1.4783
Distance: 12, Density: 1.1600
Distance: 13, Density: 0.8519
Distance: 14, Density: 1.8276
Distance: 15, Density: 1.8387
Distance: 16, Density: 0.8485
Distance: 17, Density: 1.1714
Distance: 18, Density: 1.3243
Distance: 19, Density: 1.5385
Distance: 20, Density: 1.1951
Distance: 21, Density: 1.6279
Distance: 22, Density: 1.0000
Distance: 23, Density: 1.1064
Distance: 24, Density: 1.6939
Distance: 25, Density: 1.4706
Distance: 26, Density: 0.8679
Distance: 27, Density: 0.3091
Distance: 28, Density: 0.7018
Distance: 29, Density: 0.4915
Distance: 30, Density: 0.2787
Distance: 31, Density: 0.2222
Distance: 32, Density: 0.0769
Distance: 33, Density: 0.0896
Distance: 34, Density: 0.0580
------------------------------

Step 350 Density Profile:
Distance: 0, Density: 2.0000
Distance: 1, Density: 1.3333
Distance: 2, Density: 1.6000
Distance: 3, Density: 1.8571
Distance: 4, Density: 0.7778
Distance: 5, Density: 3.0000
Distance: 6, Density: 0.4615
Distance: 7, Density: 1.0667
Distance: 8, Density: 1.4118
Distance: 9, Density: 1.4737
Distance: 10, Density: 1.1429
Distance: 11, Density: 1.8696
Distance: 12, Density: 1.4800
Distance: 13, Density: 0.8889
Distance: 14, Density: 2.0690
Distance: 15, Density: 2.0323
Distance: 16, Density: 0.9697
Distance: 17, Density: 1.3714
Distance: 18, Density: 1.4865
Distance: 19, Density: 1.7949
Distance: 20, Density: 1.5610
Distance: 21, Density: 1.8140
Distance: 22, Density: 1.0444
Distance: 23, Density: 1.2766
Distance: 24, Density: 1.6327
Distance: 25, Density: 1.5098
Distance: 26, Density: 1.0755
Distance: 27, Density: 0.3273
Distance: 28, Density: 0.5965
Distance: 29, Density: 0.4746
Distance: 30, Density: 0.3770
Distance: 31, Density: 0.2063
Distance: 32, Density: 0.1231
Distance: 33, Density: 0.1791
Distance: 34, Density: 0.0870
------------------------------

Step 400 Density Profile:
Distance: 0, Density: 2.0000
Distance: 1, Density: 1.3333
Distance: 2, Density: 1.0000
Distance: 3, Density: 1.8571
Distance: 4, Density: 0.7778
Distance: 5, Density: 2.6364
Distance: 6, Density: 0.1538
Distance: 7, Density: 1.4000
Distance: 8, Density: 1.7647
Distance: 9, Density: 1.1579
Distance: 10, Density: 1.3333
Distance: 11, Density: 2.2174
Distance: 12, Density: 1.5600
Distance: 13, Density: 0.8889
Distance: 14, Density: 2.4828
Distance: 15, Density: 2.1613
Distance: 16, Density: 1.0000
Distance: 17, Density: 1.4571
Distance: 18, Density: 1.6216
Distance: 19, Density: 2.2051
Distance: 20, Density: 1.5854
Distance: 21, Density: 1.7442
Distance: 22, Density: 1.2222
Distance: 23, Density: 1.2766
Distance: 24, Density: 1.9388
Distance: 25, Density: 1.5294
Distance: 26, Density: 1.2453
Distance: 27, Density: 0.4182
Distance: 28, Density: 0.7018
Distance: 29, Density: 0.5763
Distance: 30, Density: 0.3934
Distance: 31, Density: 0.2222
Distance: 32, Density: 0.1538
Distance: 33, Density: 0.1642
Distance: 34, Density: 0.0870
------------------------------

Step 450 Density Profile:
Distance: 0, Density: 2.0000
Distance: 1, Density: 1.3333
Distance: 2, Density: 0.4000
Distance: 3, Density: 3.0000
Distance: 4, Density: 0.8889
Distance: 5, Density: 2.6364
Distance: 6, Density: 0.3846
Distance: 7, Density: 1.8000
Distance: 8, Density: 2.0000
Distance: 9, Density: 1.4211
Distance: 10, Density: 1.7619
Distance: 11, Density: 2.1739
Distance: 12, Density: 1.6400
Distance: 13, Density: 0.9630
Distance: 14, Density: 2.4138
Distance: 15, Density: 2.1935
Distance: 16, Density: 0.9697
Distance: 17, Density: 1.7143
Distance: 18, Density: 1.8649
Distance: 19, Density: 2.8462
Distance: 20, Density: 1.6585
Distance: 21, Density: 1.9767
Distance: 22, Density: 1.2889
Distance: 23, Density: 1.1489
Distance: 24, Density: 2.0408
Distance: 25, Density: 1.4902
Distance: 26, Density: 1.0566
Distance: 27, Density: 0.4364
Distance: 28, Density: 0.8246
Distance: 29, Density: 0.6780
Distance: 30, Density: 0.2951
Distance: 31, Density: 0.2698
Distance: 32, Density: 0.2615
Distance: 33, Density: 0.1791
Distance: 34, Density: 0.1014
------------------------------

--- Final Density Profile ---
Distance: 0, Density: 6.0000
Distance: 1, Density: 1.0000
Distance: 2, Density: 0.2000
Distance: 3, Density: 4.5714
Distance: 4, Density: 1.0000
Distance: 5, Density: 2.6364
Distance: 6, Density: 0.6923
Distance: 7, Density: 1.6667
Distance: 8, Density: 2.4118
Distance: 9, Density: 1.5263
Distance: 10, Density: 1.9048
Distance: 11, Density: 2.1739
Distance: 12, Density: 1.4400
Distance: 13, Density: 1.0000
Distance: 14, Density: 2.9655
Distance: 15, Density: 2.7097
Distance: 16, Density: 0.7576
Distance: 17, Density: 1.6286
Distance: 18, Density: 2.2162
Distance: 19, Density: 2.2051
Distance: 20, Density: 1.9268
Distance: 21, Density: 1.7209
Distance: 22, Density: 1.5333
Distance: 23, Density: 1.1915
Distance: 24, Density: 2.1020
Distance: 25, Density: 1.6275
Distance: 26, Density: 1.3962
Distance: 27, Density: 0.4182
Distance: 28, Density: 0.8596
Distance: 29, Density: 0.8814
Distance: 30, Density: 0.3279
Distance: 31, Density: 0.3651
Distance: 32, Density: 0.2154
Distance: 33, Density: 0.2687
Distance: 34, Density: 0.0870
"""

# Split the raw data into blocks for each step
data_blocks = raw_data.split('------------------------------')
data_blocks = [block.strip() for block in data_blocks if block.strip()]

# Parse the data into a structured format
parsed_data = {}
core_density_data = {}
steps = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
final_step_text = 'Final Density Profile'

for i, block in enumerate(data_blocks):
    step_match = re.search(r'Step (\d+)', block)
    if step_match:
        step_num = int(step_match.group(1))
    elif 'Final Density Profile' in block:
        step_num = 500 # Assuming 'Final' means step 500
    else:
        continue # Skip if no step number found

    lines = block.split('\n')
    current_step_data = []
    core_density = None
    for line in lines:
        if 'Distance:' in line and 'Density:' in line:
            parts = line.split(',')
            distance = float(parts[0].split(':')[1].strip())
            density = float(parts[1].split(':')[1].strip())
            current_step_data.append({'Distance': distance, 'Density': density})
            if distance == 0.0:
                core_density = density
    
    if current_step_data:
        parsed_data[f'Density_Step_{step_num}'] = pd.DataFrame(current_step_data).set_index('Distance')['Density']
    
    if core_density is not None:
        core_density_data[step_num] = core_density

# Create a master DataFrame for the density profiles
df_profiles = pd.DataFrame(parsed_data)

# Fill any missing distances with NaN
all_distances = pd.Index(range(35), name='Distance')
df_profiles = df_profiles.reindex(all_distances)

# Create a DataFrame for core density fluctuation
df_core_density = pd.DataFrame(list(core_density_data.items()), columns=['Step', 'Core Density'])
df_core_density = df_core_density.sort_values('Step').reset_index(drop=True)

# Plot 1: Density Profile Evolution
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8-whitegrid')
for col in df_profiles.columns:
    step = col.split('_')[-1]
    plt.plot(df_profiles.index, df_profiles[col], label=f'Step {step}', marker='o', markersize=3)

plt.title('Density Profile Evolution Over Time', fontsize=16)
plt.xlabel('Distance from Center', fontsize=12)
plt.ylabel('Particle Density', fontsize=12)
plt.legend(title='Simulation Step', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('density_profile_evolution.png')

# Plot 2: Core Density Fluctuation
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(df_core_density['Step'], df_core_density['Core Density'], marker='o', linestyle='-')
plt.title('Core Density Fluctuation Over Time', fontsize=16)
plt.xlabel('Simulation Step', fontsize=12)
plt.ylabel('Core Density (at Distance 0)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('core_density_fluctuation.png')

print("Graphs have been generated and saved as 'density_profile_evolution.png' and 'core_density_fluctuation.png'.")
