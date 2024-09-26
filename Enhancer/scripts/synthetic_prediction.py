import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../"))
sys.path.append(parent_dir)
from train.utils import EnhancerDataset
import train.interpretation as interpretation

def extract_pwm_from_meme(meme_file, motif_name):
    with open(meme_file) as f:
        lines = f.readlines()

    # Initialize variables to capture the motif
    pwm = []
    reading_pwm = False
    
    for line in lines:
        line = line.strip()
        
        # Check if we found the desired motif
        if line.startswith('MOTIF') and motif_name in line:
            reading_pwm = True  # Start reading the PWM
            continue  # Skip to the next line
        
        # Read the PWM if we're in the correct motif section
        if reading_pwm:
            if line.startswith('letter-probability matrix:'):
                continue  # Skip the header line
            
            if line.startswith('MOTIF'):
                break  # Stop reading at the next motif
            
            if line:  # If the line is not empty
                pwm_row = list(map(float, line.split()))
                pwm.append(pwm_row)
    
    if pwm:
        print(f"PWM for {motif_name} has length: {len(pwm)}")
        return pwm  # Return the PWM matrix
    else:
        print("Motif not found.")
    
    return None  # Return None if the motif is not found

def generate_random_dna(length):
    """Generate a random DNA sequence of specified length."""
    return ''.join(random.choices('ACGT', k=length))

def insert_motif_with_distance(dna_sequence, motif, distance):
    """Insert the motif into the DNA sequence at fixed intervals."""
    motif_length = len(motif)
    new_dnas = []
    new_dnas_names = []
    
    for start in range(0, len(dna_sequence) - motif_length + 1, distance):
        new_dna = dna_sequence[:start] + motif + dna_sequence[start + motif_length:]
        new_dnas.append(new_dna)
        new_dnas_names.append(motif+'_'+str(start))

    return new_dnas,new_dnas_names

def pwm_to_motif(pwm):
    """Convert PWM to a consensus sequence (simple method)."""
    motif = ""
    for row in pwm:
        max_index = np.argmax(row)  # Get index of max probability
        motif += 'ACGT'[max_index]  # Append corresponding nucleotide
    return motif

def calculate_motif_distance(motifs, sequence):
    """Helper function that calculates distance between the end of MotifA and start of MotifB"""
    # Check if there are exactly two motifs
    if len(motifs) != 2:
        print("Warning: The 'motifs' list must contain exactly two motifs.")
        return -1
    
    motif_A = motifs[0]
    motif_B = motifs[1]
    
    # Check occurrences of each motif
    count_A = sequence.count(motif_A)
    count_B = sequence.count(motif_B)
    
    if count_A != 1 or count_B != 1:
        print("Warning: Each motif must occur exactly once in the DNA sequence.")
        return -1

    # Find the positions of the motifs
    start_A = sequence.index(motif_A)  # Start index of motif A
    end_A = start_A + len(motif_A)     # End index of motif A
    start_B = sequence.index(motif_B)   # Start index of motif B

    # Calculate the distance between the end of motif A and the start of motif B
    distance = start_B - end_A
    
    return distance

def generate_synthetic_data(dna_length: int, motif_names: list, input_dir: str, output_dir: str, distance: int, swap: bool):

    assert len(motif_names) == 2

    motifs = []
    DNA_with_motifs = []
    DNA_with_motifs_names = []
    
    # Get motif sequence from the input meme file for each motif name in the 'motif_names' list
    for motif_name in motif_names:
        motif = pwm_to_motif(extract_pwm_from_meme(input_dir, motif_name))
        print(f'{motif_name}: {motif}')
        motifs.append(motif)
    print('\n')

    # Generate a random DNA sequence for each motif, then insert each motif into different positions in the DNA
    for i, motif in enumerate(motifs):
        # Generate a random DNA sequence for each motif
        dna_sequence = generate_random_dna(dna_length)
        
        # Insert the motif at specified distances
        inserted_dnas, inserted_dnas_names = insert_motif_with_distance(dna_sequence, motif, distance)
        print(f'{motif} has {len(inserted_dnas)} {dna_length}nt DNA segments, inserted with distance {distance}')
        DNA_with_motifs.extend(inserted_dnas)
        DNA_with_motifs_names.extend(inserted_dnas_names)
    print(f'total number of different DNA segments: {len(DNA_with_motifs)}')

    # Generate and append DNA sequences without insertions to DNA_with_motifs
    dna_sequence = generate_random_dna(dna_length)
    DNA_with_motifs.append(dna_sequence)
    #DNA_with_motifs_names.append('background_dna')
    print(f'total number of different DNA segments: {len(DNA_with_motifs)}')

    DNA_seg_len = len(DNA_with_motifs)
    with open(output_dir, 'w') as fasta_file:
        for pos1 in range(DNA_seg_len):
            seg1 = DNA_with_motifs[pos1]
            seg1_name = DNA_with_motifs_names[pos1]
            for pos2 in range(DNA_seg_len):
                seg2 = DNA_with_motifs[pos2]
                seg2_name = DNA_with_motifs_names[pos2]
                for pos3 in range(DNA_seg_len):
                    seg3 = DNA_with_motifs[pos3]
                    seg3_name = DNA_with_motifs_names[pos3]
                    # Combine seg1, seg2, seg3 into a single list
                    combined_segments = [seg1, seg2, seg3]
                    if len(set(combined_segments)) == len(combined_segments): # Ensure each unique dna segment one occurs once in one synthetic DNA sequence
                        new_seq = seg1 + 'CTGA' + seg2 + 'ACCA' + seg3
                        
                        new_seq_name = seg1_name + '_' + seg2_name + '_' + seg3_name
                        fasta_file.write(f"> {new_seq_name}\n")
                        fasta_file.write(f"{new_seq}\n\n")

    return 0

def generate_synthetic_distance_data(dna_length: int, motif_names: list, input_dir: str, output_dir: str, distance: int, replicate: int, save_plot = False):
    assert len(motif_names) == 2, "There must be exactly two motifs."

    motifs = []
    # Create four lists to store four columns in the final data frame
    names = []
    sequences = []
    motif_distance = []
    replicate_list = []  # Renamed to avoid conflict
    
    # Get motif sequence from the input meme file for each motif name in the 'motif_names' list
    for motif_name in motif_names:
        motif = pwm_to_motif(extract_pwm_from_meme(input_dir, motif_name))
        print(f'{motif_name}: {motif}')
        motifs.append(motif)
    print('\n')

    for rep in range(replicate):
        # Generate DNA sequences inserted with motif A
        dna_sequence = generate_random_dna(dna_length)
        motifA_dna_sequence, motifA_dna_sequence_names = insert_motif_with_distance(dna_sequence, motifs[0], distance)
        if rep == 0:
            print(f'Motif A {motifs[0]} has {len(motifA_dna_sequence)} {dna_length}nt DNA segments, inserted with distance {distance}')

        # Generate DNA sequences inserted with motif B
        dna_sequence = generate_random_dna(dna_length)
        motifB_dna_sequence, motifB_dna_sequence_names = insert_motif_with_distance(dna_sequence, motifs[1], distance)
        if rep == 0:
            print(f'Motif B {motifs[1]} has {len(motifB_dna_sequence)} {dna_length}nt DNA segments, inserted with distance {distance}')

        # Generate DNA sequences without any insertions
        dna_sequence = generate_random_dna(dna_length)

        for motifA_index in range(len(motifA_dna_sequence)):
            segA = motifA_dna_sequence[motifA_index]
            segA_name = motifA_dna_sequence_names[motifA_index]

            for motifB_index in range(len(motifB_dna_sequence)):
                segB = motifB_dna_sequence[motifB_index]
                segB_name = motifB_dna_sequence_names[motifB_index]

                # First case: motifA, motifB, random_dna
                combined_seq = segA + 'CTGA' + segB + 'ACCA' + dna_sequence
                combined_seq_name = f"{segA_name}_{segB_name}_random_dna"
                combined_seq_distance_score = calculate_motif_distance(motifs, combined_seq)
                names.append(combined_seq_name)
                sequences.append(combined_seq)
                motif_distance.append(combined_seq_distance_score)
                replicate_list.append(rep)

                # Second case: motifA, random_dna, motifB
                combined_seq = segA + 'CTGA' + dna_sequence + 'ACCA' + segB
                combined_seq_name = f"{segA_name}_random_dna_{segB_name}"
                combined_seq_distance_score = calculate_motif_distance(motifs, combined_seq)
                names.append(combined_seq_name)
                sequences.append(combined_seq)
                motif_distance.append(combined_seq_distance_score)
                replicate_list.append(rep)

    # Create DataFrame
    result_df = pd.DataFrame({
        'name': names,
        'sequence': sequences,
        'motif_distance_score': motif_distance,
        'replicate': replicate_list
    })

    # Save DataFrame to pickle
    if save_plot == True:
        result_df.to_pickle(output_dir)

    return result_df

def motif_score_prediction(df, model, device, batch:int, target_labels: list, output_dir = None):
    result_df = df.copy()
    dataset = EnhancerDataset(df, feature_list=['motif_distance_score'], scale_mode = 'none')
    # Prepare dataloader
    dataset = DataLoader(dataset=dataset, batch_size=batch, shuffle=False)
    # Running get_explainn_predictions function to get predictions and true labels for all sequences in the given data loader
    predictions, _ = interpretation.get_explainn_predictions(dataset, model, device, isSigmoid=False)
    for i in range(len(target_labels)):
        result_df[target_labels[i]] = predictions[:,i]
    if output_dir != None:
        result_df.to_pickle(output_dir)
    return result_df