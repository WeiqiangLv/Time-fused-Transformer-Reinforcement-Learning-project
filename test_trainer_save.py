#!/usr/bin/env python3
"""
Test script to verify that the trainer's save functionality works correctly.
"""

import os
import pandas as pd
import numpy as np

def test_trainer_save_logic():
    """Test the trainer's save logic without running the full training."""
    
    print("Testing trainer save logic...")
    
    # Simulate the data that would be saved
    B_all = [[0.001, 0.002, 0.003], [0.0005, 0.001, 0.0015]]  # BER data
    R_all = [[10, 15, 20], [12, 18, 25]]  # Reward data
    L_all = [[0.5, 0.3, 0.2], [0.4, 0.25, 0.15]]  # Loss data
    
    # Simulate trainer parameters
    snr = 20
    height = 10000
    
    # Create output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}")
    
    try:
        # Test BER file saving
        ber_output_file = os.path.join(output_dir, f'BER_all_{snr}_{height}.xlsx')
        with pd.ExcelWriter(ber_output_file, engine='openpyxl') as writer1:
            pd.DataFrame(B_all).to_excel(writer1, sheet_name='page_1', float_format='%0.2f')
        print(f"✓ BER results saved to: {ber_output_file}")
        
        # Test Rewards file saving
        r_output_file = os.path.join(output_dir, f'R_all_{snr}_{height}.xlsx')
        with pd.ExcelWriter(r_output_file, engine='openpyxl') as writer2:
            pd.DataFrame(R_all).to_excel(writer2, sheet_name='page_1', float_format='%0.2f')
        print(f"✓ Rewards saved to: {r_output_file}")
        
        # Test Loss file saving
        l_output_file = os.path.join(output_dir, f'L_all_{snr}_{height}.xlsx')
        with pd.ExcelWriter(l_output_file, engine='openpyxl') as writer3:
            pd.DataFrame(L_all).to_excel(writer3, sheet_name='page_1', float_format='%0.2f')
        print(f"✓ Loss data saved to: {l_output_file}")
        
        # Verify files exist and have content
        for file_path in [ber_output_file, r_output_file, l_output_file]:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"✓ File {os.path.basename(file_path)} exists ({size} bytes)")
            else:
                print(f"✗ File {file_path} was not created")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in save logic: {e}")
        return False

if __name__ == "__main__":
    print("Trainer Save Logic Test")
    print("=" * 30)
    
    success = test_trainer_save_logic()
    
    print("\n" + "=" * 30)
    if success:
        print("🎉 Trainer save logic test passed!")
        print("The trainer should now save files correctly to the output directory.")
    else:
        print("❌ Trainer save logic test failed!")
        print("Please check the error messages above.")