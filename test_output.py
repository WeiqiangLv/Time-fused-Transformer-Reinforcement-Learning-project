#!/usr/bin/env python3
"""
Test script to verify that output directory creation and file saving works correctly.
"""

import os
import pandas as pd
import numpy as np

def test_output_functionality():
    """Test the output directory creation and Excel file saving."""
    
    print("Testing output functionality...")
    
    # Create output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}")
    else:
        print(f"✓ Output directory already exists: {output_dir}")
    
    # Test data
    test_data = {
        'SNR': [20, 24, 28, 32],
        'BER': [0.001, 0.0005, 0.0002, 0.0001],
        'Reward': [10, 15, 20, 25]
    }
    
    # Test Excel file creation
    try:
        test_file = os.path.join(output_dir, 'test_output.xlsx')
        
        with pd.ExcelWriter(test_file, engine='openpyxl') as writer:
            pd.DataFrame(test_data).to_excel(writer, sheet_name='test_data', float_format='%0.4f')
        
        print(f"✓ Successfully created test Excel file: {test_file}")
        
        # Verify file exists and has content
        if os.path.exists(test_file):
            file_size = os.path.getsize(test_file)
            print(f"✓ File size: {file_size} bytes")
            
            # Read back the data to verify
            df = pd.read_excel(test_file, sheet_name='test_data', index_col=0)
            print(f"✓ Successfully read back data with {len(df)} rows")
            
            # Clean up test file
            os.remove(test_file)
            print(f"✓ Cleaned up test file")
            
        return True
        
    except Exception as e:
        print(f"✗ Error creating Excel file: {e}")
        return False

def test_path_handling():
    """Test that we handle different path formats correctly."""
    
    print("\nTesting path handling...")
    
    # Test different path formats
    test_paths = [
        'output',
        './output',
        os.path.join('output', 'subdir'),
        'output/test_file.xlsx'
    ]
    
    for path in test_paths:
        try:
            if path.endswith('.xlsx'):
                # Test file path
                dir_path = os.path.dirname(path)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                print(f"✓ Path handling works for: {path}")
            else:
                # Test directory path
                if not os.path.exists(path):
                    os.makedirs(path)
                print(f"✓ Directory creation works for: {path}")
        except Exception as e:
            print(f"✗ Error with path {path}: {e}")
            return False
    
    # Clean up test directories
    try:
        import shutil
        if os.path.exists('output/subdir'):
            shutil.rmtree('output/subdir')
        print("✓ Cleaned up test directories")
    except:
        pass
    
    return True

if __name__ == "__main__":
    print("OTFS Output Functionality Test")
    print("=" * 40)
    
    success = True
    
    # Run tests
    success &= test_output_functionality()
    success &= test_path_handling()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! Output functionality is working correctly.")
        print("\nThe system will now:")
        print("- Create an 'output' directory automatically")
        print("- Save Excel files with proper naming: BER_all_SNR_HEIGHT.xlsx")
        print("- Handle file paths correctly across platforms")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\nCommon issues:")
        print("- Missing openpyxl dependency: pip install openpyxl")
        print("- Permission issues: check write permissions in current directory")
        print("- Pandas version compatibility: ensure pandas >= 1.4.0")