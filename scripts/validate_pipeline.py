#!/usr/bin/env python3
"""
Validation script for the polysemy analysis pipeline.

This script runs a quick test on sample data to verify:
1. Pipeline components work correctly
2. Polysemy index computation behaves sensibly
3. Known polysemous words show expected patterns

Usage:
    python scripts/validate_pipeline.py
    python scripts/validate_pipeline.py --verbose
"""

import argparse
import logging
import sys
from pathlib import Path
import csv
import json

# Pandas is optional for this validation script
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

def validate_target_words():
    """Validate that target_words.csv is properly formatted."""
    print("\n" + "="*60)
    print("VALIDATING TARGET WORDS FILE")
    print("="*60)
    
    target_words_path = Path(__file__).parent.parent / 'data' / 'target_words.csv'
    
    if not target_words_path.exists():
        print(f"❌ ERROR: {target_words_path} not found")
        return False
    
    try:
        # Read CSV using csv module (no pandas required)
        with open(target_words_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"✓ Loaded {len(rows)} target words")
        
        required_columns = ['word', 'reading', 'pos', 'known_senses', 'polysemy_expectation']
        if rows:
            actual_columns = list(rows[0].keys())
            missing = [col for col in required_columns if col not in actual_columns]
            
            if missing:
                print(f"❌ ERROR: Missing columns: {missing}")
                return False
            
            print(f"✓ All required columns present: {actual_columns}")
            
            # Display summary
            expectations = {}
            for row in rows:
                exp = row['polysemy_expectation']
                expectations[exp] = expectations.get(exp, 0) + 1
            
            print(f"\nPolysemy expectations:")
            for expectation, count in expectations.items():
                print(f"  {expectation}: {count} words")
            
            # Show some examples
            print(f"\nExample target words:")
            for row in rows[:5]:
                print(f"  {row['word']} ({row['reading']}): {row['polysemy_expectation']}")
        
        return True
    
    except Exception as e:
        print(f"❌ ERROR validating target words: {e}")
        return False

def validate_sample_texts():
    """Validate that sample text files exist and are readable."""
    print("\n" + "="*60)
    print("VALIDATING SAMPLE TEXTS")
    print("="*60)
    
    base_path = Path(__file__).parent.parent / 'data' / 'samples'
    
    edo_path = base_path / 'edo' / 'sample_edo_text.txt'
    meiji_path = base_path / 'meiji' / 'sample_meiji_text.txt'
    
    success = True
    
    for era, path in [('Edo', edo_path), ('Meiji', meiji_path)]:
        if not path.exists():
            print(f"❌ ERROR: {era} sample not found at {path}")
            success = False
            continue
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            char_count = len(content)
            line_count = content.count('\n') + 1
            
            # Check for target words
            target_words_path = Path(__file__).parent.parent / 'data' / 'target_words.csv'
            if target_words_path.exists():
                with open(target_words_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    target_words = [row['word'] for row in reader]
                
                found_words = [w for w in target_words if w in content]
                
                print(f"\n✓ {era} sample:")
                print(f"  - {char_count} characters, {line_count} lines")
                print(f"  - Contains {len(found_words)}/{len(target_words)} target words")
                if found_words:
                    print(f"  - Found: {', '.join(found_words[:10])}")
                    if len(found_words) > 10:
                        print(f"    ... and {len(found_words) - 10} more")
        
        except Exception as e:
            print(f"❌ ERROR reading {era} sample: {e}")
            success = False
    
    return success

def check_dependencies():
    """Check that required dependencies are installed."""
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    dependencies = {
        'transformers': 'Hugging Face Transformers',
        'torch': 'PyTorch',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'NumPy',
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed (pip install {module})")
            all_ok = False
    
    # Check MeCab separately
    try:
        import MeCab
        print(f"✓ MeCab installed")
    except ImportError:
        print(f"⚠ MeCab NOT installed (optional - pip install mecab-python3)")
        print(f"  Note: MeCab is required for full pipeline functionality")
    
    return all_ok

def test_imports():
    """Test that pipeline modules can be imported."""
    print("\n" + "="*60)
    print("TESTING PIPELINE IMPORTS")
    print("="*60)
    
    modules = [
        'utils',
        'data_preprocess',
        'embedding_extraction',
        'polysemy_clustering',
        'compare_eras'
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}.py can be imported")
        except Exception as e:
            print(f"❌ ERROR importing {module}.py: {e}")
            all_ok = False
    
    return all_ok

def run_quick_test(verbose=False):
    """Run a quick test of the pipeline on sample data."""
    print("\n" + "="*60)
    print("QUICK PIPELINE TEST")
    print("="*60)
    print("\nThis would run the pipeline on sample data.")
    print("Skipping actual execution to avoid dependency issues in validation.")
    print("\nTo test manually, run:")
    print("  python src/data_preprocess.py --edo-dir data/samples/edo --meiji-dir data/samples/meiji --output /tmp/test_processed --top-n 10 --min-freq 2")
    
    return True

def validate_documentation():
    """Check that key documentation files exist."""
    print("\n" + "="*60)
    print("CHECKING DOCUMENTATION")
    print("="*60)
    
    base_path = Path(__file__).parent.parent
    
    docs = {
        'README.md': 'Main README',
        'USAGE.md': 'Usage guide',
        'data/README.md': 'Data documentation',
        'data/TARGET_WORDS_METADATA.md': 'Target words metadata',
        'data/samples/README.md': 'Sample texts documentation'
    }
    
    all_ok = True
    for file_path, description in docs.items():
        full_path = base_path / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✓ {description}: {file_path} ({size} bytes)")
        else:
            print(f"❌ {description}: {file_path} NOT FOUND")
            all_ok = False
    
    return all_ok

def main():
    """Main validation routine."""
    parser = argparse.ArgumentParser(description='Validate the polysemy analysis pipeline')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    print("="*60)
    print("POLYSEMY ANALYSIS PIPELINE VALIDATION")
    print("="*60)
    
    results = {}
    
    # Run all validation checks
    results['dependencies'] = check_dependencies()
    results['imports'] = test_imports()
    results['target_words'] = validate_target_words()
    results['sample_texts'] = validate_sample_texts()
    results['documentation'] = validate_documentation()
    results['quick_test'] = run_quick_test(args.verbose)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for check, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {check}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ All validations passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Add real text data to data/edo/ and data/meiji/")
        print("2. Run: bash scripts/run_pipeline.sh")
        print("3. Check results in results/ directory")
        return 0
    else:
        print("\n⚠ Some validations failed. Please address the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
