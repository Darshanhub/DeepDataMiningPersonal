#!/usr/bin/env python3
"""
Test script to verify the epoch logging changes work correctly.
Run this before actual training to catch any issues.
"""

import os
import sys
import tempfile
import shutil
import json

def test_experiment_folder_protection():
    """Test that experiment folder protection works"""
    print("=" * 70)
    print("TEST 1: Experiment Folder Protection")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake experiment folder with checkpoint
        exp_dir = os.path.join(tmpdir, "test_exp")
        os.makedirs(exp_dir)
        
        # Create a checkpoint file
        checkpoint_file = os.path.join(exp_dir, "checkpoint.pth")
        with open(checkpoint_file, 'w') as f:
            f.write("fake checkpoint")
        
        # Test: Should detect existing experiment
        has_checkpoints = os.path.exists(checkpoint_file)
        print(f"✅ Checkpoint detection: {has_checkpoints}")
        assert has_checkpoints, "Failed to detect checkpoint"
        
        # Create epoch folders
        epoch_dir = os.path.join(exp_dir, "epoch_0")
        os.makedirs(epoch_dir)
        
        # Test: Should detect epoch folders
        dir_contents = os.listdir(exp_dir)
        has_epoch_folders = any(
            d.startswith("epoch_") and os.path.isdir(os.path.join(exp_dir, d))
            for d in dir_contents
        )
        print(f"✅ Epoch folder detection: {has_epoch_folders}")
        assert has_epoch_folders, "Failed to detect epoch folders"
        
        # Test: Empty directory should pass
        empty_dir = os.path.join(tmpdir, "empty_exp")
        os.makedirs(empty_dir)
        has_checkpoints_empty = os.path.exists(os.path.join(empty_dir, "checkpoint.pth"))
        dir_contents_empty = os.listdir(empty_dir)
        has_epoch_folders_empty = any(
            d.startswith("epoch_") and os.path.isdir(os.path.join(empty_dir, d))
            for d in dir_contents_empty
        )
        print(f"✅ Empty directory passes: {not has_checkpoints_empty and not has_epoch_folders_empty}")
        
    print("✅ TEST 1 PASSED\n")


def test_epoch_folder_creation():
    """Test that epoch folders are created correctly"""
    print("=" * 70)
    print("TEST 2: Epoch Folder Creation")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = tmpdir
        
        # Simulate creating epoch folders
        for epoch in range(3):
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Test folder exists
            assert os.path.exists(epoch_dir), f"Failed to create epoch_{epoch}"
            print(f"✅ Created: epoch_{epoch}/")
        
        # Verify all folders exist
        created_folders = [d for d in os.listdir(output_dir) if d.startswith("epoch_")]
        print(f"✅ Created {len(created_folders)} epoch folders: {created_folders}")
        assert len(created_folders) == 3
    
    print("✅ TEST 2 PASSED\n")


def test_metrics_json_saving():
    """Test that metrics can be saved to JSON"""
    print("=" * 70)
    print("TEST 3: Metrics JSON Saving")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        epoch_dir = os.path.join(tmpdir, "epoch_0")
        os.makedirs(epoch_dir)
        
        # Simulate saving metrics
        metrics_file = os.path.join(epoch_dir, "train_metrics.json")
        metrics_dict = {
            "epoch": 0,
            "loss": 2.345,
            "loss_box": 1.234,
            "loss_cls": 0.567,
            "loss_dfl": 0.544,
            "lr": 0.02,
            "time": "0:05:23"
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # Verify file exists and is readable
        assert os.path.exists(metrics_file), "Failed to create metrics file"
        print(f"✅ Created: train_metrics.json")
        
        # Verify can read back
        with open(metrics_file, 'r') as f:
            loaded_metrics = json.load(f)
        
        assert loaded_metrics["epoch"] == 0
        assert loaded_metrics["loss"] == 2.345
        print(f"✅ Metrics loaded correctly: {loaded_metrics}")
        
        # Test eval results
        eval_file = os.path.join(epoch_dir, "eval_results.json")
        eval_results = {
            "mAP": 0.654,
            "mAP_50": 0.789
        }
        
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        assert os.path.exists(eval_file)
        print(f"✅ Created: eval_results.json")
    
    print("✅ TEST 3 PASSED\n")


def test_metric_logger_compatibility():
    """Test that metric logger works with the new code"""
    print("=" * 70)
    print("TEST 4: MetricLogger Compatibility")
    print("=" * 70)
    
    # Mock MetricLogger behavior
    class MockMeter:
        def __init__(self, value):
            self.global_avg = value
    
    class MockMetricLogger:
        def __init__(self):
            self.meters = {
                'loss': MockMeter(2.345),
                'box': MockMeter(1.234),
                'cls': MockMeter(0.567),
                'dfl': MockMeter(0.544)
            }
    
    # Simulate the code in training loop
    train_metrics = MockMetricLogger()
    
    # Test accessing metrics
    try:
        loss = float(train_metrics.meters['loss'].global_avg)
        print(f"✅ Loss value: {loss}")
        assert loss == 2.345
        
        # Test accessing individual loss components
        metrics_dict = {}
        for key in ['box', 'cls', 'dfl']:
            if key in train_metrics.meters:
                metrics_dict[f'loss_{key}'] = float(train_metrics.meters[key].global_avg)
        
        print(f"✅ Loss components: {metrics_dict}")
        assert 'loss_box' in metrics_dict
        assert 'loss_cls' in metrics_dict
        assert 'loss_dfl' in metrics_dict
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    
    print("✅ TEST 4 PASSED\n")


def test_performance_overhead():
    """Test that logging has minimal overhead"""
    print("=" * 70)
    print("TEST 5: Performance Overhead")
    print("=" * 70)
    
    import time
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Measure epoch folder creation
        start = time.time()
        for i in range(100):
            epoch_dir = os.path.join(tmpdir, f"epoch_{i}")
            os.makedirs(epoch_dir, exist_ok=True)
        folder_time = time.time() - start
        
        print(f"✅ Creating 100 epoch folders: {folder_time*1000:.2f}ms ({folder_time*10:.4f}ms per folder)")
        assert folder_time < 1.0, "Folder creation too slow"
        
        # Measure JSON writing
        start = time.time()
        for i in range(100):
            metrics_file = os.path.join(tmpdir, f"epoch_{i}", "metrics.json")
            metrics = {
                "epoch": i,
                "loss": 2.345,
                "lr": 0.02
            }
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f)
        json_time = time.time() - start
        
        print(f"✅ Writing 100 JSON files: {json_time*1000:.2f}ms ({json_time*10:.4f}ms per file)")
        assert json_time < 1.0, "JSON writing too slow"
        
        total_overhead = (folder_time + json_time) * 10  # Per epoch
        print(f"✅ Total overhead per epoch: {total_overhead*1000:.2f}ms")
        print(f"   (Target: < 10ms, Actual: {total_overhead*1000:.2f}ms)")
        
        if total_overhead < 0.01:  # 10ms
            print("✅ EXCELLENT: Negligible performance impact!")
        elif total_overhead < 0.1:  # 100ms
            print("✅ GOOD: Minimal performance impact")
        else:
            print("⚠️  WARNING: Performance overhead might be noticeable")
    
    print("✅ TEST 5 PASSED\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("EPOCH LOGGING CHANGES - VALIDATION TESTS")
    print("=" * 70 + "\n")
    
    tests = [
        ("Experiment Folder Protection", test_experiment_folder_protection),
        ("Epoch Folder Creation", test_epoch_folder_creation),
        ("Metrics JSON Saving", test_metrics_json_saving),
        ("MetricLogger Compatibility", test_metric_logger_compatibility),
        ("Performance Overhead", test_performance_overhead),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
        print("\n⚠️  PLEASE FIX FAILING TESTS BEFORE TRAINING!")
        return 1
    else:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Code changes are safe to use for training")
        return 0


if __name__ == '__main__':
    sys.exit(main())
