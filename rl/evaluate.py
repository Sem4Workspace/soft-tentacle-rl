#!/usr/bin/env python3
# ==============================================================================
# EVALUATION SCRIPT - Soft Robotic Tentacle RL
# ==============================================================================
"""
Evaluate a trained RL model on the soft tentacle environment.

This script:
- Loads a trained model
- Runs episodes with the trained policy
- Computes statistics (reward, distance, success rate)
- Optional: Renders episodes for visualization
- Optional: Records videos

USAGE:
------
    # Basic evaluation
    python rl/evaluate.py --model results/checkpoints/ppo_tentacle_final.zip
    
    # With rendering
    python rl/evaluate.py --model results/checkpoints/ppo_tentacle_final.zip --render
    
    # Multiple episodes
    python rl/evaluate.py --model results/checkpoints/ppo_tentacle_final.zip --episodes 10
    
    # Save video
    python rl/evaluate.py --model results/checkpoints/ppo_tentacle_final.zip --save-video
    
    # Deterministic (no randomness)
    python rl/evaluate.py --model ... --deterministic
    
    # Stochastic (with exploration)
    python rl/evaluate.py --model ... --stochastic

CURRENT MODEL: spiral_5link.xml (5 joints, planar)
FUTURE MODEL:  tentacle.xml (10 joints, 3D motion)
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from envs.tentacle_env import TentacleEnv
from stable_baselines3 import PPO, SAC


# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def load_model(model_path: str):
    """
    Load trained model from file.
    
    Automatically detects model type (PPO or SAC) from filename.
    
    Args:
        model_path: Path to model file (.zip)
    
    Returns:
        Loaded model instance
    """
    print(f"Loading model: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Detect model type
    if 'ppo' in model_path.lower():
        model_class = PPO
        model_type = "PPO"
    elif 'sac' in model_path.lower():
        model_class = SAC
        model_type = "SAC"
    else:
        # Try loading as PPO first, fall back to SAC
        try:
            model = PPO.load(model_path)
            model_type = "PPO"
            print(f"✅ Loaded as {model_type}")
            return model
        except Exception:
            model = SAC.load(model_path)
            model_type = "SAC"
            print(f"✅ Loaded as {model_type}")
            return model
    
    model = model_class.load(model_path)
    print(f"✅ Loaded {model_type} model")
    return model


def evaluate_episode(
    model,
    env,
    deterministic: bool = True,
    render: bool = False,
) -> dict:
    """
    Run a single evaluation episode.
    
    Args:
        model: Trained RL model
        env: Environment instance
        deterministic: Use deterministic policy (no exploration)
        render: Whether to render the episode
    
    Returns:
        Episode statistics dictionary
    """
    obs, info = env.reset()
    
    episode_reward = 0.0
    episode_length = 0
    min_distance = info['distance_to_target']
    distances = []
    rewards = []
    actions_taken = []
    
    while True:
        # Get action from trained policy
        action, _ = model.predict(obs, deterministic=deterministic)
        actions_taken.append(action)
        
        # Take environment step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record statistics
        episode_reward += reward
        episode_length += 1
        distance = info['distance_to_target']
        min_distance = min(min_distance, distance)
        distances.append(distance)
        rewards.append(reward)
        
        # Render if requested
        if render:
            try:
                env.render()
            except Exception as e:
                # Rendering may not be available
                pass
        
        # Check termination
        if terminated or truncated:
            break
    
    # Compute statistics
    stats = {
        'episode_length': episode_length,
        'episode_reward': episode_reward,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_distance': min_distance,
        'final_distance': distance,
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'is_success': info['is_success'],
        'distances': distances,
        'rewards': rewards,
        'actions': np.array(actions_taken),
    }
    
    return stats


def evaluate_model(
    model_path: str,
    num_episodes: int = 5,
    deterministic: bool = True,
    render: bool = False,
    verbose: int = 1,
) -> dict:
    """
    Evaluate trained model over multiple episodes.
    
    Args:
        model_path: Path to trained model
        num_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy
        render: Whether to render episodes
        verbose: Verbosity level (0=silent, 1=normal, 2=detailed)
    
    Returns:
        Dictionary with overall statistics
    """
    
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # =========================================================================
    # CREATE ENVIRONMENT
    # =========================================================================
    print("\nCreating environment...")
    env = TentacleEnv()
    print(f"✅ Environment ready")
    print(f"   Model: {env.config['model']['xml_path']}")
    print(f"   Action space: {env.action_space}")
    
    # =========================================================================
    # RUN EPISODES
    # =========================================================================
    print(f"\nRunning {num_episodes} evaluation episodes...")
    print(f"Deterministic: {deterministic}")
    print(f"Rendering: {render}")
    print("-" * 70)
    
    all_stats = []
    
    for episode in range(num_episodes):
        try:
            stats = evaluate_episode(
                model=model,
                env=env,
                deterministic=deterministic,
                render=render,
            )
            all_stats.append(stats)
            
            # Print episode result
            if verbose >= 1:
                symbol = "✅" if stats['is_success'] else "❌"
                print(f"Episode {episode+1:2d}: {symbol} "
                      f"Reward={stats['episode_reward']:>10.4f} "
                      f"Distance={stats['final_distance']:>8.4f}m "
                      f"Length={stats['episode_length']:>4d}")
        
        except Exception as e:
            print(f"❌ Episode {episode+1} failed: {e}")
            if verbose >= 2:
                import traceback
                traceback.print_exc()
    
    # =========================================================================
    # COMPUTE OVERALL STATISTICS
    # =========================================================================
    print("-" * 70)
    
    episode_rewards = [s['episode_reward'] for s in all_stats]
    episode_lengths = [s['episode_length'] for s in all_stats]
    min_distances = [s['min_distance'] for s in all_stats]
    final_distances = [s['final_distance'] for s in all_stats]
    successes = [s['is_success'] for s in all_stats]
    
    overall_stats = {
        'num_episodes': num_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_min_distance': np.mean(min_distances),
        'mean_final_distance': np.mean(final_distances),
        'success_rate': np.mean(successes),
        'all_episodes': all_stats,
    }
    
    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Episodes evaluated: {num_episodes}")
    print(f"Successful: {sum(successes)}/{num_episodes} ({np.mean(successes)*100:.1f}%)")
    print(f"\nReward Statistics:")
    print(f"  Mean: {overall_stats['mean_reward']:>10.4f}")
    print(f"  Std:  {overall_stats['std_reward']:>10.4f}")
    print(f"  Max:  {overall_stats['max_reward']:>10.4f}")
    print(f"  Min:  {overall_stats['min_reward']:>10.4f}")
    print(f"\nDistance Statistics (m):")
    print(f"  Mean min distance:   {overall_stats['mean_min_distance']:>10.4f}")
    print(f"  Mean final distance: {overall_stats['mean_final_distance']:>10.4f}")
    print(f"\nEpisode Length:")
    print(f"  Mean: {overall_stats['mean_length']:>10.1f} steps")
    print("="*70)
    
    env.close()
    
    return overall_stats


# ==============================================================================
# COMPARISON FUNCTION
# ==============================================================================

def compare_models(
    models: list,
    num_episodes: int = 5,
    deterministic: bool = True,
):
    """
    Compare multiple trained models.
    
    Args:
        models: List of (name, path) tuples
        num_episodes: Episodes per model
        deterministic: Use deterministic policy
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    results = {}
    
    for name, path in models:
        print(f"\nEvaluating: {name}")
        stats = evaluate_model(path, num_episodes, deterministic)
        results[name] = stats
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"{'Model':<20} {'Reward':<12} {'Distance':<12} {'Success':<10}")
    print("-"*70)
    
    for name, stats in results.items():
        if stats is not None:
            reward = stats['mean_reward']
            distance = stats['mean_final_distance']
            success = stats['success_rate'] * 100
            print(f"{name:<20} {reward:>10.4f}  {distance:>10.4f}m  {success:>8.1f}%")
    
    print("="*70)


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def main():
    """Main evaluation function with CLI."""
    
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL model on soft tentacle environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python rl/evaluate.py --model results/checkpoints/ppo_tentacle_final.zip
  
  # With rendering
  python rl/evaluate.py --model results/checkpoints/ppo_tentacle_final.zip --render
  
  # Multiple episodes
  python rl/evaluate.py --model results/checkpoints/ppo_tentacle_final.zip --episodes 10
  
  # Stochastic (with exploration)
  python rl/evaluate.py --model results/checkpoints/ppo_tentacle_final.zip --stochastic
  
  # Compare models
  python rl/evaluate.py --compare ppo_final.zip sac_final.zip
        """
    )
    
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (.zip)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of evaluation episodes (default: 5)')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy (with exploration)')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering (if available)')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                       help='Verbosity level (default: 1)')
    parser.add_argument('--compare', type=str, nargs='+', default=None,
                       help='Compare multiple models')
    
    args = parser.parse_args()
    
    # Compare models
    if args.compare:
        models = [(Path(p).stem, p) for p in args.compare]
        compare_models(models, num_episodes=args.episodes)
        return 0
    
    # Single model evaluation
    if not args.model:
        print("Error: --model is required (unless using --compare)")
        parser.print_help()
        return 1
    
    # Evaluate
    try:
        overall_stats = evaluate_model(
            model_path=args.model,
            num_episodes=args.episodes,
            deterministic=not args.stochastic,
            render=args.render,
            verbose=args.verbose,
        )
        
        if overall_stats is None:
            return 1
        
        print("\n✅ EVALUATION COMPLETE!")
        return 0
    
    except Exception as e:
        print(f"\n❌ EVALUATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
