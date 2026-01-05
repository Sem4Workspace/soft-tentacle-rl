#!/usr/bin/env python3
# ==============================================================================
# PPO TRAINING SCRIPT - Soft Robotic Tentacle RL
# ==============================================================================
"""
Train a PPO (Proximal Policy Optimization) agent to control the soft tentacle.

PPO is chosen because:
- Simple and reliable
- Works well with dense reward signals
- Stable training
- Good for reaching tasks

USAGE:
------
    python rl/train_ppo.py                          # Default settings
    python rl/train_ppo.py --timesteps 500000       # Custom timesteps
    python rl/train_ppo.py --save-freq 10000        # Save every 10k steps
    python rl/train_ppo.py --verbose 2              # More verbose logging

CURRENT MODEL: spiral_5link.xml (5 joints, planar)
FUTURE MODEL:  tentacle.xml (10 joints, 3D motion)

To switch models, change 'xml_path' in configs/env.yaml
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# ==============================================================================
# CUSTOM CALLBACK FOR LOGGING
# ==============================================================================

class TentacleCallback(BaseCallback):
    """
    Custom callback for training monitoring.
    
    Logs:
    - Episode rewards
    - Episode lengths
    - Distance to target
    - Success rate
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        return True
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.verbose > 0:
            print("\n" + "="*70)
            print("TRAINING COMPLETE")
            print("="*70)


# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================

def train_ppo(
    timesteps: int = 100000,
    model_name: str = "ppo_tentacle",
    save_freq: int = 10000,
    verbose: int = 1,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple:
    """
    Train PPO agent on tentacle environment.
    
    Args:
        timesteps: Total training timesteps
        model_name: Name for saving model
        save_freq: Save checkpoint every N timesteps
        verbose: Verbosity level (0=silent, 2=detailed)
        learning_rate: PPO learning rate
        n_steps: Number of steps to collect before update
        batch_size: Batch size for training
        n_epochs: Number of epoch passes per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        (model, env, total_time_seconds)
    """
    
    print("\n" + "="*70)
    print("PPO TRAINING - SOFT ROBOTIC TENTACLE")
    print("="*70)
    
    # =========================================================================
    # SETUP DIRECTORIES
    # =========================================================================
    checkpoint_dir = project_root / "results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = project_root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # CREATE ENVIRONMENT
    # =========================================================================
    print("\n[1/5] Creating environment...")
    env = TentacleEnv()
    
    # Wrap with Monitor for automatic episode logging
    env = Monitor(env, str(log_dir / "training"))
    
    print(f"✅ Environment created")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Timesteps to train: {timesteps:,}")
    
    # =========================================================================
    # CREATE PPO MODEL
    # =========================================================================
    print("\n[2/5] Creating PPO model...")
    
    model = PPO(
        policy="MlpPolicy",              # Multi-layer perceptron policy
        env=env,
        learning_rate=learning_rate,     # Learning rate
        n_steps=n_steps,                 # Steps per update
        batch_size=batch_size,           # Batch size
        n_epochs=n_epochs,               # Epochs per update
        gamma=gamma,                     # Discount factor
        gae_lambda=gae_lambda,           # GAE lambda
        clip_range=0.2,                  # PPO clip range
        clip_range_vf=None,              # No value function clipping
        ent_coef=0.0,                    # No entropy bonus
        vf_coef=0.5,                     # Value function coefficient
        max_grad_norm=0.5,               # Gradient clipping
        use_sde=False,                   # No stochastic dynamics
        sde_sample_freq=-1,
        verbose=verbose,
        tensorboard_log=str(log_dir / "tensorboard"),
        device="auto",
    )
    
    print(f"✅ PPO model created")
    print(f"   Learning rate: {learning_rate}")
    print(f"   N steps: {n_steps}")
    print(f"   Batch size: {batch_size}")
    print(f"   N epochs: {n_epochs}")
    
    # =========================================================================
    # SETUP CALLBACKS
    # =========================================================================
    print("\n[3/5] Setting up callbacks...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoint_dir),
        name_prefix=model_name,
        verbose=verbose,
    )
    
    tentacle_callback = TentacleCallback(verbose=verbose)
    
    callbacks = [checkpoint_callback, tentacle_callback]
    
    print(f"✅ Callbacks configured")
    print(f"   Save checkpoint every {save_freq:,} steps")
    print(f"   Checkpoint directory: {checkpoint_dir}")
    
    # =========================================================================
    # TRAIN
    # =========================================================================
    print("\n[4/5] Starting training...")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=False,
        )
        
        elapsed_time = time.time() - start_time
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        elapsed_time = time.time() - start_time
    
    # =========================================================================
    # SAVE FINAL MODEL
    # =========================================================================
    print("\n[5/5] Saving final model...")
    
    model_path = checkpoint_dir / f"{model_name}_final"
    model.save(str(model_path))
    
    print(f"✅ Model saved: {model_path}.zip")
    
    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total timesteps: {timesteps:,}")
    print(f"Training time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Timesteps/second: {timesteps/elapsed_time:.0f}")
    print(f"Model saved to: {model_path}.zip")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")
    print("="*70)
    
    return model, env, elapsed_time


# ==============================================================================
# QUICK TEST AFTER TRAINING
# ==============================================================================

def test_trained_model(model, env, num_episodes: int = 3):
    """
    Quick test of trained model.
    
    Args:
        model: Trained PPO model
        env: Environment instance
        num_episodes: Number of test episodes
    """
    print("\n" + "="*70)
    print(f"TESTING TRAINED MODEL ({num_episodes} episodes)")
    print("="*70)
    
    episode_rewards = []
    episode_distances = []
    episode_successes = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        min_distance = float('inf')
        
        while True:
            # Use trained policy (deterministic)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            min_distance = min(min_distance, info['distance_to_target'])
            
            if terminated or truncated:
                success = info['is_success']
                episode_rewards.append(episode_reward)
                episode_distances.append(min_distance)
                episode_successes.append(success)
                
                print(f"Episode {episode+1}: Reward={episode_reward:.4f}, "
                      f"Min Distance={min_distance:.4f}m, Success={success}")
                break
    
    print("-" * 70)
    print(f"Average Reward: {np.mean(episode_rewards):.4f}")
    print(f"Average Min Distance: {np.mean(episode_distances):.4f}m")
    print(f"Success Rate: {np.mean(episode_successes)*100:.1f}%")
    print("="*70)


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def main():
    """Main training function with CLI."""
    
    parser = argparse.ArgumentParser(
        description="Train PPO agent on soft tentacle environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rl/train_ppo.py                              # Default: 100k steps
  python rl/train_ppo.py --timesteps 500000           # 500k steps
  python rl/train_ppo.py --learning-rate 1e-3         # Custom learning rate
  python rl/train_ppo.py --verbose 2                  # Detailed output
  python rl/train_ppo.py --no-test                    # Skip test after training
        """
    )
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps (default: 100000)')
    parser.add_argument('--save-freq', type=int, default=10000,
                       help='Save checkpoint every N steps (default: 10000)')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                       help='Verbosity level (default: 1)')
    
    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Steps per update (default: 2048)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Epochs per update (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda (default: 0.95)')
    
    # Model naming
    parser.add_argument('--model-name', type=str, default='ppo_tentacle',
                       help='Model name for saving (default: ppo_tentacle)')
    
    # Testing
    parser.add_argument('--no-test', action='store_true',
                       help='Skip testing after training')
    parser.add_argument('--test-episodes', type=int, default=3,
                       help='Number of test episodes (default: 3)')
    
    args = parser.parse_args()
    
    # Train
    try:
        model, env, elapsed_time = train_ppo(
            timesteps=args.timesteps,
            model_name=args.model_name,
            save_freq=args.save_freq,
            verbose=args.verbose,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        
        # Test trained model
        if not args.no_test:
            test_trained_model(model, env, num_episodes=args.test_episodes)
        
        env.close()
        
        print("\n✅ TRAINING COMPLETE AND SUCCESSFUL!")
        print(f"\nNext steps:")
        print(f"  1. Evaluate: python rl/evaluate.py --model results/checkpoints/{args.model_name}_final.zip")
        print(f"  2. Train SAC: python rl/train_sac.py")
        print(f"  3. Compare: both in results/")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
