"""
LLM Response Collector for EmpatheticDialogues Dataset

This script collects real LLM responses from the EmpatheticDialogues dataset
to generate training data for the BERT-based service time predictor.

Output format is compatible with preprocess_customized_dataset.py:
- model_name: str
- prompt_id: str
- prompt_content: str
- response_length: int (output token count)

Features:
- Checkpoint support for resumable collection
- Progress tracking with tqdm
- Error handling and retry logic
- Configurable sampling parameters
- File logging with rotation
- Graceful shutdown with signal handlers
- Detailed per-emotion statistics
"""

import argparse
import json
import csv
import os
import sys
import signal
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from llm.dataset_loader import EmpatheticDialoguesLoader
from llm.prompt_builder import PromptBuilder
from llm.engine import LLMEngine


def setup_logging(log_file: Optional[str] = None, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging with both console and file handlers.

    Args:
        log_file: Path to log file. If None, only console logging.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # Console handler with concise format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler with detailed format and rotation
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


class LLMResponseCollector:
    """
    Collects LLM responses from EmpatheticDialogues dataset.

    Iterates through conversations grouped by emotion, builds prompts,
    generates responses using the LLM, and saves results to CSV.
    """

    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        output_path: str,
        checkpoint_path: str,
        log_file: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        include_system_prompt: bool = True,
        include_emotion_hint: bool = False,
        max_conversation_turns: int = 2,
        log_level: str = "INFO"
    ):
        """
        Initialize the collector.

        Args:
            model_name: HuggingFace model identifier
            dataset_path: Path to EmpatheticDialogues dataset directory
            output_path: Path to output CSV file
            checkpoint_path: Path to checkpoint JSON file
            log_file: Path to log file (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            include_system_prompt: Include system prompt in prompts
            include_emotion_hint: Include emotion hints in prompts
            max_conversation_turns: Max conversation turns to include
            log_level: Logging level
        """
        # Setup logging first
        self.logger = setup_logging(log_file, log_level)
        self.log_file = log_file

        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_conversation_turns = max_conversation_turns

        # Signal handling for graceful shutdown
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Initialize components
        self.logger.info("=" * 60)
        self.logger.info("Initializing LLM Response Collector")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Dataset path: {dataset_path}")
        self.logger.info(f"Output path: {output_path}")
        self.logger.info(f"Checkpoint path: {checkpoint_path}")
        self.logger.info(f"Log file: {log_file or 'None (console only)'}")
        self.logger.info(f"Max new tokens: {max_new_tokens}")
        self.logger.info(f"Temperature: {temperature}")

        # 1. Load dataset
        self.logger.info(f"Loading EmpatheticDialogues from: {dataset_path}")
        self.dataset_loader = EmpatheticDialoguesLoader(dataset_dir=dataset_path)
        success = self.dataset_loader.load(splits=["train", "valid"])
        if not success:
            raise RuntimeError(f"Failed to load dataset from: {dataset_path}")

        # Log dataset statistics
        emotion_stats = self.dataset_loader.get_emotion_statistics()
        total_convs = sum(emotion_stats.values())
        self.logger.info(f"Dataset loaded: {total_convs} conversations, {len(emotion_stats)} emotions")

        # 2. Initialize prompt builder
        self.prompt_builder = PromptBuilder(
            include_system_prompt=include_system_prompt,
            include_emotion_hint=include_emotion_hint
        )

        # 3. Initialize LLM engine
        self.logger.info(f"Loading LLM model: {model_name}")
        self.llm_engine = LLMEngine()
        success = self.llm_engine.load_model(model_name=model_name)
        if not success:
            raise RuntimeError(f"Failed to load model: {model_name}")

        # 4. Initialize stats
        self.processed_ids: Set[str] = set()
        self.stats = {
            "total_processed": 0,
            "total_errors": 0,
            "start_time": None,
            "last_save_time": None,
            "session_start_time": None,
            "session_processed": 0,
            "session_errors": 0,
            "errors_by_type": {},
            "emotion_processed": {},
            "emotion_errors": {}
        }

        # 5. Load checkpoint if exists
        self.load_checkpoint()

        # 6. Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        self.logger.info("LLM Response Collector initialized successfully")
        self.logger.info("=" * 60)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        sig_name = signal.Signals(signum).name
        self.logger.warning(f"Received {sig_name} signal, initiating graceful shutdown...")
        self._shutdown_requested = True

    def _extract_user_context(self, conversation_data: dict, max_turns: int = 2) -> str:
        """
        Extract user context from a conversation dictionary.

        Args:
            conversation_data: Dictionary with 'conv_id' and 'utterances' keys,
                             where utterances is a list of dicts from DataFrame records
            max_turns: Maximum number of turns to include

        Returns:
            User context string
        """
        utterances = conversation_data.get('utterances', [])

        # Sort by utterance_idx to get chronological order
        sorted_utterances = sorted(utterances, key=lambda x: x.get('utterance_idx', 0))

        # Get user utterances - determine user speaker_id from first utterance
        if not sorted_utterances:
            return ''

        # In EmpatheticDialogues, the first utterance is from the user/speaker
        user_speaker_id = sorted_utterances[0].get('speaker_idx')

        user_utterances = [
            u.get('utterance', '')
            for u in sorted_utterances
            if u.get('speaker_idx') == user_speaker_id
        ]

        # Take up to max_turns user utterances
        selected = user_utterances[:max_turns]

        # Clean up text (replace _comma_ with actual comma)
        cleaned = [str(u).replace('_comma_', ',') for u in selected if u]

        # Join with newlines
        return '\n'.join(cleaned)

    def _get_all_conversations(self) -> List[Tuple[str, int, any]]:
        """
        Get all conversations as a flat list for iteration.

        Returns:
            List of (emotion, conv_idx, conversation_df) tuples
        """
        all_conversations = []

        for emotion in self.dataset_loader.get_available_emotions():
            conversations = self.dataset_loader.conversations_by_emotion.get(emotion, [])
            for conv_idx, conv_df in enumerate(conversations):
                all_conversations.append((emotion, conv_idx, conv_df))

        return all_conversations

    def _generate_prompt_id(self, emotion: str, conv_idx: int) -> str:
        """Generate unique prompt ID."""
        return f"{emotion}_{conv_idx}"

    def _log_progress_summary(self, collected: int, errors: int, elapsed_seconds: float, current_emotion: str):
        """Log detailed progress summary."""
        rate = collected / elapsed_seconds if elapsed_seconds > 0 else 0
        self.logger.info(
            f"Progress: {collected} collected, {errors} errors | "
            f"Rate: {rate:.2f} samples/sec | "
            f"Current emotion: {current_emotion}"
        )

    def _log_emotion_stats(self):
        """Log per-emotion statistics."""
        self.logger.info("Per-emotion statistics:")
        for emotion in sorted(self.stats.get("emotion_processed", {}).keys()):
            processed = self.stats["emotion_processed"].get(emotion, 0)
            errors = self.stats["emotion_errors"].get(emotion, 0)
            self.logger.info(f"  {emotion}: {processed} processed, {errors} errors")

    def collect_all(
        self,
        max_samples: Optional[int] = None,
        batch_save_interval: int = 100,
        progress_log_interval: int = 500
    ) -> int:
        """
        Collect LLM responses for all conversations.

        Args:
            max_samples: Maximum samples to collect (None for all)
            batch_save_interval: Save checkpoint every N samples
            progress_log_interval: Log detailed progress every N samples

        Returns:
            Number of samples collected
        """
        session_start = datetime.now()
        self.stats["session_start_time"] = session_start.isoformat()
        self.stats["session_processed"] = 0
        self.stats["session_errors"] = 0

        if self.stats["start_time"] is None:
            self.stats["start_time"] = session_start.isoformat()

        # Get all conversations
        all_conversations = self._get_all_conversations()
        total_available = len(all_conversations)

        self.logger.info(f"Total conversations available: {total_available}")
        self.logger.info(f"Already processed (from checkpoint): {len(self.processed_ids)}")
        self.logger.info(f"Cumulative errors (from checkpoint): {self.stats['total_errors']}")

        # Limit samples if specified
        if max_samples is not None:
            target_count = min(max_samples, total_available)
        else:
            target_count = total_available

        remaining = target_count - len(self.processed_ids)
        self.logger.info(f"Target samples: {target_count}")
        self.logger.info(f"Remaining to collect: {remaining}")

        if remaining <= 0:
            self.logger.info("Target already reached, nothing to collect")
            return 0

        # Prepare CSV file (append mode if exists)
        csv_exists = os.path.exists(self.output_path)
        csv_file = open(self.output_path, 'a', newline='', encoding='utf-8')
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=['model_name', 'prompt_id', 'prompt_content', 'response_length', 'emotion', 'execution_time']
        )

        if not csv_exists:
            csv_writer.writeheader()

        # Collection loop with progress bar
        collected = 0
        errors = 0
        current_emotion = ""
        skipped = 0

        try:
            with tqdm(total=target_count, desc="Collecting responses", unit="sample") as pbar:
                # Update progress bar for already processed
                pbar.update(len(self.processed_ids))

                for emotion, conv_idx, conv_df in all_conversations:
                    # Check for shutdown signal
                    if self._shutdown_requested:
                        self.logger.warning("Shutdown requested, stopping collection...")
                        break

                    # Check if we've reached target
                    if len(self.processed_ids) >= target_count:
                        self.logger.info(f"Target count {target_count} reached")
                        break

                    # Generate prompt ID
                    prompt_id = self._generate_prompt_id(emotion, conv_idx)

                    # Skip if already processed
                    if prompt_id in self.processed_ids:
                        skipped += 1
                        continue

                    current_emotion = emotion

                    try:
                        # Extract user context
                        user_context = self._extract_user_context(
                            conv_df,
                            max_turns=self.max_conversation_turns
                        )

                        if not user_context.strip():
                            self.logger.debug(f"Empty user context for {prompt_id}, skipping")
                            skipped += 1
                            continue

                        # Build prompt
                        prompt = self.prompt_builder.build_prompt(
                            user_context=user_context,
                            emotion=emotion,
                            arousal=None
                        )

                        # Generate response
                        result = self.llm_engine.generate(
                            prompt=prompt,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            do_sample=self.temperature > 0
                        )

                        if result["error"]:
                            error_type = type(result["error"]).__name__ if not isinstance(result["error"], str) else "GenerationError"
                            self.logger.warning(f"Generation error for {prompt_id}: {result['error']}")
                            self.logger.debug(f"Error type: {error_type}")

                            errors += 1
                            self.stats["total_errors"] += 1
                            self.stats["session_errors"] += 1
                            self.stats["errors_by_type"][error_type] = self.stats["errors_by_type"].get(error_type, 0) + 1
                            self.stats["emotion_errors"][emotion] = self.stats["emotion_errors"].get(emotion, 0) + 1
                            continue

                        # Save record
                        record = {
                            'model_name': self.model_name.split('/')[-1],  # Short name
                            'prompt_id': prompt_id,
                            'prompt_content': prompt,
                            'response_length': result['output_token_length'],
                            'emotion': emotion,
                            'execution_time': round(result['execution_time'], 4)
                        }

                        csv_writer.writerow(record)
                        csv_file.flush()  # Ensure data is written

                        # Update tracking
                        self.processed_ids.add(prompt_id)
                        collected += 1
                        self.stats["total_processed"] += 1
                        self.stats["session_processed"] += 1
                        self.stats["emotion_processed"][emotion] = self.stats["emotion_processed"].get(emotion, 0) + 1

                        pbar.update(1)
                        pbar.set_postfix({
                            'err': errors,
                            'tokens': result['output_token_length'],
                            'emotion': emotion[:8]
                        })

                        # Periodic checkpoint
                        if collected % batch_save_interval == 0:
                            self.save_checkpoint()
                            self.logger.debug(f"Checkpoint saved at {collected} samples")

                        # Periodic detailed progress log
                        if collected % progress_log_interval == 0:
                            elapsed = (datetime.now() - session_start).total_seconds()
                            self._log_progress_summary(collected, errors, elapsed, current_emotion)

                    except KeyboardInterrupt:
                        self.logger.warning("KeyboardInterrupt received, saving checkpoint...")
                        break
                    except Exception as e:
                        error_type = type(e).__name__
                        self.logger.error(f"Error processing {prompt_id}: {error_type}: {e}")
                        self.logger.debug(f"Full exception:", exc_info=True)

                        errors += 1
                        self.stats["total_errors"] += 1
                        self.stats["session_errors"] += 1
                        self.stats["errors_by_type"][error_type] = self.stats["errors_by_type"].get(error_type, 0) + 1
                        self.stats["emotion_errors"][emotion] = self.stats["emotion_errors"].get(emotion, 0) + 1
                        continue

        finally:
            csv_file.close()
            self.save_checkpoint()

            # Log final session summary
            session_duration = datetime.now() - session_start
            self.logger.info("=" * 60)
            self.logger.info("Session Summary")
            self.logger.info("=" * 60)
            self.logger.info(f"Session duration: {session_duration}")
            self.logger.info(f"Session collected: {collected}")
            self.logger.info(f"Session errors: {errors}")
            self.logger.info(f"Session skipped: {skipped}")
            if collected > 0:
                rate = collected / session_duration.total_seconds()
                self.logger.info(f"Average rate: {rate:.2f} samples/sec ({1/rate:.2f} sec/sample)")

            if self._shutdown_requested:
                self.logger.info("Collection stopped due to shutdown signal")

            self.logger.info("=" * 60)

        self.logger.info(f"Collection complete: {collected} samples collected, {errors} errors")
        return collected

    def save_checkpoint(self):
        """Save checkpoint to disk."""
        self.stats["last_save_time"] = datetime.now().isoformat()

        checkpoint_data = {
            "processed_ids": list(self.processed_ids),
            "stats": self.stats,
            "config": {
                "model_name": self.model_name,
                "output_path": self.output_path,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature
            }
        }

        try:
            os.makedirs(os.path.dirname(self.checkpoint_path) if os.path.dirname(self.checkpoint_path) else '.', exist_ok=True)
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            self.logger.debug(f"Checkpoint saved: {len(self.processed_ids)} processed IDs")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self):
        """Load checkpoint from disk if exists."""
        if not os.path.exists(self.checkpoint_path):
            self.logger.info("No checkpoint found, starting fresh")
            return

        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            self.processed_ids = set(checkpoint_data.get("processed_ids", []))

            # Load stats but preserve structure
            loaded_stats = checkpoint_data.get("stats", {})
            for key in loaded_stats:
                if key in self.stats:
                    self.stats[key] = loaded_stats[key]

            self.logger.info(f"Checkpoint loaded successfully:")
            self.logger.info(f"  - Processed IDs: {len(self.processed_ids)}")
            self.logger.info(f"  - Total processed: {self.stats.get('total_processed', 0)}")
            self.logger.info(f"  - Total errors: {self.stats.get('total_errors', 0)}")
            self.logger.info(f"  - Last save: {self.stats.get('last_save_time', 'N/A')}")

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            self.processed_ids = set()

    def reset_error_stats(self):
        """Reset error statistics (but keep processed IDs)."""
        self.logger.info("Resetting error statistics...")
        self.stats["total_errors"] = 0
        self.stats["session_errors"] = 0
        self.stats["errors_by_type"] = {}
        self.stats["emotion_errors"] = {}
        self.save_checkpoint()
        self.logger.info("Error statistics reset complete")

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return {
            **self.stats,
            "num_processed": len(self.processed_ids),
            "available_emotions": self.dataset_loader.get_available_emotions(),
            "emotion_stats": self.dataset_loader.get_emotion_statistics()
        }

    def print_detailed_stats(self):
        """Print detailed statistics to logger."""
        stats = self.get_stats()

        self.logger.info("=" * 60)
        self.logger.info("Detailed Statistics")
        self.logger.info("=" * 60)

        self.logger.info(f"Total processed: {stats['num_processed']}")
        self.logger.info(f"Total errors: {stats['total_errors']}")

        if stats.get('errors_by_type'):
            self.logger.info("Errors by type:")
            for error_type, count in sorted(stats['errors_by_type'].items(), key=lambda x: -x[1]):
                self.logger.info(f"  {error_type}: {count}")

        if stats.get('emotion_processed'):
            self.logger.info("Processed by emotion:")
            for emotion in sorted(stats['emotion_processed'].keys()):
                processed = stats['emotion_processed'].get(emotion, 0)
                total = stats['emotion_stats'].get(emotion, 0)
                pct = (processed / total * 100) if total > 0 else 0
                self.logger.info(f"  {emotion}: {processed}/{total} ({pct:.1f}%)")

        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Collect LLM responses from EmpatheticDialogues dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic collection
  python collect_llm_responses.py --max_samples 1000

  # Resume from checkpoint with logging
  python collect_llm_responses.py --max_samples 5000 --log_file ./logs/collection.log

  # Reset error statistics and continue
  python collect_llm_responses.py --reset_errors --max_samples 5000

  # Show current statistics
  python collect_llm_responses.py --show_stats
        """
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='lmsys/vicuna-13b-v1.3',
        help='HuggingFace model identifier'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='./dataset',
        help='Path to EmpatheticDialogues dataset directory'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./predictor/training/data/empathetic_responses.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./predictor/training/data/checkpoint.json',
        help='Path to checkpoint JSON file'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Path to log file (enables file logging with rotation)'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to collect (None for all)'
    )
    parser.add_argument(
        '--batch_save_interval',
        type=int,
        default=100,
        help='Save checkpoint every N samples'
    )
    parser.add_argument(
        '--progress_log_interval',
        type=int,
        default=500,
        help='Log detailed progress every N samples'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate per response'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (0.0 for greedy decoding)'
    )
    parser.add_argument(
        '--max_conversation_turns',
        type=int,
        default=2,
        help='Maximum conversation turns to include in context'
    )
    parser.add_argument(
        '--include_system_prompt',
        action='store_true',
        default=True,
        help='Include system prompt in prompts'
    )
    parser.add_argument(
        '--include_emotion_hint',
        action='store_true',
        default=False,
        help='Include emotion hints in prompts'
    )
    parser.add_argument(
        '--reset_errors',
        action='store_true',
        help='Reset error statistics before running (keeps processed IDs)'
    )
    parser.add_argument(
        '--show_stats',
        action='store_true',
        help='Show detailed statistics and exit'
    )

    args = parser.parse_args()

    # Create collector
    collector = LLMResponseCollector(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint_path,
        log_file=args.log_file,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        include_system_prompt=args.include_system_prompt,
        include_emotion_hint=args.include_emotion_hint,
        max_conversation_turns=args.max_conversation_turns,
        log_level=args.log_level
    )

    # Handle special modes
    if args.show_stats:
        collector.print_detailed_stats()
        return

    if args.reset_errors:
        collector.reset_error_stats()

    # Run collection
    collected = collector.collect_all(
        max_samples=args.max_samples,
        batch_save_interval=args.batch_save_interval,
        progress_log_interval=args.progress_log_interval
    )

    # Print final stats
    collector.print_detailed_stats()


if __name__ == '__main__':
    main()
