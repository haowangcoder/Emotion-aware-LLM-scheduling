"""
EmpatheticDialogues Dataset Loader

Loads and parses the EmpatheticDialogues dataset for generating
empathetic conversation prompts based on emotions.

Dataset structure:
- conv_id: Conversation identifier
- utterance_idx: Index of utterance in conversation
- context: Emotion label (32 categories)
- prompt: Context shown to listener
- speaker_idx: Participant identifier (two unique IDs per conversation)
- utterance: The actual utterance text
- selfeval: Self-evaluation scores
- tags: Additional tags
"""

import pandas as pd
import csv
import os
import random
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EmpatheticDialoguesLoader:
    """
    Loader for EmpatheticDialogues dataset.

    Provides methods to:
    - Load conversations from train/test/valid splits
    - Sample conversations by emotion
    - Extract user context and conversational history
    """

    def __init__(self, dataset_dir: str = "./dataset"):
        """
        Initialize dataset loader.

        Args:
            dataset_dir: Path to directory containing train.csv, test.csv, valid.csv
        """
        self.dataset_dir = dataset_dir
        self.train_df = None
        self.test_df = None
        self.valid_df = None
        self.all_df = None
        self.conversations_by_emotion = {}
        self._loaded = False

    def load(self, splits: List[str] = ["train", "valid", "test"]) -> bool:
        """
        Load dataset from CSV files.

        Args:
            splits: List of splits to load ("train", "test", "valid")

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            dfs = []

            if "train" in splits:
                train_path = os.path.join(self.dataset_dir, "train.csv")
                if os.path.exists(train_path):
                    logger.info(f"Loading train split from {train_path}")
                    self.train_df = self._read_csv_robust(train_path)
                    dfs.append(self.train_df)
                    logger.info(f"Loaded {len(self.train_df)} train utterances")

            if "valid" in splits:
                valid_path = os.path.join(self.dataset_dir, "valid.csv")
                if os.path.exists(valid_path):
                    logger.info(f"Loading valid split from {valid_path}")
                    self.valid_df = self._read_csv_robust(valid_path)
                    dfs.append(self.valid_df)
                    logger.info(f"Loaded {len(self.valid_df)} valid utterances")

            if "test" in splits:
                test_path = os.path.join(self.dataset_dir, "test.csv")
                if os.path.exists(test_path):
                    logger.info(f"Loading test split from {test_path}")
                    self.test_df = self._read_csv_robust(test_path)
                    dfs.append(self.test_df)
                    logger.info(f"Loaded {len(self.test_df)} test utterances")

            if not dfs:
                logger.error(f"No data files found in {self.dataset_dir}")
                return False

            # Combine all splits
            self.all_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Total utterances loaded: {len(self.all_df)}")

            # Parse conversations and index by emotion
            self._index_conversations_by_emotion()

            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False

    def _read_csv_robust(self, path: str) -> pd.DataFrame:
        """
        Read EmpatheticDialogues CSV robustly.

        The dataset contains occasional stray double quotes within fields which
        break the default fast (C) CSV parser. We force the Python engine and
        disable quote handling so quotes are treated as literal characters.

        Args:
            path: CSV file path

        Returns:
            pandas.DataFrame
        """
        try:
            # Use python engine and treat quotes as plain characters.
            return pd.read_csv(path, engine="python", quoting=csv.QUOTE_NONE)
        except Exception as e:
            logger.warning(
                f"Standard robust read failed for {path}: {e}. Trying manual parser fallback."
            )

        # Manual parser: split only on the first 7 commas per line and ignore extras.
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                header_line = f.readline()
                if not header_line:
                    raise ValueError("Empty file")
                columns = header_line.strip().split(",")
                if len(columns) < 8:
                    raise ValueError(
                        f"Unexpected header format with {len(columns)} columns: {columns}"
                    )

                rows = []
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    parts = line.split(",", 7)
                    if len(parts) < 8:
                        # Skip lines that are too short to contain all columns
                        continue
                    # Truncate extras beyond 8 columns (caused by corrupted tails)
                    parts = parts[:8]
                    rows.append(parts)

            df = pd.DataFrame(rows, columns=columns[:8])
            # Coerce expected numeric columns when possible
            for col in ("utterance_idx", "speaker_idx"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                    # Fill NaNs (from coercion) conservatively with -1
                    df[col] = df[col].fillna(-1).astype(int)
            return df
        except Exception as e:
            logger.warning(
                f"Manual parser fallback failed for {path}: {e}. Skipping malformed lines via pandas."
            )
            # Last resort: skip malformed lines to avoid hard failures.
            return pd.read_csv(
                path, engine="python", quoting=csv.QUOTE_NONE, on_bad_lines="skip"
            )

    def _index_conversations_by_emotion(self):
        """Index conversations by emotion for fast sampling."""
        logger.info("Indexing conversations by emotion...")

        # Group by conversation ID
        grouped = self.all_df.groupby('conv_id')

        # Build index: emotion -> list of conversations
        self.conversations_by_emotion = {}

        for conv_id, conv_df in grouped:
            # Get emotion from first utterance
            emotion = conv_df.iloc[0]['context']

            if emotion not in self.conversations_by_emotion:
                self.conversations_by_emotion[emotion] = []

            # Store conversation data
            self.conversations_by_emotion[emotion].append({
                'conv_id': conv_id,
                'utterances': conv_df.to_dict('records')
            })

        # Log statistics
        logger.info(f"Indexed {len(grouped)} conversations")
        logger.info(f"Found {len(self.conversations_by_emotion)} unique emotions")

        for emotion in sorted(self.conversations_by_emotion.keys()):
            count = len(self.conversations_by_emotion[emotion])
            logger.debug(f"  {emotion}: {count} conversations")

    def get_conversation_by_emotion(
        self,
        emotion: str,
        max_turns: int = 3,
        conversation_index: Optional[int] = None
    ) -> Optional[Tuple[Dict, int]]:
        """
        Get a conversation for the given emotion.

        Args:
            emotion: Emotion label (e.g., "excited", "sad")
            max_turns: Maximum number of conversation turns to include
            conversation_index: Specific conversation index to retrieve.
                               If None, randomly samples a conversation.

        Returns:
            Tuple of (conversation_dict, conversation_index) where conversation_dict contains:
                - emotion: Emotion label
                - conv_id: Conversation ID
                - user_utterances: List of user (speaker_idx=1) utterances
                - assistant_utterances: List of assistant (speaker_idx=0) responses
                - full_context: Combined context string
            Returns (None, -1) on error.
        """
        if not self._loaded:
            logger.error("Dataset not loaded. Call load() first.")
            return None, -1

        # Normalize emotion (lowercase, strip)
        emotion = emotion.lower().strip()

        # Check if emotion exists in dataset
        if emotion not in self.conversations_by_emotion:
            logger.warning(f"Emotion '{emotion}' not found in dataset")
            # Try to find close match
            available = list(self.conversations_by_emotion.keys())
            logger.warning(f"Available emotions: {', '.join(sorted(available)[:10])}...")
            return None, -1

        # Get conversations list for this emotion
        conversations = self.conversations_by_emotion[emotion]

        # Select conversation by index or random
        if conversation_index is not None:
            # Use specific index (with bounds checking)
            if conversation_index < 0 or conversation_index >= len(conversations):
                logger.warning(
                    f"Invalid conversation_index {conversation_index} for emotion '{emotion}' "
                    f"(valid range: 0-{len(conversations)-1})"
                )
                return None, -1
            conv = conversations[conversation_index]
            selected_index = conversation_index
        else:
            # Random selection
            selected_index = random.randint(0, len(conversations) - 1)
            conv = conversations[selected_index]

        # Extract user and assistant utterances
        user_utterances = []
        assistant_utterances = []

        utterances = conv['utterances'][:max_turns * 2]  # Limit total turns

        if not utterances:
            return None

        # Determine participant IDs for user and assistant within this conversation.
        # In EmpatheticDialogues, the first utterance (utterance_idx == 1) comes from
        # the user/speaker describing the situation; the second is the listener/assistant.
        user_speaker_id = utterances[0].get('speaker_idx')
        assistant_speaker_id = None
        if len(utterances) > 1:
            assistant_speaker_id = utterances[1].get('speaker_idx')

        for utt in utterances:
            speaker_idx = utt.get('speaker_idx')
            text = self._clean_utterance(utt.get('utterance'))

            if speaker_idx == user_speaker_id:
                user_utterances.append(text)
            elif assistant_speaker_id is None or speaker_idx == assistant_speaker_id:
                assistant_utterances.append(text)

        # Build full context (only user side for prompt)
        full_context = "\n".join(user_utterances)

        conversation_dict = {
            'emotion': emotion,
            'conv_id': conv['conv_id'],
            'user_utterances': user_utterances,
            'assistant_utterances': assistant_utterances,
            'full_context': full_context
        }

        return conversation_dict, selected_index

    def get_user_context_by_emotion(
        self,
        emotion: str,
        max_turns: int = 2,
        conversation_index: Optional[int] = None
    ) -> Optional[Tuple[str, int]]:
        """
        Get user context (speaker_idx=1 utterances) for an emotion.

        This is used to build prompts - we take the user's emotional
        expression and ask the model to generate an empathetic response.

        Args:
            emotion: Emotion label
            max_turns: Maximum user turns to include
            conversation_index: Specific conversation index to retrieve.
                               If None, randomly samples a conversation.

        Returns:
            Tuple of (user_context_string, conversation_index), or (None, -1) if not found
        """
        conv, selected_index = self.get_conversation_by_emotion(
            emotion,
            max_turns=max_turns,
            conversation_index=conversation_index
        )

        if conv is None:
            return None, -1

        # Return user side and the index
        return conv['full_context'], selected_index

    def _clean_utterance(self, text: str) -> str:
        """
        Clean utterance text.

        The dataset uses _comma_ to represent commas in text.
        We need to restore these.
        """
        if pd.isna(text):
            return ""

        text = str(text)
        # Replace special encodings
        text = text.replace("_comma_", ",")
        text = text.strip()

        return text

    def get_available_emotions(self) -> List[str]:
        """Get list of all emotions in the dataset."""
        if not self._loaded:
            return []
        return sorted(list(self.conversations_by_emotion.keys()))

    def get_emotion_statistics(self) -> Dict[str, int]:
        """Get count of conversations per emotion."""
        if not self._loaded:
            return {}

        return {
            emotion: len(convs)
            for emotion, convs in self.conversations_by_emotion.items()
        }

    def is_loaded(self) -> bool:
        """Check if dataset is loaded."""
        return self._loaded


def test_dataset_loader():
    """Test dataset loader functionality."""
    print(f"\n{'='*60}")
    print("Testing EmpatheticDialogues Dataset Loader")
    print(f"{'='*60}\n")

    # Initialize and load
    loader = EmpatheticDialoguesLoader(dataset_dir="./dataset")

    print("Loading dataset...")
    success = loader.load(splits=["train", "valid"])

    if not success:
        print("Failed to load dataset!")
        return

    print(f"\nDataset loaded successfully!")

    # Show available emotions
    emotions = loader.get_available_emotions()
    print(f"\nFound {len(emotions)} emotions:")
    print(", ".join(emotions))

    # Show statistics
    print(f"\nEmotion statistics:")
    stats = loader.get_emotion_statistics()
    for emotion in sorted(stats.keys())[:10]:  # Show first 10
        print(f"  {emotion}: {stats[emotion]} conversations")

    # Test sampling
    print(f"\n{'='*60}")
    print("Testing conversation sampling:")
    print(f"{'='*60}\n")

    test_emotions = ["excited", "sad", "anxious", "grateful"]

    for emotion in test_emotions:
        print(f"\nEmotion: {emotion}")
        print("-" * 40)

        context, conv_idx = loader.get_user_context_by_emotion(emotion, max_turns=2)

        if context:
            print(f"User context (index {conv_idx}):\n{context}")
        else:
            print(f"No conversation found for emotion: {emotion}")

        # Get full conversation
        conv, conv_idx = loader.get_conversation_by_emotion(emotion, max_turns=3)
        if conv:
            print(f"\nConversation ID: {conv['conv_id']} (index {conv_idx})")
            print(f"User utterances: {len(conv['user_utterances'])}")
            print(f"Assistant utterances: {len(conv['assistant_utterances'])}")

        print()


if __name__ == "__main__":
    # Set logging level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_dataset_loader()
