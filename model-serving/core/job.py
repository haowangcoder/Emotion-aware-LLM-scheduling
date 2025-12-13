"""
Job representation for Affect-Aware LLM Scheduling.

This module defines the Job data structure used throughout the scheduling
system. Jobs contain all information needed for affect-aware scheduling:
    - Basic scheduling fields (id, arrival time, status)
    - Emotion attributes (arousal, valence, Russell quadrant)
    - Service time predictions (BERT-based and actual)
    - Affect weight for scheduling priority
    - LLM inference fields (response, tokens, cache status)
"""


class Job:
    """
    Job representation for the scheduling system.

    Attributes:
        # === Basic Fields ===
        job_id: Unique identifier
        arrival_time: Timestamp when job arrived
        status: Current status (CREATED, PENDING, RUNNING, COMPLETED)
        execution_duration: Pre-computed/default execution duration

        # === Emotion Attributes ===
        emotion_label: Emotion category (e.g., 'sad', 'excited')
        arousal: Arousal value [-1, 1] from NRC-VAD-Lexicon
        valence: Valence value [-1, 1] from NRC-VAD-Lexicon
        russell_quadrant: Russell quadrant ('excited', 'calm', 'panic', 'depression')
        emotion_confidence: Confidence of emotion recognition [0, 1]

        # === Service Time ===
        predicted_service_time: BERT-predicted service time (seconds)
        actual_execution_duration: Measured LLM inference time

        # === Affect Weight ===
        affect_weight: Computed scheduling weight (w >= 1)
        urgency: Urgency score (u in [0, 1])

        # === LLM Inference ===
        response_text: Generated response from LLM
        output_token_length: Number of tokens in response
        conversation_context: Original prompt/context
    """

    def __init__(
        self,
        job_id,
        execution_duration,
        arrival_time,
        predicted_execution_duration=None,
        status='CREATED',
        # Emotion fields
        emotion_label=None,
        arousal=None,
        valence=None,
        russell_quadrant=None,
        emotion_confidence=1.0,
        # Service time fields
        predicted_service_time=None,
        # Affect weight fields
        affect_weight=1.0,
        urgency=0.0,
        # Legacy fields (kept for backward compatibility)
        emotion_class=None,
        valence_class=None,
    ):
        # === Basic Fields ===
        self.job_id = job_id
        self.status = status
        self.arrival_time = arrival_time
        self.execution_duration = execution_duration
        self.waiting_duration = None
        self.completion_time = None
        self.predicted_execution_duration = predicted_execution_duration
        self.curr_waiting_time = 0  # Used for starvation detection

        # === Emotion Attributes (NRC-VAD-Lexicon values) ===
        self.emotion_label = emotion_label
        self.arousal = arousal
        self.valence = valence
        self.russell_quadrant = russell_quadrant
        self.emotion_confidence = emotion_confidence

        # === Service Time (BERT Prediction) ===
        self.predicted_service_time = predicted_service_time

        # === Affect Weight (for AW-SSJF scheduling) ===
        self.affect_weight = affect_weight
        self.urgency = urgency

        # === Legacy Fields (backward compatibility) ===
        self.emotion_class = emotion_class
        self.valence_class = valence_class

        # === LLM Inference Fields ===
        self.response_text = None
        self.output_token_length = None
        self.conversation_context = None
        self.conversation_index = None
        self.actual_execution_duration = None
        self.cached = False
        self.error_msg = None
        self.fallback_used = False
        self.model_name = None

        # === Early Prompt Generation Fields ===
        self.prompt = None  # Pre-generated prompt for BERT prediction

    # === Basic Getters/Setters ===

    def get_job_id(self):
        return self.job_id

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def get_arrival_time(self):
        return self.arrival_time

    def set_waiting_duration(self, waiting_duration):
        self.waiting_duration = waiting_duration

    def get_waiting_duration(self):
        return self.waiting_duration

    def set_execution_duration(self, execution_duration):
        self.execution_duration = execution_duration

    def get_execution_duration(self):
        return self.execution_duration

    def set_completion_time(self, completion_time):
        self.completion_time = completion_time

    def get_completion_time(self):
        return self.completion_time

    def set_predicted_execution_duration(self, predicted_duration):
        self.predicted_execution_duration = predicted_duration

    def get_predicted_execution_duration(self):
        return self.predicted_execution_duration

    # === Emotion Getters/Setters ===

    def set_emotion_label(self, emotion_label):
        self.emotion_label = emotion_label

    def get_emotion_label(self):
        return self.emotion_label

    def set_arousal(self, arousal):
        self.arousal = arousal

    def get_arousal(self):
        return self.arousal

    def set_valence(self, valence):
        self.valence = valence

    def get_valence(self):
        return self.valence

    def set_russell_quadrant(self, quadrant):
        self.russell_quadrant = quadrant

    def get_russell_quadrant(self):
        return self.russell_quadrant

    def set_emotion_confidence(self, confidence):
        self.emotion_confidence = confidence

    def get_emotion_confidence(self):
        return self.emotion_confidence

    # === Service Time Getters/Setters ===

    def set_predicted_service_time(self, service_time):
        self.predicted_service_time = service_time

    def get_predicted_service_time(self):
        return self.predicted_service_time

    # === Affect Weight Getters/Setters ===

    def set_affect_weight(self, weight):
        self.affect_weight = weight

    def get_affect_weight(self):
        return self.affect_weight

    def set_urgency(self, urgency):
        self.urgency = urgency

    def get_urgency(self):
        return self.urgency

    # === Legacy Getters/Setters (backward compatibility) ===

    def set_emotion_class(self, emotion_class):
        self.emotion_class = emotion_class

    def get_emotion_class(self):
        return self.emotion_class

    def set_valence_class(self, valence_class):
        self.valence_class = valence_class

    def get_valence_class(self):
        return self.valence_class

    # === LLM Inference Getters/Setters ===

    def set_response_text(self, response_text):
        self.response_text = response_text

    def get_response_text(self):
        return self.response_text

    def set_output_token_length(self, output_token_length):
        self.output_token_length = output_token_length

    def get_output_token_length(self):
        return self.output_token_length

    def set_conversation_context(self, conversation_context):
        self.conversation_context = conversation_context

    def get_conversation_context(self):
        return self.conversation_context

    # === Early Prompt Generation Getters/Setters ===

    def set_prompt(self, prompt):
        """Set pre-generated prompt for BERT prediction."""
        self.prompt = prompt

    def get_prompt(self):
        """Get pre-generated prompt (if available)."""
        return self.prompt

    def set_actual_execution_duration(self, actual_execution_duration):
        self.actual_execution_duration = actual_execution_duration

    def get_actual_execution_duration(self):
        return self.actual_execution_duration

    def set_cached(self, cached):
        self.cached = cached

    def is_cached(self):
        return self.cached

    def set_error_msg(self, error_msg):
        self.error_msg = error_msg

    def get_error_msg(self):
        return self.error_msg

    def set_fallback_used(self, fallback_used):
        self.fallback_used = fallback_used

    def is_fallback_used(self):
        return self.fallback_used

    def set_model_name(self, model_name):
        self.model_name = model_name

    def get_model_name(self):
        return self.model_name

    # === Utility Methods ===

    def get_jct(self):
        """Get Job Completion Time (JCT = completion_time - arrival_time)."""
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None

    def to_dict(self):
        """Convert job to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'status': self.status,
            'arrival_time': self.arrival_time,
            'execution_duration': self.execution_duration,
            'waiting_duration': self.waiting_duration,
            'completion_time': self.completion_time,
            'emotion_label': self.emotion_label,
            'arousal': self.arousal,
            'valence': self.valence,
            'russell_quadrant': self.russell_quadrant,
            'emotion_confidence': self.emotion_confidence,
            'predicted_service_time': self.predicted_service_time,
            'affect_weight': self.affect_weight,
            'urgency': self.urgency,
            'output_token_length': self.output_token_length,
            'actual_execution_duration': self.actual_execution_duration,
            'cached': self.cached,
        }

    def print_info(self):
        """Print job information for debugging."""
        print('=' * 60)
        print(f'Job ID: {self.job_id} | Status: {self.status}')
        print(f'Arrival: {self.arrival_time:.2f} | '
              f'Exec Duration: {self.execution_duration:.2f} | '
              f'Waiting: {self.waiting_duration}')
        print(f'Completion: {self.completion_time}')

        if self.emotion_label is not None:
            print(f'Emotion: {self.emotion_label} | '
                  f'Arousal: {self.arousal:.3f} | '
                  f'Valence: {self.valence:.3f}')
            print(f'Quadrant: {self.russell_quadrant} | '
                  f'Confidence: {self.emotion_confidence:.2f}')

        if self.predicted_service_time is not None:
            print(f'Predicted Service Time: {self.predicted_service_time:.2f}s')

        print(f'Affect Weight: {self.affect_weight:.3f} | '
              f'Urgency: {self.urgency:.3f}')

        if self.response_text is not None:
            response_preview = self.response_text[:100] + '...' if len(self.response_text) > 100 else self.response_text
            print(f'Response: {response_preview}')
            print(f'Output Tokens: {self.output_token_length} | '
                  f'Actual Time: {self.actual_execution_duration} | '
                  f'Cached: {self.cached}')
            if self.error_msg:
                print(f'Error: {self.error_msg}')
        print('=' * 60)
