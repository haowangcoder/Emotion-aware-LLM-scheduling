"""Job representation for emotion-aware LLM scheduling."""


class Job:
    def __init__(self, job_id, execution_duration, arrival_time, predicted_execution_duration=None, status='CREATED',
                 emotion_label=None, arousal=None, emotion_class=None, valence=None, valence_class=None):
        self.job_id = job_id
        self.status = status
        self.arrival_time = arrival_time  # timestamp that the job arrives
        self.execution_duration = execution_duration
        self.waiting_duration = None
        self.completion_time = None  # timestamp that the job is finished
        self.predicted_execution_duration = predicted_execution_duration
        self.curr_waiting_time = 0  # used to avoid starvation

        # Emotion-aware fields
        self.emotion_label = emotion_label  # emotion category (e.g., 'excited', 'sad')
        self.arousal = arousal  # arousal value in [-1, 1] range
        self.emotion_class = emotion_class  # categorical label ('high', 'medium', 'low')
        self.valence = valence  # discrete valence (-0.8, 0, 0.8)
        self.valence_class = valence_class  # 'negative', 'neutral', 'positive'

        # LLM inference fields
        self.response_text = None  # Generated response from LLM
        self.output_token_length = None  # Number of tokens in response
        self.conversation_context = None  # Original prompt/context from dataset
        self.conversation_index = None  # Index of conversation in dataset (for reproducibility)
        self.actual_execution_duration = None  # Real LLM inference time (measured)
        self.cached = False  # Whether response was retrieved from cache
        self.error_msg = None  # Error message if generation failed
        self.fallback_used = False  # Whether CPU fallback was used during generation
        self.model_name = None  # Name of the model used for generation

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

    # Emotion-aware getter/setter methods
    def set_emotion_label(self, emotion_label):
        self.emotion_label = emotion_label

    def get_emotion_label(self):
        return self.emotion_label

    def set_arousal(self, arousal):
        self.arousal = arousal

    def get_arousal(self):
        return self.arousal

    def set_emotion_class(self, emotion_class):
        self.emotion_class = emotion_class

    def get_emotion_class(self):
        return self.emotion_class

    def set_valence(self, valence):
        self.valence = valence

    def get_valence(self):
        return self.valence

    def set_valence_class(self, valence_class):
        self.valence_class = valence_class

    def get_valence_class(self):
        return self.valence_class

    # LLM inference getter/setter methods
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

    def print_info(self):
        print('ID:', self.job_id, 'status:', self.status)
        print('Arrival time:', self.arrival_time,
              'Execution duration:', self.execution_duration,
              'Predicted Exec duration:', self.predicted_execution_duration,
              'Waiting duration:', self.waiting_duration,
              'Completion time:', self.completion_time)
        if self.emotion_label is not None:
            print('Emotion:', self.emotion_label,
                  'Arousal:', self.arousal,
                  'Emotion Class:', self.emotion_class,
                  'Valence:', self.valence,
                  'Valence Class:', self.valence_class)
        if self.response_text is not None:
            print('Response:', self.response_text[:100] + '...' if len(self.response_text) > 100 else self.response_text)
            print('Output tokens:', self.output_token_length,
                  'Actual exec time:', self.actual_execution_duration,
                  'Cached:', self.cached,
                  'Fallback:', self.fallback_used)
            if self.error_msg:
                print('Error:', self.error_msg)
