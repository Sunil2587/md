"""
Linguistic Micro-Pattern Features
Extracts vocabulary, pronoun, and syntactic patterns from text.
"""
import re
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import string


class LinguisticFeatureExtractor:
    """
    Extracts linguistic micro-patterns that correlate with mental health states.
    
    Features:
    - Vocabulary richness (TTR)
    - Pronoun usage patterns
    - Sentence length statistics
    - Lexical repetition
    - Cognitive complexity markers
    """
    
    # Pronoun categories
    FIRST_PERSON_SINGULAR = {'i', 'me', 'my', 'mine', 'myself'}
    FIRST_PERSON_PLURAL = {'we', 'us', 'our', 'ours', 'ourselves'}
    SECOND_PERSON = {'you', 'your', 'yours', 'yourself', 'yourselves'}
    THIRD_PERSON = {'he', 'she', 'it', 'they', 'him', 'her', 'them', 
                    'his', 'hers', 'its', 'their', 'theirs'}
    
    # Absolutist words (linked to depression/anxiety)
    ABSOLUTIST_WORDS = {
        'always', 'never', 'nothing', 'everything', 'completely',
        'totally', 'absolutely', 'constantly', 'entirely', 'whole'
    }
    
    # Cognitive process words
    COGNITIVE_WORDS = {
        'think', 'know', 'believe', 'understand', 'realize',
        'consider', 'remember', 'forget', 'wonder', 'suppose'
    }
    
    def __init__(self, min_words: int = 10):
        self.min_words = min_words
        
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract all linguistic features from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of feature names to values
        """
        # Preprocess
        words = self._tokenize(text)
        sentences = self._split_sentences(text)
        
        if len(words) < self.min_words:
            return self._empty_features()
            
        features = {}
        
        # Vocabulary features
        features.update(self._vocabulary_features(words))
        
        # Pronoun features
        features.update(self._pronoun_features(words))
        
        # Sentence features
        features.update(self._sentence_features(sentences, words))
        
        # Repetition features
        features.update(self._repetition_features(words, text))
        
        # Cognitive complexity
        features.update(self._cognitive_features(words))
        
        return features
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return [w for w in words if w and not w.isdigit()]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _vocabulary_features(self, words: List[str]) -> Dict[str, float]:
        """
        Vocabulary richness features.
        Lower TTR can indicate cognitive simplification in depression.
        """
        if not words:
            return {'ttr': 0.0, 'hapax_ratio': 0.0}
            
        unique_words = set(words)
        word_counts = Counter(words)
        
        # Type-Token Ratio
        ttr = len(unique_words) / len(words)
        
        # Hapax legomena ratio (words appearing only once)
        hapax = sum(1 for w, c in word_counts.items() if c == 1)
        hapax_ratio = hapax / len(words)
        
        return {
            'ttr': ttr,
            'hapax_ratio': hapax_ratio,
            'vocabulary_size': len(unique_words),
        }
    
    def _pronoun_features(self, words: List[str]) -> Dict[str, float]:
        """
        Pronoun usage patterns.
        Increased first-person singular usage linked to self-focused rumination.
        """
        total = len(words)
        if total == 0:
            return {
                'i_ratio': 0.0,
                'we_ratio': 0.0,
                'you_ratio': 0.0,
                'they_ratio': 0.0,
                'self_focus_score': 0.0
            }
            
        word_set = set(words)
        
        i_count = sum(1 for w in words if w in self.FIRST_PERSON_SINGULAR)
        we_count = sum(1 for w in words if w in self.FIRST_PERSON_PLURAL)
        you_count = sum(1 for w in words if w in self.SECOND_PERSON)
        they_count = sum(1 for w in words if w in self.THIRD_PERSON)
        
        total_pronouns = i_count + we_count + you_count + they_count
        
        # Self-focus: I / (I + we + they)
        social_pronouns = i_count + we_count + they_count
        self_focus = i_count / social_pronouns if social_pronouns > 0 else 0
        
        return {
            'i_ratio': i_count / total,
            'we_ratio': we_count / total,
            'you_ratio': you_count / total,
            'they_ratio': they_count / total,
            'pronoun_density': total_pronouns / total,
            'self_focus_score': self_focus
        }
    
    def _sentence_features(
        self, 
        sentences: List[str], 
        words: List[str]
    ) -> Dict[str, float]:
        """
        Sentence-level features.
        Reduced variance may indicate cognitive rigidity.
        """
        if not sentences:
            return {
                'mean_sentence_length': 0.0,
                'std_sentence_length': 0.0,
                'sentence_length_range': 0.0
            }
            
        lengths = [len(s.split()) for s in sentences]
        
        return {
            'mean_sentence_length': np.mean(lengths),
            'std_sentence_length': np.std(lengths),
            'sentence_length_range': max(lengths) - min(lengths),
            'num_sentences': len(sentences)
        }
    
    def _repetition_features(
        self, 
        words: List[str], 
        text: str
    ) -> Dict[str, float]:
        """
        Lexical repetition features.
        High repetition may indicate fixation or rumination.
        """
        if len(words) < 2:
            return {'repetition_ratio': 0.0, 'bigram_repetition': 0.0}
            
        word_counts = Counter(words)
        repeated = sum(c - 1 for c in word_counts.values() if c > 1)
        
        # Bigram repetition
        bigrams = list(zip(words[:-1], words[1:]))
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(c - 1 for c in bigram_counts.values() if c > 1)
        
        return {
            'repetition_ratio': repeated / len(words),
            'bigram_repetition': repeated_bigrams / len(bigrams) if bigrams else 0
        }
    
    def _cognitive_features(self, words: List[str]) -> Dict[str, float]:
        """
        Cognitive complexity and absolutist thinking.
        """
        total = len(words)
        if total == 0:
            return {
                'absolutist_ratio': 0.0,
                'cognitive_ratio': 0.0
            }
            
        absolutist = sum(1 for w in words if w in self.ABSOLUTIST_WORDS)
        cognitive = sum(1 for w in words if w in self.COGNITIVE_WORDS)
        
        return {
            'absolutist_ratio': absolutist / total,
            'cognitive_ratio': cognitive / total
        }
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dict when text is too short"""
        return {
            'ttr': 0.0,
            'hapax_ratio': 0.0,
            'vocabulary_size': 0.0,
            'i_ratio': 0.0,
            'we_ratio': 0.0,
            'you_ratio': 0.0,
            'they_ratio': 0.0,
            'pronoun_density': 0.0,
            'self_focus_score': 0.0,
            'mean_sentence_length': 0.0,
            'std_sentence_length': 0.0,
            'sentence_length_range': 0.0,
            'num_sentences': 0.0,
            'repetition_ratio': 0.0,
            'bigram_repetition': 0.0,
            'absolutist_ratio': 0.0,
            'cognitive_ratio': 0.0
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return list(self._empty_features().keys())
