import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from swift.plugin.orm import McqORM, VqaBleuReward, VqaBertReward

class TestMultiTaskORM(unittest.TestCase):

    def setUp(self):
        self.completions = ["Answer1", "Answer2", "Answer3"]
        self.solutions = ["<answer>Answer1</answer>", "Reference2", "<answer>Answer3</answer>"]
        self.tasks = ["mcq", "vqa", "cls"]

    def test_mcq_orm(self):
        orm = McqORM()
        # Should filter out 'vqa' (index 1), so index 1 should be None
        # Index 0 ('mcq') should be 1.0 (match)
        # Index 2 ('cls') should be 1.0 (match) because cls is treated like mcq
        rewards = orm(self.completions, self.solutions, task=self.tasks)
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)
        self.assertIsNone(rewards[1])
        self.assertEqual(rewards[2], 1.0)

    @patch('swift.plugin.orm.VqaBleuReward.__init__', return_value=None)
    @patch('jieba.cut')
    @patch('nltk.translate.bleu_score.sentence_bleu')
    def test_vqa_bleu_reward(self, mock_bleu, mock_cut, mock_init):
        # We need to manually set the instance because __init__ is mocked
        orm = VqaBleuReward()
        
        # Mock behavior
        mock_cut.side_effect = lambda x: list(x) # simple char filtering
        mock_bleu.return_value = 0.5
        
        rewards = orm(self.completions, self.solutions, task=self.tasks)
        
        self.assertEqual(len(rewards), 3)
        self.assertIsNone(rewards[0])
        self.assertEqual(rewards[1], 0.5)
        self.assertIsNone(rewards[2])

    @patch('swift.plugin.orm.VqaBertReward._load_model')
    def test_vqa_bert_reward(self, mock_load):
        orm = VqaBertReward(model_name_or_path="dummy")
        
        # Mocking internal components to avoid actual model run
        orm.tokenizer = MagicMock()
        orm.model = MagicMock()
        orm.meanpooling = MagicMock(return_value=MagicMock())
        
        # Mock internal calls
        with patch('torch.no_grad'), \
             patch('torch.nn.functional.normalize'), \
             patch('torch.nn.functional.cosine_similarity') as mock_cos:
             
            mock_cos.return_value.item.return_value = 0.8
            
            rewards = orm(self.completions, self.solutions, task=self.tasks)
            
            self.assertEqual(len(rewards), 3)
            self.assertIsNone(rewards[0])
            self.assertEqual(rewards[1], 0.8)
            self.assertIsNone(rewards[2])

if __name__ == '__main__':
    unittest.main()
