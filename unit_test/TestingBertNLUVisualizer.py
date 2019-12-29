#!/usr/bin/python

# import modules used here -- sys is a very standard one
import sys
from bertviz.pytorch_transformers_attn import BertModel, BertTokenizer
from bertviz.head_view import show
from transformers.tokenization_bert_nlu import BertNLUTokenizer
from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertForPreTraining as BertModel
from transformers.modeling_bert_nlu import BertNLUForPreTraining as BertNLUModel
from bertviz.head_view import show

def call_html():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))

def test_bert():
	model_type = 'bert'
	model_version = 'bert-base-cased'
	do_lower_case = False
	model = BertModel.from_pretrained(model_version, output_attentions=True)
	tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
	sentence_a = "The cat sat on the mat"
	sentence_b = "The cat lay on the rug"
	call_html()
	show(model, model_type, tokenizer, sentence_a, sentence_b)

def test_bert_nlu():
	model_type = 'bert-nlu'

	model_path = '../transformers/output/pretrain/bert-nlu-gpu/checkpoint-50000'
	model_version = 'bert-base-cased'

	do_lower_case = False
	model = BertNLUModel.from_pretrained(model_path, output_attentions=True)
	tokenizer = BertNLUTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
	sentence_a = "The cat sat on the mat"
	sentence_b = "The cat lay on the rug"
	call_html()
	show(model, model_type, tokenizer, sentence_a, sentence_b=sentence_b)
# Gather our code in a main() function
def main():
	test_bert()


  # Command line args are in sys.argv[1], sys.argv[2] ..
  # sys.argv[0] is the script name itself and can be ignored

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()