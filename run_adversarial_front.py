import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
from transformers import AutoModel
from squad_adv_mod import *
from datasets import Dataset

NUM_PREPROCESSING_WORKERS = 2

def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.


    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    #argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
    #                  help="""This argument specifies which task to train/evaluate on.
    #    Pass "nli" for natural language inference or "qa" for question answering.
    #    By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument specifies the dataset used""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset == 'AddSent':
       dataset_train = datasets.load_dataset('squad_adversarial','AddSent',split='validation[0:2500]')
       dataset_validation = datasets.load_dataset('squad_adversarial','AddSent',split='validation[2500:]')
    else:
       dataset_train = datasets.load_dataset('squad_adversarial','AddOneSent',split='validation[0:1200]')
       dataset_validation = datasets.load_dataset('squad_adversarial','AddOneSent',split='validation[1200:]')

    train_size = len(dataset_train)
    validation_size = len(dataset_validation)

    data_train_dict = {'id':[], 'title':[], 'context':[], 'question':[], 'answers':[]}
    for i in range(train_size):
        ex = dataset_train[i]
        ex = move_to_the_front(ex)
        data_train_dict['id'].append(ex['id'])
        data_train_dict['title'].append(ex['title'])
        data_train_dict['context'].append(ex['context'])
        data_train_dict['question'].append(ex['question'])
        data_train_dict['answers'].append(ex['answers'])
    dataset_train = Dataset.from_dict(data_train_dict)

    data_validation_dict = {'id':[], 'title':[], 'context':[], 'question':[], 'answers':[]}
    for i in range(validation_size):
        ex = dataset_validation[i]
        ex = move_to_the_front(ex)
        data_validation_dict['id'].append(ex['id'])
        data_validation_dict['title'].append(ex['title'])
        data_validation_dict['context'].append(ex['context'])
        data_validation_dict['question'].append(ex['question'])
        data_validation_dict['answers'].append(ex['answers'])
    dataset_validation = Dataset.from_dict(data_validation_dict)


    # Here we select the right model fine-tuning head

    model_class = AutoModelForQuestionAnswering
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained('/content/drive/MyDrive/fp_2/output_AddSent_train',local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained('google/electra-small-discriminator', use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
    prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")

    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset_train
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        eval_dataset = dataset_validation
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None

    # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
    # to enable the question-answering specific evaluation metrics
    trainer_class = QuestionAnsweringTrainer
    eval_kwargs['eval_examples'] = eval_dataset
    metric = datasets.load_metric('squad')
    compute_metrics = lambda eval_preds: metric.compute(
        predictions=eval_preds.predictions, references=eval_preds.label_ids)

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:

            predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
            for example in eval_dataset:
                example_with_prediction = dict(example)
                example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                f.write(json.dumps(example_with_prediction))
                f.write('\n')

if __name__ == "__main__":
    main()
